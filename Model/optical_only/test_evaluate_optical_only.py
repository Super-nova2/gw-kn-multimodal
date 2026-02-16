#!/usr/bin/env python3
"""
Evaluation script for optical-only KN classifier.

Notes:
  - By default this can evaluate on the same dataset used in training
    (data leakage may exist; intentionally ignored for now).
  - Metrics follow the project classification utilities (AUROC, AUPRC, F1, ECE).
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.amp import autocast
from torch.utils.data import DataLoader
from tqdm import tqdm


SCRIPT_DIR = Path(__file__).resolve().parent
MODEL_DIR = SCRIPT_DIR.parent
if str(MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(MODEL_DIR))

from data_loader import OpticalBinaryDataset
from metrics import compute_classification_metrics
from model import OpticalKNClassifier


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate optical-only KN classifier")
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--config", type=str, default=None, help="Optional JSON config for path fallback.")

    p.add_argument("--pos_data_path", type=str, default="/fred/oz016/bgao_kn/data/LSST_KN_BNS/combined_dataset_with_neg_gw.h5")
    p.add_argument("--neg_data_path", type=str, default="/fred/oz016/bgao_kn/data/ELASTICC2_TRAIN/negative_dataset.h5")
    p.add_argument("--neg_group", type=str, default="ELASTICC2_TRAIN/optical_data")

    p.add_argument("--batch_size", type=int, default=1024)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--pin_memory", type=int, default=1)
    p.add_argument("--persistent_workers", type=int, default=0)
    p.add_argument("--prefetch_factor", type=int, default=2)

    p.add_argument("--max_pos_samples", type=int, default=100000)
    p.add_argument("--max_neg_samples", type=int, default=100000)
    p.add_argument("--sample_seed", type=int, default=42)

    p.add_argument("--target_recall", type=float, default=None)
    p.add_argument("--output_dir", type=str, default="eval_results/optical_only")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--no_plots", action="store_true")
    return p.parse_args()


def load_json(path):
    if path is None:
        return {}
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Config must be JSON object: {path}")
    return data


def choose_value(cli_value, config_dict, ckpt_args, key, default=None):
    if cli_value is not None:
        return cli_value
    if key in config_dict and config_dict[key] is not None:
        return config_dict[key]
    if key in ckpt_args and ckpt_args[key] is not None:
        return ckpt_args[key]
    return default


def build_ref_time(batch_size, n_ref, ref_start, ref_end, device, dtype):
    ref = torch.linspace(ref_start, ref_end, n_ref, dtype=dtype, device=device)
    return ref.unsqueeze(0).repeat(batch_size, 1)


def select_threshold_for_target_recall(probs, labels, target_recall=0.98):
    thresholds = torch.linspace(1.0, 0.0, 1001, device=probs.device)
    best = None
    fallback = None

    for thr in thresholds:
        preds = probs >= thr
        tp = ((preds == 1) & (labels == 1)).sum().item()
        fp = ((preds == 1) & (labels == 0)).sum().item()
        tn = ((preds == 0) & (labels == 0)).sum().item()
        fn = ((preds == 0) & (labels == 1)).sum().item()

        recall = tp / max(1, tp + fn)
        precision = tp / max(1, tp + fp)
        fpr = fp / max(1, fp + tn)

        candidate = {
            "threshold": float(thr.item()),
            "recall": float(recall),
            "precision": float(precision),
            "fpr": float(fpr),
            "tp": int(tp),
            "fp": int(fp),
            "tn": int(tn),
            "fn": int(fn),
        }

        if fallback is None or candidate["recall"] > fallback["recall"] or (
            abs(candidate["recall"] - fallback["recall"]) < 1e-12
            and candidate["precision"] > fallback["precision"]
        ):
            fallback = candidate

        if recall >= target_recall:
            best = candidate
            break

    if best is None:
        best = fallback
        best["meets_target_recall"] = False
    else:
        best["meets_target_recall"] = True
    best["target_recall"] = float(target_recall)
    return best


def load_model(checkpoint_path, device, config_dict):
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    ckpt_args = ckpt.get("args", {})
    if not isinstance(ckpt_args, dict):
        ckpt_args = {}

    def _get(key, default):
        if key in ckpt_args and ckpt_args[key] is not None:
            return ckpt_args[key]
        if key in config_dict and config_dict[key] is not None:
            return config_dict[key]
        return default

    model = OpticalKNClassifier(
        optical_input_dim=6,
        ref_time_dim=_get("ref_dim", 64),
        enc_dim=_get("enc_dim", 64),
        opt_dropout=_get("opt_dropout", 0.1),
        feature_dropout=_get("feature_dropout", 0.0),
        head_hidden_dim=_get("head_hidden_dim", None),
        head_dropout=_get("head_dropout", 0.2),
        include_coords=bool(_get("include_coords", False)),
    )

    state_dict = ckpt.get("model_state_dict", ckpt)
    if not isinstance(state_dict, dict):
        raise ValueError("Checkpoint does not contain valid state_dict.")
    state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
    missing, unexpected = model.load_state_dict(state_dict, strict=False)

    model.to(device)
    model.eval()
    print(
        f"Loaded checkpoint: {checkpoint_path} | epoch={ckpt.get('epoch', '?')} "
        f"| missing={len(missing)} unexpected={len(unexpected)}"
    )
    return model, ckpt_args


def build_eval_dataset(args, ckpt_args, config_dict):
    pos_data_path = choose_value(args.pos_data_path, config_dict, ckpt_args, "pos_data_path", default=None)
    neg_data_path = choose_value(args.neg_data_path, config_dict, ckpt_args, "neg_data_path", default=None)
    neg_group = choose_value(
        args.neg_group, config_dict, ckpt_args, "neg_group", default="ELASTICC2_TRAIN/optical_data"
    )
    include_coords = bool(choose_value(None, config_dict, ckpt_args, "include_coords", default=False))

    if pos_data_path is None or neg_data_path is None:
        raise ValueError(
            "pos_data_path/neg_data_path missing. Provide via CLI or --config, "
            "or ensure they are stored in checkpoint args."
        )
    if not os.path.exists(pos_data_path):
        raise FileNotFoundError(f"Positive data file not found: {pos_data_path}")
    if not os.path.exists(neg_data_path):
        raise FileNotFoundError(f"Negative data file not found: {neg_data_path}")

    base_dataset = OpticalBinaryDataset(
        pos_h5_path=pos_data_path,
        neg_h5_path=neg_data_path,
        neg_group=neg_group,
        include_coords=include_coords,
        cache_in_memory=False,
    )

    pos_indices = np.arange(base_dataset.n_pos, dtype=np.int64)
    neg_indices = np.arange(base_dataset.n_neg, dtype=np.int64)
    rng = np.random.default_rng(args.sample_seed)

    if args.max_pos_samples is not None and args.max_pos_samples < len(pos_indices):
        pos_indices = rng.choice(pos_indices, size=int(args.max_pos_samples), replace=False)
        pos_indices = np.sort(pos_indices)
    if args.max_neg_samples is not None and args.max_neg_samples < len(neg_indices):
        neg_indices = rng.choice(neg_indices, size=int(args.max_neg_samples), replace=False)
        neg_indices = np.sort(neg_indices)

    dataset = OpticalBinaryDataset(
        pos_h5_path=pos_data_path,
        neg_h5_path=neg_data_path,
        neg_group=neg_group,
        pos_indices=pos_indices,
        neg_indices=neg_indices,
        include_coords=include_coords,
        cache_in_memory=False,
    )

    print(
        "Evaluation dataset: "
        f"n_pos={dataset.n_pos}, n_neg={dataset.n_neg}, total={len(dataset)} | "
        f"same source as training may cause leakage (ignored by request)."
    )
    return dataset, {
        "pos_data_path": pos_data_path,
        "neg_data_path": neg_data_path,
        "neg_group": neg_group,
        "include_coords": include_coords,
    }


def build_eval_loader(dataset, args):
    if args.num_workers > 0:
        return DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=bool(args.pin_memory),
            persistent_workers=bool(args.persistent_workers),
            prefetch_factor=args.prefetch_factor,
        )
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=bool(args.pin_memory),
    )


@torch.no_grad()
def run_evaluation(model, loader, device, n_ref, ref_start, ref_end, target_recall):
    criterion = nn.BCEWithLogitsLoss()

    if device.type == "cuda":
        use_bf16 = torch.cuda.is_bf16_supported()
        amp_dtype = torch.bfloat16 if use_bf16 else torch.float16
    else:
        amp_dtype = torch.float32

    all_probs = []
    all_labels = []
    total_loss = 0.0
    n_batches = 0
    ref_time_cache = None

    for batch in tqdm(loader, desc="Evaluating"):
        opt_t, opt_v, opt_mask, opt_err, opt_coords, labels = batch
        opt_t = opt_t.to(device, non_blocking=True)
        opt_v = opt_v.to(device, non_blocking=True)
        opt_mask = opt_mask.to(device, non_blocking=True)
        opt_err = opt_err.to(device, non_blocking=True)
        opt_coords = opt_coords.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True).float()

        batch_size = opt_t.size(0)
        if (
            ref_time_cache is None
            or ref_time_cache.shape[0] != batch_size
            or ref_time_cache.dtype != opt_t.dtype
        ):
            ref_time_cache = build_ref_time(batch_size, n_ref, ref_start, ref_end, device, opt_t.dtype)

        with autocast(device_type="cuda", dtype=amp_dtype, enabled=(device.type == "cuda")):
            logits = model(opt_coords, opt_t, opt_v, ref_time_cache, opt_mask, opt_err).squeeze(-1)
            loss = criterion(logits, labels)

        probs = torch.sigmoid(logits.float())
        total_loss += loss.item()
        n_batches += 1
        all_probs.append(probs.detach().cpu())
        all_labels.append(labels.detach().cpu().long())

    if n_batches == 0:
        raise RuntimeError("No batches were evaluated.")

    probs = torch.cat(all_probs, dim=0)
    labels = torch.cat(all_labels, dim=0)
    cls = compute_classification_metrics(probs, labels)
    op = select_threshold_for_target_recall(probs, labels, target_recall=target_recall)

    results = {
        "loss": float(total_loss / n_batches),
        **cls,
        "op_threshold": op["threshold"],
        "op_recall": op["recall"],
        "op_precision": op["precision"],
        "op_fpr": op["fpr"],
        "op_meets_target_recall": bool(op["meets_target_recall"]),
        "target_recall": float(target_recall),
    }
    return results, probs.numpy(), labels.numpy()


def generate_plots(probs, labels, results, output_dir):
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available, skipping plots")
        return

    os.makedirs(output_dir, exist_ok=True)
    probs = np.asarray(probs, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)

    # ROC
    sorted_idx = np.argsort(-probs)
    sorted_labels = labels[sorted_idx]
    n_pos = int(sorted_labels.sum())
    n_neg = int(len(sorted_labels) - n_pos)
    if n_pos > 0 and n_neg > 0:
        tpr = np.cumsum(sorted_labels) / n_pos
        fpr = np.cumsum(1 - sorted_labels) / n_neg
        tpr = np.concatenate([[0.0], tpr])
        fpr = np.concatenate([[0.0], fpr])
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.plot(fpr, tpr, lw=2)
        ax.plot([0, 1], [0, 1], "k--", lw=1)
        ax.set_title(f"ROC Curve (AUROC={results.get('auroc', 0.0):.4f})")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        fig.savefig(os.path.join(output_dir, "roc_curve.png"), dpi=150, bbox_inches="tight")
        plt.close(fig)

    # PR
    if n_pos > 0:
        tp_cum = np.cumsum(sorted_labels)
        ranks = np.arange(1, len(sorted_labels) + 1)
        precision = tp_cum / np.maximum(ranks, 1)
        recall = tp_cum / max(n_pos, 1)
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.plot(recall, precision, lw=2)
        ax.set_title(f"Precision-Recall Curve (AUPRC={results.get('auprc', 0.0):.4f})")
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        fig.savefig(os.path.join(output_dir, "pr_curve.png"), dpi=150, bbox_inches="tight")
        plt.close(fig)

    # Calibration
    n_bins = 10
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_accs = []
    bin_confs = []
    for i in range(n_bins):
        low, high = edges[i], edges[i + 1]
        if i < n_bins - 1:
            mask = (probs >= low) & (probs < high)
        else:
            mask = (probs >= low) & (probs <= high)
        if mask.any():
            bin_accs.append(labels[mask].mean())
            bin_confs.append(probs[mask].mean())
    if bin_accs:
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.bar(bin_confs, bin_accs, width=0.08, alpha=0.7, label="Model")
        ax.plot([0, 1], [0, 1], "k--", lw=1, label="Perfect")
        ax.set_title(f"Calibration (ECE={results.get('ece', 0.0):.4f})")
        ax.set_xlabel("Mean Predicted Probability")
        ax.set_ylabel("Fraction of Positives")
        ax.legend()
        fig.savefig(os.path.join(output_dir, "calibration.png"), dpi=150, bbox_inches="tight")
        plt.close(fig)

    # Logits/probability distribution (NO BOTTOM FILL)
    pos_probs = probs[labels == 1]
    neg_probs = probs[labels == 0]
    fig, ax = plt.subplots(figsize=(10, 6))
    bins = np.linspace(0.0, 1.0, 61)
    if len(pos_probs) > 0:
        ax.hist(
            pos_probs,
            bins=bins,
            histtype="step",
            linewidth=2.0,
            color="#27ae60",
            label=f"Positive (KN) n={len(pos_probs)}",
        )
    if len(neg_probs) > 0:
        ax.hist(
            neg_probs,
            bins=bins,
            histtype="step",
            linewidth=2.0,
            color="#c0392b",
            label=f"Negative (non-KN) n={len(neg_probs)}",
        )
    op_thr = float(results.get("op_threshold", 0.5))
    ax.axvline(0.5, color="black", linestyle="--", linewidth=1.0, alpha=0.6, label="thr=0.5")
    ax.axvline(op_thr, color="#34495e", linestyle="-.", linewidth=1.2, alpha=0.8, label=f"op_thr={op_thr:.3f}")
    ax.set_title("Optical-only Match Probability Distribution")
    ax.set_xlabel("Predicted Probability")
    ax.set_ylabel("Count")
    ax.set_xlim(0.0, 1.0)
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.legend()
    fig.savefig(os.path.join(output_dir, "prob_distribution.png"), dpi=180, bbox_inches="tight")
    plt.close(fig)

    print(f"Plots saved to {output_dir}/")


def main():
    args = parse_args()
    config_dict = load_json(args.config)

    requested_device = args.device
    if requested_device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU.")
        device = torch.device("cpu")
    else:
        device = torch.device(requested_device)
    print(f"Using device: {device}")

    model, ckpt_args = load_model(args.checkpoint, device, config_dict)
    dataset, data_meta = build_eval_dataset(args, ckpt_args, config_dict)
    loader = build_eval_loader(dataset, args)

    n_ref = int(choose_value(None, config_dict, ckpt_args, "n_ref", default=64))
    ref_start = float(choose_value(None, config_dict, ckpt_args, "ref_start", default=-0.3))
    ref_end = float(choose_value(None, config_dict, ckpt_args, "ref_end", default=0.6))
    target_recall = float(
        choose_value(args.target_recall, config_dict, ckpt_args, "target_recall", default=0.98)
    )

    metrics, probs, labels = run_evaluation(
        model=model,
        loader=loader,
        device=device,
        n_ref=n_ref,
        ref_start=ref_start,
        ref_end=ref_end,
        target_recall=target_recall,
    )

    os.makedirs(args.output_dir, exist_ok=True)
    results = {
        "classification": metrics,
        "meta": {
            "checkpoint": args.checkpoint,
            "device": str(device),
            "n_samples": int(len(labels)),
            "n_pos": int((labels == 1).sum()),
            "n_neg": int((labels == 0).sum()),
            "n_ref": n_ref,
            "ref_start": ref_start,
            "ref_end": ref_end,
            "batch_size": int(args.batch_size),
            **data_meta,
        },
    }

    out_json = os.path.join(args.output_dir, "eval_results.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("\n=== Optical-only Evaluation Summary ===")
    print(f"loss: {metrics['loss']:.4f}")
    print(f"AUROC: {metrics.get('auroc', 0.0):.4f}")
    print(f"AUPRC: {metrics.get('auprc', 0.0):.4f}")
    print(f"F1(opt): {metrics.get('f1_optimal', 0.0):.4f}")
    print(
        f"op@recall>={metrics['target_recall']:.3f}: "
        f"thr={metrics['op_threshold']:.3f}, "
        f"recall={metrics['op_recall']:.4f}, "
        f"precision={metrics['op_precision']:.4f}, "
        f"fpr={metrics['op_fpr']:.4f}"
    )
    print(f"Saved: {out_json}")

    if not args.no_plots:
        generate_plots(probs, labels, metrics, args.output_dir)


if __name__ == "__main__":
    main()
