#!/usr/bin/env python3
"""
Train frozen probes on optical-only latent features to measure nuisance leakage.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader


SCRIPT_DIR = Path(__file__).resolve().parent
MODEL_DIR = SCRIPT_DIR.parent / "Model"
if str(MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(MODEL_DIR))

from data_loader import OpticalBinaryDataset
from model import OpticalKNClassifier


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Probe cadence leakage from optical-only latent features.")
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--config", type=str, default=None)
    p.add_argument("--pos_data_path", type=str, default=None)
    p.add_argument("--neg_data_path", type=str, default=None)
    p.add_argument("--neg_group", type=str, default=None)
    p.add_argument("--batch_size", type=int, default=1024)
    p.add_argument("--max_pos_samples", type=int, default=20000)
    p.add_argument("--max_neg_samples", type=int, default=20000)
    p.add_argument("--sample_seed", type=int, default=42)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--output_json", type=str, default=None)
    return p.parse_args()


def load_json(path):
    if path is None:
        return {}
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data if isinstance(data, dict) else {}


def choose_value(cli_value, config_dict, ckpt_args, key, default=None):
    if cli_value is not None:
        return cli_value
    if key in config_dict and config_dict[key] is not None:
        return config_dict[key]
    if key in ckpt_args and ckpt_args[key] is not None:
        return ckpt_args[key]
    return default


def bucketize_n_det_np(n_det: np.ndarray) -> np.ndarray:
    out = np.zeros_like(n_det, dtype=np.int64)
    out[n_det == 4] = 1
    out[(n_det >= 5) & (n_det <= 6)] = 2
    out[(n_det >= 7) & (n_det <= 12)] = 3
    out[n_det >= 13] = 4
    return out


def bucketize_n_bands_np(n_bands: np.ndarray) -> np.ndarray:
    out = np.zeros_like(n_bands, dtype=np.int64)
    out[n_bands == 2] = 1
    out[n_bands == 3] = 2
    out[n_bands >= 4] = 3
    return out


def bucketize_t_span_np(t_span: np.ndarray) -> np.ndarray:
    out = np.zeros_like(t_span, dtype=np.int64)
    out[t_span > 0.01] = 1
    out[t_span > 0.05] = 2
    out[t_span > 0.2] = 3
    out[t_span > 0.6] = 4
    return out


def load_model(checkpoint_path: str, device: torch.device, config_dict: Dict[str, object]):
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    ckpt_args = ckpt.get("args", {})
    if not isinstance(ckpt_args, dict):
        ckpt_args = {}
    state_dict = ckpt.get("model_state_dict", ckpt)
    cleaned = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
    has_universal_aux = (
        bool(ckpt_args.get("universal_train_enable", False))
        or any(k.startswith("projection_head.") for k in cleaned.keys())
        or any(k.startswith("adv_head_n_det.") for k in cleaned.keys())
    )
    model = OpticalKNClassifier(
        optical_input_dim=6,
        ref_time_dim=int(choose_value(None, config_dict, ckpt_args, "ref_dim", 64)),
        enc_dim=int(choose_value(None, config_dict, ckpt_args, "enc_dim", 128)),
        opt_dropout=float(choose_value(None, config_dict, ckpt_args, "opt_dropout", 0.1)),
        feature_dropout=float(choose_value(None, config_dict, ckpt_args, "feature_dropout", 0.0)),
        head_hidden_dim=choose_value(None, config_dict, ckpt_args, "head_hidden_dim", None),
        head_dropout=float(choose_value(None, config_dict, ckpt_args, "head_dropout", 0.2)),
        universal_aux_enable=bool(has_universal_aux),
        proj_dim=64,
        adv_hidden_dim=int(
            choose_value(None, config_dict, ckpt_args, "head_hidden_dim", None)
            or choose_value(None, config_dict, ckpt_args, "enc_dim", 128)
        ),
    )
    model.load_state_dict(cleaned, strict=True)
    model.to(device)
    model.eval()
    return model, ckpt_args


def build_ref_time(batch_size, n_ref, ref_start, ref_end, device, dtype):
    ref = torch.linspace(ref_start, ref_end, n_ref, dtype=dtype, device=device)
    return ref.unsqueeze(0).repeat(batch_size, 1)


def main() -> None:
    args = parse_args()
    config_dict = load_json(args.config)
    requested_device = args.device
    if requested_device == "cuda" and not torch.cuda.is_available():
        device = torch.device("cpu")
    else:
        device = torch.device(requested_device)

    model, ckpt_args = load_model(args.checkpoint, device, config_dict)
    pos_data_path = choose_value(args.pos_data_path, config_dict, ckpt_args, "eval_pos_data_path", None)
    neg_data_path = choose_value(args.neg_data_path, config_dict, ckpt_args, "eval_neg_data_path", None)
    if pos_data_path is None:
        pos_data_path = choose_value(args.pos_data_path, config_dict, ckpt_args, "pos_data_path", None)
    if neg_data_path is None:
        neg_data_path = choose_value(args.neg_data_path, config_dict, ckpt_args, "neg_data_path", None)
    neg_group = choose_value(args.neg_group, config_dict, ckpt_args, "eval_neg_group", None)
    if neg_group is None:
        neg_group = choose_value(args.neg_group, config_dict, ckpt_args, "neg_group", "Tutorial/optical_data")

    base = OpticalBinaryDataset(
        pos_h5_path=pos_data_path,
        neg_h5_path=neg_data_path,
        neg_group=neg_group,
        cache_in_memory=False,
        load_meta_features=True,
        return_prefix_aux=True,
    )
    rng = np.random.default_rng(int(args.sample_seed))
    pos_idx = np.arange(base.n_pos, dtype=np.int64)
    neg_idx = np.arange(base.n_neg, dtype=np.int64)
    if args.max_pos_samples is not None and args.max_pos_samples < len(pos_idx):
        pos_idx = np.sort(rng.choice(pos_idx, size=int(args.max_pos_samples), replace=False))
    if args.max_neg_samples is not None and args.max_neg_samples < len(neg_idx):
        neg_idx = np.sort(rng.choice(neg_idx, size=int(args.max_neg_samples), replace=False))
    dataset = OpticalBinaryDataset(
        pos_h5_path=pos_data_path,
        neg_h5_path=neg_data_path,
        neg_group=neg_group,
        pos_indices=pos_idx,
        neg_indices=neg_idx,
        cache_in_memory=False,
        load_meta_features=True,
        return_prefix_aux=True,
    )
    loader = DataLoader(dataset, batch_size=int(args.batch_size), shuffle=False, num_workers=0)

    n_ref = int(choose_value(None, config_dict, ckpt_args, "n_ref", 64))
    ref_start = float(choose_value(None, config_dict, ckpt_args, "ref_start", -0.3))
    ref_end = float(choose_value(None, config_dict, ckpt_args, "ref_end", 0.6))

    feats = []
    bucket_det = []
    bucket_band = []
    bucket_span = []
    with torch.no_grad():
        for batch in loader:
            opt_t, opt_v, opt_mask, opt_err, labels, slot_is_detection = batch
            opt_t = opt_t.to(device)
            opt_v = opt_v.to(device)
            opt_mask = opt_mask.to(device)
            opt_err = opt_err.to(device)
            slot_is_detection = slot_is_detection.to(device)
            ref_time = build_ref_time(opt_t.size(0), n_ref, ref_start, ref_end, device, opt_t.dtype)
            out = model.forward_with_aux(opt_t, opt_v, ref_time, opt_mask, opt_err, return_aux=True)
            feat = out["joint_feat"].detach().cpu().numpy()
            valid_rows = (opt_mask.sum(dim=-1) > 0)
            n_det = ((slot_is_detection > 0) & valid_rows).sum(dim=1).detach().cpu().numpy()
            n_bands = (opt_mask.sum(dim=1) > 0).sum(dim=1).detach().cpu().numpy()
            t_np = opt_t.detach().cpu().numpy()
            valid_np = valid_rows.detach().cpu().numpy()
            t_span = np.zeros((t_np.shape[0],), dtype=np.float32)
            for i in range(t_np.shape[0]):
                if valid_np[i].any():
                    vals = t_np[i][valid_np[i]]
                    t_span[i] = float(np.max(vals) - np.min(vals))
            feats.append(feat)
            bucket_det.append(bucketize_n_det_np(n_det))
            bucket_band.append(bucketize_n_bands_np(n_bands))
            bucket_span.append(bucketize_t_span_np(t_span))

    X = np.concatenate(feats, axis=0)
    y_det = np.concatenate(bucket_det, axis=0)
    y_band = np.concatenate(bucket_band, axis=0)
    y_span = np.concatenate(bucket_span, axis=0)

    results = {}
    for name, y in [("n_det_bucket", y_det), ("n_bands_bucket", y_band), ("t_span_bucket", y_span)]:
        if np.unique(y).size < 2:
            results[name] = {"accuracy": 1.0, "n_classes": int(np.unique(y).size)}
            continue
        counts = np.bincount(y.astype(np.int64))
        stratify = y if np.all(counts[counts > 0] >= 2) else None
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=42, stratify=stratify)
        clf = LogisticRegression(max_iter=1000, multi_class="auto")
        clf.fit(X_tr, y_tr)
        pred = clf.predict(X_te)
        results[name] = {
            "accuracy": float(accuracy_score(y_te, pred)),
            "n_classes": int(np.unique(y).size),
        }

    payload = {
        "checkpoint": args.checkpoint,
        "n_samples": int(X.shape[0]),
        "results": results,
    }
    if args.output_json:
        Path(args.output_json).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
