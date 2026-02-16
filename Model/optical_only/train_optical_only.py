#!/usr/bin/env python3
"""
Train an optical-only KN classifier using the existing optical encoder.

Two-stage schedule:
1) Freeze optical encoder, train new classifier head.
2) Unfreeze optical encoder and fine-tune with a smaller encoder LR.
"""

import argparse
import datetime
import gc
import json
import os
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
try:
    from torch.utils.tensorboard import SummaryWriter
except Exception:  # pragma: no cover - runtime environment dependent
    SummaryWriter = None
from tqdm import tqdm


SCRIPT_DIR = Path(__file__).resolve().parent
MODEL_DIR = SCRIPT_DIR.parent
if str(MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(MODEL_DIR))

from data_loader import create_optical_binary_dataloaders
from metrics import compute_classification_metrics
from model import OpticalKNClassifier


def build_ref_time(batch_size, n_ref, ref_start, ref_end, device, dtype):
    ref = torch.linspace(ref_start, ref_end, n_ref, dtype=dtype, device=device)
    return ref.unsqueeze(0).repeat(batch_size, 1)


def augment_optical_data(
    opt_t,
    opt_v,
    opt_mask,
    opt_err,
    training=True,
    time_jitter=0.0,
    flux_noise=0.0,
    obs_dropout=0.0,
    band_dropout=0.0,
):
    if not training:
        return opt_t, opt_v, opt_mask, opt_err

    if time_jitter > 0:
        time_mask = (opt_mask.sum(dim=-1) > 0).float()
        opt_t = opt_t + torch.randn_like(opt_t) * time_jitter * time_mask

    if flux_noise > 0:
        if opt_err is not None:
            noise = torch.randn_like(opt_v) * (opt_err * flux_noise)
        else:
            noise = torch.randn_like(opt_v) * flux_noise
        opt_v = opt_v + noise * opt_mask

    if obs_dropout > 0:
        drop_mask = (torch.rand_like(opt_mask) < obs_dropout) & (opt_mask > 0)
        if drop_mask.any():
            opt_mask = opt_mask.masked_fill(drop_mask, 0)
            opt_v = opt_v.masked_fill(drop_mask, 0.0)
            if opt_err is not None:
                opt_err = opt_err.masked_fill(drop_mask, 0.0)

    if band_dropout > 0:
        band_mask = torch.rand(opt_v.size(0), opt_v.size(2), device=opt_v.device) < band_dropout
        if band_mask.any():
            band_mask = band_mask[:, None, :]
            opt_mask = opt_mask.masked_fill(band_mask, 0)
            opt_v = opt_v.masked_fill(band_mask, 0.0)
            if opt_err is not None:
                opt_err = opt_err.masked_fill(band_mask, 0.0)

    return opt_t, opt_v, opt_mask, opt_err


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


def run_eval(model, loader, device, args, criterion, amp_dtype):
    model.eval()
    losses = 0.0
    n_batches = 0
    all_probs = []
    all_labels = []
    ref_time_cache = None

    with torch.no_grad():
        for batch in loader:
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
                ref_time_cache = build_ref_time(
                    batch_size, args.n_ref, args.ref_start, args.ref_end, device, opt_t.dtype
                )

            with autocast(device_type="cuda", dtype=amp_dtype, enabled=(device.type == "cuda")):
                logits = model(opt_coords, opt_t, opt_v, ref_time_cache, opt_mask, opt_err).squeeze(-1)
                loss = criterion(logits, labels)

            probs = torch.sigmoid(logits.float())
            losses += loss.item()
            n_batches += 1
            all_probs.append(probs.detach().cpu())
            all_labels.append(labels.detach().cpu().long())

    if n_batches == 0:
        model.train()
        return None

    probs = torch.cat(all_probs, dim=0)
    labels = torch.cat(all_labels, dim=0)
    cls_metrics = compute_classification_metrics(probs, labels)
    op = select_threshold_for_target_recall(probs, labels, target_recall=args.target_recall)

    model.train()
    return {
        "loss": losses / n_batches,
        "auroc": float(cls_metrics.get("auroc", 0.0)),
        "auprc": float(cls_metrics.get("auprc", 0.0)),
        "f1_optimal": float(cls_metrics.get("f1_optimal", 0.0)),
        "op_threshold": op["threshold"],
        "op_recall": op["recall"],
        "op_precision": op["precision"],
        "op_fpr": op["fpr"],
        "op_meets_target_recall": bool(op["meets_target_recall"]),
    }


def train_one_epoch(model, loader, optimizer, device, args, criterion, scaler, amp_dtype):
    model.train()
    total_loss = 0.0
    n_batches = 0
    ref_time_cache = None

    pbar = tqdm(loader, desc="Train", mininterval=0.0, miniters=100)
    for batch in pbar:
        opt_t, opt_v, opt_mask, opt_err, opt_coords, labels = batch
        opt_t = opt_t.to(device, non_blocking=True)
        opt_v = opt_v.to(device, non_blocking=True)
        opt_mask = opt_mask.to(device, non_blocking=True)
        opt_err = opt_err.to(device, non_blocking=True)
        opt_coords = opt_coords.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True).float()

        opt_t, opt_v, opt_mask, opt_err = augment_optical_data(
            opt_t,
            opt_v,
            opt_mask,
            opt_err,
            training=True,
            time_jitter=args.opt_aug_time_jitter,
            flux_noise=args.opt_aug_noise,
            obs_dropout=args.opt_aug_dropout,
            band_dropout=args.opt_aug_band_dropout,
        )

        batch_size = opt_t.size(0)
        if (
            ref_time_cache is None
            or ref_time_cache.shape[0] != batch_size
            or ref_time_cache.dtype != opt_t.dtype
        ):
            ref_time_cache = build_ref_time(
                batch_size, args.n_ref, args.ref_start, args.ref_end, device, opt_t.dtype
            )

        optimizer.zero_grad(set_to_none=True)
        with autocast(device_type="cuda", dtype=amp_dtype, enabled=(device.type == "cuda")):
            logits = model(opt_coords, opt_t, opt_v, ref_time_cache, opt_mask, opt_err).squeeze(-1)
            loss = criterion(logits, labels)

        scaler.scale(loss).backward()
        if args.grad_clip_norm > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.grad_clip_norm)
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        n_batches += 1
        if n_batches % 50 == 0:
            pbar.set_postfix({"loss": f"{(total_loss / n_batches):.4f}"})

    pbar.close()
    return total_loss / max(1, n_batches)


def save_checkpoint(path, model, optimizer, epoch, args, val_metrics):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(
        {
            "epoch": int(epoch),
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
            "val_metrics": val_metrics,
            "args": vars(args),
        },
        path,
    )


def init_tensorboard_writer(args, save_root):
    if bool(args.disable_tensorboard):
        print("TensorBoard logging disabled by --disable_tensorboard.")
        return None

    if SummaryWriter is None:
        print("TensorBoard is unavailable (missing tensorboard package). Skipping TB logging.")
        return None

    log_dir = Path(args.tb_log_dir) if args.tb_log_dir else (save_root / "tb_logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(log_dir), flush_secs=int(args.tb_flush_secs))
    writer.add_text("run/config", json.dumps(vars(args), indent=2), global_step=0)
    print(f"TensorBoard log dir: {log_dir}")
    return writer


def build_stage_optimizer(model, args, freeze_encoder):
    if freeze_encoder:
        model.set_encoder_trainable(False)
        return torch.optim.AdamW(
            model.classifier.parameters(),
            lr=args.lr_head_stage1,
            weight_decay=args.weight_decay,
        )

    model.set_encoder_trainable(True)
    return torch.optim.AdamW(
        [
            {"params": model.classifier.parameters(), "lr": args.lr_head_stage2},
            {"params": model.optical_encoder.parameters(), "lr": args.lr_encoder_stage2},
        ],
        weight_decay=args.weight_decay,
    )


def train(args):
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        use_bf16 = torch.cuda.is_bf16_supported()
        amp_dtype = torch.bfloat16 if use_bf16 else torch.float16
        scaler = GradScaler(enabled=(not use_bf16))
        print(f"Running on device: {device} | AMP dtype={amp_dtype}")
    else:
        amp_dtype = torch.float32
        scaler = GradScaler(enabled=False)
        print(f"Running on device: {device} | No AMP (CPU)")

    train_loader, val_loader, steps_per_epoch, val_steps = create_optical_binary_dataloaders(
        pos_h5_path=args.pos_data_path,
        neg_h5_path=args.neg_data_path,
        neg_group=args.neg_group,
        batch_size=args.batch_size,
        val_batch_size=args.val_batch_size,
        steps_per_epoch=args.steps_per_epoch,
        val_steps_per_epoch=args.val_steps_per_epoch,
        val_split=args.val_split,
        split_seed=args.split_seed,
        num_workers=args.num_workers,
        pin_memory=bool(args.pin_memory),
        persistent_workers=bool(args.persistent_workers),
        prefetch_factor=args.prefetch_factor,
        include_coords=bool(args.include_coords),
        cache_in_memory=bool(args.cache_in_memory),
    )
    print(f"Train Steps/Epoch: {steps_per_epoch} | Val Steps/Epoch: {val_steps}")

    model = OpticalKNClassifier(
        optical_input_dim=6,
        ref_time_dim=args.ref_dim,
        enc_dim=args.enc_dim,
        opt_dropout=args.opt_dropout,
        feature_dropout=args.feature_dropout,
        head_hidden_dim=args.head_hidden_dim,
        head_dropout=args.head_dropout,
        include_coords=bool(args.include_coords),
    ).to(device)

    if args.pretrained_albef_ckpt:
        if not os.path.exists(args.pretrained_albef_ckpt):
            raise FileNotFoundError(f"Pretrained checkpoint not found: {args.pretrained_albef_ckpt}")
        ckpt = torch.load(args.pretrained_albef_ckpt, map_location=device)
        state_dict = ckpt.get("model_state_dict", ckpt)
        missing, unexpected = model.load_optical_encoder_from_albef_state_dict(state_dict, strict=False)
        print(
            "Loaded optical encoder from ALBEF checkpoint "
            f"(missing={len(missing)}, unexpected={len(unexpected)})."
        )

    criterion = nn.BCEWithLogitsLoss()

    stage_plan = [
        ("stage1_head_only", int(args.epochs_stage1), True),
        ("stage2_finetune", int(args.epochs_stage2), False),
    ]

    best_score = float("-inf")
    best_metrics = {}
    best_epoch = -1
    global_epoch = 0
    no_improve = 0

    save_root = Path(args.ckpt_path) / "optical_only"
    save_root.mkdir(parents=True, exist_ok=True)
    tb_writer = init_tensorboard_writer(args, save_root)

    optimizer = None
    for stage_name, stage_epochs, freeze_encoder in stage_plan:
        if stage_epochs <= 0:
            continue

        optimizer = build_stage_optimizer(model, args, freeze_encoder=freeze_encoder)
        if stage_name == "stage2_finetune":
            no_improve = 0

        print(
            f"\n[{stage_name}] epochs={stage_epochs} freeze_encoder={freeze_encoder} "
            f"lr_head={optimizer.param_groups[0]['lr']}"
        )

        for _ in range(stage_epochs):
            global_epoch += 1
            train_loss = train_one_epoch(
                model, train_loader, optimizer, device, args, criterion, scaler, amp_dtype
            )
            val_metrics = run_eval(model, val_loader, device, args, criterion, amp_dtype)
            if val_metrics is None:
                continue

            if tb_writer is not None:
                tb_writer.add_scalar("train/loss", float(train_loss), global_epoch)
                tb_writer.add_scalar("train/lr_head", float(optimizer.param_groups[0]["lr"]), global_epoch)
                if len(optimizer.param_groups) > 1:
                    tb_writer.add_scalar("train/lr_encoder", float(optimizer.param_groups[1]["lr"]), global_epoch)
                tb_writer.add_scalar(
                    "train/stage_id",
                    1.0 if stage_name == "stage1_head_only" else 2.0,
                    global_epoch,
                )

                tb_writer.add_scalar("val/loss", float(val_metrics["loss"]), global_epoch)
                tb_writer.add_scalar("val/auroc", float(val_metrics["auroc"]), global_epoch)
                tb_writer.add_scalar("val/auprc", float(val_metrics["auprc"]), global_epoch)
                tb_writer.add_scalar("val/f1_optimal", float(val_metrics["f1_optimal"]), global_epoch)
                tb_writer.add_scalar("val/op_threshold", float(val_metrics["op_threshold"]), global_epoch)
                tb_writer.add_scalar("val/op_recall", float(val_metrics["op_recall"]), global_epoch)
                tb_writer.add_scalar("val/op_precision", float(val_metrics["op_precision"]), global_epoch)
                tb_writer.add_scalar("val/op_fpr", float(val_metrics["op_fpr"]), global_epoch)
                tb_writer.add_scalar(
                    "val/op_meets_target_recall",
                    1.0 if val_metrics["op_meets_target_recall"] else 0.0,
                    global_epoch,
                )

            score = float(val_metrics["auroc"] + val_metrics["auprc"])
            if tb_writer is not None:
                tb_writer.add_scalar("val/score_auroc_plus_auprc", score, global_epoch)
            improved = score > (best_score + args.early_stop_min_delta)

            print(
                f"Epoch {global_epoch} | stage={stage_name} "
                f"| train_loss={train_loss:.4f} | val_loss={val_metrics['loss']:.4f} "
                f"| AUROC={val_metrics['auroc']:.4f} | AUPRC={val_metrics['auprc']:.4f} "
                f"| thr={val_metrics['op_threshold']:.3f} "
                f"| recall={val_metrics['op_recall']:.4f} "
                f"| precision={val_metrics['op_precision']:.4f} "
                f"| fpr={val_metrics['op_fpr']:.4f}"
            )

            if improved:
                best_score = score
                best_epoch = global_epoch
                best_metrics = {
                    "stage": stage_name,
                    "epoch": global_epoch,
                    "train_loss": train_loss,
                    **val_metrics,
                }
                save_checkpoint(
                    str(save_root / "optical_only_best.pth"),
                    model,
                    optimizer,
                    global_epoch,
                    args,
                    best_metrics,
                )
                no_improve = 0
                print(
                    f"Saved best checkpoint: {save_root / 'optical_only_best.pth'} "
                    f"(auroc+auprc={best_score:.4f})"
                )
                if tb_writer is not None:
                    tb_writer.add_scalar("best/score_auroc_plus_auprc", float(best_score), global_epoch)
                    tb_writer.add_scalar("best/op_precision", float(val_metrics["op_precision"]), global_epoch)
                    tb_writer.add_scalar("best/auroc", float(val_metrics["auroc"]), global_epoch)
                    tb_writer.add_scalar("best/auprc", float(val_metrics["auprc"]), global_epoch)
                    tb_writer.add_scalar("best/epoch", float(best_epoch), global_epoch)
            else:
                if stage_name == "stage2_finetune":
                    no_improve += 1
                    if args.early_stop_patience > 0 and no_improve >= args.early_stop_patience:
                        print("Early stopping triggered in stage2.")
                        break

            if device.type == "cuda":
                torch.cuda.empty_cache()
            gc.collect()

        if stage_name == "stage2_finetune" and args.early_stop_patience > 0 and no_improve >= args.early_stop_patience:
            break

    final_metrics = {
        "best_epoch": best_epoch,
        "best_auroc_plus_auprc": best_score,
        "best_precision_at_target_recall": float(best_metrics.get("op_precision", 0.0)) if best_metrics else 0.0,
        **best_metrics,
    }
    with open(save_root / "train_summary.json", "w") as f:
        json.dump(final_metrics, f, indent=2)

    if optimizer is not None:
        save_checkpoint(
            str(save_root / "optical_only_last.pth"),
            model,
            optimizer,
            global_epoch,
            args,
            final_metrics,
        )

    if tb_writer is not None:
        tb_writer.add_text("run/final_metrics", json.dumps(final_metrics, indent=2), global_step=global_epoch)
        tb_writer.flush()
        tb_writer.close()

    print("\nTraining complete.")
    print(f"Best epoch: {best_epoch}")
    print(f"Summary: {save_root / 'train_summary.json'}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None, help="Optional JSON config file.")
    parser.add_argument("--pos_data_path", type=str, default=None)
    parser.add_argument("--neg_data_path", type=str, default=None)
    parser.add_argument("--neg_group", type=str, default="ELASTICC2_TRAIN/optical_data")
    parser.add_argument("--ckpt_path", type=str, default=None)
    parser.add_argument("--pretrained_albef_ckpt", type=str, default=None)

    parser.add_argument("--epochs_stage1", type=int, default=5)
    parser.add_argument("--epochs_stage2", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--val_batch_size", type=int, default=512)
    parser.add_argument("--steps_per_epoch", type=int, default=None)
    parser.add_argument("--val_steps_per_epoch", type=int, default=None)
    parser.add_argument("--val_split", type=float, default=0.1)
    parser.add_argument("--split_seed", type=int, default=42)

    parser.add_argument("--lr_head_stage1", type=float, default=1e-3)
    parser.add_argument("--lr_head_stage2", type=float, default=3e-4)
    parser.add_argument("--lr_encoder_stage2", type=float, default=3e-5)
    parser.add_argument("--weight_decay", type=float, default=1e-3)
    parser.add_argument("--grad_clip_norm", type=float, default=1.0)

    parser.add_argument("--n_ref", type=int, default=64)
    parser.add_argument("--ref_start", type=float, default=-0.3)
    parser.add_argument("--ref_end", type=float, default=0.6)
    parser.add_argument("--ref_dim", type=int, default=64)
    parser.add_argument("--enc_dim", type=int, default=64)

    parser.add_argument("--opt_dropout", type=float, default=0.1)
    parser.add_argument("--feature_dropout", type=float, default=0.0)
    parser.add_argument("--head_hidden_dim", type=int, default=None)
    parser.add_argument("--head_dropout", type=float, default=0.2)
    parser.add_argument("--include_coords", action="store_true")

    parser.add_argument("--opt_aug_noise", type=float, default=0.0)
    parser.add_argument("--opt_aug_time_jitter", type=float, default=0.0)
    parser.add_argument("--opt_aug_dropout", type=float, default=0.0)
    parser.add_argument("--opt_aug_band_dropout", type=float, default=0.0)

    parser.add_argument("--target_recall", type=float, default=0.98)
    parser.add_argument("--early_stop_patience", type=int, default=5)
    parser.add_argument("--early_stop_min_delta", type=float, default=1e-4)

    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--pin_memory", type=int, default=1)
    parser.add_argument("--persistent_workers", type=int, default=1)
    parser.add_argument("--prefetch_factor", type=int, default=4)
    parser.add_argument("--cache_in_memory", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tb_log_dir", type=str, default=None)
    parser.add_argument("--tb_flush_secs", type=int, default=30)
    parser.add_argument("--disable_tensorboard", action="store_true")

    pre_args, _ = parser.parse_known_args()
    if pre_args.config is not None:
        config_path = Path(pre_args.config)
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        with config_path.open("r", encoding="utf-8") as f:
            config = json.load(f)
        if not isinstance(config, dict):
            raise ValueError("Config JSON must be an object.")
        parser.set_defaults(**config)

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    for key in ("pos_data_path", "neg_data_path", "ckpt_path"):
        if getattr(args, key) is None:
            raise ValueError(f"{key} must be provided via CLI or --config.")
    if not os.path.exists(args.pos_data_path):
        raise FileNotFoundError(f"Positive data file not found: {args.pos_data_path}")
    if not os.path.exists(args.neg_data_path):
        raise FileNotFoundError(f"Negative data file not found: {args.neg_data_path}")
    os.makedirs(args.ckpt_path, exist_ok=True)

    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] Starting optical-only KN training")
    print(json.dumps(vars(args), indent=2))
    train(args)
