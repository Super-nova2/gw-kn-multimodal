from data_loader import create_training_dataloader, create_train_val_dataloaders
from model import GWOpticalALBEFModel
from tqdm import tqdm
import torch
from torch.utils.tensorboard import SummaryWriter
import h5py
import os
import datetime
import argparse
import math
import warnings
warnings.filterwarnings("ignore", "Wswiglal-redir-stdio")


def build_ref_time(batch_size, n_ref, ref_start, ref_end, device, dtype):
    ref = torch.linspace(ref_start, ref_end, n_ref, dtype=dtype, device=device)
    return ref.unsqueeze(0).repeat(batch_size, 1)

def sample_easy_negatives(batch_size, device):
    if batch_size < 2:
        return None
    shift = int(torch.randint(1, batch_size, (1,), device=device).item())
    return (torch.arange(batch_size, device=device) + shift) % batch_size

def augment_gw_data(gw_s, gw_m, training=True,
                    noise_std=0.05, scalar_jitter=0.02, channel_dropout_prob=0.1):
    """
    GW数据增强函数，增加训练样本的有效多样性。

    Args:
        gw_s: GW scalar features [Batch, 7] (质量、自旋等参数)
        gw_m: GW skymap [Batch, 7, 19200] (MOC skymap序列)
        training: 是否在训练模式（验证时不增强）
        noise_std: skymap高斯噪声标准差
        scalar_jitter: scalar参数扰动比例
        channel_dropout_prob: 随机dropout某个skymap通道的概率

    Returns:
        augmented gw_s, gw_m
    """
    if not training:
        return gw_s, gw_m

    # 1. Skymap高斯噪声注入
    # 添加小幅随机噪声，模拟观测不确定性
    skymap_noise = torch.randn_like(gw_m) * noise_std
    gw_m = gw_m + skymap_noise

    # 2. Scalar参数扰动
    # 对质量、自旋、距离等参数添加小幅随机扰动
    # 使用乘性噪声保持参数的量纲
    scalar_multiplier = 1.0 + (torch.rand_like(gw_s) - 0.5) * 2 * scalar_jitter
    gw_s = gw_s * scalar_multiplier

    # 3. Skymap通道Dropout（可选）
    # 随机将某个通道（如distance信息）置零，增加鲁棒性
    if torch.rand(1).item() < channel_dropout_prob:
        # 随机选择一个通道（0-6），但避免dropout概率通道(index 4)
        channel_idx = torch.randint(0, gw_m.size(1), (1,)).item()
        if channel_idx != 4:  # 保留概率通道
            gw_m[:, channel_idx, :] = 0

    return gw_s, gw_m

def build_lr_scheduler(optimizer, args, steps_per_epoch, start_step):
    if args.lr_scheduler == "none":
        return None
    total_steps = max(1, args.epochs * steps_per_epoch)
    warmup_steps = max(0, args.warmup_epochs * steps_per_epoch)
    min_lr = args.min_lr if args.min_lr is not None else 0.0
    min_lr_ratio = min(min_lr / args.lr, 1.0)

    def lr_lambda(step):
        if warmup_steps > 0 and step < warmup_steps:
            return float(step + 1) / float(max(1, warmup_steps))
        progress = (step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch=start_step - 1)

def apply_temperature_schedule(model, args, epoch):
    if args.temp_schedule == "learned":
        return None
    if args.temp_schedule == "fixed":
        target_temp = args.temp_init
    else:
        progress = epoch / float(max(1, args.epochs - 1))
        target_temp = args.temp_final + 0.5 * (args.temp_init - args.temp_final) * (1.0 + math.cos(math.pi * progress))
    target_temp = max(args.temp_min, min(args.temp_max, target_temp))
    model.log_temp.data.fill_(math.log(target_temp))
    return target_temp

def clamp_temperature(model, args):
    if args.temp_schedule != "learned":
        return
    min_log = math.log(args.temp_min)
    max_log = math.log(args.temp_max)
    model.log_temp.data.clamp_(min_log, max_log)

def evaluate(model, val_loader, device, args, epoch):
    model.eval()
    has_negatives = args.neg_data_path is not None
    ref_time_cache = None

    val_total = 0.0
    val_itc = 0.0
    val_cls = 0.0
    val_itc_acc = 0.0
    val_pos_acc = 0.0
    val_hard_acc = 0.0
    val_neg_acc = 0.0
    val_batches = 0

    with torch.no_grad():
        for batch_data in val_loader:
            if has_negatives:
                (gw_s, gw_m, opt_t, opt_v, opt_mask, opt_err, opt_coords, gw_indices,
                 neg_t, neg_v, neg_mask, neg_err, neg_coords) = batch_data
            else:
                gw_s, gw_m, opt_t, opt_v, opt_mask, opt_err, opt_coords, gw_indices = batch_data

            gw_s = gw_s.to(device, non_blocking=True)
            gw_m = gw_m.to(device, non_blocking=True)
            opt_t = opt_t.to(device, non_blocking=True)
            opt_v = opt_v.to(device, non_blocking=True)
            opt_mask = opt_mask.to(device, non_blocking=True)
            opt_err = opt_err.to(device, non_blocking=True)
            opt_coords = opt_coords.to(device, non_blocking=True)
            gw_indices = gw_indices.to(device, non_blocking=True).long()

            if torch.isnan(opt_v).any() or torch.isinf(opt_v).any():
                continue
            if torch.isnan(opt_err).any() or torch.isinf(opt_err).any():
                continue

            if has_negatives:
                neg_t = neg_t.to(device, non_blocking=True)
                neg_v = neg_v.to(device, non_blocking=True)
                neg_mask = neg_mask.to(device, non_blocking=True)
                neg_err = neg_err.to(device, non_blocking=True)
                neg_coords = neg_coords.to(device, non_blocking=True)
                if torch.isnan(neg_v).any() or torch.isinf(neg_v).any():
                    continue
                if torch.isnan(neg_err).any() or torch.isinf(neg_err).any():
                    continue

            batch_size = gw_s.size(0)
            if (
                ref_time_cache is None
                or ref_time_cache.shape[0] != batch_size
                or ref_time_cache.dtype != opt_t.dtype
            ):
                ref_time_cache = build_ref_time(
                    batch_size, args.n_ref, args.ref_start, args.ref_end, device, opt_t.dtype
                )
            opt_ref_t = ref_time_cache

            g, z_l, h_l = model.encode(
                gw_s, gw_m, opt_coords, opt_t, opt_v, opt_ref_t, opt_mask, opt_err
            )
            itc_loss, sim_g2o = model.compute_itc_loss(
                g, z_l, gw_indices, mask=args.mask_itc
            )

            logits_pos = model.fusion_logits(g, h_l)
            labels_pos = torch.ones(batch_size, device=device, dtype=torch.long)
            labels_neg = torch.zeros(batch_size, device=device, dtype=torch.long)
            pos_loss = model.cls_criterion(logits_pos, labels_pos)

            use_hard_neg = epoch >= args.hard_neg_start_epoch
            if use_hard_neg:
                neg_idx = model.sample_hard_negatives(sim_g2o, gw_indices)
            else:
                neg_idx = sample_easy_negatives(batch_size, device)

            if neg_idx is None:
                hard_loss = torch.zeros((), device=device)
                logits_hard = logits_pos.detach()
            else:
                h_l_hard = h_l[neg_idx]
                logits_hard = model.fusion_logits(g, h_l_hard)
                hard_loss = model.cls_criterion(logits_hard, labels_neg)

            if has_negatives:
                _, h_l_neg = model.encode_optical(
                    neg_coords, neg_t, neg_v, opt_ref_t, neg_mask, neg_err
                )
                logits_neg = model.fusion_logits(g, h_l_neg)
                neg_loss = model.cls_criterion(logits_neg, labels_neg)
                cls_loss = (pos_loss + hard_loss + neg_loss) / 3.0
            else:
                cls_loss = 0.5 * (pos_loss + hard_loss)

            total_loss = args.itc_weight * itc_loss + args.cls_weight * cls_loss

            val_total += total_loss.item()
            val_itc += itc_loss.item()
            val_cls += cls_loss.item()

            itc_preds = sim_g2o.argmax(dim=1)
            itc_acc = (itc_preds == torch.arange(batch_size, device=device)).float().mean().item()
            pos_acc = (logits_pos.argmax(dim=1) == labels_pos).float().mean().item()
            hard_acc = (logits_hard.argmax(dim=1) == labels_neg).float().mean().item()
            neg_acc = 0.0
            if has_negatives:
                neg_acc = (logits_neg.argmax(dim=1) == labels_neg).float().mean().item()

            val_itc_acc += itc_acc
            val_pos_acc += pos_acc
            val_hard_acc += hard_acc
            val_neg_acc += neg_acc
            val_batches += 1

    if val_batches == 0:
        model.train()
        return None

    metrics = {
        "total": val_total / val_batches,
        "itc": val_itc / val_batches,
        "cls": val_cls / val_batches,
        "itc_acc": val_itc_acc / val_batches,
        "pos_acc": val_pos_acc / val_batches,
        "hard_acc": val_hard_acc / val_batches,
        "neg_acc": val_neg_acc / val_batches
    }

    model.train()
    return metrics


def train(args):
    print("Training in Float32 precision.")
    if args.temp_final is None:
        args.temp_final = args.temp_init
    if args.temp_min <= 0 or args.temp_max <= 0:
        raise ValueError("temp_min and temp_max must be > 0.")
    if args.temp_min >= args.temp_max:
        raise ValueError("temp_min must be < temp_max.")

    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
    print(f"Running on device: {device}")

    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = os.path.join(os.path.dirname(args.ckpt_path), "tb_logs", f"run_albef_{timestamp}")
    writer = SummaryWriter(log_dir=log_dir)
    print(f"TensorBoard logging started at: {log_dir}")

    val_loader = None
    if args.val_split is not None and 0 < args.val_split < 1:
        train_loader, val_loader, steps_per_epoch, val_steps = create_train_val_dataloaders(
            h5_path=args.data_path,
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
            cache_in_memory=bool(args.cache_in_memory),
            negative_h5_path=args.neg_data_path,
            negative_group=args.neg_group
        )
        print(f"Train Steps/Epoch: {steps_per_epoch} | Val Steps/Epoch: {val_steps}")
    else:
        if args.steps_per_epoch is not None:
            steps_per_epoch = args.steps_per_epoch
        else:
            with h5py.File(args.data_path, 'r') as f:
                total_optical = f['events/optical_data/values'].shape[0]
            steps_per_epoch = total_optical // args.batch_size
            print(f"Dataset Size: {total_optical} | Steps/Epoch: {steps_per_epoch}")

        train_loader = create_training_dataloader(
            h5_path=args.data_path,
            batch_size=args.batch_size,
            steps_per_epoch=steps_per_epoch,
            num_workers=args.num_workers,
            pin_memory=bool(args.pin_memory),
            persistent_workers=bool(args.persistent_workers),
            prefetch_factor=args.prefetch_factor,
            cache_in_memory=bool(args.cache_in_memory),
            negative_h5_path=args.neg_data_path,
            negative_group=args.neg_group
        )

    model = GWOpticalALBEFModel(
        gw_scalar_dim=7,
        gw_skymap_channels=7,
        optical_input_dim=6,
        ref_time_dim=args.ref_dim,
        enc_dim=args.enc_dim,
        proj_dim=args.proj_dim,
        fusion_attn_dim=args.fusion_attn_dim,
        fusion_hidden_dim=args.fusion_hidden_dim,
        temp_init=args.temp_init,
        temp_min=args.temp_min,
        temp_max=args.temp_max,
        gw_dropout=args.gw_dropout,
        opt_dropout=args.opt_dropout,
        fusion_dropout=args.fusion_dropout,
        label_smoothing=args.label_smoothing,
        use_lightweight_gw=getattr(args, 'use_lightweight_gw', False)
    ).to(device)

    if args.use_lightweight_gw:
        print("Using lightweight GW encoder (~100K params) to prevent overfitting.")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    start_epoch = 0
    global_step = 0
    if args.resume is not None:
        if not os.path.exists(args.resume):
            raise FileNotFoundError(f"Resume checkpoint not found: {args.resume}")
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"], strict=True)
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = ckpt.get("epoch", -1) + 1
        global_step = start_epoch * steps_per_epoch
        print(f"Resumed from {args.resume} at epoch {start_epoch}.")
    model.train()
    has_negatives = args.neg_data_path is not None
    if args.temp_schedule != "learned":
        model.log_temp.requires_grad_(False)

    pbar_update_every = 500
    best_val = None
    epochs_no_improve = 0
    lr_scheduler = build_lr_scheduler(optimizer, args, steps_per_epoch, global_step)

    for epoch in range(start_epoch, args.epochs):
        epoch_total = 0.0
        epoch_itc = 0.0
        epoch_cls = 0.0

        ref_time_cache = None
        apply_temperature_schedule(model, args, epoch)

        pbar = tqdm(
            train_loader,
            desc=f"Epoch {epoch+1}/{args.epochs}",
            mininterval=0,
            miniters=pbar_update_every
        )
        
        for batch_idx, batch_data in enumerate(pbar):
            if has_negatives:
                (gw_s, gw_m, opt_t, opt_v, opt_mask, opt_err, opt_coords, gw_indices,
                 neg_t, neg_v, neg_mask, neg_err, neg_coords) = batch_data
            else:
                gw_s, gw_m, opt_t, opt_v, opt_mask, opt_err, opt_coords, gw_indices = batch_data

            gw_s = gw_s.to(device, non_blocking=True)
            gw_m = gw_m.to(device, non_blocking=True)
            # 应用GW数据增强（仅训练时）
            gw_s, gw_m = augment_gw_data(
                gw_s, gw_m, training=True,
                noise_std=args.gw_aug_noise,
                scalar_jitter=args.gw_aug_jitter,
                channel_dropout_prob=args.gw_aug_dropout
            )
            opt_t = opt_t.to(device, non_blocking=True)
            opt_v = opt_v.to(device, non_blocking=True)
            opt_mask = opt_mask.to(device, non_blocking=True)
            opt_err = opt_err.to(device, non_blocking=True)
            opt_coords = opt_coords.to(device, non_blocking=True)
            gw_indices = gw_indices.to(device, non_blocking=True).long()

            if torch.isnan(opt_v).any() or torch.isinf(opt_v).any():
                print("NaN or Inf detected in optical values. Skipping batch.")
                continue
            if torch.isnan(opt_err).any() or torch.isinf(opt_err).any():
                print("NaN or Inf detected in optical errors. Skipping batch.")
                continue

            if has_negatives:
                neg_t = neg_t.to(device, non_blocking=True)
                neg_v = neg_v.to(device, non_blocking=True)
                neg_mask = neg_mask.to(device, non_blocking=True)
                neg_err = neg_err.to(device, non_blocking=True)
                neg_coords = neg_coords.to(device, non_blocking=True)
                if torch.isnan(neg_v).any() or torch.isinf(neg_v).any():
                    print("NaN or Inf detected in negative optical values. Skipping batch.")
                    continue
                if torch.isnan(neg_err).any() or torch.isinf(neg_err).any():
                    print("NaN or Inf detected in negative optical errors. Skipping batch.")
                    continue

            batch_size = gw_s.size(0)

            if (
                ref_time_cache is None
                or ref_time_cache.shape[0] != batch_size
                or ref_time_cache.dtype != opt_t.dtype
            ):
                ref_time_cache = build_ref_time(
                    batch_size, args.n_ref, args.ref_start, args.ref_end, device, opt_t.dtype
                )
            opt_ref_t = ref_time_cache

            optimizer.zero_grad(set_to_none=True)

            g, z_l, h_l = model.encode(
                gw_s, gw_m, opt_coords, opt_t, opt_v, opt_ref_t, opt_mask, opt_err
            )
            itc_loss, sim_g2o = model.compute_itc_loss(
                g, z_l, gw_indices, mask=args.mask_itc
            )

            logits_pos = model.fusion_logits(g, h_l)
            labels_pos = torch.ones(batch_size, device=device, dtype=torch.long)
            labels_neg = torch.zeros(batch_size, device=device, dtype=torch.long)
            pos_loss = model.cls_criterion(logits_pos, labels_pos)

            use_hard_neg = epoch >= args.hard_neg_start_epoch
            if use_hard_neg:
                neg_idx = model.sample_hard_negatives(sim_g2o, gw_indices)
            else:
                neg_idx = sample_easy_negatives(batch_size, device)

            if neg_idx is None:
                hard_loss = torch.zeros((), device=device)
                logits_hard = logits_pos.detach()
            else:
                h_l_hard = h_l[neg_idx]
                logits_hard = model.fusion_logits(g, h_l_hard)
                hard_loss = model.cls_criterion(logits_hard, labels_neg)

            if has_negatives:
                _, h_l_neg = model.encode_optical(
                    neg_coords, neg_t, neg_v, opt_ref_t, neg_mask, neg_err
                )
                logits_neg = model.fusion_logits(g, h_l_neg)
                neg_loss = model.cls_criterion(logits_neg, labels_neg)
                cls_loss = (pos_loss + hard_loss + neg_loss) / 3.0
            else:
                cls_loss = 0.5 * (pos_loss + hard_loss)

            total_loss = args.itc_weight * itc_loss + args.cls_weight * cls_loss
            total_loss.backward()
            # 添加梯度裁剪防止梯度爆炸
            if args.grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.grad_clip_norm)
            optimizer.step()
            if lr_scheduler is not None:
                lr_scheduler.step()
            clamp_temperature(model, args)

            total_val = total_loss.item()
            itc_val = itc_loss.item()
            pos_loss_val = pos_loss.item()
            hard_loss_val = hard_loss.item()
            if has_negatives:
                neg_loss_val = neg_loss.item()
            cls_val = cls_loss.item()
            epoch_total += total_val
            epoch_itc += itc_val
            epoch_cls += cls_val

            with torch.no_grad():
                itc_preds = sim_g2o.argmax(dim=1)
                itc_acc = (itc_preds == torch.arange(batch_size, device=device)).float().mean().item()
                pos_acc = (logits_pos.argmax(dim=1) == labels_pos).float().mean().item()
                hard_acc = (logits_hard.argmax(dim=1) == labels_neg).float().mean().item()
                neg_acc = None
                if has_negatives:
                    neg_acc = (logits_neg.argmax(dim=1) == labels_neg).float().mean().item()
                current_temp = model.log_temp.exp().item()

            if batch_idx % 10 == 0:
                writer.add_scalar('Train/Batch_Total_Loss', total_val, global_step)
                writer.add_scalar('Train/Batch_ITC_Loss', itc_val, global_step)
                writer.add_scalar('Train/Batch_CLS_Loss', cls_val, global_step)
                writer.add_scalar('Train/Batch_Pos_Loss', pos_loss_val, global_step)
                writer.add_scalar('Train/Batch_HardNeg_Loss', hard_loss_val, global_step)
                if has_negatives:
                    writer.add_scalar('Train/Batch_ExtraNeg_Loss', neg_loss_val, global_step)
                writer.add_scalar('Train/Batch_ITC_Acc', itc_acc, global_step)
                writer.add_scalar('Train/Batch_Pos_Acc', pos_acc, global_step)
                writer.add_scalar('Train/Batch_HardNeg_Acc', hard_acc, global_step)
                if has_negatives and neg_acc is not None:
                    writer.add_scalar('Train/Batch_ExtraNeg_Acc', neg_acc, global_step)
                writer.add_scalar('Train/Temperature', current_temp, global_step)
                writer.add_scalar('Train/Learning_Rate', optimizer.param_groups[0]['lr'], global_step)

            if batch_idx % 100 == 0:
                postfix = {
                    'Total': f"{total_val:.4f}",
                    'ITC': f"{itc_val:.4f}",
                    'CLS': f"{cls_val:.4f}",
                    'ITC_Acc': f"{itc_acc:.2f}"
                }
                pbar.set_postfix(postfix)

            global_step += 1

        avg_total = epoch_total / len(train_loader)
        avg_itc = epoch_itc / len(train_loader)
        avg_cls = epoch_cls / len(train_loader)
        print(
            f"Epoch {epoch+1} Complete. Avg Total: {avg_total:.4f} | "
            f"ITC: {avg_itc:.4f} | CLS: {avg_cls:.4f}"
        )

        writer.add_scalar('Train/Epoch_Total_Loss', avg_total, epoch)
        writer.add_scalar('Train/Epoch_ITC_Loss', avg_itc, epoch)
        writer.add_scalar('Train/Epoch_CLS_Loss', avg_cls, epoch)

        stop_early = False
        if val_loader is not None:
            val_metrics = evaluate(model, val_loader, device, args, epoch)
            if val_metrics is not None:
                print(
                    f"Val Avg Total: {val_metrics['total']:.4f} | "
                    f"ITC: {val_metrics['itc']:.4f} | CLS: {val_metrics['cls']:.4f}"
                )
                writer.add_scalar('Val/Epoch_Total_Loss', val_metrics['total'], epoch)
                writer.add_scalar('Val/Epoch_ITC_Loss', val_metrics['itc'], epoch)
                writer.add_scalar('Val/Epoch_CLS_Loss', val_metrics['cls'], epoch)
                writer.add_scalar('Val/Epoch_ITC_Acc', val_metrics['itc_acc'], epoch)
                writer.add_scalar('Val/Epoch_Pos_Acc', val_metrics['pos_acc'], epoch)
                writer.add_scalar('Val/Epoch_HardNeg_Acc', val_metrics['hard_acc'], epoch)
                if args.neg_data_path is not None:
                    writer.add_scalar('Val/Epoch_ExtraNeg_Acc', val_metrics['neg_acc'], epoch)

                if args.early_stop_patience > 0:
                    if best_val is None or val_metrics['total'] < best_val - args.early_stop_min_delta:
                        best_val = val_metrics['total']
                        epochs_no_improve = 0
                        best_ckpt = os.path.join(args.ckpt_path, "ALBEF", "albef_best.pth")
                        os.makedirs(os.path.dirname(best_ckpt), exist_ok=True)
                        torch.save({
                            'epoch': epoch,
                            'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'loss': val_metrics['total'],
                        }, best_ckpt)
                        print(f"Saved best checkpoint: {best_ckpt}")
                    else:
                        epochs_no_improve += 1
                        if epochs_no_improve >= args.early_stop_patience:
                            print("Early stopping triggered.")
                            stop_early = True

        checkpoint_path = os.path.join(args.ckpt_path, "ALBEF", f"albef_epoch_{epoch+1}.pth")
        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_total,
        }, checkpoint_path)
        if stop_early:
            break

    writer.close()


if __name__ == "__main__":
    """
    Example usage:
    python ML+GW+KN/Model/ALBEF_train.py --data_path data/LSST_KN_BNS/combined_dataset.h5 \
        --epochs 2 --batch_size 32 --steps_per_epoch 10 --ckpt_path data/model/checkpoints
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="training_data.h5")
    parser.add_argument("--neg_data_path", type=str, default=None)
    parser.add_argument("--neg_group", type=str, default="events/optical_data")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--steps_per_epoch", type=int, default=None)
    parser.add_argument("--ckpt_path", type=str, default=None)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay for AdamW optimizer")
    parser.add_argument("--grad_clip_norm", type=float, default=1.0, help="Max norm for gradient clipping (0 to disable)")
    parser.add_argument("--lr_scheduler", type=str, default="none", choices=["none", "cosine"])
    parser.add_argument("--warmup_epochs", type=int, default=0)
    parser.add_argument("--min_lr", type=float, default=0.0)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--pin_memory", type=int, default=1)
    parser.add_argument("--persistent_workers", type=int, default=1)
    parser.add_argument("--prefetch_factor", type=int, default=4)
    parser.add_argument("--cache_in_memory", action='store_true')
    parser.add_argument("--val_split", type=float, default=0.1)
    parser.add_argument("--val_batch_size", type=int, default=None)
    parser.add_argument("--val_steps_per_epoch", type=int, default=None)
    parser.add_argument("--split_seed", type=int, default=42)
    parser.add_argument("--early_stop_patience", type=int, default=10)
    parser.add_argument("--early_stop_min_delta", type=float, default=1e-4)
    parser.add_argument("--n_ref", type=int, default=64)
    parser.add_argument("--ref_start", type=float, default=-0.3)
    parser.add_argument("--ref_end", type=float, default=0.6)
    parser.add_argument("--ref_dim", type=int, default=64)
    parser.add_argument("--enc_dim", type=int, default=128)
    parser.add_argument("--proj_dim", type=int, default=256)
    parser.add_argument("--fusion_attn_dim", type=int, default=None)
    parser.add_argument("--fusion_hidden_dim", type=int, default=None)
    parser.add_argument("--fusion_dropout", type=float, default=0.1)
    parser.add_argument("--label_smoothing", type=float, default=0.0, help="Label smoothing for classification loss (0 to disable)")
    parser.add_argument("--temp_init", type=float, default=0.07)
    parser.add_argument("--temp_final", type=float, default=None)
    parser.add_argument("--temp_min", type=float, default=0.01)
    parser.add_argument("--temp_max", type=float, default=100.0)
    parser.add_argument("--temp_schedule", type=str, default="learned", choices=["learned", "fixed", "cosine"])
    parser.add_argument("--gw_dropout", type=float, default=0.1)
    parser.add_argument("--opt_dropout", type=float, default=0.1)
    parser.add_argument("--itc_weight", type=float, default=1.0)
    parser.add_argument("--cls_weight", type=float, default=1.0)
    parser.add_argument("--mask_itc", action='store_true', help="Mask same-event pairs in ITC loss")
    parser.add_argument("--hard_neg_start_epoch", type=int, default=0)
    parser.add_argument("--use_lightweight_gw", action='store_true',
                        help="Use lightweight GW encoder (~100K params) instead of ResNet-18 (~11M params) to prevent overfitting on small GW datasets")
    # GW数据增强参数
    parser.add_argument("--gw_aug_noise", type=float, default=0.05,
                        help="GW skymap augmentation noise std (default: 0.05)")
    parser.add_argument("--gw_aug_jitter", type=float, default=0.02,
                        help="GW scalar augmentation jitter ratio (default: 0.02)")
    parser.add_argument("--gw_aug_dropout", type=float, default=0.1,
                        help="GW channel dropout probability (default: 0.1)")

    args = parser.parse_args()

    if os.path.exists(args.data_path):
        if args.ckpt_path is None:
            raise ValueError("ckpt_path must be provided.")
        os.makedirs(args.ckpt_path, exist_ok=True)
        train(args)
    else:
        print("Data file not found.")
