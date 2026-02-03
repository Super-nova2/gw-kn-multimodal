from data_loader import (
    create_training_dataloader,
    create_train_val_dataloaders,
    create_supcon_dataloaders,
)
from model import GWOpticalALBEFModel
from tqdm import tqdm
import torch
import torch.nn.functional as F
from torch.amp import autocast, GradScaler
from torch.utils.tensorboard import SummaryWriter
import h5py
import os
import datetime
import argparse
import math
import gc
import json
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

    # 1. Skymap probability augmentation in log space
    # Channel 4 stores 100 * probdensity * dA (probability mass per pixel).
    # Convert to normalized probability, perturb in log space, renormalize.
    eps = 1e-30
    dP = gw_m[:, 4, :]  # [Batch, 19200]
    dP_sum = dP.sum(dim=-1, keepdim=True).clamp(min=eps)
    p = dP / dP_sum  # normalized probability
    noise = torch.randn_like(p)
    p_perturbed = p * torch.exp(noise * noise_std)
    p_perturbed = p_perturbed / p_perturbed.sum(dim=-1, keepdim=True).clamp(min=eps)
    # Scale back to original dP sum
    gw_m = gw_m.clone()
    gw_m[:, 4, :] = p_perturbed * dP_sum

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

def augment_optical_data(opt_t, opt_v, opt_mask, opt_err, training=True,
                         time_jitter=0.0, flux_noise=0.0,
                         obs_dropout=0.0, band_dropout=0.0):
    """
    Optical data augmentation to improve generalization.
    """
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

def compute_cls_weight(args, epoch):
    if epoch < args.cls_start_epoch:
        return 0.0
    if args.cls_ramp_epochs <= 0:
        return args.cls_weight
    progress = (epoch - args.cls_start_epoch + 1) / float(args.cls_ramp_epochs)
    return args.cls_weight * min(1.0, progress)

def compute_itc_weight(args, epoch):
    if args.itc_decay_epochs <= 0 or args.itc_decay_ratio <= 0:
        return args.itc_weight
    if epoch < args.itc_decay_start_epoch:
        return args.itc_weight
    progress = (epoch - args.itc_decay_start_epoch + 1) / float(args.itc_decay_epochs)
    progress = min(1.0, progress)
    return args.itc_weight * (1.0 - args.itc_decay_ratio * progress)

def compute_hard_neg_ratio(args, epoch):
    if epoch < args.hard_neg_start_epoch:
        return 0.0
    if args.hard_neg_ramp_epochs <= 0:
        return 1.0
    progress = (epoch - args.hard_neg_start_epoch + 1) / float(args.hard_neg_ramp_epochs)
    return min(1.0, progress)

def set_requires_grad(module, requires_grad):
    for param in module.parameters():
        param.requires_grad = requires_grad

def apply_freeze_schedule(model, args, epoch):
    freeze_enc = args.freeze_encoder_epochs > 0 and epoch < args.freeze_encoder_epochs
    freeze_itc = args.freeze_itc_epochs > 0 and epoch < args.freeze_itc_epochs

    set_requires_grad(model.gw_encoder, not freeze_enc)
    set_requires_grad(model.optical_encoder, not freeze_enc)

    set_requires_grad(model.gw_proj, not freeze_itc)
    set_requires_grad(model.opt_proj, not freeze_itc)
    if args.temp_schedule == "learned":
        model.log_temp.requires_grad_(not freeze_itc)
    else:
        model.log_temp.requires_grad_(False)

def compute_weighted_cls_loss(pos_loss, hard_loss, neg_loss, has_negatives, args):
    pos_weight = args.cls_pos_weight
    neg_weight = args.cls_neg_weight
    extra_neg_weight = args.cls_extra_neg_weight

    if has_negatives:
        denom = max(1e-8, pos_weight + neg_weight + extra_neg_weight)
        return (
            pos_weight * pos_loss + neg_weight * hard_loss + extra_neg_weight * neg_loss
        ) / denom

    denom = max(1e-8, pos_weight + neg_weight)
    return (pos_weight * pos_loss + neg_weight * hard_loss) / denom

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

def evaluate(model, val_loader, device, args, epoch, amp_dtype=torch.float32):
    from metrics import (compute_retrieval_metrics,
                         compute_classification_metrics,
                         compute_embedding_metrics)

    model.eval()
    has_negatives = args.neg_data_path is not None
    ref_time_cache = None
    cls_weight = compute_cls_weight(args, epoch)
    itc_weight = compute_itc_weight(args, epoch)
    hard_neg_ratio = compute_hard_neg_ratio(args, epoch)

    val_total = 0.0
    val_itc = 0.0
    val_cls = 0.0
    val_itc_acc = 0.0
    val_pos_acc = 0.0
    val_hard_acc = 0.0
    val_neg_acc = 0.0
    val_total_acc = 0.0
    val_batches = 0

    # Accumulators for new metrics (computed globally after all batches)
    all_cls_probs = []    # predicted P(match) for classification metrics
    all_cls_labels = []   # ground-truth labels for classification metrics
    all_cls_sources = []  # source tag per sample ('pos', 'hard', 'neg')
    all_feat_g = []       # L2-normalized GW embeddings for embedding metrics
    all_feat_o = []       # L2-normalized optical embeddings
    all_gw_indices = []   # GW event indices
    retrieval_metrics_accum = []  # per-batch retrieval metric dicts

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

            if has_negatives:
                neg_t = neg_t.to(device, non_blocking=True)
                neg_v = neg_v.to(device, non_blocking=True)
                neg_mask = neg_mask.to(device, non_blocking=True)
                neg_err = neg_err.to(device, non_blocking=True)
                neg_coords = neg_coords.to(device, non_blocking=True)

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

            with autocast(device_type='cuda', dtype=amp_dtype, enabled=(device.type == 'cuda')):
                g, z_l, h_l = model.encode(
                    gw_s, gw_m, opt_coords, opt_t, opt_v, opt_ref_t, opt_mask, opt_err
                )
                if args.itc_loss_type == "supcon":
                    itc_loss, sim_g2o = model.compute_supcon_loss(
                        g, z_l, gw_indices, margin=args.supcon_margin
                    )
                else:
                    itc_loss, sim_g2o = model.compute_itc_loss(
                        g, z_l, gw_indices, mask=args.mask_itc
                    )

                # Extract L2-normalized projected features for embedding metrics
                feat_g = F.normalize(model.gw_proj(g), p=2, dim=1, eps=1e-8)
                feat_o = F.normalize(model.opt_proj(z_l), p=2, dim=1, eps=1e-8)

                logits_pos = model.fusion_logits(g, h_l)
                labels_pos = torch.ones(batch_size, device=device, dtype=torch.long)
                labels_neg = torch.zeros(batch_size, device=device, dtype=torch.long)
                pos_loss = model.cls_criterion(logits_pos, labels_pos)

                easy_idx = sample_easy_negatives(batch_size, device)
                if easy_idx is None:
                    neg_idx = None
                elif hard_neg_ratio <= 0:
                    neg_idx = easy_idx
                else:
                    if args.semi_hard:
                        hard_idx = model.sample_semi_hard_negatives(sim_g2o, gw_indices, margin=args.semi_hard_margin)
                    else:
                        hard_idx = model.sample_hard_negatives(sim_g2o, gw_indices)
                    if hard_neg_ratio >= 1:
                        neg_idx = hard_idx
                    else:
                        choose_hard = torch.rand(batch_size, device=device) < hard_neg_ratio
                        neg_idx = torch.where(choose_hard, hard_idx, easy_idx)

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
                else:
                    neg_loss = None

                cls_loss = compute_weighted_cls_loss(
                    pos_loss, hard_loss, neg_loss, has_negatives, args
                )

                total_loss = itc_weight * itc_loss + cls_weight * cls_loss

            val_total += total_loss.item()
            val_itc += itc_loss.item()
            val_cls += cls_loss.item()

            # --- Per-batch retrieval metrics ---
            sim_g2o_d = sim_g2o.detach()
            batch_ret = compute_retrieval_metrics(
                sim_g2o_d, gw_indices, ks=(1, 5, 10)
            )
            retrieval_metrics_accum.append(batch_ret)

            # --- ITC accuracy ---
            itc_preds = sim_g2o_d.argmax(dim=1)
            anchor_gw = gw_indices
            pred_gw = gw_indices[itc_preds]
            itc_acc = (anchor_gw == pred_gw).float().mean().item()

            pos_acc = (logits_pos.argmax(dim=1) == labels_pos).float().mean().item()
            hard_acc = (logits_hard.argmax(dim=1) == labels_neg).float().mean().item()
            neg_acc = 0.0
            if has_negatives:
                neg_acc = (logits_neg.argmax(dim=1) == labels_neg).float().mean().item()
                total_acc = (pos_acc + hard_acc + neg_acc) / 3.0
            else:
                total_acc = (pos_acc + hard_acc) / 2.0

            val_itc_acc += itc_acc
            val_pos_acc += pos_acc
            val_hard_acc += hard_acc
            val_neg_acc += neg_acc
            val_total_acc += total_acc
            val_batches += 1

            # --- Accumulate for global classification + embedding metrics ---
            pos_probs = torch.softmax(logits_pos.detach().float(), dim=1)[:, 1]
            all_cls_probs.append(pos_probs.cpu())
            all_cls_labels.append(labels_pos.cpu())
            all_cls_sources.extend(['pos'] * batch_size)

            hard_probs = torch.softmax(logits_hard.detach().float(), dim=1)[:, 1]
            all_cls_probs.append(hard_probs.cpu())
            all_cls_labels.append(labels_neg[:batch_size].cpu())
            all_cls_sources.extend(['hard'] * batch_size)

            if has_negatives:
                neg_probs_val = torch.softmax(logits_neg.detach().float(), dim=1)[:, 1]
                all_cls_probs.append(neg_probs_val.cpu())
                all_cls_labels.append(labels_neg[:batch_size].cpu())
                all_cls_sources.extend(['neg'] * batch_size)

            all_feat_g.append(feat_g.detach().cpu())
            all_feat_o.append(feat_o.detach().cpu())
            all_gw_indices.append(gw_indices.cpu())

            # Memory cleanup in validation loop
            del g, z_l, h_l, sim_g2o, sim_g2o_d, itc_loss, cls_loss, total_loss
            del logits_pos, logits_hard, pos_loss, hard_loss, feat_g, feat_o
            if has_negatives:
                del h_l_neg, logits_neg, neg_loss

    if val_batches == 0:
        model.train()
        return None

    # --- Compute global metrics ---
    # Average per-batch retrieval metrics
    retrieval = {}
    if retrieval_metrics_accum:
        for key in retrieval_metrics_accum[0]:
            retrieval[key] = sum(d[key] for d in retrieval_metrics_accum) / len(retrieval_metrics_accum)

    # Global classification metrics
    cls_metrics = compute_classification_metrics(
        torch.cat(all_cls_probs),
        torch.cat(all_cls_labels),
        all_sources=all_cls_sources
    )

    # Global embedding metrics
    emb_metrics = compute_embedding_metrics(
        torch.cat(all_feat_g),
        torch.cat(all_feat_o),
        torch.cat(all_gw_indices),
    )

    metrics = {
        "total": val_total / val_batches,
        "itc": val_itc / val_batches,
        "cls": val_cls / val_batches,
        "itc_acc": val_itc_acc / val_batches,
        "pos_acc": val_pos_acc / val_batches,
        "hard_acc": val_hard_acc / val_batches,
        "neg_acc": val_neg_acc / val_batches,
        "total_acc": val_total_acc / val_batches,
        "retrieval": retrieval,
        "classification": cls_metrics,
        "embedding": emb_metrics,
    }

    model.train()
    # Memory cleanup after validation
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    return metrics


def train(args):
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
        use_bf16 = torch.cuda.is_bf16_supported()
        amp_dtype = torch.bfloat16 if use_bf16 else torch.float16
        scaler = GradScaler(enabled=(not use_bf16))
        print(f"Running on device: {device} | AMP dtype={amp_dtype}")
    else:
        amp_dtype = torch.float32
        scaler = GradScaler(enabled=False)
        print(f"Running on device: {device} | No AMP (CPU)")

    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = os.path.join(os.path.dirname(args.ckpt_path), "tb_logs", f"run_albef_{timestamp}")
    writer = SummaryWriter(log_dir=log_dir)
    print(f"TensorBoard logging started at: {log_dir}")

    val_loader = None
    if args.val_split is not None and 0 < args.val_split < 1:
        if args.itc_loss_type == "supcon":
            train_loader, val_loader, steps_per_epoch, val_steps = create_supcon_dataloaders(
                h5_path=args.data_path,
                batch_size=args.batch_size,
                samples_per_gw=args.samples_per_gw,
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
                negative_group=args.neg_group,
                min_lc_per_gw=args.min_lc_per_gw
            )
            print(
                f"SupCon mode: {args.samples_per_gw} samples/GW, "
                f"{args.batch_size // args.samples_per_gw} GW/batch"
            )
        else:
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
        proj_dropout=getattr(args, 'proj_dropout', 0.0),
        feature_dropout=args.feature_dropout,
        fusion_dropout=args.fusion_dropout,
        label_smoothing=args.label_smoothing,
        itc_label_smoothing=args.itc_label_smoothing,
        use_lightweight_gw=getattr(args, 'use_lightweight_gw', False)
    ).to(device)

    if args.use_lightweight_gw:
        print("Using lightweight GW encoder (~100K params) to prevent overfitting.")

    if hasattr(torch, 'compile'):
        model = torch.compile(model)
        print("Model compiled with torch.compile() for optimized execution.")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    start_epoch = 0
    global_step = 0
    if args.resume is not None and args.pretrained is not None:
        raise ValueError("resume and pretrained are mutually exclusive.")
    if args.pretrained is not None:
        if not os.path.exists(args.pretrained):
            raise FileNotFoundError(f"Pretrained checkpoint not found: {args.pretrained}")
        ckpt = torch.load(args.pretrained, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"], strict=True)
        print(f"Loaded pretrained weights from {args.pretrained}.")
    elif args.resume is not None:
        if not os.path.exists(args.resume):
            raise FileNotFoundError(f"Resume checkpoint not found: {args.resume}")
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"], strict=True)
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if 'scaler_state_dict' in ckpt:
            scaler.load_state_dict(ckpt['scaler_state_dict'])
        start_epoch = ckpt.get("epoch", -1) + 1
        global_step = start_epoch * steps_per_epoch
        print(f"Resumed from {args.resume} at epoch {start_epoch}.")
    model.train()
    has_negatives = args.neg_data_path is not None
    apply_freeze_schedule(model, args, start_epoch)

    pbar_update_every = 500
    best_val = None
    best_val_metrics = {}
    epochs_no_improve = 0
    lr_scheduler = build_lr_scheduler(optimizer, args, steps_per_epoch, global_step)

    for epoch in range(start_epoch, args.epochs):
        epoch_total = 0.0
        epoch_itc = 0.0
        epoch_cls = 0.0

        ref_time_cache = None
        apply_temperature_schedule(model, args, epoch)
        apply_freeze_schedule(model, args, epoch)
        cls_weight = compute_cls_weight(args, epoch)
        itc_weight = compute_itc_weight(args, epoch)
        hard_neg_ratio = compute_hard_neg_ratio(args, epoch)

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

            if has_negatives:
                neg_t = neg_t.to(device, non_blocking=True)
                neg_v = neg_v.to(device, non_blocking=True)
                neg_mask = neg_mask.to(device, non_blocking=True)
                neg_err = neg_err.to(device, non_blocking=True)
                neg_coords = neg_coords.to(device, non_blocking=True)

            opt_t, opt_v, opt_mask, opt_err = augment_optical_data(
                opt_t, opt_v, opt_mask, opt_err, training=True,
                time_jitter=args.opt_aug_time_jitter,
                flux_noise=args.opt_aug_noise,
                obs_dropout=args.opt_aug_dropout,
                band_dropout=args.opt_aug_band_dropout
            )
            if has_negatives:
                neg_t, neg_v, neg_mask, neg_err = augment_optical_data(
                    neg_t, neg_v, neg_mask, neg_err, training=True,
                    time_jitter=args.opt_aug_time_jitter,
                    flux_noise=args.opt_aug_noise,
                    obs_dropout=args.opt_aug_dropout,
                    band_dropout=args.opt_aug_band_dropout
                )

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

            with autocast(device_type='cuda', dtype=amp_dtype, enabled=(device.type == 'cuda')):
                g, z_l, h_l = model.encode(
                    gw_s, gw_m, opt_coords, opt_t, opt_v, opt_ref_t, opt_mask, opt_err
                )
                if args.itc_loss_type == "supcon":
                    itc_loss, sim_g2o = model.compute_supcon_loss(
                        g, z_l, gw_indices, margin=args.supcon_margin
                    )
                else:
                    itc_loss, sim_g2o = model.compute_itc_loss(
                        g, z_l, gw_indices, mask=args.mask_itc
                    )

                logits_pos = model.fusion_logits(g, h_l)
                labels_pos = torch.ones(batch_size, device=device, dtype=torch.long)
                labels_neg = torch.zeros(batch_size, device=device, dtype=torch.long)
                pos_loss = model.cls_criterion(logits_pos, labels_pos)

                easy_idx = sample_easy_negatives(batch_size, device)
                if easy_idx is None:
                    neg_idx = None
                elif hard_neg_ratio <= 0:
                    neg_idx = easy_idx
                else:
                    if args.semi_hard:
                        hard_idx = model.sample_semi_hard_negatives(sim_g2o, gw_indices, margin=args.semi_hard_margin)
                    else:
                        hard_idx = model.sample_hard_negatives(sim_g2o, gw_indices)
                    if hard_neg_ratio >= 1:
                        neg_idx = hard_idx
                    else:
                        choose_hard = torch.rand(batch_size, device=device) < hard_neg_ratio
                        neg_idx = torch.where(choose_hard, hard_idx, easy_idx)

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
                else:
                    neg_loss = None

                cls_loss = compute_weighted_cls_loss(
                    pos_loss, hard_loss, neg_loss, has_negatives, args
                )

                # 分阶段训练：cls_start_epoch之前只训练ITC
                total_loss = itc_weight * itc_loss + cls_weight * cls_loss

            scaler.scale(total_loss).backward()
            if args.grad_clip_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.grad_clip_norm)
            scaler.step(optimizer)
            scaler.update()
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

            # Detach similarity matrix to prevent gradient graph retention
            sim_g2o_detached = sim_g2o.detach()
            with torch.no_grad():
                # ITC accuracy: accept any optical sample from the same GW event
                itc_preds = sim_g2o_detached.argmax(dim=1)
                pred_gw = gw_indices[itc_preds]
                itc_acc = (gw_indices == pred_gw).float().mean().item()

                pos_acc = (logits_pos.argmax(dim=1) == labels_pos).float().mean().item()
                hard_acc = (logits_hard.argmax(dim=1) == labels_neg).float().mean().item()
                neg_acc = None
                if has_negatives:
                    neg_acc = (logits_neg.argmax(dim=1) == labels_neg).float().mean().item()
                current_temp = model.log_temp.exp().item()

            if batch_idx % 50 == 0:
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
                writer.add_scalar('Train/Itc_Weight', itc_weight, global_step)
                writer.add_scalar('Train/Cls_Weight', cls_weight, global_step)
                writer.add_scalar('Train/HardNeg_Ratio', hard_neg_ratio, global_step)
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

            # Memory cleanup: delete large tensors to free memory
            del g, z_l, h_l, sim_g2o, sim_g2o_detached, itc_loss, total_loss
            del logits_pos, logits_hard, pos_loss, hard_loss, cls_loss
            if has_negatives:
                del h_l_neg, logits_neg, neg_loss
            
            # Periodic aggressive memory cleanup every 50 batches
            if batch_idx % 50 == 0:
                if device.type == 'cuda':
                    torch.cuda.empty_cache()
                gc.collect()

            global_step += 1

        # End of epoch cleanup
        pbar.close()
        del pbar
        
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
        
        # Flush tensorboard to prevent memory accumulation
        writer.flush()

        stop_early = False
        if val_loader is not None:
            val_metrics = evaluate(model, val_loader, device, args, epoch, amp_dtype=amp_dtype)
            if val_metrics is not None:
                print(
                    f"Val Avg Total: {val_metrics['total']:.4f} | "
                    f"ITC: {val_metrics['itc']:.4f} | CLS: {val_metrics['cls']:.4f}"
                )
                writer.add_scalar('Val/Epoch_Total_Loss', val_metrics['total'], epoch)
                writer.add_scalar('Val/Epoch_ITC_Loss', val_metrics['itc'], epoch)
                writer.add_scalar('Val/Epoch_CLS_Loss', val_metrics['cls'], epoch)
                # Retrieval metrics
                ret = val_metrics.get('retrieval', {})
                for k, v in ret.items():
                    writer.add_scalar(f'Val/Retrieval/{k}', v, epoch)

                # Classification metrics
                cls_m = val_metrics.get('classification', {})
                for k in ('auroc', 'auprc', 'f1_optimal', 'f1_threshold', 'ece'):
                    if k in cls_m:
                        writer.add_scalar(f'Val/Classification/{k}', cls_m[k], epoch)
                for k in ('acc_pos', 'acc_hard', 'acc_neg', 'acc_neg_gw', 'acc_total'):
                    if k in cls_m:
                        writer.add_scalar(f'Val/Classification/{k}', cls_m[k], epoch)

                # Embedding quality metrics
                emb = val_metrics.get('embedding', {})
                for k, v in emb.items():
                    writer.add_scalar(f'Val/Embedding/{k}', v, epoch)

                # Print summary of new metrics
                r1 = ret.get('g2o_recall_at_1', 0)
                r5 = ret.get('g2o_recall_at_5', 0)
                mrr = ret.get('g2o_mrr', 0)
                auroc = cls_m.get('auroc', 0)
                auprc = cls_m.get('auprc', 0)
                align = emb.get('alignment', 0)
                print(
                    f"  Retrieval: R@1={r1:.4f} R@5={r5:.4f} MRR={mrr:.4f} | "
                    f"CLS: AUROC={auroc:.4f} AUPRC={auprc:.4f} | "
                    f"Emb: align={align:.4f}"
                )

                if args.early_stop_patience > 0:
                    current_acc = val_metrics.get('classification', {}).get('acc_total', 0)
                    if best_val is None or current_acc > best_val + args.early_stop_min_delta:
                        best_val = current_acc
                        best_val_metrics = {
                            "best_val_acc_total": current_acc,
                            "best_val_loss": val_metrics['total'],
                            "best_epoch": epoch,
                            "val_itc_loss": val_metrics.get('itc', 0),
                            "val_cls_loss": val_metrics.get('cls', 0),
                            "val_recall_at_1": val_metrics.get('retrieval', {}).get('g2o_recall_at_1', 0),
                            "val_recall_at_5": val_metrics.get('retrieval', {}).get('g2o_recall_at_5', 0),
                            "val_mrr": val_metrics.get('retrieval', {}).get('g2o_mrr', 0),
                            "val_auroc": val_metrics.get('classification', {}).get('auroc', 0),
                            "val_auprc": val_metrics.get('classification', {}).get('auprc', 0),
                        }
                        epochs_no_improve = 0
                        best_ckpt = os.path.join(args.ckpt_path, "ALBEF", "albef_best.pth")
                        os.makedirs(os.path.dirname(best_ckpt), exist_ok=True)
                        torch.save({
                            'epoch': epoch,
                            'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'scaler_state_dict': scaler.state_dict(),
                            'acc_total': current_acc,
                            'loss': val_metrics['total'],
                            'args': vars(args),
                        }, best_ckpt)
                        print(f"Saved best checkpoint (acc_total={current_acc:.4f}): {best_ckpt}")
                    else:
                        epochs_no_improve += 1
                        if epochs_no_improve >= args.early_stop_patience:
                            print("Early stopping triggered.")
                            stop_early = True

        if not getattr(args, 'skip_epoch_checkpoints', False):
            checkpoint_path = os.path.join(args.ckpt_path, "ALBEF", f"albef_epoch_{epoch+1}.pth")
            os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scaler_state_dict': scaler.state_dict(),
                'loss': avg_total,
                'args': vars(args),
            }, checkpoint_path)
        
        # End-of-epoch memory cleanup
        if device.type == 'cuda':
            torch.cuda.empty_cache()
        gc.collect()
        
        if stop_early:
            break

    writer.close()

    # Write trial results JSON for HPO collection
    if best_val_metrics:
        best_val_metrics["final_epoch"] = epoch
        result_path = os.path.join(args.ckpt_path, "ALBEF", "trial_results.json")
        os.makedirs(os.path.dirname(result_path), exist_ok=True)
        with open(result_path, "w") as f:
            json.dump(best_val_metrics, f, indent=2)
        print(f"Trial results saved to: {result_path}")


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
    parser.add_argument("--pretrained", type=str, default=None,
                        help="Load model weights from checkpoint without optimizer state")
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
    parser.add_argument("--proj_dropout", type=float, default=0.0,
                        help="Projection head dropout to prevent ITC overfitting (default: 0.0)")
    parser.add_argument("--feature_dropout", type=float, default=0.0,
                        help="Dropout applied to encoder features before ITC/CLS heads")
    parser.add_argument("--freeze_encoder_epochs", type=int, default=0,
                        help="Freeze GW/optical encoders for N initial epochs")
    parser.add_argument("--freeze_itc_epochs", type=int, default=0,
                        help="Freeze ITC projection heads/temp for N initial epochs")
    parser.add_argument("--itc_weight", type=float, default=1.0)
    parser.add_argument("--cls_weight", type=float, default=1.0)
    parser.add_argument("--cls_pos_weight", type=float, default=1.0,
                        help="Positive class weight for CLS loss")
    parser.add_argument("--cls_neg_weight", type=float, default=1.0,
                        help="Negative class weight for CLS loss")
    parser.add_argument("--cls_extra_neg_weight", type=float, default=1.0,
                        help="Extra negative class weight for CLS loss")
    parser.add_argument("--cls_ramp_epochs", type=int, default=0,
                        help="Epochs to ramp CLS weight from 0 to cls_weight (0 to disable)")
    parser.add_argument("--itc_decay_start_epoch", type=int, default=0,
                        help="Epoch to start decaying ITC weight (ignored if itc_decay_epochs <= 0)")
    parser.add_argument("--itc_decay_epochs", type=int, default=0,
                        help="Epochs to decay ITC weight (0 to disable)")
    parser.add_argument("--itc_decay_ratio", type=float, default=0.0,
                        help="Fractional decay of ITC weight by the end of itc_decay_epochs")
    parser.add_argument("--itc_label_smoothing", type=float, default=0.0,
                        help="Label smoothing for ITC loss (0 to disable)")
    parser.add_argument("--itc_loss_type", type=str, default="infonce",
                        choices=["infonce", "supcon"],
                        help="ITC loss type: 'infonce' (original) or 'supcon' (supervised contrastive)")
    parser.add_argument("--supcon_margin", type=float, default=0.0,
                        help="Margin for SupCon loss to enforce separation between positives and negatives (default: 0.0)")
    parser.add_argument("--samples_per_gw", type=int, default=4,
                        help="Number of optical samples per GW event for SupCon (default: 4)")
    parser.add_argument("--min_lc_per_gw", type=int, default=2,
                        help="Minimum light curves required for a GW to be eligible for SupCon")
    parser.add_argument("--mask_itc", action='store_true', help="Mask same-event pairs in ITC loss")
    parser.add_argument("--hard_neg_start_epoch", type=int, default=0)
    parser.add_argument("--hard_neg_ramp_epochs", type=int, default=0,
                        help="Epochs to ramp hard negative ratio to 1.0 (0 to disable)")
    parser.add_argument("--semi_hard", action='store_true',
                        help="Use semi-hard negative sampling instead of hardest negative")
    parser.add_argument("--semi_hard_margin", type=float, default=0.2,
                        help="Relative margin for semi-hard band: sim in [(1-m)*pos_median, pos_median], m in (0,1)")
    parser.add_argument("--cls_start_epoch", type=int, default=0,
                        help="Epoch to start CLS training. Before this epoch, only ITC loss is used (for staged training)")
    parser.add_argument("--use_lightweight_gw", action='store_true',
                        help="Use lightweight GW encoder (~100K params) instead of ResNet-18 (~11M params) to prevent overfitting on small GW datasets")
    # GW数据增强参数
    parser.add_argument("--gw_aug_noise", type=float, default=0.05,
                        help="GW skymap augmentation noise std (default: 0.05)")
    parser.add_argument("--gw_aug_jitter", type=float, default=0.02,
                        help="GW scalar augmentation jitter ratio (default: 0.02)")
    parser.add_argument("--gw_aug_dropout", type=float, default=0.1,
                        help="GW channel dropout probability (default: 0.1)")
    # Optical data augmentation parameters
    parser.add_argument("--opt_aug_noise", type=float, default=0.0,
                        help="Optical flux noise scale relative to errors (default: 0.0)")
    parser.add_argument("--opt_aug_time_jitter", type=float, default=0.0,
                        help="Optical time jitter std (default: 0.0)")
    parser.add_argument("--opt_aug_dropout", type=float, default=0.0,
                        help="Optical observation dropout probability (default: 0.0)")
    parser.add_argument("--opt_aug_band_dropout", type=float, default=0.0,
                        help="Optical band dropout probability (default: 0.0)")
    parser.add_argument("--hpo_trial_number", type=int, default=None,
                        help="Optuna trial number (set automatically by HPO, not for manual use)")
    parser.add_argument("--skip_epoch_checkpoints", action='store_true',
                        help="Skip per-epoch checkpoint saves (only save best checkpoint). Useful for HPO.")

    args = parser.parse_args()

    if os.path.exists(args.data_path):
        if args.ckpt_path is None:
            raise ValueError("ckpt_path must be provided.")
        os.makedirs(args.ckpt_path, exist_ok=True)
        train(args)
    else:
        print("Data file not found.")
