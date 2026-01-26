import argparse
import datetime
import math
import os

import h5py
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from data_loader_v1 import create_train_val_dataloaders, create_mixed_gw_dataloaders
from model_v1 import GWOpticalFusionModel


def build_ref_time(batch_size, n_ref, ref_start, ref_end, device, dtype):
    ref = torch.linspace(ref_start, ref_end, n_ref, dtype=dtype, device=device)
    return ref.unsqueeze(0).repeat(batch_size, 1)


def sample_easy_negatives(batch_size, device, gw_indices=None):
    if batch_size < 2:
        return None
    if gw_indices is None:
        shift = int(torch.randint(1, batch_size, (1,), device=device).item())
        return (torch.arange(batch_size, device=device) + shift) % batch_size

    neg_indices = torch.empty(batch_size, dtype=torch.long, device=device)
    for i in range(batch_size):
        valid = torch.where(gw_indices != gw_indices[i])[0]
        if valid.numel() == 0:
            return None
        rand_idx = torch.randint(0, valid.numel(), (1,), device=device).item()
        neg_indices[i] = valid[rand_idx]
    return neg_indices


def build_lr_scheduler(optimizer, args, steps_per_epoch, start_step):
    if args.lr_scheduler == "none":
        return None
    total_steps = max(1, args.epochs * steps_per_epoch)
    warmup_steps = max(0, args.warmup_epochs * steps_per_epoch)
    min_lr_ratio = min(args.min_lr / args.lr, 1.0)

    def lr_lambda(step):
        if warmup_steps > 0 and step < warmup_steps:
            return float(step + 1) / float(max(1, warmup_steps))
        progress = (step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch=start_step - 1)


def compute_weighted_cls_loss(pos_loss, neg_loss, extra_neg_loss, args, has_extra):
    pos_weight = args.pos_weight
    neg_weight = args.neg_weight
    extra_weight = args.extra_neg_weight

    if has_extra and extra_neg_loss is not None:
        denom = max(1e-8, pos_weight + neg_weight + extra_weight)
        return (pos_weight * pos_loss + neg_weight * neg_loss + extra_weight * extra_neg_loss) / denom

    denom = max(1e-8, pos_weight + neg_weight)
    return (pos_weight * pos_loss + neg_weight * neg_loss) / denom


def evaluate(model, val_loader, device, args, epoch):
    model.eval()
    has_negatives = args.neg_data_path is not None
    use_neg_gw = args.use_neg_gw

    val_loss = 0.0
    val_pos_acc = 0.0
    val_neg_acc = 0.0
    val_extra_acc = 0.0
    val_batches = 0

    with torch.no_grad():
        for batch_data in val_loader:
            is_neg_gw_batch = None
            if use_neg_gw:
                if has_negatives:
                    (
                        gw_input, opt_time, opt_val, opt_mask, opt_err, opt_coords, _gw_idx,
                        neg_time, neg_val, neg_mask, neg_err, neg_coords, is_neg_gw_batch
                    ) = batch_data
                else:
                    gw_input, opt_time, opt_val, opt_mask, opt_err, opt_coords, _gw_idx, is_neg_gw_batch = batch_data
                    neg_time = neg_val = neg_mask = neg_err = neg_coords = None
            else:
                if has_negatives:
                    (
                        gw_input, opt_time, opt_val, opt_mask, opt_err, opt_coords, _gw_idx,
                        neg_time, neg_val, neg_mask, neg_err, neg_coords
                    ) = batch_data
                else:
                    gw_input, opt_time, opt_val, opt_mask, opt_err, opt_coords, _gw_idx = batch_data
                    neg_time = neg_val = neg_mask = neg_err = neg_coords = None

            gw_input = gw_input.to(device, non_blocking=True)
            opt_time = opt_time.to(device, non_blocking=True)
            opt_val = opt_val.to(device, non_blocking=True)
            opt_mask = opt_mask.to(device, non_blocking=True)
            opt_err = opt_err.to(device, non_blocking=True)
            opt_coords = opt_coords.to(device, non_blocking=True)

            batch_size = gw_input.size(0)
            opt_ref_t = build_ref_time(batch_size, args.n_ref, args.ref_start, args.ref_end, device, opt_time.dtype)

            g, _, h_l = model.encode(gw_input, opt_coords, opt_time, opt_val, opt_ref_t, opt_mask, opt_err)

            labels_pos = torch.ones(batch_size, device=device, dtype=torch.long)
            labels_neg = torch.zeros(batch_size, device=device, dtype=torch.long)
            gw_indices = _gw_idx.to(device) if torch.is_tensor(_gw_idx) else torch.tensor(_gw_idx, device=device)
            if is_neg_gw_batch is not None:
                is_neg_gw_batch = is_neg_gw_batch.to(device, non_blocking=True)
                if is_neg_gw_batch.any():
                    labels_pos = torch.where(is_neg_gw_batch, labels_neg, labels_pos)
                    gw_indices = gw_indices.clone()
                    neg_positions = torch.where(is_neg_gw_batch)[0]
                    gw_indices[neg_positions] = -(neg_positions + 1).to(gw_indices.dtype)

            logits_pos = model.fusion_logits(g, h_l)
            pos_loss = model.cls_criterion(logits_pos, labels_pos)
            pos_acc = (logits_pos.argmax(dim=1) == labels_pos).float().mean().item()

            neg_idx = sample_easy_negatives(batch_size, device, gw_indices=gw_indices)
            if neg_idx is None:
                neg_loss = pos_loss * 0.0
                neg_acc = 0.0
            else:
                h_l_neg = h_l[neg_idx]
                logits_neg = model.fusion_logits(g, h_l_neg)
                neg_loss = model.cls_criterion(logits_neg, labels_neg)
                neg_acc = (logits_neg.argmax(dim=1) == labels_neg).float().mean().item()

            extra_neg_loss = None
            extra_acc = 0.0
            if has_negatives:
                neg_time = neg_time.to(device, non_blocking=True)
                neg_val = neg_val.to(device, non_blocking=True)
                neg_mask = neg_mask.to(device, non_blocking=True)
                neg_err = neg_err.to(device, non_blocking=True)
                neg_coords = neg_coords.to(device, non_blocking=True)

                _, h_l_neg_extra = model.encode_optical(
                    neg_coords, neg_time, neg_val, opt_ref_t, neg_mask, neg_err
                )
                logits_neg_extra = model.fusion_logits(g, h_l_neg_extra)
                extra_neg_loss = model.cls_criterion(logits_neg_extra, labels_neg)
                extra_acc = (logits_neg_extra.argmax(dim=1) == labels_neg).float().mean().item()

            cls_loss = compute_weighted_cls_loss(pos_loss, neg_loss, extra_neg_loss, args, has_negatives)

            val_loss += cls_loss.item()
            val_pos_acc += pos_acc
            val_neg_acc += neg_acc
            val_extra_acc += extra_acc
            val_batches += 1

    if val_batches == 0:
        return {
            "loss": 0.0,
            "pos_acc": 0.0,
            "neg_acc": 0.0,
            "extra_acc": 0.0
        }

    return {
        "loss": val_loss / val_batches,
        "pos_acc": val_pos_acc / val_batches,
        "neg_acc": val_neg_acc / val_batches,
        "extra_acc": val_extra_acc / val_batches
    }


def train_with_config(
    config: dict,
    data_path: str,
    neg_data_path: str = None,
    neg_group: str = "events/optical_data",
    max_epochs: int = 50,
    trial=None,
    verbose: bool = False
):
    """
    Train model with given config dict, return best validation loss.
    Used for hyperparameter optimization with Optuna.

    Args:
        config: Dictionary containing hyperparameters
        data_path: Path to training data HDF5 file
        neg_data_path: Path to negative samples HDF5 file
        neg_group: Group name in negative HDF5 file
        max_epochs: Maximum number of training epochs
        trial: Optuna trial object for pruning (optional)
        verbose: Whether to print progress

    Returns:
        best_val_loss: Best validation loss achieved during training
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with h5py.File(data_path, 'r') as f:
        gw_scalar_dim = f['events/gw_data/scalars'].shape[1]
        gw_pixel_dim = f['events/optical_data/gw_pixel_features'].shape[1]
    gw_input_dim = gw_scalar_dim + gw_pixel_dim

    # Extract config values with defaults
    batch_size = config.get('batch_size', 128)
    samples_per_gw = config.get('samples_per_gw', 8)
    val_split = config.get('val_split', 0.2)
    neg_gw_ratio = config.get('neg_gw_ratio', 0.2)
    use_neg_gw = config.get('use_neg_gw', True)

    n_ref = config.get('n_ref', 64)
    ref_start = config.get('ref_start', -0.3)
    ref_end = config.get('ref_end', 0.6)
    ref_dim = config.get('ref_dim', 64)
    enc_dim = config.get('enc_dim', 128)
    optical_dim = config.get('optical_dim', 6)
    fusion_attn_dim = config.get('fusion_attn_dim', None)
    fusion_hidden_dim = config.get('fusion_hidden_dim', None)
    fusion_dropout = config.get('fusion_dropout', 0.1)
    gw_dropout = config.get('gw_dropout', 0.1)
    opt_dropout = config.get('opt_dropout', 0.1)
    label_smoothing = config.get('label_smoothing', 0.0)

    lr = config.get('lr', 1e-4)
    weight_decay = config.get('weight_decay', 1e-3)
    grad_clip_norm = config.get('grad_clip_norm', 1.0)
    lr_scheduler_type = config.get('lr_scheduler', 'none')
    warmup_epochs = config.get('warmup_epochs', 0)
    min_lr = config.get('min_lr', 0.0)

    pos_weight = config.get('pos_weight', 1.0)
    neg_weight = config.get('neg_weight', 0.5)
    extra_neg_weight = config.get('extra_neg_weight', 0.5)

    # Create a simple namespace for args compatibility
    class Args:
        pass
    args_obj = Args()
    args_obj.lr = lr
    args_obj.lr_scheduler = lr_scheduler_type
    args_obj.warmup_epochs = warmup_epochs
    args_obj.min_lr = min_lr
    args_obj.epochs = max_epochs
    args_obj.pos_weight = pos_weight
    args_obj.neg_weight = neg_weight
    args_obj.extra_neg_weight = extra_neg_weight
    args_obj.n_ref = n_ref
    args_obj.ref_start = ref_start
    args_obj.ref_end = ref_end
    args_obj.grad_clip_norm = grad_clip_norm
    args_obj.neg_data_path = neg_data_path
    args_obj.use_neg_gw = use_neg_gw

    # Create data loaders
    if use_neg_gw:
        train_loader, val_loader, steps_per_epoch, _ = create_mixed_gw_dataloaders(
            data_path,
            batch_size=batch_size,
            samples_per_gw=samples_per_gw,
            val_split=val_split,
            split_seed=42,
            num_workers=8,
            pin_memory=True,
            persistent_workers=True,
            prefetch_factor=8,
            negative_h5_path=neg_data_path,
            negative_group=neg_group,
            cache_in_memory=True,
            neg_gw_ratio=neg_gw_ratio
        )
    else:
        train_loader, val_loader, steps_per_epoch, _ = create_train_val_dataloaders(
            data_path,
            batch_size=batch_size,
            val_split=val_split,
            split_seed=42,
            num_workers=8,
            pin_memory=True,
            persistent_workers=True,
            prefetch_factor=8,
            negative_h5_path=neg_data_path,
            negative_group=neg_group,
            cache_in_memory=True
        )

    model = GWOpticalFusionModel(
        gw_input_dim=gw_input_dim,
        optical_input_dim=optical_dim,
        ref_time_dim=ref_dim,
        enc_dim=enc_dim,
        fusion_attn_dim=fusion_attn_dim,
        fusion_hidden_dim=fusion_hidden_dim,
        fusion_dropout=fusion_dropout,
        gw_dropout=gw_dropout,
        opt_dropout=opt_dropout,
        label_smoothing=label_smoothing
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    lr_scheduler = build_lr_scheduler(optimizer, args_obj, steps_per_epoch, 0)

    best_val_loss = float('inf')
    has_negatives = neg_data_path is not None

    for epoch in range(max_epochs):
        model.train()
        epoch_loss = 0.0

        iterator = train_loader if not verbose else tqdm(train_loader, desc=f"Epoch {epoch+1}")
        for batch_data in iterator:
            is_neg_gw_batch = None
            if use_neg_gw:
                if has_negatives:
                    (
                        gw_input, opt_time, opt_val, opt_mask, opt_err, opt_coords, _gw_idx,
                        neg_time, neg_val, neg_mask, neg_err, neg_coords, is_neg_gw_batch
                    ) = batch_data
                else:
                    gw_input, opt_time, opt_val, opt_mask, opt_err, opt_coords, _gw_idx, is_neg_gw_batch = batch_data
                    neg_time = neg_val = neg_mask = neg_err = neg_coords = None
            else:
                if has_negatives:
                    (
                        gw_input, opt_time, opt_val, opt_mask, opt_err, opt_coords, _gw_idx,
                        neg_time, neg_val, neg_mask, neg_err, neg_coords
                    ) = batch_data
                else:
                    gw_input, opt_time, opt_val, opt_mask, opt_err, opt_coords, _gw_idx = batch_data
                    neg_time = neg_val = neg_mask = neg_err = neg_coords = None

            gw_input = gw_input.to(device, non_blocking=True)
            opt_time = opt_time.to(device, non_blocking=True)
            opt_val = opt_val.to(device, non_blocking=True)
            opt_mask = opt_mask.to(device, non_blocking=True)
            opt_err = opt_err.to(device, non_blocking=True)
            opt_coords = opt_coords.to(device, non_blocking=True)

            batch_size_actual = gw_input.size(0)
            opt_ref_t = build_ref_time(batch_size_actual, n_ref, ref_start, ref_end, device, opt_time.dtype)

            g, _, h_l = model.encode(gw_input, opt_coords, opt_time, opt_val, opt_ref_t, opt_mask, opt_err)

            labels_pos = torch.ones(batch_size_actual, device=device, dtype=torch.long)
            labels_neg = torch.zeros(batch_size_actual, device=device, dtype=torch.long)
            gw_indices = _gw_idx.to(device) if torch.is_tensor(_gw_idx) else torch.tensor(_gw_idx, device=device)

            if is_neg_gw_batch is not None:
                is_neg_gw_batch = is_neg_gw_batch.to(device, non_blocking=True)
                if is_neg_gw_batch.any():
                    labels_pos = torch.where(is_neg_gw_batch, labels_neg, labels_pos)
                    gw_indices = gw_indices.clone()
                    neg_positions = torch.where(is_neg_gw_batch)[0]
                    gw_indices[neg_positions] = -(neg_positions + 1).to(gw_indices.dtype)

            logits_pos = model.fusion_logits(g, h_l)
            pos_loss = model.cls_criterion(logits_pos, labels_pos)

            neg_idx = sample_easy_negatives(batch_size_actual, device, gw_indices=gw_indices)
            if neg_idx is None:
                neg_loss = pos_loss * 0.0
            else:
                h_l_neg = h_l[neg_idx]
                logits_neg = model.fusion_logits(g, h_l_neg)
                neg_loss = model.cls_criterion(logits_neg, labels_neg)

            extra_neg_loss = None
            if has_negatives:
                neg_time = neg_time.to(device, non_blocking=True)
                neg_val = neg_val.to(device, non_blocking=True)
                neg_mask = neg_mask.to(device, non_blocking=True)
                neg_err = neg_err.to(device, non_blocking=True)
                neg_coords = neg_coords.to(device, non_blocking=True)

                _, h_l_neg_extra = model.encode_optical(
                    neg_coords, neg_time, neg_val, opt_ref_t, neg_mask, neg_err
                )
                logits_neg_extra = model.fusion_logits(g, h_l_neg_extra)
                extra_neg_loss = model.cls_criterion(logits_neg_extra, labels_neg)

            cls_loss = compute_weighted_cls_loss(pos_loss, neg_loss, extra_neg_loss, args_obj, has_negatives)

            optimizer.zero_grad(set_to_none=True)
            cls_loss.backward()
            if grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
            optimizer.step()
            if lr_scheduler is not None:
                lr_scheduler.step()

            epoch_loss += cls_loss.item()

        # Validation
        val_metrics = evaluate(model, val_loader, device, args_obj, epoch)
        val_loss = val_metrics['loss']

        if val_loss < best_val_loss:
            best_val_loss = val_loss

        if verbose:
            print(f"Epoch {epoch+1}: train_loss={epoch_loss/len(train_loader):.4f}, val_loss={val_loss:.4f}")

        # Optuna pruning
        if trial is not None:
            trial.report(val_loss, epoch)
            if trial.should_prune():
                import optuna
                raise optuna.TrialPruned()

    return best_val_loss


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    with h5py.File(args.data_path, 'r') as f:
        gw_scalar_dim = f['events/gw_data/scalars'].shape[1]
        gw_pixel_dim = f['events/optical_data/gw_pixel_features'].shape[1]

    gw_input_dim = gw_scalar_dim + gw_pixel_dim
    print(f"GW input dim: {gw_input_dim} (scalars={gw_scalar_dim}, pixel={gw_pixel_dim})")

    if args.use_neg_gw:
        train_loader, val_loader, steps_per_epoch, val_steps = create_mixed_gw_dataloaders(
            args.data_path,
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
            negative_h5_path=args.neg_data_path,
            negative_group=args.neg_group,
            cache_in_memory=args.cache_in_memory,
            neg_gw_ratio=args.neg_gw_ratio
        )
    else:
        train_loader, val_loader, steps_per_epoch, val_steps = create_train_val_dataloaders(
            args.data_path,
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
            negative_h5_path=args.neg_data_path,
            negative_group=args.neg_group,
            cache_in_memory=args.cache_in_memory
        )

    model = GWOpticalFusionModel(
        gw_input_dim=gw_input_dim,
        optical_input_dim=args.optical_dim,
        ref_time_dim=args.ref_dim,
        enc_dim=args.enc_dim,
        fusion_attn_dim=args.fusion_attn_dim,
        fusion_hidden_dim=args.fusion_hidden_dim,
        fusion_dropout=args.fusion_dropout,
        gw_dropout=args.gw_dropout,
        opt_dropout=args.opt_dropout,
        label_smoothing=args.label_smoothing
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    lr_scheduler = build_lr_scheduler(optimizer, args, steps_per_epoch, 0)

    log_dir = args.log_dir
    if log_dir is None:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        log_dir = os.path.join(args.ckpt_path, "logs", f"run_{timestamp}")
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir)

    best_val = None
    epochs_no_improve = 0
    global_step = 0

    has_negatives = args.neg_data_path is not None
    use_neg_gw = args.use_neg_gw

    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0.0
        epoch_pos_acc = 0.0
        epoch_neg_acc = 0.0
        epoch_extra_acc = 0.0

        progress = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        for batch_idx, batch_data in enumerate(progress):
            is_neg_gw_batch = None
            if use_neg_gw:
                if has_negatives:
                    (
                        gw_input, opt_time, opt_val, opt_mask, opt_err, opt_coords, _gw_idx,
                        neg_time, neg_val, neg_mask, neg_err, neg_coords, is_neg_gw_batch
                    ) = batch_data
                else:
                    gw_input, opt_time, opt_val, opt_mask, opt_err, opt_coords, _gw_idx, is_neg_gw_batch = batch_data
                    neg_time = neg_val = neg_mask = neg_err = neg_coords = None
            else:
                if has_negatives:
                    (
                        gw_input, opt_time, opt_val, opt_mask, opt_err, opt_coords, _gw_idx,
                        neg_time, neg_val, neg_mask, neg_err, neg_coords
                    ) = batch_data
                else:
                    gw_input, opt_time, opt_val, opt_mask, opt_err, opt_coords, _gw_idx = batch_data
                    neg_time = neg_val = neg_mask = neg_err = neg_coords = None

            gw_input = gw_input.to(device, non_blocking=True)
            opt_time = opt_time.to(device, non_blocking=True)
            opt_val = opt_val.to(device, non_blocking=True)
            opt_mask = opt_mask.to(device, non_blocking=True)
            opt_err = opt_err.to(device, non_blocking=True)
            opt_coords = opt_coords.to(device, non_blocking=True)

            batch_size = gw_input.size(0)
            opt_ref_t = build_ref_time(batch_size, args.n_ref, args.ref_start, args.ref_end, device, opt_time.dtype)

            g, _, h_l = model.encode(gw_input, opt_coords, opt_time, opt_val, opt_ref_t, opt_mask, opt_err)

            labels_pos = torch.ones(batch_size, device=device, dtype=torch.long)
            labels_neg = torch.zeros(batch_size, device=device, dtype=torch.long)
            gw_indices = _gw_idx.to(device) if torch.is_tensor(_gw_idx) else torch.tensor(_gw_idx, device=device)
            if is_neg_gw_batch is not None:
                is_neg_gw_batch = is_neg_gw_batch.to(device, non_blocking=True)
                if is_neg_gw_batch.any():
                    labels_pos = torch.where(is_neg_gw_batch, labels_neg, labels_pos)
                    gw_indices = gw_indices.clone()
                    neg_positions = torch.where(is_neg_gw_batch)[0]
                    gw_indices[neg_positions] = -(neg_positions + 1).to(gw_indices.dtype)

            logits_pos = model.fusion_logits(g, h_l)
            pos_loss = model.cls_criterion(logits_pos, labels_pos)
            pos_acc = (logits_pos.argmax(dim=1) == labels_pos).float().mean().item()

            neg_idx = sample_easy_negatives(batch_size, device, gw_indices=gw_indices)
            if neg_idx is None:
                neg_loss = pos_loss * 0.0
                neg_acc = 0.0
            else:
                h_l_neg = h_l[neg_idx]
                logits_neg = model.fusion_logits(g, h_l_neg)
                neg_loss = model.cls_criterion(logits_neg, labels_neg)
                neg_acc = (logits_neg.argmax(dim=1) == labels_neg).float().mean().item()

            extra_neg_loss = None
            extra_acc = 0.0
            if has_negatives:
                neg_time = neg_time.to(device, non_blocking=True)
                neg_val = neg_val.to(device, non_blocking=True)
                neg_mask = neg_mask.to(device, non_blocking=True)
                neg_err = neg_err.to(device, non_blocking=True)
                neg_coords = neg_coords.to(device, non_blocking=True)

                _, h_l_neg_extra = model.encode_optical(
                    neg_coords, neg_time, neg_val, opt_ref_t, neg_mask, neg_err
                )
                logits_neg_extra = model.fusion_logits(g, h_l_neg_extra)
                extra_neg_loss = model.cls_criterion(logits_neg_extra, labels_neg)
                extra_acc = (logits_neg_extra.argmax(dim=1) == labels_neg).float().mean().item()

            cls_loss = compute_weighted_cls_loss(pos_loss, neg_loss, extra_neg_loss, args, has_negatives)

            optimizer.zero_grad(set_to_none=True)
            cls_loss.backward()
            if args.grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm)
            optimizer.step()
            if lr_scheduler is not None:
                lr_scheduler.step()

            epoch_loss += cls_loss.item()
            epoch_pos_acc += pos_acc
            epoch_neg_acc += neg_acc
            epoch_extra_acc += extra_acc

            if batch_idx % args.log_every == 0:
                writer.add_scalar('Train/Batch_Loss', cls_loss.item(), global_step)
                writer.add_scalar('Train/Batch_Pos_Acc', pos_acc, global_step)
                writer.add_scalar('Train/Batch_Neg_Acc', neg_acc, global_step)
                if has_negatives:
                    writer.add_scalar('Train/Batch_Extra_Acc', extra_acc, global_step)

            progress.set_postfix({
                'loss': f"{cls_loss.item():.4f}",
                'pos_acc': f"{pos_acc:.2f}",
                'neg_acc': f"{neg_acc:.2f}"
            })

            global_step += 1

        avg_loss = epoch_loss / len(train_loader)
        avg_pos = epoch_pos_acc / len(train_loader)
        avg_neg = epoch_neg_acc / len(train_loader)
        avg_extra = epoch_extra_acc / len(train_loader)

        writer.add_scalar('Train/Epoch_Loss', avg_loss, epoch)
        writer.add_scalar('Train/Epoch_Pos_Acc', avg_pos, epoch)
        writer.add_scalar('Train/Epoch_Neg_Acc', avg_neg, epoch)
        if has_negatives:
            writer.add_scalar('Train/Epoch_Extra_Acc', avg_extra, epoch)

        print(f"Epoch {epoch+1}: loss={avg_loss:.4f} pos_acc={avg_pos:.3f} neg_acc={avg_neg:.3f}")

        if val_loader is not None:
            val_metrics = evaluate(model, val_loader, device, args, epoch)
            writer.add_scalar('Val/Epoch_Loss', val_metrics['loss'], epoch)
            writer.add_scalar('Val/Epoch_Pos_Acc', val_metrics['pos_acc'], epoch)
            writer.add_scalar('Val/Epoch_Neg_Acc', val_metrics['neg_acc'], epoch)
            if has_negatives:
                writer.add_scalar('Val/Epoch_Extra_Acc', val_metrics['extra_acc'], epoch)
            print(
                f"Val: loss={val_metrics['loss']:.4f} pos_acc={val_metrics['pos_acc']:.3f} "
                f"neg_acc={val_metrics['neg_acc']:.3f}"
            )

            if best_val is None or val_metrics['loss'] < best_val - args.early_stop_min_delta:
                best_val = val_metrics['loss']
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= args.early_stop_patience:
                    print("Early stopping triggered.")
                    break

        if args.ckpt_path is not None:
            os.makedirs(args.ckpt_path, exist_ok=True)
            checkpoint_path = os.path.join(args.ckpt_path, f"fusion_epoch_{epoch+1}.pth")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss
            }, checkpoint_path)

    writer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--neg_data_path", type=str, default=None)
    parser.add_argument("--neg_group", type=str, default="events/optical_data")
    parser.add_argument("--use_neg_gw", action='store_true',
                        help="Include negative GW events (BNS without KN) in training")
    parser.add_argument("--neg_gw_ratio", type=float, default=0.2,
                        help="Ratio of negative GW samples per batch (default: 0.2)")
    parser.add_argument("--samples_per_gw", type=int, default=4,
                        help="Number of optical samples per positive GW event per batch")
    parser.add_argument("--ckpt_path", type=str, default=None)
    parser.add_argument("--log_dir", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--steps_per_epoch", type=int, default=None)
    parser.add_argument("--val_steps_per_epoch", type=int, default=None)
    parser.add_argument("--val_batch_size", type=int, default=None)
    parser.add_argument("--val_split", type=float, default=0.1)
    parser.add_argument("--split_seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--pin_memory", type=int, default=1)
    parser.add_argument("--persistent_workers", type=int, default=1)
    parser.add_argument("--prefetch_factor", type=int, default=4)
    parser.add_argument("--cache_in_memory", action='store_true')
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--grad_clip_norm", type=float, default=1.0)
    parser.add_argument("--lr_scheduler", type=str, default="none", choices=["none", "cosine"])
    parser.add_argument("--warmup_epochs", type=int, default=0)
    parser.add_argument("--min_lr", type=float, default=0.0)
    parser.add_argument("--n_ref", type=int, default=64)
    parser.add_argument("--ref_start", type=float, default=-0.3)
    parser.add_argument("--ref_end", type=float, default=0.6)
    parser.add_argument("--ref_dim", type=int, default=64)
    parser.add_argument("--enc_dim", type=int, default=128)
    parser.add_argument("--optical_dim", type=int, default=6)
    parser.add_argument("--fusion_attn_dim", type=int, default=None)
    parser.add_argument("--fusion_hidden_dim", type=int, default=None)
    parser.add_argument("--fusion_dropout", type=float, default=0.1)
    parser.add_argument("--gw_dropout", type=float, default=0.1)
    parser.add_argument("--opt_dropout", type=float, default=0.1)
    parser.add_argument("--label_smoothing", type=float, default=0.0)
    parser.add_argument("--pos_weight", type=float, default=1.0)
    parser.add_argument("--neg_weight", type=float, default=1.0)
    parser.add_argument("--extra_neg_weight", type=float, default=1.0)
    parser.add_argument("--early_stop_patience", type=int, default=10)
    parser.add_argument("--early_stop_min_delta", type=float, default=1e-4)
    parser.add_argument("--log_every", type=int, default=50)

    args = parser.parse_args()

    if not os.path.exists(args.data_path):
        raise FileNotFoundError(f"Data file not found: {args.data_path}")
    if args.ckpt_path is None:
        raise ValueError("ckpt_path must be provided.")

    train(args)
