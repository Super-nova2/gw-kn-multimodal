"""
Comprehensive evaluation script for GW-KN ALBEF model.

Supports:
    - Batch-mode retrieval metrics (same as training eval)
    - Gallery-mode retrieval (realistic candidate pool evaluation)
    - Classification metrics (AUROC, AUPRC, F1, ECE)
    - Embedding quality metrics (alignment, uniformity)
    - Visualizations (ROC, PR curve, t-SNE, calibration, logits distribution)

New Features:
    - Support for external negative samples (non-KN transients)
    - Logits distribution plot for three pair types:
      * Positive pairs (GW, KN): matched GW-KN optical pairs
      * Easy negatives (GW, nonKN): GW paired with non-KN transients
      * Hard negatives (GW_wrong, KN): mismatched GW paired with KN optical

Usage:
    python test_evaluate.py \\
        --checkpoint /path/to/albef_best.pth \\
        --test_data_path /fred/oz016/bgao_kn/data/LSST_KN_BNS/combined_dataset_with_neg_gw.h5 \\
        --neg_data_path /fred/oz016/bgao_kn/data/ELASTICC2_TRAIN/negative_dataset.h5 \\
        --neg_group ELASTICC2_TRAIN/optical_data \\
        --output_dir /path/to/eval_results
"""

import argparse
import json
import os
import sys
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from torch.amp import autocast
from tqdm import tqdm

# Add parent dir for local imports
sys.path.insert(0, str(Path(__file__).resolve().parent))

from data_loader import (
    RelationalHDF5Dataset,
    BalancedGWBatchedSampler,
    _build_dataloader,
)
from model import GWOpticalALBEFModel
from metrics import (
    compute_retrieval_metrics,
    compute_classification_metrics,
    compute_embedding_metrics,
)


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate GW-KN ALBEF model")

    # Checkpoint
    p.add_argument("--checkpoint", type=str, required=True)

    # Data source
    p.add_argument("--test_data_path", type=str, required=True,
                   help="Independent test HDF5 file")
    p.add_argument("--neg_data_path", type=str,
                   default="/fred/oz016/bgao_kn/data/ELASTICC2_TRAIN/negative_dataset.h5",
                   help="Path to negative (non-KN transient) HDF5 file")
    p.add_argument("--neg_group", type=str, default="ELASTICC2_TRAIN/optical_data",
                   help="HDF5 group path for negative optical data")
    p.add_argument("--n_neg_samples", type=int, default=5000,
                   help="Number of negative samples to use for logits distribution")
    p.add_argument("--test_steps", type=int, default=None,
                   help="Number of test sampling steps. If None, auto-computed to match n_neg_samples")

    # Eval parameters
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--gallery_sizes", type=str, default="100,500,1000",
                   help="Comma-separated gallery sizes for gallery-mode eval")
    p.add_argument("--gallery_trials", type=int, default=10,
                   help="Number of random gallery compositions per GW event")

    # Training config (required if checkpoint lacks saved 'args')
    p.add_argument("--config", type=str, default=None,
                   help="Path to training config JSON (e.g. args/ALBEF_supcon.json)")

    # Output
    p.add_argument("--output_dir", type=str, default="eval_results")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--no_plots", action="store_true",
                   help="Skip generating plots (useful on headless machines)")

    return p.parse_args()


def load_model(args, device):
    """Load model from checkpoint, reconstructing architecture from saved args or config."""
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)

    # Priority: checkpoint saved args > config JSON file
    saved_args = ckpt.get("args", {})
    if not saved_args and args.config:
        with open(args.config) as f:
            saved_args = json.load(f)
        print(f"Loaded config from {args.config}")
    elif not saved_args:
        print("WARNING: Checkpoint has no saved args and no --config provided. "
              "Using defaults which may not match training.")

    def _get(key, default):
        return saved_args.get(key, default)

    model_args = {
        "enc_dim": _get("enc_dim", 128),
        "proj_dim": _get("proj_dim", 256),
        "n_ref": _get("n_ref", 64),
        "ref_start": _get("ref_start", -0.3),
        "ref_end": _get("ref_end", 0.6),
        "ref_dim": _get("ref_dim", 64),
        "use_lightweight_gw": _get("use_lightweight_gw", False),
        "gw_dropout": _get("gw_dropout", 0.0),
        "opt_dropout": _get("opt_dropout", 0.0),
        "proj_dropout": _get("proj_dropout", 0.0),
        "fusion_dropout": _get("fusion_dropout", 0.0),
        "feature_dropout": _get("feature_dropout", 0.0),
        "label_smoothing": _get("label_smoothing", 0.0),
        "temp_init": _get("temp_init", 0.07),
        "temp_min": _get("temp_min", 0.01),
        "temp_max": _get("temp_max", 100.0),
        "itc_label_smoothing": _get("itc_label_smoothing", 0.0),
        "fusion_attn_dim": _get("fusion_attn_dim", None),
        "fusion_hidden_dim": _get("fusion_hidden_dim", None),
    }

    model = GWOpticalALBEFModel(
        enc_dim=model_args["enc_dim"],
        proj_dim=model_args["proj_dim"],
        ref_time_dim=model_args["ref_dim"],
        use_lightweight_gw=model_args["use_lightweight_gw"],
        gw_dropout=model_args["gw_dropout"],
        opt_dropout=model_args["opt_dropout"],
        proj_dropout=model_args["proj_dropout"],
        fusion_dropout=model_args["fusion_dropout"],
        feature_dropout=model_args["feature_dropout"],
        label_smoothing=model_args["label_smoothing"],
        temp_init=model_args["temp_init"],
        temp_min=model_args["temp_min"],
        temp_max=model_args["temp_max"],
        itc_label_smoothing=model_args["itc_label_smoothing"],
        fusion_attn_dim=model_args["fusion_attn_dim"],
        fusion_hidden_dim=model_args["fusion_hidden_dim"],
    )
    # Strip _orig_mod. prefix from torch.compile'd checkpoints
    state_dict = ckpt["model_state_dict"]
    state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    print(f"Loaded checkpoint from epoch {ckpt.get('epoch', '?')}, "
          f"loss={ckpt.get('loss', '?')}")
    return model, model_args, saved_args


def build_test_dataloader(args, saved_args):
    """Build test dataloader from independent test HDF5.
    
    The number of steps is controlled by args.test_steps. If None,
    it is auto-computed to match args.n_neg_samples (so that the number
    of sampled light curves is comparable to the number of negative samples).
    """
    from data_loader import build_gw_to_lc_mapping

    dataset = RelationalHDF5Dataset(
        args.test_data_path,
        negative_h5_path=args.neg_data_path,
        negative_group=args.neg_group,
    )
    gw_to_lc = build_gw_to_lc_mapping(args.test_data_path)
    n_gw = len(gw_to_lc)
    batch_size = min(args.batch_size, n_gw)
    
    # Compute steps: either user-specified or auto-computed to match n_neg_samples
    if args.test_steps is not None:
        steps = args.test_steps
    else:
        # Auto-compute: target n_samples ~ n_neg_samples
        # Each step samples batch_size light curves
        target_samples = args.n_neg_samples
        steps = max(1, (target_samples + batch_size - 1) // batch_size)
    
    print(f"Test sampling: {steps} steps x {batch_size} batch_size = ~{steps * batch_size} samples")
    
    sampler = BalancedGWBatchedSampler(
        gw_to_lc, batch_size=batch_size,
        steps_per_epoch=steps,
    )

    loader = _build_dataloader(
        dataset, sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
        prefetch_factor=2 if args.num_workers > 0 else None,
    )
    return loader, dataset


def build_ref_time(batch_size, n_ref, ref_start, ref_end, device, dtype):
    ref = torch.linspace(ref_start, ref_end, n_ref, dtype=dtype, device=device)
    return ref.unsqueeze(0).repeat(batch_size, 1)


def load_negative_optical_samples(neg_data_path, neg_group, n_samples=5000, seed=42):
    """Load negative optical samples (non-KN transients) from external HDF5 file.
    
    Args:
        neg_data_path: Path to negative dataset HDF5 file
        neg_group: HDF5 group path for optical data
        n_samples: Number of samples to load
        seed: Random seed for sampling
        
    Returns:
        dict with keys: 'values', 'times', 'masks', 'errors', 'coordinates', 'types'
    """
    if neg_data_path is None or not os.path.exists(neg_data_path):
        print(f"WARNING: Negative data path not found: {neg_data_path}")
        return None
    
    rng = np.random.default_rng(seed)
    
    with h5py.File(neg_data_path, 'r') as f:
        grp = f[neg_group]
        total_samples = grp['values'].shape[0]
        
        # Randomly sample indices
        n_samples = min(n_samples, total_samples)
        sample_indices = rng.choice(total_samples, size=n_samples, replace=False)
        sample_indices = np.sort(sample_indices)  # Sort for efficient HDF5 access
        
        print(f"Loading {n_samples} negative optical samples from {neg_data_path}")
        
        # Load data
        neg_data = {
            'values': torch.from_numpy(grp['values'][sample_indices]),
            'times': torch.from_numpy(grp['times'][sample_indices]),
            'masks': torch.from_numpy(grp['masks'][sample_indices]),
            'errors': torch.from_numpy(grp['errors'][sample_indices]),
            'coordinates': torch.from_numpy(grp['coordinates'][sample_indices]),
        }
        
        # Load types if available
        if 'types' in grp:
            neg_data['types'] = [grp['types'][i].decode() if isinstance(grp['types'][i], bytes) 
                                 else grp['types'][i] for i in sample_indices]
        
    return neg_data


@torch.no_grad()
def extract_all_embeddings(model, loader, device, model_args):
    """Run full forward pass, collecting embeddings and predictions.

    For classification evaluation, both matched (positive) and mismatched
    (negative) GW-optical pairs are created.  Negatives are formed by
    shifting the optical sequence within each batch so that each GW is
    paired with an unrelated optical LC.
    """
    all_feat_g = []
    all_feat_o = []
    all_gw_indices = []
    all_logits = []     # pos + neg logits
    all_labels = []     # 1 for matched, 0 for mismatched
    all_sim_blocks = []
    all_gw_idx_blocks = []

    ref_time_cache = None
    n_ref = model_args["n_ref"]
    ref_start = model_args["ref_start"]
    ref_end = model_args["ref_end"]

    for batch_data in tqdm(loader, desc="Extracting embeddings"):
        # Unpack
        if len(batch_data) == 13:
            (gw_s, gw_m, opt_t, opt_v, opt_mask, opt_err, opt_coords,
             gw_indices, neg_t, neg_v, neg_mask, neg_err, neg_coords) = batch_data
        else:
            gw_s, gw_m, opt_t, opt_v, opt_mask, opt_err, opt_coords, gw_indices = batch_data[:8]

        gw_s = gw_s.to(device)
        gw_m = gw_m.to(device)
        opt_t = opt_t.to(device)
        opt_v = opt_v.to(device)
        opt_mask = opt_mask.to(device)
        opt_err = opt_err.to(device)
        opt_coords = opt_coords.to(device)
        gw_indices = gw_indices.to(device).long()

        batch_size = gw_s.size(0)
        if (ref_time_cache is None or ref_time_cache.shape[0] != batch_size
                or ref_time_cache.dtype != opt_t.dtype):
            ref_time_cache = build_ref_time(
                batch_size, n_ref, ref_start, ref_end, device, opt_t.dtype
            )
        opt_ref_t = ref_time_cache

        with autocast(device_type='cuda', dtype=torch.float16,
                      enabled=(device.type == 'cuda')):
            g, z_l, h_l = model.encode(
                gw_s, gw_m, opt_coords, opt_t, opt_v, opt_ref_t, opt_mask, opt_err
            )
            feat_g = F.normalize(model.gw_proj(g), p=2, dim=1, eps=1e-8)
            feat_o = F.normalize(model.opt_proj(z_l), p=2, dim=1, eps=1e-8)

            # Positive pairs (matched GW-optical)
            logits_pos = model.fusion_logits(g, h_l)

            # Negative pairs (shift optical by 1 so each GW pairs with wrong optical)
            shift = 1
            h_l_neg = torch.roll(h_l, shifts=shift, dims=0)
            logits_neg = model.fusion_logits(g, h_l_neg)

            # Similarity matrix for retrieval
            temperature = model.log_temp.exp().clamp(
                min=model.temp_min, max=model.temp_max
            )
            sim_g2o = torch.matmul(feat_g, feat_o.T) / temperature

        all_feat_g.append(feat_g.float().cpu())
        all_feat_o.append(feat_o.float().cpu())
        all_gw_indices.append(gw_indices.cpu())
        # Concatenate pos and neg logits + labels
        all_logits.append(logits_pos.float().cpu())
        all_logits.append(logits_neg.float().cpu())
        all_labels.append(torch.ones(batch_size, dtype=torch.long))
        all_labels.append(torch.zeros(batch_size, dtype=torch.long))
        all_sim_blocks.append(sim_g2o.float().cpu())
        all_gw_idx_blocks.append(gw_indices.cpu())

    return {
        "feat_g": torch.cat(all_feat_g),
        "feat_o": torch.cat(all_feat_o),
        "gw_indices": torch.cat(all_gw_indices),
        "logits": torch.cat(all_logits),
        "labels": torch.cat(all_labels),
        "sim_blocks": all_sim_blocks,
        "gw_idx_blocks": all_gw_idx_blocks,
    }


@torch.no_grad()
def extract_triplet_logits(model, loader, device, model_args, neg_optical_data, 
                           shuffle_gw=False, shuffle_seed=42):
    """Extract logits for three types of sample pairs:
    
    1. Positive pairs (GW, KN): matched GW-KN optical pairs
    2. Easy negatives (GW, nonKN): GW paired with non-KN transients
    3. Semi-hard negatives (GW_wrong, KN): mismatched GW paired with KN optical
       - Uses similarity-based semi-hard negative mining: for each optical,
         select the most similar wrong GW that is still less similar than
         the correct GW (below positive similarity).
    
    Args:
        model: Trained ALBEF model
        loader: DataLoader for test data
        device: Torch device
        model_args: Model configuration dict
        neg_optical_data: Dict with negative optical samples from external file
        shuffle_gw: If True, randomly shuffle GW within each batch (ablation test)
        shuffle_seed: Random seed for GW shuffling
        
    Returns:
        dict with logits_positive, logits_easy_neg, logits_hard_neg, and types
    """
    logits_positive = []    # (GW, matched KN)
    logits_easy_neg = []    # (GW, non-KN transient)
    logits_hard_neg = []    # (wrong GW, KN) - semi-hard
    easy_neg_types = []     # Type of non-KN transient
    
    n_ref = model_args["n_ref"]
    ref_start = model_args["ref_start"]
    ref_end = model_args["ref_end"]
    
    # RNG for GW shuffling
    if shuffle_gw:
        shuffle_rng = torch.Generator(device='cpu')
        shuffle_rng.manual_seed(shuffle_seed)
    
    # Pre-process negative optical data if available
    has_neg_optical = neg_optical_data is not None
    neg_idx = 0
    
    if has_neg_optical:
        neg_values = neg_optical_data['values'].to(device)
        neg_times = neg_optical_data['times'].to(device)
        neg_masks = neg_optical_data['masks'].to(device)
        neg_errors = neg_optical_data['errors'].to(device)
        neg_coords = neg_optical_data['coordinates'].to(device)
        neg_types_list = neg_optical_data.get('types', ['unknown'] * len(neg_values))
        total_neg = len(neg_values)
    
    for batch_data in tqdm(loader, desc="Extracting triplet logits"):
        # Unpack - get positive KN data
        if len(batch_data) >= 8:
            gw_s, gw_m, opt_t, opt_v, opt_mask, opt_err, opt_coords, gw_indices = batch_data[:8]
        else:
            continue
            
        gw_s = gw_s.to(device)
        gw_m = gw_m.to(device)
        opt_t = opt_t.to(device)
        opt_v = opt_v.to(device)
        opt_mask = opt_mask.to(device)
        opt_err = opt_err.to(device)
        opt_coords = opt_coords.to(device)
        gw_indices_dev = gw_indices.to(device).long()
        
        batch_size = gw_s.size(0)
        
        # GW shuffle ablation: randomly permute GW within batch
        if shuffle_gw:
            perm = torch.randperm(batch_size, generator=shuffle_rng)
            gw_s = gw_s[perm]
            gw_m = gw_m[perm]
            # Note: gw_indices_dev is NOT shuffled to keep semi-hard mining consistent
        
        ref_time = build_ref_time(batch_size, n_ref, ref_start, ref_end, device, opt_t.dtype)
        
        with autocast(device_type='cuda', dtype=torch.float16,
                      enabled=(device.type == 'cuda')):
            # Encode GW and KN optical
            g, z_l, h_l = model.encode(
                gw_s, gw_m, opt_coords, opt_t, opt_v, ref_time, opt_mask, opt_err
            )
            
            # 1. Positive pairs: matched GW-KN
            logits_pos = model.fusion_logits(g, h_l)
            logits_positive.append(logits_pos.float().cpu())
            
            # 2. Semi-hard negatives: use similarity-based semi-hard negative mining
            # Compute projected features for similarity matrix
            feat_g = F.normalize(model.gw_proj(g), p=2, dim=1, eps=1e-8)
            feat_o = F.normalize(model.opt_proj(z_l), p=2, dim=1, eps=1e-8)
            
            # Compute similarity matrix (O2G direction: for each optical, find wrong GW)
            temperature = model.log_temp.exp().clamp(min=model.temp_min, max=model.temp_max)
            sim_o2g = torch.matmul(feat_o, feat_g.T) / temperature  # [batch_opt, batch_gw]
            
            # Sample semi-hard negatives: for each optical, find the most similar 
            # wrong GW that is still below positive similarity
            semi_hard_gw_idx = sample_semi_hard_negatives_for_optical(
                sim_o2g, gw_indices_dev
            )
            
            # Get semi-hard negative GW features and compute logits
            g_semihard = g[semi_hard_gw_idx]
            logits_hard = model.fusion_logits(g_semihard, h_l)
            logits_hard_neg.append(logits_hard.float().cpu())
            
            # 3. Easy negatives: correct GW paired with non-KN transients
            if has_neg_optical:
                # Get batch of negative optical samples
                batch_neg_indices = []
                batch_neg_types = []
                for i in range(batch_size):
                    idx = neg_idx % total_neg
                    batch_neg_indices.append(idx)
                    batch_neg_types.append(neg_types_list[idx])
                    neg_idx += 1
                
                # Get negative optical data for this batch
                neg_v_batch = neg_values[batch_neg_indices]
                neg_t_batch = neg_times[batch_neg_indices]
                neg_m_batch = neg_masks[batch_neg_indices]
                neg_e_batch = neg_errors[batch_neg_indices]
                neg_c_batch = neg_coords[batch_neg_indices]
                
                # Encode negative optical with the same GW
                ref_time_neg = build_ref_time(batch_size, n_ref, ref_start, ref_end, 
                                              device, neg_t_batch.dtype)
                _, _, h_l_neg = model.encode(
                    gw_s, gw_m, neg_c_batch, neg_t_batch, neg_v_batch, 
                    ref_time_neg, neg_m_batch, neg_e_batch
                )
                
                logits_easy = model.fusion_logits(g, h_l_neg)
                logits_easy_neg.append(logits_easy.float().cpu())
                easy_neg_types.extend(batch_neg_types)
    
    result = {
        "logits_positive": torch.cat(logits_positive) if logits_positive else None,
        "logits_hard_neg": torch.cat(logits_hard_neg) if logits_hard_neg else None,
    }
    
    if has_neg_optical and logits_easy_neg:
        result["logits_easy_neg"] = torch.cat(logits_easy_neg)
        result["easy_neg_types"] = easy_neg_types
    else:
        result["logits_easy_neg"] = None
        result["easy_neg_types"] = []
    
    return result


def sample_semi_hard_negatives_for_optical(sim_o2g, gw_indices):
    """
    Sample semi-hard negative GW indices for each optical sample.
    
    For each optical sample i, find the GW j that:
    1. Is NOT from the same GW event as optical i
    2. Has similarity < positive similarity (sim with correct GW)
    3. Has the highest similarity among all such negatives
    
    This implements semi-hard negative mining: selecting the hardest negative
    that is still easier than the positive, which provides informative gradients.
    
    Reference: model.py sample_semi_hard_negatives
    
    Args:
        sim_o2g: Similarity matrix [batch_opt, batch_gw], Optical-to-GW similarities
        gw_indices: GW event indices for each sample [batch]
        
    Returns:
        semi_hard_idx: Indices of semi-hard negative GW for each optical sample [batch]
    """
    sim = sim_o2g.detach()
    batch_size = sim.size(0)
    device = sim.device
    
    # Create same-event mask: gw_indices[i] == gw_indices[j]
    same_event = gw_indices.unsqueeze(0) == gw_indices.unsqueeze(1)  # [batch, batch]
    
    # Get positive similarity for each sample (diagonal of same-event pairs)
    # For test set with unique GW per sample, this is the diagonal
    pos_sim = sim.diagonal()  # [batch] - similarity with correct GW
    
    # Mask out same-event pairs from negative candidates
    neg_sim = sim.masked_fill(same_event, -1e9)
    
    # For each optical, find negatives with sim < pos_sim
    # Then select the one with maximum similarity (semi-hard)
    result = torch.zeros(batch_size, dtype=torch.long, device=device)
    
    for i in range(batch_size):
        # Find negatives below positive similarity
        below_mask = (neg_sim[i] < pos_sim[i]) & (neg_sim[i] > -1e8)
        below_idx = below_mask.nonzero(as_tuple=False).squeeze(-1)
        
        if below_idx.numel() > 0:
            # Select the negative with highest similarity (closest to positive)
            best_local = neg_sim[i, below_idx].argmax()
            result[i] = below_idx[best_local]
        else:
            # Fallback: all negatives exceed pos_sim, pick random negative
            neg_idx = (~same_event[i]).nonzero(as_tuple=False).squeeze(-1)
            if neg_idx.numel() > 0:
                pick = torch.randint(0, neg_idx.numel(), (1,), device=device)
                result[i] = neg_idx[pick]
            else:
                # Edge case: no negatives available (shouldn't happen)
                result[i] = (i + 1) % batch_size
    
    return result


def evaluate_retrieval_batch_mode(embeddings, ks=(1, 5, 10)):
    """Per-batch retrieval metrics (same as training eval)."""
    results_accum = []
    for sim, gi in zip(embeddings["sim_blocks"], embeddings["gw_idx_blocks"]):
        r = compute_retrieval_metrics(sim, gi, ks=ks)
        results_accum.append(r)

    avg = {}
    if results_accum:
        for key in results_accum[0]:
            avg[key] = sum(d[key] for d in results_accum) / len(results_accum)
    return avg


def evaluate_retrieval_gallery_mode(embeddings, gallery_sizes, n_trials=10,
                                    seed=42):
    """
    Realistic retrieval: for each test GW, build a candidate pool of N optical
    samples (1 correct + N-1 distractors) and rank by cosine similarity.
    """
    feat_g = embeddings["feat_g"]
    feat_o = embeddings["feat_o"]
    gw_indices = embeddings["gw_indices"]

    unique_gw = torch.unique(gw_indices)
    rng = np.random.default_rng(seed)

    results = {}
    for N in gallery_sizes:
        recalls = {1: [], 5: [], 10: []}
        mrrs = []

        for trial in range(n_trials):
            for gw_id in unique_gw:
                gw_mask = gw_indices == gw_id
                other_mask = gw_indices != gw_id

                if gw_mask.sum() == 0 or other_mask.sum() == 0:
                    continue

                # Pick one GW embedding as query
                gw_idxs = torch.where(gw_mask)[0]
                query_idx = gw_idxs[rng.integers(len(gw_idxs))]
                query = feat_g[query_idx]

                # Pick one correct optical
                correct_idx = gw_idxs[rng.integers(len(gw_idxs))]
                correct_opt = feat_o[correct_idx]

                # Pick N-1 distractors from other events
                other_idxs = torch.where(other_mask)[0].numpy()
                n_distract = min(N - 1, len(other_idxs))
                distract_idxs = rng.choice(other_idxs, size=n_distract,
                                           replace=False)
                distract_opt = feat_o[distract_idxs]

                # Build gallery: correct at position 0
                gallery = torch.cat([correct_opt.unsqueeze(0), distract_opt])
                sims = torch.matmul(gallery, query)
                ranked = sims.argsort(descending=True)

                # Position of the correct answer (index 0) in ranking
                correct_rank = (ranked == 0).nonzero(as_tuple=True)[0].item()

                for k in recalls:
                    recalls[k].append(1.0 if correct_rank < k else 0.0)
                mrrs.append(1.0 / (correct_rank + 1))

        for k in recalls:
            key = f"gallery_{N}_recall_at_{k}"
            results[key] = float(np.mean(recalls[k])) if recalls[k] else 0.0
        results[f"gallery_{N}_mrr"] = float(np.mean(mrrs)) if mrrs else 0.0

    return results


def evaluate_classification(embeddings):
    """Global classification metrics from fusion head (pos + neg pairs)."""
    logits = embeddings["logits"]
    labels = embeddings["labels"]
    probs = torch.softmax(logits.float(), dim=1)[:, 1]
    return compute_classification_metrics(probs, labels)


def evaluate_embeddings(embeddings):
    """Embedding quality metrics."""
    return compute_embedding_metrics(
        embeddings["feat_g"],
        embeddings["feat_o"],
        embeddings["gw_indices"],
    )


def generate_plots(embeddings, results, output_dir):
    """Generate evaluation plots."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available, skipping plots")
        return

    os.makedirs(output_dir, exist_ok=True)

    # --- 1. ROC Curve ---
    logits = embeddings["logits"]
    labels = embeddings["labels"]
    probs = torch.softmax(logits.float(), dim=1)[:, 1].numpy()
    labels_np = labels.numpy()

    sorted_idx = np.argsort(-probs)
    sorted_labels = labels_np[sorted_idx]
    n_pos = sorted_labels.sum()
    n_neg = len(sorted_labels) - n_pos

    if n_pos > 0 and n_neg > 0:
        tpr = np.cumsum(sorted_labels) / n_pos
        fpr = np.cumsum(1 - sorted_labels) / n_neg
        tpr = np.concatenate([[0], tpr])
        fpr = np.concatenate([[0], fpr])

        fig, ax = plt.subplots(figsize=(6, 6))
        ax.plot(fpr, tpr, lw=2)
        ax.plot([0, 1], [0, 1], "k--", lw=1)
        auroc = results.get("classification", {}).get("auroc", 0)
        ax.set_title(f"ROC Curve (AUROC={auroc:.4f})")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        fig.savefig(os.path.join(output_dir, "roc_curve.png"), dpi=150,
                    bbox_inches="tight")
        plt.close(fig)

    # --- 2. Precision-Recall Curve ---
    if n_pos > 0:
        tp_cum = np.cumsum(sorted_labels)
        ranks = np.arange(1, len(sorted_labels) + 1)
        precision = tp_cum / ranks
        recall = tp_cum / n_pos

        fig, ax = plt.subplots(figsize=(6, 6))
        ax.plot(recall, precision, lw=2)
        auprc = results.get("classification", {}).get("auprc", 0)
        ax.set_title(f"Precision-Recall Curve (AUPRC={auprc:.4f})")
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        fig.savefig(os.path.join(output_dir, "pr_curve.png"), dpi=150,
                    bbox_inches="tight")
        plt.close(fig)

    # --- 3. Calibration Plot ---
    n_bins = 10
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_accs = []
    bin_confs = []
    bin_counts = []
    for i in range(n_bins):
        low, high = bin_edges[i], bin_edges[i + 1]
        mask = (probs >= low) & (probs < high) if i < n_bins - 1 else (probs >= low) & (probs <= high)
        if mask.sum() > 0:
            bin_accs.append(labels_np[mask].mean())
            bin_confs.append(probs[mask].mean())
            bin_counts.append(mask.sum())

    if bin_accs:
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.bar(bin_confs, bin_accs, width=0.08, alpha=0.7, label="Model")
        ax.plot([0, 1], [0, 1], "k--", lw=1, label="Perfect")
        ece = results.get("classification", {}).get("ece", 0)
        ax.set_title(f"Calibration (ECE={ece:.4f})")
        ax.set_xlabel("Mean Predicted Probability")
        ax.set_ylabel("Fraction of Positives")
        ax.legend()
        fig.savefig(os.path.join(output_dir, "calibration.png"), dpi=150,
                    bbox_inches="tight")
        plt.close(fig)

    # --- 4. Gallery-mode Recall@1 vs Pool Size ---
    gallery_results = results.get("retrieval_gallery", {})
    if gallery_results:
        sizes = sorted(set(
            int(k.split("_")[1]) for k in gallery_results if "recall_at_1" in k
        ))
        r1_values = [gallery_results.get(f"gallery_{s}_recall_at_1", 0)
                     for s in sizes]

        if sizes:
            fig, ax = plt.subplots(figsize=(7, 5))
            ax.plot(sizes, r1_values, "o-", lw=2)
            ax.set_xlabel("Gallery Size (number of candidates)")
            ax.set_ylabel("Recall@1")
            ax.set_title("Recall@1 vs Candidate Pool Size")
            ax.set_xscale("log")
            ax.grid(True, alpha=0.3)
            fig.savefig(os.path.join(output_dir, "recall_vs_gallery.png"),
                        dpi=150, bbox_inches="tight")
            plt.close(fig)

    # --- 5. t-SNE of embeddings ---
    try:
        from sklearn.manifold import TSNE
        n_sample = min(2000, len(embeddings["feat_g"]))
        perm = np.random.RandomState(42).permutation(len(embeddings["feat_g"]))[:n_sample]
        fg = embeddings["feat_g"][perm].numpy()
        fo = embeddings["feat_o"][perm].numpy()
        gi = embeddings["gw_indices"][perm].numpy()

        combined = np.concatenate([fg, fo])
        tsne = TSNE(n_components=2, random_state=42, perplexity=30)
        coords = tsne.fit_transform(combined)

        fig, ax = plt.subplots(figsize=(10, 8))
        # Color by GW event, shape by modality
        unique_events = np.unique(gi)
        colors = plt.cm.tab20(np.linspace(0, 1, min(20, len(unique_events))))

        for i, ev in enumerate(unique_events[:20]):
            mask = gi == ev
            c = colors[i % len(colors)]
            idx_gw = np.where(mask)[0]
            idx_opt = np.where(mask)[0] + n_sample
            ax.scatter(coords[idx_gw, 0], coords[idx_gw, 1],
                       c=[c], marker="^", s=40, alpha=0.7)
            ax.scatter(coords[idx_opt, 0], coords[idx_opt, 1],
                       c=[c], marker="o", s=30, alpha=0.7)

        ax.set_title("t-SNE Embedding Space (^=GW, o=Optical)")
        fig.savefig(os.path.join(output_dir, "tsne_embeddings.png"), dpi=150,
                    bbox_inches="tight")
        plt.close(fig)
    except ImportError:
        print("sklearn not available, skipping t-SNE plot")

    print(f"Plots saved to {output_dir}/")


def generate_logits_distribution_plot(triplet_logits, output_dir):
    """Generate logits distribution plot for three types of sample pairs.
    
    Plot types:
    1. Positive pairs (GW, KN): matched GW-KN optical pairs
    2. Easy negatives (GW, nonKN): GW paired with non-KN transients  
    3. Hard negatives (GW_wrong, KN): mismatched GW paired with KN optical
    
    Args:
        triplet_logits: Dict with logits_positive, logits_easy_neg, logits_hard_neg
        output_dir: Directory to save plots
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.ticker import MaxNLocator
    except ImportError:
        print("matplotlib not available, skipping logits distribution plot")
        return
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract probabilities (class 1 = match probability)
    probs_pos = None
    probs_easy = None
    probs_hard = None
    
    if triplet_logits.get("logits_positive") is not None:
        probs_pos = torch.softmax(triplet_logits["logits_positive"].float(), dim=1)[:, 1].numpy()
    if triplet_logits.get("logits_easy_neg") is not None:
        probs_easy = torch.softmax(triplet_logits["logits_easy_neg"].float(), dim=1)[:, 1].numpy()
    if triplet_logits.get("logits_hard_neg") is not None:
        probs_hard = torch.softmax(triplet_logits["logits_hard_neg"].float(), dim=1)[:, 1].numpy()
    
    # --- 1. Main distribution plot ---
    fig, ax = plt.subplots(figsize=(10, 6))
    
    bins = np.linspace(0, 1, 51)
    alpha = 0.6
    
    if probs_pos is not None:
        ax.hist(probs_pos, bins=bins, alpha=alpha, label=f'Positive (GW, KN) n={len(probs_pos)}', 
                color='#2ecc71', edgecolor='white', linewidth=0.5)
    if probs_easy is not None:
        ax.hist(probs_easy, bins=bins, alpha=alpha, label=f'Easy Neg (GW, nonKN) n={len(probs_easy)}',
                color='#3498db', edgecolor='white', linewidth=0.5)
    if probs_hard is not None:
        ax.hist(probs_hard, bins=bins, alpha=alpha, label=f'Hard Neg (GW_wrong, KN) n={len(probs_hard)}',
                color='#e74c3c', edgecolor='white', linewidth=0.5)
    
    ax.set_xlabel('Match Probability (Softmax Output)', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title('Classification Head Logits Distribution\nby Sample Pair Type', fontsize=14)
    ax.legend(loc='upper center', fontsize=10)
    ax.set_xlim(0, 1)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    
    # Add vertical line at 0.5 threshold
    ax.axvline(x=0.5, color='black', linestyle='--', linewidth=1.5, alpha=0.7, label='Threshold=0.5')
    
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "logits_distribution_triplet.png"), dpi=200,
                bbox_inches="tight")
    plt.close(fig)
    
    # --- 2. KDE density plot for better visualization ---
    try:
        from scipy import stats
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        x_range = np.linspace(0, 1, 200)
        
        if probs_pos is not None and len(probs_pos) > 1:
            kde_pos = stats.gaussian_kde(probs_pos, bw_method=0.05)
            ax.plot(x_range, kde_pos(x_range), color='#27ae60', linewidth=2.5,
                   label=f'Positive (GW, KN)')
            
        if probs_easy is not None and len(probs_easy) > 1:
            kde_easy = stats.gaussian_kde(probs_easy, bw_method=0.05)
            ax.plot(x_range, kde_easy(x_range), color='#2980b9', linewidth=2.5,
                   label=f'Easy Neg (GW, nonKN)')
            
        if probs_hard is not None and len(probs_hard) > 1:
            kde_hard = stats.gaussian_kde(probs_hard, bw_method=0.05)
            ax.plot(x_range, kde_hard(x_range), color='#c0392b', linewidth=2.5,
                   label=f'Semi-Hard Neg (GW_wrong, KN)')
        
        ax.set_xlabel('Match Probability (Softmax Output)', fontsize=12)
        ax.set_ylabel('Density', fontsize=12)
        ax.set_title('Classification Head Output Distribution (KDE)\nby Sample Pair Type', fontsize=14)
        ax.legend(loc='upper center', fontsize=10)
        ax.set_xlim(0, 1)
        ax.axvline(x=0.5, color='black', linestyle='--', linewidth=1.5, alpha=0.7)
        ax.grid(True, alpha=0.3, linestyle='--')
        
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, "logits_distribution_kde.png"), dpi=200,
                    bbox_inches="tight")
        plt.close(fig)
    except ImportError:
        print("scipy not available, skipping KDE plot")
    
    # Print summary statistics
    print("\n--- Logits Distribution Summary ---")
    if probs_pos is not None:
        print(f"  Positive (GW, KN):     mean={np.mean(probs_pos):.4f}  std={np.std(probs_pos):.4f}  "
              f"median={np.median(probs_pos):.4f}  n={len(probs_pos)}")
    if probs_easy is not None:
        print(f"  Easy Neg (GW, nonKN):  mean={np.mean(probs_easy):.4f}  std={np.std(probs_easy):.4f}  "
              f"median={np.median(probs_easy):.4f}  n={len(probs_easy)}")
    if probs_hard is not None:
        print(f"  Hard Neg (GW_wrong, KN): mean={np.mean(probs_hard):.4f}  std={np.std(probs_hard):.4f}  "
              f"median={np.median(probs_hard):.4f}  n={len(probs_hard)}")
    
    print(f"Logits distribution plots saved to {output_dir}/")


def generate_gw_shuffle_comparison_plot(triplet_logits_normal, triplet_logits_shuffle, output_dir):
    """Generate comparison plot for GW-shuffle ablation test.
    
    Compares logits distributions before and after shuffling GW inputs.
    If distributions are similar after shuffle, the model doesn't use GW info effectively.
    
    Args:
        triplet_logits_normal: Dict with logits from normal forward pass
        triplet_logits_shuffle: Dict with logits from GW-shuffled forward pass
        output_dir: Directory to save plots
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from scipy import stats
    except ImportError:
        print("matplotlib/scipy not available, skipping GW-shuffle comparison plot")
        return
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract probabilities
    def get_probs(logits_dict, key):
        if logits_dict.get(key) is not None:
            return torch.softmax(logits_dict[key].float(), dim=1)[:, 1].numpy()
        return None
    
    probs_pos_normal = get_probs(triplet_logits_normal, "logits_positive")
    probs_easy_normal = get_probs(triplet_logits_normal, "logits_easy_neg")
    probs_hard_normal = get_probs(triplet_logits_normal, "logits_hard_neg")
    
    probs_pos_shuffle = get_probs(triplet_logits_shuffle, "logits_positive")
    probs_easy_shuffle = get_probs(triplet_logits_shuffle, "logits_easy_neg")
    probs_hard_shuffle = get_probs(triplet_logits_shuffle, "logits_hard_neg")
    
    # --- Combined KDE comparison plot ---
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    x_range = np.linspace(0, 1, 200)
    
    pair_types = [
        ('Positive (GW, KN)', probs_pos_normal, probs_pos_shuffle, '#27ae60', '#2ecc71'),
        ('Easy Neg (GW, nonKN)', probs_easy_normal, probs_easy_shuffle, '#2980b9', '#3498db'),
        ('Semi-Hard Neg (GW_wrong, KN)', probs_hard_normal, probs_hard_shuffle, '#c0392b', '#e74c3c'),
    ]
    
    for ax, (title, probs_n, probs_s, color_n, color_s) in zip(axes, pair_types):
        if probs_n is not None and len(probs_n) > 1:
            kde_n = stats.gaussian_kde(probs_n, bw_method=0.05)
            ax.plot(x_range, kde_n(x_range), color=color_n, linewidth=2.5,
                   label=f'Normal (μ={np.mean(probs_n):.3f})')
        
        if probs_s is not None and len(probs_s) > 1:
            kde_s = stats.gaussian_kde(probs_s, bw_method=0.05)
            ax.plot(x_range, kde_s(x_range), color=color_s, linewidth=2.5, linestyle='--',
                   label=f'GW-Shuffle (μ={np.mean(probs_s):.3f})')
        
        ax.set_xlabel('Match Probability', fontsize=11)
        ax.set_ylabel('Density', fontsize=11)
        ax.set_title(title, fontsize=12)
        ax.legend(loc='upper right', fontsize=9)
        ax.set_xlim(0, 1)
        ax.axvline(x=0.5, color='black', linestyle=':', linewidth=1, alpha=0.5)
        ax.grid(True, alpha=0.3, linestyle='--')
    
    fig.suptitle('GW-Shuffle Ablation Test: Effect of Randomizing GW Input', fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "gw_shuffle_comparison.png"), dpi=200,
                bbox_inches="tight")
    plt.close(fig)
    
    # --- All 6 distributions on one plot ---
    fig, ax = plt.subplots(figsize=(12, 6))
    
    if probs_pos_normal is not None and len(probs_pos_normal) > 1:
        kde = stats.gaussian_kde(probs_pos_normal, bw_method=0.05)
        ax.plot(x_range, kde(x_range), color='#27ae60', linewidth=2.5,
               label='Positive - Normal')
    if probs_pos_shuffle is not None and len(probs_pos_shuffle) > 1:
        kde = stats.gaussian_kde(probs_pos_shuffle, bw_method=0.05)
        ax.plot(x_range, kde(x_range), color='#27ae60', linewidth=2.5, linestyle='--',
               label='Positive - Shuffle')
    
    if probs_easy_normal is not None and len(probs_easy_normal) > 1:
        kde = stats.gaussian_kde(probs_easy_normal, bw_method=0.05)
        ax.plot(x_range, kde(x_range), color='#2980b9', linewidth=2.5,
               label='Easy Neg - Normal')
    if probs_easy_shuffle is not None and len(probs_easy_shuffle) > 1:
        kde = stats.gaussian_kde(probs_easy_shuffle, bw_method=0.05)
        ax.plot(x_range, kde(x_range), color='#2980b9', linewidth=2.5, linestyle='--',
               label='Easy Neg - Shuffle')
    
    if probs_hard_normal is not None and len(probs_hard_normal) > 1:
        kde = stats.gaussian_kde(probs_hard_normal, bw_method=0.05)
        ax.plot(x_range, kde(x_range), color='#c0392b', linewidth=2.5,
               label='Semi-Hard Neg - Normal')
    if probs_hard_shuffle is not None and len(probs_hard_shuffle) > 1:
        kde = stats.gaussian_kde(probs_hard_shuffle, bw_method=0.05)
        ax.plot(x_range, kde(x_range), color='#c0392b', linewidth=2.5, linestyle='--',
               label='Semi-Hard Neg - Shuffle')
    
    ax.set_xlabel('Match Probability (Softmax Output)', fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.set_title('GW-Shuffle Ablation: All Distributions Comparison\n(Solid=Normal, Dashed=GW-Shuffled)', fontsize=14)
    ax.legend(loc='upper right', fontsize=10, ncol=2)
    ax.set_xlim(0, 1)
    ax.axvline(x=0.5, color='black', linestyle=':', linewidth=1.5, alpha=0.7)
    ax.grid(True, alpha=0.3, linestyle='--')
    
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "gw_shuffle_all_distributions.png"), dpi=200,
                bbox_inches="tight")
    plt.close(fig)
    
    # Print comparison statistics
    print("\n" + "=" * 70)
    print("GW-SHUFFLE ABLATION TEST RESULTS")
    print("=" * 70)
    print("\nIf distributions are similar after shuffle, model doesn't use GW info.")
    print("If Positive (green) drops significantly, model IS using GW info.\n")
    
    def print_comparison(name, probs_n, probs_s):
        if probs_n is not None and probs_s is not None:
            delta_mean = np.mean(probs_s) - np.mean(probs_n)
            delta_std = np.std(probs_s) - np.std(probs_n)
            # KS test for distribution difference
            ks_stat, ks_pval = stats.ks_2samp(probs_n, probs_s)
            print(f"  {name}:")
            print(f"    Normal:  mean={np.mean(probs_n):.4f}  std={np.std(probs_n):.4f}")
            print(f"    Shuffle: mean={np.mean(probs_s):.4f}  std={np.std(probs_s):.4f}")
            print(f"    Δmean={delta_mean:+.4f}  Δstd={delta_std:+.4f}")
            print(f"    KS-test: stat={ks_stat:.4f}  p-value={ks_pval:.2e}")
            if ks_pval < 0.05:
                print(f"    → Distributions are SIGNIFICANTLY DIFFERENT (p<0.05)")
            else:
                print(f"    → Distributions are NOT significantly different (p≥0.05)")
            print()
    
    print_comparison("Positive (GW, KN)", probs_pos_normal, probs_pos_shuffle)
    print_comparison("Easy Neg (GW, nonKN)", probs_easy_normal, probs_easy_shuffle)
    print_comparison("Semi-Hard Neg (GW_wrong, KN)", probs_hard_normal, probs_hard_shuffle)
    
    print("=" * 70)
    print(f"GW-shuffle comparison plots saved to {output_dir}/")


def save_results(results, output_dir):
    """Save results as JSON."""
    os.makedirs(output_dir, exist_ok=True)

    # Convert numpy types for JSON serialization
    def convert(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        return obj

    results_clean = convert(results)
    path = os.path.join(output_dir, "eval_results.json")
    with open(path, "w") as f:
        json.dump(results_clean, f, indent=2)
    print(f"Results saved to {path}")


def print_summary(results):
    """Print a concise summary of all results."""
    print("\n" + "=" * 70)
    print("EVALUATION SUMMARY")
    print("=" * 70)

    # Retrieval (batch mode)
    ret = results.get("retrieval_batch", {})
    if ret:
        print("\n--- Retrieval (Batch Mode) ---")
        print(f"  GW→Opt  R@1={ret.get('g2o_recall_at_1', 0):.4f}  "
              f"R@5={ret.get('g2o_recall_at_5', 0):.4f}  "
              f"R@10={ret.get('g2o_recall_at_10', 0):.4f}  "
              f"MRR={ret.get('g2o_mrr', 0):.4f}  "
              f"mAP={ret.get('g2o_map', 0):.4f}")
        print(f"  Opt→GW  R@1={ret.get('o2g_recall_at_1', 0):.4f}  "
              f"R@5={ret.get('o2g_recall_at_5', 0):.4f}  "
              f"MRR={ret.get('o2g_mrr', 0):.4f}")

    # Gallery mode
    gal = results.get("retrieval_gallery", {})
    if gal:
        print("\n--- Retrieval (Gallery Mode) ---")
        sizes = sorted(set(
            int(k.split("_")[1]) for k in gal if "recall_at_1" in k
        ))
        for s in sizes:
            r1 = gal.get(f"gallery_{s}_recall_at_1", 0)
            r5 = gal.get(f"gallery_{s}_recall_at_5", 0)
            mrr = gal.get(f"gallery_{s}_mrr", 0)
            print(f"  Pool={s:5d}  R@1={r1:.4f}  R@5={r5:.4f}  MRR={mrr:.4f}")

    # Classification
    cls = results.get("classification", {})
    if cls:
        print("\n--- Classification ---")
        print(f"  AUROC={cls.get('auroc', 0):.4f}  "
              f"AUPRC={cls.get('auprc', 0):.4f}  "
              f"F1={cls.get('f1_optimal', 0):.4f} (t={cls.get('f1_threshold', 0):.2f})  "
              f"ECE={cls.get('ece', 0):.4f}")
        tp = cls.get('tp', 0)
        fp = cls.get('fp', 0)
        tn = cls.get('tn', 0)
        fn = cls.get('fn', 0)
        print(f"  Confusion (t=0.5): TP={tp} FP={fp} TN={tn} FN={fn}")

    # Embedding
    emb = results.get("embedding", {})
    if emb:
        print("\n--- Embedding Quality ---")
        print(f"  Alignment={emb.get('alignment', 0):.4f}  "
              f"Uniformity(GW)={emb.get('uniformity_gw', 0):.4f}  "
              f"Uniformity(Opt)={emb.get('uniformity_opt', 0):.4f}")
        print(f"  Inter-modal gap={emb.get('inter_modal_gap', 0):.4f}  "
              f"Intra-sim={emb.get('intra_sim', 0):.4f}  "
              f"Inter-sim={emb.get('inter_sim', 0):.4f}")

    print("=" * 70)


def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    gallery_sizes = [int(s) for s in args.gallery_sizes.split(",")]

    os.makedirs(args.output_dir, exist_ok=True)

    # Load model
    model, model_args, saved_args = load_model(args, device)

    # Build test dataloader
    loader, dataset = build_test_dataloader(args, saved_args)
    print(f"Test set: {len(loader)} batches")

    # Extract all embeddings
    embeddings = extract_all_embeddings(model, loader, device, model_args)
    n_samples = len(embeddings["feat_g"])
    n_unique_gw = len(torch.unique(embeddings["gw_indices"]))
    print(f"Extracted {n_samples} samples from {n_unique_gw} unique GW events")

    # Compute all metrics
    results = {}
    print("\nComputing batch-mode retrieval metrics...")
    results["retrieval_batch"] = evaluate_retrieval_batch_mode(embeddings)

    print("Computing gallery-mode retrieval metrics...")
    results["retrieval_gallery"] = evaluate_retrieval_gallery_mode(
        embeddings, gallery_sizes, n_trials=args.gallery_trials
    )

    print("Computing classification metrics...")
    results["classification"] = evaluate_classification(embeddings)

    print("Computing embedding quality metrics...")
    results["embedding"] = evaluate_embeddings(embeddings)

    # Load negative optical samples for triplet logits analysis
    neg_optical_data = None
    if args.neg_data_path and os.path.exists(args.neg_data_path):
        neg_optical_data = load_negative_optical_samples(
            args.neg_data_path, 
            args.neg_group, 
            n_samples=args.n_neg_samples
        )

    # Extract triplet logits for distribution analysis
    triplet_logits = None
    triplet_logits_shuffle = None
    if neg_optical_data is not None or True:  # Always extract for hard negatives
        print("\nExtracting triplet logits for distribution analysis...")
        # Rebuild loader to iterate again
        loader2, _ = build_test_dataloader(args, saved_args)
        triplet_logits = extract_triplet_logits(
            model, loader2, device, model_args, neg_optical_data,
            shuffle_gw=False
        )
        
        # GW-shuffle ablation test
        print("\nExtracting triplet logits with GW-SHUFFLE (ablation test)...")
        loader3, _ = build_test_dataloader(args, saved_args)
        triplet_logits_shuffle = extract_triplet_logits(
            model, loader3, device, model_args, neg_optical_data,
            shuffle_gw=True, shuffle_seed=42
        )

    # Save and display
    save_results(results, args.output_dir)
    print_summary(results)

    if not args.no_plots:
        print("\nGenerating plots...")
        generate_plots(embeddings, results, args.output_dir)
        
        # Generate triplet logits distribution plot
        if triplet_logits is not None:
            print("\nGenerating logits distribution plots...")
            generate_logits_distribution_plot(triplet_logits, args.output_dir)
        
        # Generate GW-shuffle comparison plot
        if triplet_logits is not None and triplet_logits_shuffle is not None:
            print("\nGenerating GW-shuffle comparison plots...")
            generate_gw_shuffle_comparison_plot(
                triplet_logits, triplet_logits_shuffle, args.output_dir
            )


if __name__ == "__main__":
    "python test_evaluate.py --checkpoint /fred/oz016/bgao_kn/data/model/checkpoints/supcon_v1/ALBEF/albef_best.pth --test_data_path /fred/oz016/bgao_kn/data/LSST_KN_BNS/combined_dataset_with_neg_gw.h5 --output_dir eval_results"
    main()
