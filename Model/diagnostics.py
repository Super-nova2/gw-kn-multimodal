"""
Diagnostic script for GW-Optical ALBEF training.

Loads one batch from each dataloader mode and checks:
1. Event-ID distributions and batch composition
2. False-negative rate for contrastive (in-batch)
3. Cross-modal positive mask correctness
4. SpatialEmbedding coordinate consistency
5. Embedding variance (collapse detection) and cosine similarity statistics

Usage:
    python diagnostics.py --data_path /path/to/combined_dataset.h5 \
        [--neg_data_path /path/to/neg.h5] [--batch_size 32] [--samples_per_gw 4]
"""

import sys
import os
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import math

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data_loader import (
    build_gw_to_lc_mapping,
    RelationalHDF5Dataset,
    BalancedGWBatchedSampler,
    MultiPositiveGWBatchedSampler,
    split_gw_map,
    _build_dataloader,
)
from model import GWOpticalALBEFModel


def header(title):
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")


def check_1_batch_composition(batch_data, has_negatives):
    """Check event-ID distributions."""
    header("1. Batch Composition & Event-ID Distribution")

    if has_negatives:
        (gw_s, gw_m, opt_t, opt_v, opt_mask, opt_err, opt_coords,
         gw_indices, neg_t, neg_v, neg_mask, neg_err, neg_coords) = batch_data
    else:
        gw_s, gw_m, opt_t, opt_v, opt_mask, opt_err, opt_coords, gw_indices = batch_data

    B = gw_s.size(0)
    gw_indices = gw_indices.long()
    unique_gw = gw_indices.unique()

    print(f"Batch size:           {B}")
    print(f"Unique GW indices:    {len(unique_gw)}")
    print(f"GW index range:       [{gw_indices.min().item()}, {gw_indices.max().item()}]")

    # Duplicates
    counts = torch.bincount(gw_indices - gw_indices.min())
    dup_mask = counts > 1
    n_dup_events = dup_mask.sum().item()
    print(f"Events with >1 sample: {n_dup_events}  (samples_per_gw > 1 mode)")

    # Data integrity
    print(f"\nOptical values  - NaN: {torch.isnan(opt_v).any().item()}, "
          f"Inf: {torch.isinf(opt_v).any().item()}")
    print(f"Optical errors  - NaN: {torch.isnan(opt_err).any().item()}, "
          f"Inf: {torch.isinf(opt_err).any().item()}")
    print(f"Optical mask    - mean fill rate: {opt_mask.mean().item():.4f}")
    print(f"GW scalars      - NaN: {torch.isnan(gw_s).any().item()}, "
          f"Inf: {torch.isinf(gw_s).any().item()}")
    print(f"GW skymap       - NaN: {torch.isnan(gw_m).any().item()}, "
          f"Inf: {torch.isinf(gw_m).any().item()}")

    return gw_s, gw_m, opt_t, opt_v, opt_mask, opt_err, opt_coords, gw_indices


def check_2_false_negative_rate(gw_indices):
    """Compute false-negative rate: fraction of (anchor, negative) pairs
    where negative.event_id == anchor.event_id."""
    header("2. False-Negative Rate (In-Batch)")

    B = gw_indices.size(0)
    same_event = gw_indices.unsqueeze(0) == gw_indices.unsqueeze(1)  # [B, B]
    diag = torch.eye(B, dtype=torch.bool)
    off_diag_same = same_event & ~diag

    total_off_diag = B * (B - 1)
    n_false_neg = off_diag_same.sum().item()
    rate = n_false_neg / total_off_diag if total_off_diag > 0 else 0

    print(f"Total off-diagonal pairs:  {total_off_diag}")
    print(f"Same-event off-diag pairs: {n_false_neg}")
    print(f"False-negative rate:       {rate:.6f}")

    if rate > 0:
        print("WARNING: Non-zero false-negative rate! Same-event samples "
              "appear as negatives in contrastive.")
        rows, cols = torch.where(off_diag_same)
        for r, c in zip(rows[:5].tolist(), cols[:5].tolist()):
            print(f"  Pair ({r},{c}): gw_idx={gw_indices[r].item()} == {gw_indices[c].item()}")
    else:
        print("OK: No in-batch false negatives detected.")


def check_3_cross_modal_positive_mask(gw_indices):
    """Verify SupCon positive mask includes cross-modal pairs."""
    header("3. Cross-Modal Positive Mask Correctness (SupCon)")

    B = gw_indices.size(0)
    labels = torch.cat([gw_indices, gw_indices], dim=0)  # [2B]
    labels_eq = labels.unsqueeze(0) == labels.unsqueeze(1)  # [2B, 2B]
    mask_pos = labels_eq.float()
    mask_pos.fill_diagonal_(0)

    # Check a few GW anchors (first half, positions 0..B-1)
    print("Checking first 3 GW anchors for cross-modal positives:")
    for i in range(min(3, B)):
        pos_indices = torch.where(mask_pos[i] > 0)[0].tolist()
        gw_positives = [p for p in pos_indices if p < B]
        opt_positives = [p - B for p in pos_indices if p >= B]

        print(f"\n  GW anchor {i} (event={gw_indices[i].item()}):")
        print(f"    Within-GW positives (indices <B):  {gw_positives}")
        print(f"    Cross-modal optical positives:      {opt_positives}")

        if len(opt_positives) == 0:
            print("    WARNING: No cross-modal positives! SupCon won't align GW<->Opt.")
        else:
            for oi in opt_positives[:3]:
                print(f"      opt[{oi}] event={gw_indices[oi].item()} "
                      f"{'MATCH' if gw_indices[oi] == gw_indices[i] else 'MISMATCH!'}")

    # Check optical anchors (second half, positions B..2B-1)
    print("\nChecking first 3 Optical anchors for cross-modal positives:")
    for i in range(min(3, B)):
        anchor_idx = B + i
        pos_indices = torch.where(mask_pos[anchor_idx] > 0)[0].tolist()
        gw_positives = [p for p in pos_indices if p < B]
        opt_positives = [p - B for p in pos_indices if p >= B]

        print(f"\n  Opt anchor {i} (event={gw_indices[i].item()}):")
        print(f"    Cross-modal GW positives:           {gw_positives}")
        print(f"    Within-Opt positives (indices >=B):  {opt_positives}")

        if len(gw_positives) == 0:
            print("    WARNING: No cross-modal GW positives! Alignment broken.")


def check_4_spatial_embedding_consistency():
    """Verify SpatialEmbedding coordinate convention vs skymap convention."""
    header("4. SpatialEmbedding Coordinate Consistency Check")

    from model import SpatialEmbedding

    # Test with actual model forward pass
    emb = SpatialEmbedding(output_dim=64)
    emb.eval()

    # Test point: equator (RA=0, Dec=0)
    coords_equator = torch.tensor([[0.0, 0.0]])  # RA=0, Dec=0
    coords_npole = torch.tensor([[0.0, 90.0]])    # north pole

    theta_eq = (90.0 - 0.0) * (math.pi / 180.0)  # pi/2
    phi_eq = 0.0

    # Expected Cartesian (HEALPix standard)
    x_correct = math.sin(theta_eq) * math.cos(phi_eq)  # 1.0
    z_correct = math.cos(theta_eq)                       # 0.0

    print(f"Equator (RA=0, Dec=0):")
    print(f"  Expected Cartesian: x={x_correct:.4f}, z={z_correct:.4f}")

    # Check the actual conversion in model code
    ra = coords_equator[:, 0]
    dec = coords_equator[:, 1]
    theta = (90.0 - dec) * (math.pi / 180.0)
    phi = ra * (math.pi / 180.0)
    x = torch.sin(theta) * torch.cos(phi)
    y = torch.sin(theta) * torch.sin(phi)
    z = torch.cos(theta)
    print(f"  Model Cartesian:    x={x.item():.4f}, y={y.item():.4f}, z={z.item():.4f}")

    if abs(x.item() - x_correct) < 1e-6 and abs(z.item() - z_correct) < 1e-6:
        print("  OK: SpatialEmbedding coordinates match HEALPix convention.")
    else:
        print("  BUG: SpatialEmbedding coordinates do NOT match HEALPix convention!")


def check_5_embedding_stats(model, gw_s, gw_m, opt_t, opt_v, opt_mask,
                            opt_err, opt_coords, device):
    """Check embedding variance, norms, cosine similarity stats."""
    header("5. Embedding Statistics (Collapse Detection)")

    model.eval()
    B = gw_s.size(0)
    n_ref = 64
    opt_ref_t = torch.linspace(-0.3, 0.6, n_ref).unsqueeze(0).repeat(B, 1).to(device)

    with torch.no_grad():
        g, z_l, h_l = model.encode(
            gw_s.to(device), gw_m.to(device), opt_coords.to(device),
            opt_t.to(device), opt_v.to(device), opt_ref_t,
            opt_mask.to(device), opt_err.to(device)
        )
        feat_g = F.normalize(model.gw_proj(g), p=2, dim=1, eps=1e-8)
        feat_o = F.normalize(model.opt_proj(z_l), p=2, dim=1, eps=1e-8)

    print("GW encoder output (g):")
    print(f"  shape: {g.shape}, norm mean: {g.norm(dim=1).mean():.4f}, "
          f"std per dim: {g.std(dim=0).mean():.6f}")
    print(f"  any NaN: {torch.isnan(g).any().item()}, "
          f"any Inf: {torch.isinf(g).any().item()}")

    print("\nOptical encoder output (z_l):")
    print(f"  shape: {z_l.shape}, norm mean: {z_l.norm(dim=1).mean():.4f}, "
          f"std per dim: {z_l.std(dim=0).mean():.6f}")

    print("\nGW projected (L2-normalized):")
    print(f"  shape: {feat_g.shape}, std per dim: {feat_g.std(dim=0).mean():.6f}")
    if feat_g.std(dim=0).mean() < 1e-3:
        print("  WARNING: Very low variance — possible embedding collapse!")

    print("\nOptical projected (L2-normalized):")
    print(f"  shape: {feat_o.shape}, std per dim: {feat_o.std(dim=0).mean():.6f}")
    if feat_o.std(dim=0).mean() < 1e-3:
        print("  WARNING: Very low variance — possible embedding collapse!")

    # Cosine similarity stats
    sim = torch.matmul(feat_g, feat_o.T)  # [B, B]
    diag_sim = sim.diag()
    off_diag = sim.masked_fill(torch.eye(B, device=device).bool(), float('nan'))

    print(f"\nCosine similarity (GW->Opt):")
    print(f"  Diagonal (matched pairs):     mean={diag_sim.mean():.4f}, "
          f"std={diag_sim.std():.4f}")
    print(f"  Off-diagonal (unmatched):      mean={off_diag.nanmean():.4f}, "
          f"std={off_diag[~off_diag.isnan()].std():.4f}")

    gap = diag_sim.mean() - off_diag.nanmean()
    print(f"  Gap (diag - off-diag):        {gap:.4f}")
    if gap < 0.05:
        print("  WARNING: Very small gap — contrastive may not be working!")

    # GW-GW similarity (should be diverse, not collapsed)
    sim_gg = torch.matmul(feat_g, feat_g.T)
    off_diag_gg = sim_gg.masked_fill(torch.eye(B, device=device).bool(), float('nan'))
    print(f"\nGW-GW cosine similarity (off-diag): mean={off_diag_gg.nanmean():.4f}")
    if off_diag_gg.nanmean() > 0.95:
        print("  WARNING: GW embeddings are nearly identical — collapsed!")

    sim_oo = torch.matmul(feat_o, feat_o.T)
    off_diag_oo = sim_oo.masked_fill(torch.eye(B, device=device).bool(), float('nan'))
    print(f"Opt-Opt cosine similarity (off-diag): mean={off_diag_oo.nanmean():.4f}")
    if off_diag_oo.nanmean() > 0.95:
        print("  WARNING: Optical embeddings are nearly identical — collapsed!")

    # Temperature
    temp = model.log_temp.exp().item()
    print(f"\nModel temperature: {temp:.4f}")


def main():
    parser = argparse.ArgumentParser(description="ALBEF Training Diagnostics")
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--neg_data_path", type=str, default=None)
    parser.add_argument("--neg_group", type=str, default="events/optical_data")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--samples_per_gw", type=int, default=4)
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to model checkpoint for embedding stats")
    args = parser.parse_args()

    has_negatives = args.neg_data_path is not None

    print("=" * 70)
    print("  GW-Optical ALBEF Diagnostics")
    print("=" * 70)
    print(f"Data path:     {args.data_path}")
    print(f"Neg data path: {args.neg_data_path}")
    print(f"Batch size:    {args.batch_size}")
    print(f"Samples/GW:    {args.samples_per_gw}")

    # Build mapping
    gw_map = build_gw_to_lc_mapping(args.data_path)
    train_map, val_map = split_gw_map(gw_map, 0.1, 42)

    # Create dataset
    dataset = RelationalHDF5Dataset(
        args.data_path,
        negative_h5_path=args.neg_data_path,
        negative_group=args.neg_group,
        cache_in_memory=False,
    )

    # Create appropriate sampler
    if args.samples_per_gw > 1:
        sampler = MultiPositiveGWBatchedSampler(
            gw_to_lc_map=train_map,
            batch_size=args.batch_size,
            samples_per_gw=args.samples_per_gw,
            steps_per_epoch=1,
        )
    else:
        sampler = BalancedGWBatchedSampler(
            gw_to_lc_map=train_map,
            batch_size=args.batch_size,
            steps_per_epoch=1,
        )

    loader = _build_dataloader(dataset, sampler, num_workers=0,
                               pin_memory=False, persistent_workers=False,
                               prefetch_factor=2)

    # Get one batch
    batch_data = next(iter(loader))

    # Run checks 1-4 (no model needed)
    (gw_s, gw_m, opt_t, opt_v, opt_mask, opt_err, opt_coords,
     gw_indices) = check_1_batch_composition(batch_data, has_negatives)

    gw_indices = gw_indices.long()

    check_2_false_negative_rate(gw_indices)
    check_3_cross_modal_positive_mask(gw_indices)
    check_4_spatial_embedding_consistency()

    # Check 5: embedding stats (requires model)
    if args.checkpoint is not None and os.path.exists(args.checkpoint):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        ckpt = torch.load(args.checkpoint, map_location=device)
        ckpt_args = argparse.Namespace(**ckpt.get("args", {}))
        model = GWOpticalALBEFModel(
            enc_dim=getattr(ckpt_args, "enc_dim", 128),
            proj_dim=getattr(ckpt_args, "proj_dim", 256),
            ref_time_dim=getattr(ckpt_args, "ref_dim", 64),
            use_lightweight_gw=getattr(ckpt_args, "use_lightweight_gw", False),
        ).to(device)
        model.load_state_dict(ckpt["model_state_dict"], strict=False)
        check_5_embedding_stats(
            model, gw_s, gw_m, opt_t, opt_v, opt_mask, opt_err, opt_coords, device
        )
    else:
        print("\n\nSkipping Check 5 (embedding stats): no --checkpoint provided.")

    header("DONE")
    print("Review the output above for warnings and bugs.\n")


if __name__ == "__main__":
    main()
