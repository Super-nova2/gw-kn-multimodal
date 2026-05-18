"""
Evaluation metrics for the GW-KN ALBEF model.

Three categories:
    1. Retrieval metrics (contrastive/ITC branch): Recall@K, MRR, mAP
    2. Classification metrics (fusion branch): AUROC, AUPRC, F1, ECE
    3. Embedding quality metrics: alignment, uniformity, inter-modal gap
"""

import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# 1. Retrieval Metrics
# ---------------------------------------------------------------------------

def compute_retrieval_metrics(sim_g2o, gw_indices, is_neg_gw=None, ks=(1, 5, 10)):
    """
    Compute retrieval metrics from the GW-optical similarity matrix.

    In the multi-positive setting (samples_per_gw > 1), a prediction is
    correct if the predicted optical sample belongs to the same GW event
    as the query GW anchor.

    Args:
        sim_g2o: Similarity matrix [B, B] (GW queries x optical gallery).
        gw_indices: GW event index for each sample [B].
        is_neg_gw: Boolean mask [B] where True = negative GW position.
        ks: Tuple of K values for Recall@K.

    Returns:
        Dictionary with g2o and o2g retrieval metrics.
    """
    device = sim_g2o.device
    B = sim_g2o.size(0)

    if is_neg_gw is not None and is_neg_gw.any():
        pos_mask = ~is_neg_gw
    else:
        pos_mask = torch.ones(B, dtype=torch.bool, device=device)

    results = {}

    # GW → Optical direction
    g2o = _retrieval_metrics_one_direction(
        sim_g2o, gw_indices, pos_mask, ks, device
    )
    for k, v in g2o.items():
        results[f"g2o_{k}"] = v

    # Optical → GW direction
    o2g = _retrieval_metrics_one_direction(
        sim_g2o.T, gw_indices, pos_mask, ks, device
    )
    for k, v in o2g.items():
        results[f"o2g_{k}"] = v

    return results


def _retrieval_metrics_one_direction(sim, gw_indices, anchor_mask, ks, device):
    """Compute Recall@K, MRR, mAP for one direction of retrieval."""
    B = sim.size(0)
    G = sim.size(1)  # gallery size (may differ from B for extended galleries)
    max_k = max(ks)

    # Filter to valid anchors
    valid_idx = torch.where(anchor_mask)[0]
    if len(valid_idx) == 0:
        return {f"recall_at_{k}": 0.0 for k in ks} | {"mrr": 0.0, "map": 0.0}

    sim_valid = sim[valid_idx]  # [N_valid, G]
    anchor_gw = gw_indices[valid_idx]  # [N_valid]

    # For each anchor, build positive mask across gallery
    # positive[i, j] = True if gallery item j is a correct match for anchor i
    positive = anchor_gw.unsqueeze(1) == gw_indices.unsqueeze(0)  # [N_valid, G]

    # Sort gallery by descending similarity
    sorted_indices = sim_valid.argsort(dim=1, descending=True)  # [N_valid, G]

    # Gather relevance labels in sorted order
    sorted_relevant = positive.gather(1, sorted_indices)  # [N_valid, G]

    n_valid = len(valid_idx)
    recall_at_k = {}
    for k in ks:
        actual_k = min(k, G)
        hits = sorted_relevant[:, :actual_k].any(dim=1).float()
        recall_at_k[f"recall_at_{k}"] = hits.mean().item()

    # MRR: reciprocal rank of first positive
    ranks = torch.arange(1, G + 1, device=device).unsqueeze(0).expand(n_valid, -1).float()
    # Mask non-relevant to inf rank
    first_pos_rank = torch.where(
        sorted_relevant,
        ranks,
        torch.full_like(ranks, float('inf'))
    ).min(dim=1).values
    mrr = (1.0 / first_pos_rank).mean().item()

    # mAP: mean average precision
    # cumulative sum of relevant items at each position
    cum_relevant = sorted_relevant.float().cumsum(dim=1)  # [N_valid, B]
    precision_at_k = cum_relevant / ranks  # precision at each rank
    # Only count positions where the item is relevant
    ap = (precision_at_k * sorted_relevant.float()).sum(dim=1)
    # Normalize by number of positives per query
    n_pos = positive.float().sum(dim=1).clamp_min(1)
    ap = ap / n_pos
    map_val = ap.mean().item()

    return recall_at_k | {"mrr": mrr, "map": map_val}


# ---------------------------------------------------------------------------
# 2. Classification Metrics
# ---------------------------------------------------------------------------

def compute_classification_metrics(all_probs, all_labels, all_sources=None,
                                   n_bins=10):
    """
    Compute classification metrics from accumulated predictions.

    Args:
        all_probs: Predicted probability of positive class [N].
        all_labels: Ground-truth binary labels [N] (1=match, 0=no-match).
        all_sources: Optional string labels per sample ('pos', 'hard', 'neg',
                     'neg_gw') for per-source breakdown [N].
        n_bins: Number of bins for ECE.

    Returns:
        Dictionary with auroc, auprc, f1_optimal, f1_threshold, ece,
        confusion matrix, and per-source accuracy.
    """
    results = {}

    probs_raw = all_probs.float()
    labels_raw = all_labels
    finite = torch.isfinite(probs_raw) & torch.isfinite(labels_raw.float())
    n_dropped = int((~finite).sum().item())

    probs = probs_raw[finite]
    labels = labels_raw[finite].long()
    if all_sources is not None:
        keep = finite.detach().cpu().tolist()
        all_sources = [src for src, is_finite in zip(all_sources, keep) if is_finite]

    N = len(probs)

    if N == 0:
        return {"auroc": 0.0, "auprc": 0.0, "f1_optimal": 0.0,
                "f1_threshold": 0.5, "ece": 0.0,
                "n_dropped_nonfinite": n_dropped}

    results["n_dropped_nonfinite"] = n_dropped

    # --- AUROC ---
    results["auroc"] = _auroc(probs, labels)

    # --- AUPRC ---
    results["auprc"] = _auprc(probs, labels)

    # --- F1 at optimal threshold ---
    f1, threshold = _optimal_f1(probs, labels)
    results["f1_optimal"] = f1
    results["f1_threshold"] = threshold

    # --- ECE ---
    results["ece"] = _expected_calibration_error(probs, labels, n_bins)

    # --- Confusion matrix at threshold = 0.5 ---
    preds_05 = (probs >= 0.5).long()
    tp = ((preds_05 == 1) & (labels == 1)).sum().item()
    fp = ((preds_05 == 1) & (labels == 0)).sum().item()
    tn = ((preds_05 == 0) & (labels == 0)).sum().item()
    fn = ((preds_05 == 0) & (labels == 1)).sum().item()
    results["tp"] = tp
    results["fp"] = fp
    results["tn"] = tn
    results["fn"] = fn

    # --- Derived confusion-matrix metrics ---
    results["accuracy"] = (tp + tn) / max(tp + fp + tn + fn, 1)
    results["precision"] = tp / max(tp + fp, 1)
    results["recall"] = tp / max(tp + fn, 1)
    prec, rec = results["precision"], results["recall"]
    results["f1"] = 2.0 * prec * rec / max(prec + rec, 1e-12)

    # --- Per-source accuracy ---
    if all_sources is not None:
        for src in set(all_sources):
            mask = [s == src for s in all_sources]
            mask = torch.tensor(mask, dtype=torch.bool, device=probs.device)
            if mask.sum() > 0:
                src_preds = (probs[mask] >= 0.5).long()
                src_labels = labels[mask]
                src_acc = (src_preds == src_labels).float().mean().item()
                results[f"acc_{src}"] = src_acc
        source_accs = [v for k, v in results.items() if k.startswith('acc_')]
        if source_accs:
            results['acc_total'] = sum(source_accs) / len(source_accs)

    return results


def _grouped_binary_counts(probs, labels):
    """Return positive/negative counts per distinct score, sorted descending."""
    sorted_indices = probs.argsort(descending=True)
    sorted_scores = probs[sorted_indices]
    sorted_labels = labels[sorted_indices].float()

    _, counts = torch.unique_consecutive(sorted_scores, return_counts=True)
    ends = counts.cumsum(0)
    starts = torch.cat([torch.zeros(1, dtype=ends.dtype, device=ends.device), ends[:-1]])

    pos_counts = torch.stack([sorted_labels[int(s.item()):int(e.item())].sum() for s, e in zip(starts, ends)])
    total_counts = counts.to(dtype=pos_counts.dtype)
    neg_counts = total_counts - pos_counts
    return pos_counts, neg_counts


def _auroc(probs, labels):
    """Compute tie-aware AUROC using grouped score thresholds."""
    n_pos = labels.sum().float()
    n_neg = (labels == 0).sum().float()
    if n_pos.item() == 0 or n_neg.item() == 0:
        return 0.0

    pos_counts, neg_counts = _grouped_binary_counts(probs, labels)
    tpr = pos_counts.cumsum(0) / n_pos
    fpr = neg_counts.cumsum(0) / n_neg

    tpr = torch.cat([torch.zeros(1, device=tpr.device), tpr])
    fpr = torch.cat([torch.zeros(1, device=fpr.device), fpr])
    return torch.trapezoid(tpr, fpr).item()


def _auprc(probs, labels):
    """Compute tie-aware average precision using grouped score thresholds."""
    n_pos = labels.sum().float()
    if n_pos.item() == 0:
        return 0.0

    pos_counts, neg_counts = _grouped_binary_counts(probs, labels)
    cum_pos = pos_counts.cumsum(0)
    cum_total = (pos_counts + neg_counts).cumsum(0)
    precision = cum_pos / cum_total.clamp_min(1.0)
    recall = cum_pos / n_pos
    prev_recall = torch.cat([torch.zeros(1, device=recall.device), recall[:-1]])
    return ((recall - prev_recall) * precision).sum().item()


def _optimal_f1(probs, labels):
    """Find the threshold maximizing F1 score."""
    thresholds = torch.linspace(0.0, 1.0, 101, device=probs.device)
    best_f1 = 0.0
    best_threshold = 0.5

    for t in thresholds:
        preds = (probs >= t).long()
        tp = ((preds == 1) & (labels == 1)).sum().float()
        fp = ((preds == 1) & (labels == 0)).sum().float()
        fn = ((preds == 0) & (labels == 1)).sum().float()
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)
        if f1.item() > best_f1:
            best_f1 = f1.item()
            best_threshold = t.item()

    return best_f1, best_threshold


def _expected_calibration_error(probs, labels, n_bins=10):
    """Compute Expected Calibration Error."""
    bin_boundaries = torch.linspace(0.0, 1.0, n_bins + 1, device=probs.device)
    ece = 0.0
    N = len(probs)

    for i in range(n_bins):
        low, high = bin_boundaries[i], bin_boundaries[i + 1]
        if i == n_bins - 1:
            mask = (probs >= low) & (probs <= high)
        else:
            mask = (probs >= low) & (probs < high)
        n_bin = mask.sum().item()
        if n_bin == 0:
            continue
        avg_confidence = probs[mask].mean().item()
        avg_accuracy = labels[mask].float().mean().item()
        ece += (n_bin / N) * abs(avg_accuracy - avg_confidence)

    return ece


# ---------------------------------------------------------------------------
# 3. Embedding Quality Metrics
# ---------------------------------------------------------------------------

def compute_embedding_metrics(feat_g, feat_o, gw_indices, is_neg_gw=None,
                              max_pairs=5000):
    """
    Compute embedding quality metrics.

    Args:
        feat_g: L2-normalized GW embeddings [N, D].
        feat_o: L2-normalized optical embeddings [N, D].
        gw_indices: GW event index for each sample [N].
        is_neg_gw: Boolean mask [N] where True = negative GW.
        max_pairs: Max number of random pairs for uniformity (memory control).

    Returns:
        Dictionary with alignment, uniformity, inter-modal gap, and
        intra/inter class similarity.
    """
    device = feat_g.device
    N = feat_g.size(0)

    if is_neg_gw is not None and is_neg_gw.any():
        pos_mask = ~is_neg_gw
    else:
        pos_mask = torch.ones(N, dtype=torch.bool, device=device)

    pos_idx = torch.where(pos_mask)[0]
    if len(pos_idx) < 2:
        return {"alignment": 0.0, "uniformity_gw": 0.0, "uniformity_opt": 0.0,
                "inter_modal_gap": 0.0, "intra_sim": 0.0, "inter_sim": 0.0}

    fg = feat_g[pos_idx]
    fo = feat_o[pos_idx]
    gi = gw_indices[pos_idx]

    results = {}

    # --- Alignment: mean squared L2 distance of positive pairs ---
    # Positive pair = (feat_g[i], feat_o[i]) for matched positions
    diff = fg - fo
    results["alignment"] = (diff * diff).sum(dim=1).mean().item()

    # --- Uniformity: log E[exp(-2 * ||x - y||^2)] for random pairs ---
    n_sample = min(max_pairs, len(pos_idx))
    perm = torch.randperm(len(pos_idx), device=device)[:n_sample]

    fg_sample = fg[perm]
    fo_sample = fo[perm]

    # GW uniformity
    results["uniformity_gw"] = _uniformity(fg_sample)
    # Optical uniformity
    results["uniformity_opt"] = _uniformity(fo_sample)

    # --- Inter-modal gap ---
    # Mean cosine distance between matched GW and optical
    cos_sim = (fg * fo).sum(dim=1)
    results["inter_modal_gap"] = (1.0 - cos_sim).mean().item()

    # --- Intra-class vs inter-class cosine similarity ---
    # Sample a subset for efficiency
    n_sample = min(1000, len(pos_idx))
    perm = torch.randperm(len(pos_idx), device=device)[:n_sample]
    fg_s = fg[perm]
    fo_s = fo[perm]
    gi_s = gi[perm]

    # Cosine similarity matrix (GW query → optical gallery)
    sim = torch.matmul(fg_s, fo_s.T)  # [n_sample, n_sample]

    same_event = gi_s.unsqueeze(0) == gi_s.unsqueeze(1)  # [n_sample, n_sample]
    diff_event = ~same_event

    if same_event.sum() > 0:
        results["intra_sim"] = sim[same_event].mean().item()
    else:
        results["intra_sim"] = 0.0

    if diff_event.sum() > 0:
        results["inter_sim"] = sim[diff_event].mean().item()
    else:
        results["inter_sim"] = 0.0

    return results


def _uniformity(features, t=2.0):
    """
    Compute uniformity loss: log E[exp(-t * ||x - y||^2)].

    Reference: Wang & Isola, ICML 2020.
    """
    n = features.size(0)
    if n < 2:
        return 0.0

    # Pairwise squared distances
    sq_pdist = torch.cdist(features, features, p=2).pow(2)

    # Exclude diagonal (self-pairs)
    mask = ~torch.eye(n, dtype=torch.bool, device=features.device)
    sq_pdist = sq_pdist[mask]

    uniformity = torch.log(torch.exp(-t * sq_pdist).mean() + 1e-8).item()
    return uniformity
