from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import h5py
import numpy as np
import torch


TABLE_METRIC_LABELS = ["R@1", "R@5", "R@10", "MRR"]
TABLE_METRIC_KEYS = ["recall_at_1", "recall_at_5", "recall_at_10", "mrr"]
PLOT_DPI = 300
PLOT_FONT_BASE = 13
RETRIEVAL_CURVES_FIGSIZE = (15, 4.8)
PLOT_METHOD_LABELS = {
    "skymap-only": "Skymap-only",
    "optical-only": "Optical-only",
    "full": "MAGIKS",
    "Full": "MAGIKS",
    "w/o \u5bf9\u6bd4\u5b66\u4e60": "w/o Contrastive Learning",
    "w/o \u4ea4\u53c9\u6ce8\u610f\u529b": "w/o Cross-Attention",
    "w/o \u878d\u5408\u5206\u652f": "w/o Fusion Branch",
    "\u5168\u6a21\u6001": "MAGIKS",
    "\u5168\u6a21\u6001 + \u56f0\u96be\u6837\u672c\u6316\u6398": "MAGIKS + Hard Mining",
}


def _plot_method_label(method: str) -> str:
    return PLOT_METHOD_LABELS.get(str(method), str(method))


def _plot_method_draw_order(method: str) -> tuple:
    """Sort key: 'full' variants draw last (on top of other curves)."""
    s = str(method)
    is_full = "全模态" in s or "Full" in s
    return (int(is_full), s)


def _remove_stale_pdf(path: Path) -> None:
    if path.exists():
        path.unlink()


def _as_numpy_1d(value: Any, *, dtype: np.dtype) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        arr = value.detach().cpu().numpy()
    else:
        arr = np.asarray(value)
    return np.asarray(arr, dtype=dtype).reshape(-1)


def resolve_optional_path(base_dir: Path, value: Any) -> str | None:
    if value in (None, "", "null"):
        return None
    path = Path(str(value)).expanduser()
    if not path.is_absolute():
        path = (base_dir / path).resolve()
    return str(path)


def resolve_checkpoint_path(checkpoint_path: str, model_type: str) -> str:
    path = Path(checkpoint_path).expanduser()
    if path.is_file():
        return str(path.resolve())
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint path not found: {checkpoint_path}")
    if not path.is_dir():
        raise FileNotFoundError(f"Unsupported checkpoint path: {checkpoint_path}")

    preferred_names = ["optical_only_best.pth", "best.pth"] if model_type == "optical" else ["albef_best.pth", "best.pth"]
    for filename in preferred_names:
        matches = sorted(path.rglob(filename))
        if matches:
            return str(matches[0].resolve())

    epoch_candidates: List[Tuple[int, Path]] = []
    epoch_patterns = [
        re.compile(r"albef_epoch_(\d+)\.pth$"),
        re.compile(r"optical_only_epoch_(\d+)\.pth$"),
    ]
    for candidate in path.rglob("*.pth"):
        for pattern in epoch_patterns:
            match = pattern.fullmatch(candidate.name)
            if match:
                epoch_candidates.append((int(match.group(1)), candidate))
                break
    if epoch_candidates:
        return str(max(epoch_candidates, key=lambda item: item[0])[1].resolve())

    fallback_names = ["optical_only_last.pth", "albef_last.pth", "last.pth"]
    for filename in fallback_names:
        matches = sorted(path.rglob(filename))
        if matches:
            return str(matches[0].resolve())

    any_pth = sorted(path.rglob("*.pth"))
    if any_pth:
        return str(any_pth[0].resolve())

    raise FileNotFoundError(f"No checkpoint file found under directory: {checkpoint_path}")


def build_comparison_model_specs(cfg: Mapping[str, Any], cfg_dir: Path) -> List[Dict[str, Any]]:
    model_specs: List[Dict[str, Any]] = []
    valid_scoring = {"optical", "logits", "contrastive", "auto", "skymap"}
    for raw_spec in cfg.get("models", []):
        spec = dict(raw_spec)
        spec["name"] = str(spec["name"])
        spec["type"] = str(spec["type"])
        spec["checkpoint"] = resolve_optional_path(cfg_dir, spec.get("checkpoint"))
        spec["config"] = resolve_optional_path(cfg_dir, spec.get("config"))
        if spec["type"] in {"optical", "multimodal"} and spec["checkpoint"] is None:
            raise ValueError(f"models[{spec['name']}] is missing checkpoint")
        if spec["type"] == "multimodal" and spec["config"] is None:
            raise ValueError(f"Multimodal model '{spec['name']}' requires a config path")

        if spec["type"] == "skymap":
            spec["resolved_checkpoint"] = None
        else:
            spec["resolved_checkpoint"] = resolve_checkpoint_path(spec["checkpoint"], spec["type"])
        spec["resolved_config"] = spec.get("config")

        if "scoring" not in spec:
            if spec["type"] == "optical":
                spec["scoring"] = "optical"
            elif spec["type"] == "skymap":
                spec["scoring"] = "skymap"
            else:
                spec["scoring"] = "auto"
        if spec["scoring"] not in valid_scoring:
            raise ValueError(
                f"models[{spec['name']}] has invalid scoring '{spec['scoring']}', "
                f"must be one of {valid_scoring}"
            )
        model_specs.append(spec)

    if not model_specs:
        raise ValueError("Comparison config must define a non-empty models array")
    return model_specs


def _as_torch_coords(value: Any) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        coords = value.detach().cpu().to(torch.float32)
    else:
        coords = torch.as_tensor(value, dtype=torch.float32)
    if coords.ndim != 2 or coords.shape[1] != 2:
        raise ValueError(f"Expected coordinate array with shape [N, 2], got {tuple(coords.shape)}")
    return coords


def _coords_to_radians(coords: torch.Tensor) -> torch.Tensor:
    if coords.numel() == 0:
        return coords
    max_abs = float(coords.detach().abs().max().item())
    if max_abs > (2 * np.pi + 1e-3):
        return coords * (np.pi / 180.0)
    return coords


def _coords_to_unit_xyz(coords: torch.Tensor) -> torch.Tensor:
    coords = _coords_to_radians(coords.to(torch.float32))
    if coords.numel() == 0:
        return torch.empty((0, 3), dtype=torch.float32, device=coords.device)
    ra = coords[:, 0]
    dec = coords[:, 1]
    return torch.stack(
        [
            torch.cos(dec) * torch.cos(ra),
            torch.cos(dec) * torch.sin(ra),
            torch.sin(dec),
        ],
        dim=-1,
    )


def compute_credible_levels_single_gw(gw_skymap: torch.Tensor, opt_coords: torch.Tensor) -> torch.Tensor:
    opt_xyz = _coords_to_unit_xyz(opt_coords)

    pix_xyz = gw_skymap[:3, :].to(torch.float32)
    dot = torch.matmul(opt_xyz, pix_xyz)
    nearest_idx = dot.argmax(dim=-1)

    dA = gw_skymap[3, :].to(torch.float32)
    dP = gw_skymap[4, :].to(torch.float32)
    return _credible_levels_from_probability_density_torch(dP, dA, nearest_idx)


def _nearest_pixel_indices_from_xyz(opt_xyz: torch.Tensor, pixel_xyz: torch.Tensor, chunk_size: int = 4096) -> np.ndarray:
    pixel_xyz = pixel_xyz.to(torch.float32)

    nearest_chunks: List[torch.Tensor] = []
    for start in range(0, opt_xyz.shape[0], int(chunk_size)):
        chunk = opt_xyz[start : start + int(chunk_size)]
        nearest_chunks.append(torch.matmul(chunk, pixel_xyz).argmax(dim=-1).cpu())
    if not nearest_chunks:
        return np.empty((0,), dtype=np.int64)
    return torch.cat(nearest_chunks).numpy().astype(np.int64, copy=False)


def _nearest_pixel_indices(opt_coords: torch.Tensor, pixel_xyz: torch.Tensor, chunk_size: int = 4096) -> np.ndarray:
    return _nearest_pixel_indices_from_xyz(_coords_to_unit_xyz(opt_coords), pixel_xyz, chunk_size=int(chunk_size))


def _credible_levels_from_probability_density_torch(
    dP: torch.Tensor,
    dA: torch.Tensor,
    nearest_idx: torch.Tensor,
) -> torch.Tensor:
    dP = torch.nan_to_num(dP.to(torch.float32), nan=0.0, posinf=0.0, neginf=0.0).clamp_min(0.0)
    dA = torch.nan_to_num(dA.to(torch.float32), nan=0.0, posinf=0.0, neginf=0.0).clamp_min(0.0)
    total_probability = dP.sum().clamp_min(torch.finfo(dP.dtype).eps)
    density = dP / dA.clamp_min(torch.finfo(dP.dtype).eps)
    density_at_opt = density[nearest_idx]
    credible_mass = torch.where(
        density.unsqueeze(0) >= density_at_opt.unsqueeze(-1),
        dP.unsqueeze(0),
        torch.zeros_like(dP).unsqueeze(0),
    ).sum(dim=-1)
    return credible_mass / total_probability


def _credible_levels_from_probability_density_numpy(
    dP: np.ndarray,
    dA: np.ndarray,
    nearest_idx: np.ndarray,
) -> np.ndarray:
    dP_clean = np.nan_to_num(np.asarray(dP, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    dP_clean = np.maximum(dP_clean, 0.0)
    dA_clean = np.nan_to_num(np.asarray(dA, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    dA_clean = np.maximum(dA_clean, 0.0)
    total_probability = float(dP_clean.sum())
    if total_probability <= 0.0:
        return np.full(np.asarray(nearest_idx).shape, np.nan, dtype=np.float64)

    density = np.divide(dP_clean, dA_clean, out=np.zeros_like(dP_clean), where=dA_clean > 0.0)
    order = np.argsort(density, kind="mergesort")
    sorted_density = density[order]
    cumulative = np.concatenate(([0.0], np.cumsum(dP_clean[order], dtype=np.float64)))
    thresholds = density[np.asarray(nearest_idx, dtype=np.int64)]
    first_ge = np.searchsorted(sorted_density, thresholds, side="left")
    credible_mass = total_probability - cumulative[first_ge]
    return credible_mass / total_probability


def _cell_credible_levels_from_probability_density_numpy(dP: np.ndarray, dA: np.ndarray) -> np.ndarray:
    dP_clean = np.nan_to_num(np.asarray(dP, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    dP_clean = np.maximum(dP_clean, 0.0)
    dA_clean = np.nan_to_num(np.asarray(dA, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    dA_clean = np.maximum(dA_clean, 0.0)
    total_probability = float(dP_clean.sum())
    if total_probability <= 0.0:
        return np.full(dP_clean.shape, np.nan, dtype=np.float64)

    density = np.divide(dP_clean, dA_clean, out=np.zeros_like(dP_clean), where=dA_clean > 0.0)
    order = np.argsort(density, kind="mergesort")
    sorted_density = density[order]
    cumulative = np.concatenate(([0.0], np.cumsum(dP_clean[order], dtype=np.float64)))
    first_ge = np.searchsorted(sorted_density, density, side="left")
    credible_mass = total_probability - cumulative[first_ge]
    return credible_mass / total_probability


def _unit_xyz_to_radec_degrees(xyz: np.ndarray) -> np.ndarray:
    xyz = np.asarray(xyz, dtype=np.float64)
    if xyz.ndim != 2 or xyz.shape[1] != 3:
        raise ValueError(f"Expected xyz with shape [N, 3], got {xyz.shape}")
    norm = np.linalg.norm(xyz, axis=1)
    norm = np.where(norm > 0.0, norm, 1.0)
    x = xyz[:, 0] / norm
    y = xyz[:, 1] / norm
    z = np.clip(xyz[:, 2] / norm, -1.0, 1.0)
    ra = np.degrees(np.arctan2(y, x)) % 360.0
    dec = np.degrees(np.arcsin(z))
    return np.stack([ra, dec], axis=1).astype(np.float32)


def _infer_negative_pool_size(neg_optical_data: Mapping[str, Any]) -> int:
    for key in ("times", "coordinates", "values", "zero_time_mjd_cls_base", "zero_time_mjd_base"):
        if key in neg_optical_data:
            value = neg_optical_data[key]
            return int(value.shape[0] if hasattr(value, "shape") else len(value))
    raise KeyError("Cannot infer negative pool size; expected one of times/coordinates/values/zero_time_mjd fields.")


def build_time_sky_candidate_sequence(
    *,
    anchor_time_mjd: float,
    candidate_zero_time_mjd: Any,
    candidate_credible_levels: Any,
    time_window_days: float,
    credible_level_max: float,
    seed: int,
) -> Dict[str, np.ndarray]:
    candidate_times = _as_numpy_1d(candidate_zero_time_mjd, dtype=np.float64)
    candidate_credible = _as_numpy_1d(candidate_credible_levels, dtype=np.float64)
    if candidate_times.shape != candidate_credible.shape:
        raise ValueError("candidate_zero_time_mjd and candidate_credible_levels must have the same shape.")

    if not np.isfinite(anchor_time_mjd):
        return {
            "candidate_indices": np.empty((0,), dtype=np.int64),
            "credible_levels": np.empty((0,), dtype=np.float32),
            "abs_dt_days": np.empty((0,), dtype=np.float32),
        }

    signed_dt_days = candidate_times - float(anchor_time_mjd)
    keep = (
        np.isfinite(candidate_times)
        & np.isfinite(candidate_credible)
        & (signed_dt_days >= 0.0)
        & (signed_dt_days <= float(time_window_days))
        & (candidate_credible <= float(credible_level_max))
    )

    valid_idx = np.nonzero(keep)[0].astype(np.int64, copy=False)
    if valid_idx.size == 0:
        return {
            "candidate_indices": np.empty((0,), dtype=np.int64),
            "credible_levels": np.empty((0,), dtype=np.float32),
            "abs_dt_days": np.empty((0,), dtype=np.float32),
        }

    rng = np.random.default_rng(int(seed))
    tie_break = rng.random(valid_idx.size)
    order = np.lexsort(
        (
            tie_break,
            candidate_credible[valid_idx],
            signed_dt_days[valid_idx],
        )
    )
    ordered_idx = valid_idx[order]
    return {
        "candidate_indices": ordered_idx.astype(np.int64, copy=False),
        "credible_levels": np.asarray(candidate_credible[ordered_idx], dtype=np.float32),
        "abs_dt_days": np.asarray(signed_dt_days[ordered_idx], dtype=np.float32),
    }


def build_time_sky_candidate_sequences(
    *,
    test_data_path: str,
    unique_gw_ids: Sequence[int],
    neg_optical_data: Mapping[str, Any],
    n_trials: int,
    seed: int,
    time_window_days: float,
    credible_level_max: float,
    zero_time_field: str = "zero_time_mjd_cls_base",
) -> Tuple[Dict[Tuple[int, int], Dict[str, np.ndarray]], Dict[int, torch.Tensor], Dict[int, float]]:
    if zero_time_field not in neg_optical_data:
        # Try fallback fields in order
        for fallback in ("first_detection_mjd", "zero_time_mjd_base"):
            if fallback in neg_optical_data:
                zero_time_field = fallback
                break
        else:
            raise KeyError(f"Negative optical data is missing '{zero_time_field}' and fallbacks.")
    if "coordinates" not in neg_optical_data:
        raise KeyError("Negative optical data is missing 'coordinates' required for skymap filtering.")

    unique_gw = [int(gw_id) for gw_id in unique_gw_ids]
    neg_times = _as_numpy_1d(neg_optical_data[zero_time_field], dtype=np.float64)
    neg_coords = _as_torch_coords(neg_optical_data["coordinates"])
    neg_opt_xyz = _coords_to_unit_xyz(neg_coords)

    if neg_times.shape[0] != neg_coords.shape[0]:
        raise ValueError("Negative optical times and coordinates must have the same first dimension.")
    if not unique_gw:
        return {}, {}, {}

    with h5py.File(test_data_path, "r") as f:
        if "events/gw_data/event_time_mjd" not in f:
            raise KeyError(f"Missing required field 'events/gw_data/event_time_mjd' in {test_data_path}")
        if "events/gw_data/skymaps" not in f:
            raise KeyError(f"Missing required field 'events/gw_data/skymaps' in {test_data_path}")

        gw_event_times = np.asarray(f["events/gw_data/event_time_mjd"][unique_gw], dtype=np.float64)
        gw_skymaps_np = np.asarray(f["events/gw_data/skymaps"][unique_gw], dtype=np.float32)

    candidate_sequences: Dict[Tuple[int, int], Dict[str, np.ndarray]] = {}
    gw_skymaps: Dict[int, torch.Tensor] = {}
    gw_time_lookup: Dict[int, float] = {}

    for local_idx, gw_id in enumerate(unique_gw):
        gw_skymap = torch.from_numpy(gw_skymaps_np[local_idx]).to(torch.float32)
        gw_skymaps[int(gw_id)] = gw_skymap
        anchor_time = float(gw_event_times[local_idx])
        gw_time_lookup[int(gw_id)] = anchor_time

        dA = np.asarray(gw_skymaps_np[local_idx, 3, :], dtype=np.float64)
        dP = np.asarray(gw_skymaps_np[local_idx, 4, :], dtype=np.float64)
        candidate_credible = np.full(neg_times.shape, np.nan, dtype=np.float64)
        if np.isfinite(anchor_time):
            time_mask = (
                np.isfinite(neg_times)
                & (neg_times >= anchor_time)
                & (neg_times <= anchor_time + float(time_window_days))
            )
            if np.any(time_mask):
                event_pixel_idx = _nearest_pixel_indices_from_xyz(
                    neg_opt_xyz[time_mask],
                    gw_skymap[:3, :],
                )
                candidate_credible[time_mask] = _credible_levels_from_probability_density_numpy(
                    dP,
                    dA,
                    event_pixel_idx,
                )

        for trial in range(int(n_trials)):
            seq_seed = int(seed) + 7919 * int(trial) + 104729 * int(gw_id)
            candidate_sequences[(int(trial), int(gw_id))] = build_time_sky_candidate_sequence(
                anchor_time_mjd=anchor_time,
                candidate_zero_time_mjd=neg_times,
                candidate_credible_levels=candidate_credible,
                time_window_days=float(time_window_days),
                credible_level_max=float(credible_level_max),
                seed=seq_seed,
            )

    return candidate_sequences, gw_skymaps, gw_time_lookup


def build_synthetic_time_sky_candidate_sequences(
    *,
    test_data_path: str,
    unique_gw_ids: Sequence[int],
    neg_optical_data: Mapping[str, Any],
    gallery_sizes: Sequence[int],
    n_trials: int,
    seed: int,
    time_window_days: float,
    credible_level_max: float,
) -> Tuple[Dict[Tuple[int, int], Dict[str, np.ndarray]], Dict[int, torch.Tensor], Dict[int, float]]:
    unique_gw = [int(gw_id) for gw_id in unique_gw_ids]
    gallery_sizes = [int(size) for size in gallery_sizes]
    max_neg_needed = max([max(int(size) - 1, 0) for size in gallery_sizes] or [0])
    neg_pool_size = _infer_negative_pool_size(neg_optical_data)
    if max_neg_needed > 0 and neg_pool_size <= 0:
        raise ValueError("Synthetic gallery construction requires at least one negative optical sample.")
    if not unique_gw:
        return {}, {}, {}

    with h5py.File(test_data_path, "r") as f:
        if "events/gw_data/event_time_mjd" not in f:
            raise KeyError(f"Missing required field 'events/gw_data/event_time_mjd' in {test_data_path}")
        if "events/gw_data/skymaps" not in f:
            raise KeyError(f"Missing required field 'events/gw_data/skymaps' in {test_data_path}")

        gw_event_times = np.asarray(f["events/gw_data/event_time_mjd"][unique_gw], dtype=np.float64)
        gw_skymaps_np = np.asarray(f["events/gw_data/skymaps"][unique_gw], dtype=np.float32)

    candidate_sequences: Dict[Tuple[int, int], Dict[str, np.ndarray]] = {}
    gw_skymaps: Dict[int, torch.Tensor] = {}
    gw_time_lookup: Dict[int, float] = {}

    for local_idx, gw_id in enumerate(unique_gw):
        gw_skymap_np = gw_skymaps_np[local_idx]
        gw_skymap = torch.from_numpy(gw_skymap_np).to(torch.float32)
        gw_skymaps[int(gw_id)] = gw_skymap
        anchor_time = float(gw_event_times[local_idx])
        gw_time_lookup[int(gw_id)] = anchor_time
        if not np.isfinite(anchor_time):
            raise ValueError(f"GW event {gw_id} has non-finite event_time_mjd; cannot synthesize time-conditioned gallery.")

        dA = np.asarray(gw_skymap_np[3, :], dtype=np.float64)
        dP = np.asarray(gw_skymap_np[4, :], dtype=np.float64)
        cell_credible = _cell_credible_levels_from_probability_density_numpy(dP, dA)
        eligible = np.isfinite(cell_credible) & (cell_credible <= float(credible_level_max)) & np.isfinite(dA) & (dA > 0.0)
        eligible_idx = np.nonzero(eligible)[0].astype(np.int64, copy=False)
        if eligible_idx.size == 0 and max_neg_needed > 0:
            raise ValueError(
                f"GW event {gw_id} has no skymap cells with credible_level <= {float(credible_level_max):.3f}."
            )
        eligible_area = np.asarray(dA[eligible_idx], dtype=np.float64)
        eligible_area = np.maximum(np.nan_to_num(eligible_area, nan=0.0, posinf=0.0, neginf=0.0), 0.0)
        area_sum = float(eligible_area.sum())
        cell_prob = eligible_area / area_sum if area_sum > 0.0 else None

        for trial in range(int(n_trials)):
            seq_seed = int(seed) + 7919 * int(trial) + 104729 * int(gw_id)
            rng = np.random.default_rng(seq_seed)
            if max_neg_needed <= 0:
                neg_indices = np.empty((0,), dtype=np.int64)
                synthetic_abs_dt = np.empty((0,), dtype=np.float32)
                synthetic_times = np.empty((0,), dtype=np.float64)
                synthetic_coords = np.empty((0, 2), dtype=np.float32)
                synthetic_credible = np.empty((0,), dtype=np.float32)
            else:
                replace_neg = bool(neg_pool_size < max_neg_needed)
                neg_indices = rng.choice(
                    int(neg_pool_size),
                    size=int(max_neg_needed),
                    replace=replace_neg,
                ).astype(np.int64, copy=False)
                delta_t = rng.uniform(
                    0.0,
                    float(time_window_days),
                    size=int(max_neg_needed),
                )
                sampled_cells = rng.choice(
                    eligible_idx,
                    size=int(max_neg_needed),
                    replace=True,
                    p=cell_prob,
                ).astype(np.int64, copy=False)
                synthetic_abs_dt = delta_t.astype(np.float32)
                synthetic_times = (anchor_time + delta_t).astype(np.float64)
                synthetic_coords = _unit_xyz_to_radec_degrees(gw_skymap_np[:3, sampled_cells].T)
                synthetic_credible = np.asarray(cell_credible[sampled_cells], dtype=np.float32)

            candidate_sequences[(int(trial), int(gw_id))] = {
                "candidate_indices": neg_indices,
                "credible_levels": synthetic_credible,
                "abs_dt_days": synthetic_abs_dt,
                "synthetic_coordinates": synthetic_coords,
                "synthetic_zero_time_mjd_cls_base": synthetic_times,
            }

    return candidate_sequences, gw_skymaps, gw_time_lookup


def build_prefixed_gallery_specs(
    *,
    gw_positive_indices: Mapping[int, Any],
    candidate_sequences: Mapping[Tuple[int, int], Mapping[str, Any]],
    gallery_sizes: Sequence[int],
    n_trials: int,
    seed: int,
    include_undersized: bool,
) -> Tuple[Dict[Tuple[int, int, int], Dict[str, Any]], List[int]]:
    unique_gw = sorted(int(gw_id) for gw_id in gw_positive_indices.keys())
    galleries: Dict[Tuple[int, int, int], Dict[str, Any]] = {}
    gallery_sizes = [int(size) for size in gallery_sizes]

    for trial in range(int(n_trials)):
        trial_rng = np.random.default_rng(int(seed) + 15485863 * int(trial))
        for gw_id in unique_gw:
            gw_pos_idx = _as_numpy_1d(gw_positive_indices.get(int(gw_id), []), dtype=np.int64)
            if gw_pos_idx.size == 0:
                raise ValueError(f"GW event {gw_id} has no positive optical samples available for gallery construction.")
            positive_index = int(gw_pos_idx[trial_rng.integers(gw_pos_idx.size)])

            seq = candidate_sequences.get((int(trial), int(gw_id)), {})
            neg_idx_full = _as_numpy_1d(seq.get("candidate_indices", []), dtype=np.int64)
            neg_cred_full = _as_numpy_1d(seq.get("credible_levels", []), dtype=np.float32)
            neg_dt_full = _as_numpy_1d(seq.get("abs_dt_days", []), dtype=np.float32)
            neg_coords_full = None
            if "synthetic_coordinates" in seq:
                neg_coords_full = np.asarray(seq["synthetic_coordinates"], dtype=np.float32).reshape(-1, 2)
            neg_time_full = None
            if "synthetic_zero_time_mjd_cls_base" in seq:
                neg_time_full = _as_numpy_1d(seq["synthetic_zero_time_mjd_cls_base"], dtype=np.float64)

            for gallery_size in gallery_sizes:
                requested = int(gallery_size)
                target_neg = max(requested - 1, 0)
                available_neg = int(neg_idx_full.shape[0])
                if target_neg > available_neg and not bool(include_undersized):
                    continue
                take_neg = min(target_neg, available_neg)
                actual_gallery_size = 1 + int(take_neg)
                gallery_spec = {
                    "positive_index": positive_index,
                    "negative_indices": neg_idx_full[:take_neg].astype(np.int64, copy=False),
                    "negative_credible_levels": neg_cred_full[:take_neg].astype(np.float32, copy=False),
                    "negative_abs_dt_days": neg_dt_full[:take_neg].astype(np.float32, copy=False),
                    "requested_gallery_size": requested,
                    "actual_gallery_size": actual_gallery_size,
                    "coverage_met": bool(actual_gallery_size >= requested),
                    "is_undersized": bool(actual_gallery_size < requested),
                }
                if neg_coords_full is not None:
                    gallery_spec["negative_synthetic_coordinates"] = neg_coords_full[:take_neg].astype(np.float32, copy=False)
                if neg_time_full is not None:
                    gallery_spec["negative_synthetic_zero_time_mjd_cls_base"] = neg_time_full[:take_neg].astype(np.float64, copy=False)
                galleries[(requested, int(trial), int(gw_id))] = gallery_spec

    return galleries, unique_gw


def extract_gallery_negative_abs_dt_days(
    gallery_spec: Mapping[str, Any],
    *,
    gw_event_time_mjd: Optional[float],
    n_negative: int,
) -> Optional[np.ndarray]:
    """Return per-negative |dt| for a gallery spec when time metadata is available."""
    n_negative = int(n_negative)
    if n_negative <= 0:
        return np.asarray([], dtype=np.float32)

    if "negative_abs_dt_days" in gallery_spec:
        dt = _as_numpy_1d(gallery_spec["negative_abs_dt_days"], dtype=np.float32)[:n_negative]
        if dt.shape[0] == n_negative:
            return np.abs(dt).astype(np.float32, copy=False)

    if "negative_synthetic_zero_time_mjd_cls_base" not in gallery_spec:
        return None
    if gw_event_time_mjd is None or not np.isfinite(float(gw_event_time_mjd)):
        return None

    times = _as_numpy_1d(gallery_spec["negative_synthetic_zero_time_mjd_cls_base"], dtype=np.float64)[:n_negative]
    if times.shape[0] != n_negative:
        return None
    return np.abs(times - float(gw_event_time_mjd)).astype(np.float32, copy=False)


def score_all_galleries_skymap(
    *,
    positive_bank: Mapping[str, Any],
    galleries: Mapping[Tuple[int, int, int], Mapping[str, Any]],
    gw_skymaps: Mapping[int, torch.Tensor],
) -> Dict[Tuple[int, int, int], Dict[str, Any]]:
    if "opt_coords" not in positive_bank:
        raise KeyError("positive_bank must contain 'opt_coords' for skymap-only scoring.")
    positive_coords = _as_torch_coords(positive_bank["opt_coords"])

    pos_credible_cache: Dict[Tuple[int, int], float] = {}
    outcomes: Dict[Tuple[int, int, int], Dict[str, Any]] = {}

    for key, gallery_spec in galleries.items():
        requested_gallery_size, trial, gw_id = key
        pos_index = int(gallery_spec["positive_index"])
        cache_key = (int(gw_id), pos_index)
        if cache_key not in pos_credible_cache:
            pos_coords = positive_coords[pos_index].unsqueeze(0)
            pos_cred = compute_credible_levels_single_gw(gw_skymaps[int(gw_id)], pos_coords)
            pos_credible_cache[cache_key] = float(pos_cred.squeeze(0).item())

        pos_score = 1.0 - float(pos_credible_cache[cache_key])
        neg_scores = 1.0 - _as_numpy_1d(gallery_spec.get("negative_credible_levels", []), dtype=np.float64)
        scores = np.concatenate([np.asarray([pos_score], dtype=np.float64), neg_scores], axis=0)
        ranked = np.argsort(-scores, kind="mergesort")

        outcomes[(int(requested_gallery_size), int(trial), int(gw_id))] = {
            "rank": int(np.where(ranked == 0)[0][0]),
            "requested_gallery_size": int(gallery_spec["requested_gallery_size"]),
            "actual_gallery_size": int(gallery_spec["actual_gallery_size"]),
            "coverage_met": bool(gallery_spec["coverage_met"]),
            "is_undersized": bool(gallery_spec["is_undersized"]),
        }

    return outcomes


def aggregate_gallery_outcomes(
    *,
    outcomes: Mapping[Tuple[int, int, int], Mapping[str, Any]],
    gallery_sizes: Sequence[int],
    n_trials: int,
    unique_gw: Sequence[int],
    gw_source_map: Mapping[int, str] | None = None,
) -> Tuple[Dict[str, float], Dict[str, Dict[str, float]], Dict[str, Dict[str, Any]]]:
    metrics: Dict[str, float] = {}
    coverage_stats: Dict[str, Dict[str, Any]] = {}
    by_source: Dict[str, Dict[int, Dict[str, Any]]] = {}

    for gallery_size in [int(size) for size in gallery_sizes]:
        recalls = {1: [], 5: [], 10: []}
        mrrs: List[float] = []
        fill_ratios: List[float] = []
        full_coverage_flags: List[float] = []
        actual_sizes: List[int] = []

        for trial in range(int(n_trials)):
            for gw_id in [int(gw) for gw in unique_gw]:
                key = (int(gallery_size), int(trial), int(gw_id))
                if key not in outcomes:
                    continue
                outcome = outcomes[key]
                rank = int(outcome["rank"])
                actual_gallery_size = int(outcome["actual_gallery_size"])
                coverage_met = bool(outcome.get("coverage_met", actual_gallery_size >= int(gallery_size)))
                fill_ratio = min(
                    1.0,
                    max(0.0, float(actual_gallery_size) / float(max(int(gallery_size), 1))),
                )

                for k in recalls:
                    recalls[k].append(1.0 if rank < k else 0.0)
                mrrs.append(1.0 / float(rank + 1))
                fill_ratios.append(fill_ratio)
                full_coverage_flags.append(1.0 if coverage_met else 0.0)
                actual_sizes.append(actual_gallery_size)

                if gw_source_map is not None:
                    source_label = gw_source_map.get(int(gw_id), "unknown")
                    if source_label not in by_source:
                        by_source[source_label] = {}
                    if int(gallery_size) not in by_source[source_label]:
                        by_source[source_label][int(gallery_size)] = {
                            "recalls": {1: [], 5: [], 10: []},
                            "mrrs": [],
                        }
                    source_bucket = by_source[source_label][int(gallery_size)]
                    for k in source_bucket["recalls"]:
                        source_bucket["recalls"][k].append(1.0 if rank < k else 0.0)
                    source_bucket["mrrs"].append(1.0 / float(rank + 1))

        for k, values in recalls.items():
            metrics[f"gallery_{gallery_size}_recall_at_{k}"] = float(np.mean(values)) if values else 0.0
        metrics[f"gallery_{gallery_size}_mrr"] = float(np.mean(mrrs)) if mrrs else 0.0

        coverage_key = f"gallery_{gallery_size}"
        if actual_sizes:
            full_coverage_count = int(sum(1 for flag in full_coverage_flags if flag > 0.0))
            fill_ratio_mean = float(np.mean(fill_ratios))
            coverage_stats[coverage_key] = {
                "coverage": fill_ratio_mean,
                "fill_ratio_mean": fill_ratio_mean,
                "full_coverage": float(np.mean(full_coverage_flags)),
                "n_queries_total": int(len(actual_sizes)),
                "n_queries_covered": full_coverage_count,
                "n_queries_full_coverage": full_coverage_count,
                "effective_gallery_size_mean": float(np.mean(actual_sizes)),
                "effective_gallery_size_min": int(np.min(actual_sizes)),
                "effective_gallery_size_max": int(np.max(actual_sizes)),
            }
        else:
            coverage_stats[coverage_key] = {
                "coverage": 0.0,
                "fill_ratio_mean": 0.0,
                "full_coverage": 0.0,
                "n_queries_total": 0,
                "n_queries_covered": 0,
                "n_queries_full_coverage": 0,
                "effective_gallery_size_mean": 0.0,
                "effective_gallery_size_min": 0,
                "effective_gallery_size_max": 0,
            }

    source_metrics: Dict[str, Dict[str, float]] = {}
    for source_label, size_acc in by_source.items():
        source_metrics[source_label] = {}
        for gallery_size, acc in size_acc.items():
            for k, vals in acc["recalls"].items():
                source_metrics[source_label][f"gallery_{gallery_size}_recall_at_{k}"] = float(np.mean(vals)) if vals else 0.0
            source_metrics[source_label][f"gallery_{gallery_size}_mrr"] = float(np.mean(acc["mrrs"])) if acc["mrrs"] else 0.0

    return metrics, source_metrics, coverage_stats


def build_curve_rows(
    *,
    method_name: str,
    gallery_sizes: Sequence[int],
    retrieval_metrics: Mapping[str, Any],
    coverage_stats: Mapping[str, Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for gallery_size in [int(size) for size in gallery_sizes]:
        coverage_key = f"gallery_{gallery_size}"
        coverage_info = coverage_stats.get(coverage_key, {})
        for metric_label, metric_key in zip(TABLE_METRIC_LABELS, TABLE_METRIC_KEYS):
            rows.append(
                {
                    "method": str(method_name),
                    "gallery_size_target": int(gallery_size),
                    "gallery_size_actual": float(coverage_info.get("effective_gallery_size_mean", float(gallery_size))),
                    "coverage": float(coverage_info.get("coverage", 0.0)),
                    "fill_ratio_mean": float(coverage_info.get("fill_ratio_mean", coverage_info.get("coverage", 0.0))),
                    "full_coverage": float(coverage_info.get("full_coverage", coverage_info.get("coverage", 0.0))),
                    "metric_name": metric_label,
                    "metric_value": float(retrieval_metrics.get(f"gallery_{gallery_size}_{metric_key}", 0.0)),
                }
            )
    return rows


def plot_retrieval_curves(curve_rows: Sequence[Mapping[str, Any]], output_dir: Path | str) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.rcParams.update({"font.size": PLOT_FONT_BASE, "font.family": "serif", "font.serif": ["Times New Roman", "STIXGeneral", "DejaVu Serif"]})
    except Exception:
        return

    rows = list(curve_rows)
    if not rows:
        return

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    methods = sorted({str(row["method"]) for row in rows}, key=_plot_method_draw_order)
    metrics = ["R@1", "R@10", "MRR"]
    fig, axes = plt.subplots(1, len(metrics), figsize=RETRIEVAL_CURVES_FIGSIZE, sharex=False, sharey=False)
    if len(metrics) == 1:
        axes = [axes]

    for ax, metric_name in zip(axes, metrics):
        metric_rows = [row for row in rows if str(row.get("metric_name")) == metric_name]
        for method in methods:
            method_rows = sorted(
                [row for row in metric_rows if str(row.get("method")) == method],
                key=lambda row: int(row["gallery_size_target"]),
            )
            if not method_rows:
                continue
            ax.plot(
                [int(row["gallery_size_target"]) for row in method_rows],
                [float(row["metric_value"]) for row in method_rows],
                marker="o",
                linewidth=2,
                label=_plot_method_label(method),
            )
        ax.set_xscale("log")
        ax.set_xlabel("Gallery Size")
        ax.set_ylabel(metric_name)
        ax.set_title(f"{metric_name} vs Gallery Size")
        ax.grid(True, alpha=0.3)

    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=max(1, min(4, len(labels))), frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    fig.savefig(output_dir / "retrieval_curves.png", dpi=PLOT_DPI, bbox_inches="tight")
    fig.savefig(output_dir / "retrieval_curves.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_retrieval_coverage(curve_rows: Sequence[Mapping[str, Any]], output_dir: Path | str) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.rcParams.update({"font.size": PLOT_FONT_BASE, "font.family": "serif", "font.serif": ["Times New Roman", "STIXGeneral", "DejaVu Serif"]})
    except Exception:
        return

    rows = list(curve_rows)
    if not rows:
        return

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    by_method: Dict[str, Dict[int, Dict[str, float]]] = {}
    for row in rows:
        method = str(row["method"])
        gallery_size = int(row["gallery_size_target"])
        by_method.setdefault(method, {})
        by_method[method][gallery_size] = {
            "coverage": float(row.get("coverage", 0.0)),
            "gallery_size_actual": float(row.get("gallery_size_actual", gallery_size)),
        }

    fig, ax = plt.subplots(figsize=(7, 5))
    for method in sorted(by_method, key=_plot_method_draw_order):
        ordered = sorted(by_method[method].items(), key=lambda item: int(item[0]))
        ax.plot(
            [gallery_size for gallery_size, _ in ordered],
            [payload["coverage"] for _, payload in ordered],
            marker="o",
            linewidth=2,
            label=_plot_method_label(method),
        )

    ax.set_xscale("log")
    ax.set_xlabel("Gallery Size")
    ax.set_ylabel("Mean Fill Ratio")
    ax.set_ylim(0.0, 1.05)
    ax.set_title("Mean Gallery Fill Ratio Under Candidate Constraints")
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_dir / "retrieval_coverage.png", dpi=PLOT_DPI, bbox_inches="tight")
    _remove_stale_pdf(output_dir / "retrieval_coverage.pdf")
    plt.close(fig)
