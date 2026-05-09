import h5py
import numpy as np
import pandas as pd
from astropy.io import fits
from tqdm import tqdm
import hashlib
import os
import sys
import healpy as hp
from ligo.skymap.io.fits import read_sky_map
from ligo.skymap.moc import uniq2nest, uniq2pixarea
import warnings
warnings.filterwarnings("ignore", "Wswiglal-redir-stdio")
import torch
from collections import defaultdict
from typing import Dict, List, Iterator, Optional, Tuple, Sequence
from torch.utils.data import Dataset, DataLoader, Sampler
from pathlib import Path
import subprocess

try:
    from optical_prefix import (
        DEFAULT_PREFIX_DET_SUPPORT,
        apply_prefix_right_censoring_numpy,
        build_prefix_manifest_entries,
        parse_prefix_det_support,
    )
except ModuleNotFoundError as exc:
    if exc.name != "optical_prefix":
        raise
    # Support importlib-loaded entry scripts where Model/ is not on sys.path.
    this_dir = str(Path(__file__).resolve().parent)
    if this_dir not in sys.path:
        sys.path.insert(0, this_dir)
    from optical_prefix import (
        DEFAULT_PREFIX_DET_SUPPORT,
        apply_prefix_right_censoring_numpy,
        build_prefix_manifest_entries,
        parse_prefix_det_support,
    )

try:
    from lightcurve_merge import (
        MERGE_FLUX_DOMAIN,
        MERGE_MODE,
        MERGE_WINDOW_HOURS,
        merge_photometry_psfflux,
    )
except ModuleNotFoundError as exc:
    if exc.name != "lightcurve_merge":
        raise
    this_dir = str(Path(__file__).resolve().parent)
    if this_dir not in sys.path:
        sys.path.insert(0, this_dir)
    from lightcurve_merge import (
        MERGE_FLUX_DOMAIN,
        MERGE_MODE,
        MERGE_WINDOW_HOURS,
        merge_photometry_psfflux,
    )


LEGACY_FULL_WINDOW_START = -0.3
LEGACY_FULL_WINDOW_END = 0.6
RUNTIME_WINDOW_TOL = 1.0e-8


def _normalize_runtime_input_window(
    window_start: Optional[float],
    window_end: Optional[float],
) -> Tuple[Optional[float], Optional[float]]:
    if window_start is None and window_end is None:
        return None, None
    if window_start is None or window_end is None:
        raise ValueError("runtime input window requires both start and end or neither.")
    start = float(window_start)
    end = float(window_end)
    if not np.isfinite(start) or not np.isfinite(end):
        raise ValueError("runtime input window bounds must be finite.")
    if end <= start:
        raise ValueError(f"Invalid runtime input window: start={start}, end={end}")
    return start, end


def _float_close(a: Optional[float], b: Optional[float], tol: float = RUNTIME_WINDOW_TOL) -> bool:
    if a is None or b is None:
        return False
    return abs(float(a) - float(b)) <= float(tol)


def _read_root_time_window_attrs(h5_path: str) -> Tuple[Optional[float], Optional[float]]:
    if h5_path is None or (not os.path.exists(h5_path)):
        return None, None
    with h5py.File(h5_path, "r") as f:
        if "time_window_start" in f.attrs and "time_window_end" in f.attrs:
            return float(f.attrs["time_window_start"]), float(f.attrs["time_window_end"])
    return None, None


def _runtime_input_window_is_active_for_bounds(
    window_start: Optional[float],
    window_end: Optional[float],
    dataset_window_start: Optional[float],
    dataset_window_end: Optional[float],
) -> bool:
    start, end = _normalize_runtime_input_window(window_start, window_end)
    if start is None or end is None:
        return False
    baseline_start = LEGACY_FULL_WINDOW_START if dataset_window_start is None else float(dataset_window_start)
    baseline_end = LEGACY_FULL_WINDOW_END if dataset_window_end is None else float(dataset_window_end)
    return not (_float_close(start, baseline_start) and _float_close(end, baseline_end))


def runtime_input_window_is_active_for_h5(
    h5_path: Optional[str],
    window_start: Optional[float],
    window_end: Optional[float],
) -> bool:
    dataset_window_start, dataset_window_end = _read_root_time_window_attrs(h5_path) if h5_path else (None, None)
    return _runtime_input_window_is_active_for_bounds(
        window_start,
        window_end,
        dataset_window_start,
        dataset_window_end,
    )


def build_effective_input_window_metadata(
    ref_start: float,
    ref_end: float,
    *,
    runtime_input_window_start: Optional[float] = None,
    runtime_input_window_end: Optional[float] = None,
    dataset_window_start: Optional[float] = None,
    dataset_window_end: Optional[float] = None,
) -> Dict[str, object]:
    runtime_start, runtime_end = _normalize_runtime_input_window(
        runtime_input_window_start,
        runtime_input_window_end,
    )
    effective_start = float(ref_start if runtime_start is None else runtime_start)
    effective_end = float(ref_end if runtime_end is None else runtime_end)
    runtime_override_active = _runtime_input_window_is_active_for_bounds(
        effective_start,
        effective_end,
        dataset_window_start,
        dataset_window_end,
    )
    return {
        "runtime_override_active": bool(runtime_override_active),
        "effective_input_window_start_scaled": float(effective_start),
        "effective_input_window_end_scaled": float(effective_end),
        "effective_input_window_start_days": float(effective_start * 100.0),
        "effective_input_window_end_days": float(effective_end * 100.0),
        "reference_grid_start_scaled": float(ref_start),
        "reference_grid_end_scaled": float(ref_end),
    }


def apply_runtime_input_window_torch(
    opt_time: torch.Tensor,
    opt_val: torch.Tensor,
    opt_mask: torch.Tensor,
    opt_err: Optional[torch.Tensor] = None,
    slot_is_detection: Optional[torch.Tensor] = None,
    *,
    window_start: Optional[float] = None,
    window_end: Optional[float] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
    start, end = _normalize_runtime_input_window(window_start, window_end)
    if start is None or end is None:
        return opt_time, opt_val, opt_mask, opt_err, slot_is_detection

    keep_rows = (opt_time >= float(start)) & (opt_time <= float(end))
    keep_rows_3d = keep_rows.unsqueeze(-1)
    cropped_time = torch.where(keep_rows, opt_time, torch.zeros_like(opt_time))
    cropped_val = torch.where(keep_rows_3d, opt_val, torch.zeros_like(opt_val))
    cropped_mask = torch.where(keep_rows_3d, opt_mask, torch.zeros_like(opt_mask))
    cropped_err = None if opt_err is None else torch.where(keep_rows_3d, opt_err, torch.zeros_like(opt_err))
    cropped_det = None
    if slot_is_detection is not None:
        cropped_det = torch.where(keep_rows, slot_is_detection, torch.zeros_like(slot_is_detection))
    return cropped_time, cropped_val, cropped_mask, cropped_err, cropped_det


def apply_runtime_input_window_numpy(
    opt_time: np.ndarray,
    opt_val: np.ndarray,
    opt_mask: np.ndarray,
    opt_err: Optional[np.ndarray] = None,
    slot_is_detection: Optional[np.ndarray] = None,
    *,
    window_start: Optional[float] = None,
    window_end: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
    start, end = _normalize_runtime_input_window(window_start, window_end)
    if start is None or end is None:
        return opt_time, opt_val, opt_mask, opt_err, slot_is_detection

    keep_rows = (np.asarray(opt_time, dtype=np.float32) >= float(start)) & (
        np.asarray(opt_time, dtype=np.float32) <= float(end)
    )
    out_time = np.asarray(opt_time, dtype=np.float32).copy()
    out_val = np.asarray(opt_val, dtype=np.float32).copy()
    out_mask = np.asarray(opt_mask, dtype=np.float32).copy()
    out_err = None if opt_err is None else np.asarray(opt_err, dtype=np.float32).copy()
    out_det = None if slot_is_detection is None else np.asarray(slot_is_detection, dtype=np.float32).copy()

    out_time[~keep_rows] = 0.0
    out_val[~keep_rows, :] = 0.0
    out_mask[~keep_rows, :] = 0.0
    if out_err is not None:
        out_err[~keep_rows, :] = 0.0
    if out_det is not None:
        out_det[~keep_rows] = 0.0
    return out_time, out_val, out_mask, out_err, out_det


def _format_runtime_window_token(value: float) -> str:
    sign = "m" if float(value) < 0 else "p"
    scaled = int(round(abs(float(value)) * 10000.0))
    return f"{sign}{scaled:06d}"


def _runtime_window_cache_path(
    h5_path: str,
    group: str,
    window_start: float,
    window_end: float,
) -> Path:
    source = Path(h5_path)
    digest = hashlib.blake2b(
        f"{Path(h5_path).resolve()}|{group}|{float(window_start):.8f}|{float(window_end):.8f}".encode("utf-8"),
        digest_size=8,
    ).hexdigest()
    group_token = str(group).strip("/").replace("/", "__")
    cache_dir = source.parent / ".runtime_window_meta"
    filename = (
        f"{source.stem}.{group_token}.win_{_format_runtime_window_token(window_start)}_"
        f"{_format_runtime_window_token(window_end)}.{digest}.npz"
    )
    return cache_dir / filename


def _compute_runtime_window_meta_chunk(
    times: np.ndarray,
    masks: np.ndarray,
    slot_is_detection: Optional[np.ndarray],
    *,
    window_start: float,
    window_end: float,
) -> Dict[str, np.ndarray]:
    times_np = np.asarray(times, dtype=np.float32)
    masks_np = np.asarray(masks, dtype=np.float32)
    row_valid = masks_np.sum(axis=-1) > 0
    keep = row_valid & (times_np >= float(window_start)) & (times_np <= float(window_end))
    keep_3d = keep[..., None] & (masks_np > 0)

    n_obs = keep.sum(axis=1).astype(np.float32, copy=False)
    if slot_is_detection is not None:
        slot_det_np = np.asarray(slot_is_detection, dtype=np.float32)
        n_det = ((slot_det_np > 0) & keep).sum(axis=1).astype(np.float32, copy=False)
    else:
        n_det = n_obs.astype(np.float32, copy=True)

    band_hits = keep_3d.any(axis=1)
    n_bands = band_hits.sum(axis=1).astype(np.float32, copy=False)
    single_band_id = np.full((times_np.shape[0],), -1, dtype=np.int16)
    single_mask = n_bands == 1
    if np.any(single_mask):
        single_band_id[single_mask] = np.argmax(band_hits[single_mask], axis=1).astype(np.int16, copy=False)

    t_span = np.zeros((times_np.shape[0],), dtype=np.float32)
    has_obs = n_obs > 0
    if np.any(has_obs):
        first_idx = np.argmax(keep, axis=1)
        last_idx = keep.shape[1] - 1 - np.argmax(keep[:, ::-1], axis=1)
        rows = np.flatnonzero(has_obs)
        t_span[rows] = (
            times_np[rows, last_idx[rows]] - times_np[rows, first_idx[rows]]
        ).astype(np.float32, copy=False)

    return {
        "n_obs": n_obs,
        "n_det": n_det,
        "n_bands": n_bands,
        "t_span": t_span,
        "single_band_id": single_band_id,
    }


def _load_or_compute_runtime_window_meta_arrays(
    h5_path: str,
    group: str,
    *,
    window_start: Optional[float],
    window_end: Optional[float],
    chunk_size: int = 8192,
) -> dict:
    start, end = _normalize_runtime_input_window(window_start, window_end)
    if start is None or end is None or (not runtime_input_window_is_active_for_h5(h5_path, start, end)):
        with h5py.File(h5_path, "r") as f:
            return OpticalBinaryDataset._load_or_compute_meta_arrays(f, group, chunk_size=chunk_size)

    cache_path = _runtime_window_cache_path(h5_path, group, float(start), float(end))
    if cache_path.exists():
        with np.load(cache_path, allow_pickle=False) as npz:
            return {
                "n_obs": np.asarray(npz["n_obs"], dtype=np.float32).reshape(-1),
                "n_det": np.asarray(npz["n_det"], dtype=np.float32).reshape(-1),
                "n_bands": np.asarray(npz["n_bands"], dtype=np.float32).reshape(-1),
                "t_span": np.asarray(npz["t_span"], dtype=np.float32).reshape(-1),
                "single_band_id": np.asarray(npz["single_band_id"], dtype=np.int16).reshape(-1),
            }

    with h5py.File(h5_path, "r") as f:
        if group not in f:
            raise KeyError(f"Group '{group}' not found in {h5_path}")
        grp = f[group]
        n_total = int(grp["values"].shape[0])
        ds_masks = grp["masks"]
        ds_times = grp["times"]
        ds_slot_is_detection = grp["slot_is_detection"] if "slot_is_detection" in grp else None

        n_obs_all = np.zeros((n_total,), dtype=np.float32)
        n_det_all = np.zeros((n_total,), dtype=np.float32)
        n_bands_all = np.zeros((n_total,), dtype=np.float32)
        t_span_all = np.zeros((n_total,), dtype=np.float32)
        single_band_id_all = np.full((n_total,), -1, dtype=np.int16)

        for s in range(0, n_total, int(chunk_size)):
            e = min(s + int(chunk_size), n_total)
            chunk = _compute_runtime_window_meta_chunk(
                times=np.asarray(ds_times[s:e], dtype=np.float32),
                masks=np.asarray(ds_masks[s:e], dtype=np.float32),
                slot_is_detection=(
                    np.asarray(ds_slot_is_detection[s:e], dtype=np.float32)
                    if ds_slot_is_detection is not None
                    else None
                ),
                window_start=float(start),
                window_end=float(end),
            )
            n_obs_all[s:e] = chunk["n_obs"]
            n_det_all[s:e] = chunk["n_det"]
            n_bands_all[s:e] = chunk["n_bands"]
            t_span_all[s:e] = chunk["t_span"]
            single_band_id_all[s:e] = chunk["single_band_id"]

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache_path,
        n_obs=n_obs_all,
        n_det=n_det_all,
        n_bands=n_bands_all,
        t_span=t_span_all,
        single_band_id=single_band_id_all,
        window_start=np.float32(start),
        window_end=np.float32(end),
    )
    return {
        "n_obs": n_obs_all,
        "n_det": n_det_all,
        "n_bands": n_bands_all,
        "t_span": t_span_all,
        "single_band_id": single_band_id_all,
    }


def _normalize_day_windows(windows) -> List[float]:
    if windows is None:
        return []
    if isinstance(windows, str):
        parts = [p.strip() for p in windows.split(",")]
        vals = [float(p) for p in parts if p]
    else:
        vals = [float(v) for v in windows]
    vals = [v for v in vals if np.isfinite(v) and v > 0]
    if not vals:
        return []
    vals = sorted(set(vals))
    return vals


def _geometric_level_probs(n_levels: int) -> np.ndarray:
    if n_levels <= 0:
        return np.zeros((0,), dtype=np.float64)
    weights = np.asarray([0.5 ** i for i in range(n_levels)], dtype=np.float64)
    s = float(weights.sum())
    if s <= 0:
        return np.full((n_levels,), 1.0 / float(n_levels), dtype=np.float64)
    return weights / s


def _resolve_noncache_loader_policy(
    *,
    cache_in_memory: bool,
    usage: str,
    num_workers: int,
    pin_memory: bool,
    persistent_workers: bool,
    prefetch_factor: int,
) -> Dict[str, object]:
    resolved = {
        "num_workers": int(num_workers),
        "pin_memory": bool(pin_memory),
        "persistent_workers": bool(persistent_workers),
        "prefetch_factor": int(prefetch_factor),
        "ram_reclaim_active": False,
    }
    if cache_in_memory:
        return resolved

    resolved["ram_reclaim_active"] = True
    resolved["persistent_workers"] = False
    if str(usage) in {"val", "ood"}:
        resolved["pin_memory"] = False
    elif resolved["num_workers"] <= 0:
        resolved["pin_memory"] = False
    return resolved


def _log_loader_runtime_policy(
    *,
    label: str,
    cache_in_memory: bool,
    usage: str,
    policy: Dict[str, object],
) -> None:
    if cache_in_memory:
        print(
            f"{label}: cache_in_memory=True, preserving existing loader policy "
            f"(usage={usage}, num_workers={int(policy['num_workers'])}, "
            f"pin_memory={int(bool(policy['pin_memory']))}, "
            f"persistent_workers={int(bool(policy['persistent_workers']))}, "
            f"prefetch_factor={int(policy['prefetch_factor'])})."
        )
        return

    print(
        f"{label}: cache_in_memory=False, RAM-reclaim loader policy active "
        f"(usage={usage}, num_workers={int(policy['num_workers'])}, "
        f"pin_memory={int(bool(policy['pin_memory']))}, "
        f"persistent_workers={int(bool(policy['persistent_workers']))}, "
        f"prefetch_factor={int(policy['prefetch_factor'])})."
    )


def build_gw_to_lc_mapping(h5_path: str):
    """
    Scans the HDF5 file to build a mapping from GW Event Index to Light Curve Indices.
    This is required for the Balanced Sampler.
    
    Args:
        h5_path: Path to the HDF5 file.
        
    Returns:
        gw_to_lc_map: Dictionary {gw_idx: np.array([lc_idx_1, lc_idx_2, ...])}
    """
    print(f"Building GW-to-Optical index mapping from {h5_path}...")
    with h5py.File(h5_path, 'r') as f:
        # Load the parent_gw_idx array into memory (it's essentially a list of integers)
        # Shape: [Total_Optical_Samples]
        all_parent_indices = f['events/optical_data/parent_gw_idx'][:]
        
    gw_to_lc_map = defaultdict(list)
    for lc_idx, gw_idx in enumerate(all_parent_indices):
        gw_to_lc_map[gw_idx].append(lc_idx)
        
    # Convert lists to numpy arrays for faster random sampling later
    final_map = {k: np.array(v) for k, v in gw_to_lc_map.items()}
    
    print(f"Mapping complete. Found {len(final_map)} unique GW events.")
    return final_map


class RelationalHDF5Dataset(Dataset):
    """
    PyTorch Dataset for the relational HDF5 structure.
    Reads optical data by index and fetches the corresponding unique GW data.
    Optional negative optical samples can be drawn from a separate HDF5 file.
    """
    def __init__(
        self,
        h5_path: str,
        negative_h5_path: str = None,
        negative_group: str = "events/optical_data",
        cache_in_memory: bool = False,
        use_neg_gw: bool = False,
        return_zero_time_mjd: bool = False,
        nonkn_cls_base_field: str = "zero_time_mjd_cls_base",
        extra_negative_timeaware_enable: bool = False,
        extra_negative_timeaware_windows_days=None,
        extra_negative_timeaware_min_candidates: int = 1,
        extra_negative_timeaware_seed: int = 42,
        opt_input_window_start: Optional[float] = None,
        opt_input_window_end: Optional[float] = None,
    ):
        super().__init__()
        self.h5_path = h5_path
        self.h5_file = None
        self.negative_h5_path = negative_h5_path
        self.negative_group = negative_group
        self.neg_file = None
        self.neg_length = None
        self.cache_in_memory = cache_in_memory
        self.data_cache = None
        self.neg_cache = None
        self.return_zero_time_mjd = bool(return_zero_time_mjd)
        self.nonkn_cls_base_field = str(nonkn_cls_base_field)
        self.has_opt_zero_time_mjd_base = False
        self.has_opt_first_detection_mjd = False
        self.has_gw_event_time_mjd = False
        self.has_neg_zero_time_mjd_base = False
        self.has_neg_zero_time_mjd_cls_base = False
        self.extra_negative_timeaware_enable = bool(extra_negative_timeaware_enable)
        self.extra_negative_timeaware_windows_days = _normalize_day_windows(extra_negative_timeaware_windows_days)
        self.extra_negative_timeaware_min_candidates = max(1, int(extra_negative_timeaware_min_candidates))
        self.extra_negative_timeaware_seed = int(extra_negative_timeaware_seed)
        self.extra_negative_timeaware_level_probs = _geometric_level_probs(
            len(self.extra_negative_timeaware_windows_days)
        )
        self.extra_negative_timeaware_active = False
        self.gw_event_time_mjd_np = None
        self.neg_cls_time_sorted = None
        self.neg_cls_sorted_to_orig_idx = None

        # Negative GW support (BNS events without KN)
        self.use_neg_gw = use_neg_gw
        self.neg_gw_indices = None
        self.n_pos_gw = None
        self.opt_input_window_start, self.opt_input_window_end = _normalize_runtime_input_window(
            opt_input_window_start,
            opt_input_window_end,
        )
        pos_dataset_window_start, pos_dataset_window_end = _read_root_time_window_attrs(self.h5_path)
        self.opt_input_window_active = _runtime_input_window_is_active_for_bounds(
            self.opt_input_window_start,
            self.opt_input_window_end,
            pos_dataset_window_start,
            pos_dataset_window_end,
        )
        self.neg_input_window_active = False

        # Open file temporarily to get dataset length
        with h5py.File(h5_path, 'r') as f:
            self.length = f['events/optical_data/values'].shape[0]
            self.has_opt_zero_time_mjd_base = "events/optical_data/zero_time_mjd_base" in f
            self.has_opt_first_detection_mjd = "events/optical_data/first_detection_mjd" in f
            self.has_gw_event_time_mjd = "events/gw_data/event_time_mjd" in f
            if self.has_gw_event_time_mjd:
                self.gw_event_time_mjd_np = np.asarray(
                    f["events/gw_data/event_time_mjd"][:], dtype=np.float64
                ).reshape(-1)

            if self.return_zero_time_mjd and (not self.has_opt_zero_time_mjd_base) and (not self.has_gw_event_time_mjd):
                raise KeyError(
                    "return_zero_time_mjd=True requires either "
                    "'events/optical_data/zero_time_mjd_base' or "
                    "'events/gw_data/event_time_mjd' in positive dataset."
                )
            if self.cache_in_memory:
                print(f"Caching positive dataset in memory from {h5_path} (as tensors)...")
                self.data_cache = {
                    "opt_val": torch.from_numpy(f['events/optical_data/values'][:]),
                    "opt_err": torch.from_numpy(f['events/optical_data/errors'][:]),
                    "opt_mask": torch.from_numpy(f['events/optical_data/masks'][:]),
                    "opt_time": torch.from_numpy(f['events/optical_data/times'][:]),
                    "opt_coords": torch.from_numpy(f['events/optical_data/coordinates'][:]),
                    "parent_gw_idx": f['events/optical_data/parent_gw_idx'][:],  # keep numpy for indexing
                    "gw_scalar": torch.from_numpy(f['events/gw_data/scalars'][:]),
                    "gw_skymap": torch.from_numpy(f['events/gw_data/skymaps'][:])
                }
                if self.has_opt_zero_time_mjd_base:
                    self.data_cache["opt_zero_time_mjd_base"] = torch.from_numpy(
                        f["events/optical_data/zero_time_mjd_base"][:]
                    )
                if self.has_opt_first_detection_mjd:
                    self.data_cache["opt_first_detection_mjd"] = torch.from_numpy(
                        f["events/optical_data/first_detection_mjd"][:]
                    )
                elif self.return_zero_time_mjd and self.has_gw_event_time_mjd:
                    self.data_cache["gw_event_time_mjd"] = torch.from_numpy(
                        f["events/gw_data/event_time_mjd"][:]
                    )
                # Validate data integrity once at load time
                assert not torch.isnan(self.data_cache["opt_val"]).any(), "NaN found in cached optical values"
                assert not torch.isinf(self.data_cache["opt_val"]).any(), "Inf found in cached optical values"
                assert not torch.isnan(self.data_cache["opt_err"]).any(), "NaN found in cached optical errors"
                assert not torch.isinf(self.data_cache["opt_err"]).any(), "Inf found in cached optical errors"
                # Move tensors to shared memory to avoid CoW duplication in forked workers
                for k, v in self.data_cache.items():
                    if isinstance(v, torch.Tensor):
                        self.data_cache[k] = v.share_memory_()

            # Load negative GW data if available and requested
            if use_neg_gw and 'events/gw_data/has_kn' in f:
                has_kn = f['events/gw_data/has_kn'][:]
                self.neg_gw_indices = np.where(has_kn == 0)[0]
                self.n_pos_gw = int(np.sum(has_kn == 1))
                print(f"Loaded {len(self.neg_gw_indices)} negative GW events (no KN)")

                if self.cache_in_memory:
                    # Cache negative GW data separately for efficient sampling (as tensors)
                    self.data_cache["neg_gw_scalar"] = torch.from_numpy(f['events/gw_data/scalars'][self.neg_gw_indices]).share_memory_()
                    self.data_cache["neg_gw_skymap"] = torch.from_numpy(f['events/gw_data/skymaps'][self.neg_gw_indices]).share_memory_()
        
        if self.negative_h5_path is not None:
            neg_dataset_window_start, neg_dataset_window_end = _read_root_time_window_attrs(self.negative_h5_path)
            self.neg_input_window_active = _runtime_input_window_is_active_for_bounds(
                self.opt_input_window_start,
                self.opt_input_window_end,
                neg_dataset_window_start,
                neg_dataset_window_end,
            )
            with h5py.File(self.negative_h5_path, 'r') as f:
                if self.negative_group not in f:
                    raise KeyError(f"Negative group '{self.negative_group}' not found in {self.negative_h5_path}")
                self.neg_length = f[f"{self.negative_group}/values"].shape[0]
                self.has_neg_zero_time_mjd_base = f"{self.negative_group}/zero_time_mjd_base" in f
                self.has_neg_zero_time_mjd_cls_base = (
                    f"{self.negative_group}/{self.nonkn_cls_base_field}" in f
                )
                if self.return_zero_time_mjd and not self.has_neg_zero_time_mjd_base:
                    raise KeyError(
                        f"return_zero_time_mjd=True requires '{self.negative_group}/zero_time_mjd_base' "
                        f"in negative dataset: {self.negative_h5_path}"
                    )
                if self.return_zero_time_mjd and not self.has_neg_zero_time_mjd_cls_base:
                    raise KeyError(
                        f"return_zero_time_mjd=True requires '{self.negative_group}/{self.nonkn_cls_base_field}' "
                        f"in negative dataset: {self.negative_h5_path}"
                    )
                if self.cache_in_memory:
                    print(f"Caching negative dataset in memory from {self.negative_h5_path} (as tensors)...")
                    self.neg_cache = {
                        "neg_val": torch.from_numpy(f[f"{self.negative_group}/values"][:]),
                        "neg_err": torch.from_numpy(f[f"{self.negative_group}/errors"][:]),
                        "neg_mask": torch.from_numpy(f[f"{self.negative_group}/masks"][:]),
                        "neg_time": torch.from_numpy(f[f"{self.negative_group}/times"][:]),
                        "neg_coords": torch.from_numpy(f[f"{self.negative_group}/coordinates"][:])
                    }
                    if self.has_neg_zero_time_mjd_base:
                        self.neg_cache["neg_zero_time_mjd_base"] = torch.from_numpy(
                            f[f"{self.negative_group}/zero_time_mjd_base"][:]
                        )
                    if self.has_neg_zero_time_mjd_cls_base:
                        self.neg_cache["neg_zero_time_mjd_cls_base"] = torch.from_numpy(
                            f[f"{self.negative_group}/{self.nonkn_cls_base_field}"][:]
                        )
                    # Validate negative data integrity
                    assert not torch.isnan(self.neg_cache["neg_val"]).any(), "NaN found in cached negative values"
                    assert not torch.isinf(self.neg_cache["neg_val"]).any(), "Inf found in cached negative values"
                    # Move to shared memory to avoid CoW duplication in forked workers
                    for k, v in self.neg_cache.items():
                        if isinstance(v, torch.Tensor):
                            self.neg_cache[k] = v.share_memory_()
                if self.extra_negative_timeaware_enable:
                    if not self.has_neg_zero_time_mjd_cls_base:
                        print(
                            "WARNING: extra-negative time-aware sampling disabled because "
                            f"'{self.negative_group}/{self.nonkn_cls_base_field}' is missing."
                        )
                    else:
                        neg_cls = np.asarray(
                            f[f"{self.negative_group}/{self.nonkn_cls_base_field}"][:],
                            dtype=np.float64,
                        ).reshape(-1)
                        valid = np.isfinite(neg_cls)
                        if valid.any():
                            valid_idx = np.nonzero(valid)[0].astype(np.int64, copy=False)
                            neg_cls_valid = neg_cls[valid]
                            order = np.argsort(neg_cls_valid, kind="mergesort")
                            self.neg_cls_time_sorted = neg_cls_valid[order]
                            self.neg_cls_sorted_to_orig_idx = valid_idx[order]
                        else:
                            print(
                                "WARNING: extra-negative time-aware sampling disabled because "
                                "non-KN cls base times are all non-finite."
                            )

        self.extra_negative_timeaware_active = bool(
            self.extra_negative_timeaware_enable
            and self.negative_h5_path is not None
            and self.neg_length is not None
            and self.neg_length > 0
            and self.gw_event_time_mjd_np is not None
            and self.neg_cls_time_sorted is not None
            and self.neg_cls_sorted_to_orig_idx is not None
            and len(self.extra_negative_timeaware_windows_days) > 0
        )
        if self.extra_negative_timeaware_enable and not self.extra_negative_timeaware_active:
            print("WARNING: extra-negative time-aware sampling requested but inactive; fallback to uniform random.")
        elif self.extra_negative_timeaware_active:
            print(
                "Extra-negative time-aware sampling enabled: "
                f"windows={self.extra_negative_timeaware_windows_days}, "
                f"min_candidates={self.extra_negative_timeaware_min_candidates}, "
                f"valid_neg={int(self.neg_cls_time_sorted.shape[0])}/{int(self.neg_length)}"
            )

    def close(self) -> int:
        closed = 0
        for attr_name in ("h5_file", "neg_file"):
            handle = getattr(self, attr_name, None)
            if handle is None:
                continue
            try:
                handle.close()
                closed += 1
            except Exception:
                pass
            finally:
                setattr(self, attr_name, None)
        return closed

    def __del__(self):
        self.close()

    def _sample_uniform_neg_idx(self) -> int:
        return int(np.random.randint(0, int(self.neg_length)))

    def _sample_timeaware_negative_idx(self, gw_idx: int) -> int:
        if not self.extra_negative_timeaware_active:
            return self._sample_uniform_neg_idx()

        gw_idx = int(gw_idx)
        if gw_idx < 0 or gw_idx >= int(self.gw_event_time_mjd_np.shape[0]):
            return self._sample_uniform_neg_idx()

        anchor = float(self.gw_event_time_mjd_np[gw_idx])
        if not np.isfinite(anchor):
            return self._sample_uniform_neg_idx()

        times = self.neg_cls_time_sorted
        sorted_to_orig = self.neg_cls_sorted_to_orig_idx
        windows = self.extra_negative_timeaware_windows_days
        min_cand = int(self.extra_negative_timeaware_min_candidates)
        # One-sided after-GW windows: [0, w0], then (previous, current].
        window_edges = anchor + np.asarray(windows, dtype=np.float64)
        lefts = np.empty((len(windows),), dtype=np.int64)
        rights = np.searchsorted(times, window_edges, side="right")

        eligible_levels = []
        level_counts = []
        for i in range(len(windows)):
            if i == 0:
                lefts[i] = np.searchsorted(times, anchor, side="left")
            else:
                lefts[i] = np.searchsorted(times, anchor + float(windows[i - 1]), side="right")
            count = max(0, int(rights[i]) - int(lefts[i]))
            if count >= min_cand:
                eligible_levels.append(i)
                level_counts.append(count)

        if not eligible_levels:
            return self._sample_uniform_neg_idx()

        pri = self.extra_negative_timeaware_level_probs[np.asarray(eligible_levels, dtype=np.int64)]
        pri_sum = float(pri.sum())
        if pri_sum <= 0:
            pri = np.full((len(eligible_levels),), 1.0 / float(len(eligible_levels)), dtype=np.float64)
        else:
            pri = pri / pri_sum
        picked = int(np.random.choice(np.asarray(eligible_levels, dtype=np.int64), p=pri))

        outer_l = int(lefts[picked])
        outer_r = int(rights[picked])
        count = max(0, outer_r - outer_l)
        if count <= 0:
            return self._sample_uniform_neg_idx()

        sorted_idx = outer_l + int(np.random.randint(0, count))
        return int(sorted_to_orig[sorted_idx])
            
    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        """
        Args:
            idx: Index of the light curve (optical data).
        """
        neg_gw_local_idx = None
        if isinstance(idx, (tuple, list)):
            if len(idx) != 2:
                raise TypeError("Expected index int or (opt_idx, neg_gw_local_idx) tuple.")
            opt_idx, neg_gw_local_idx = idx
        elif isinstance(idx, np.ndarray):
            if idx.shape == ():
                opt_idx = int(idx)
            elif idx.size == 2:
                opt_idx, neg_gw_local_idx = idx.tolist()
            else:
                raise TypeError("Expected index int or (opt_idx, neg_gw_local_idx) tuple.")
        else:
            opt_idx = idx

        opt_zero_time_mjd_base = None
        if self.data_cache is None:
            # Lazy loading: Open file only when needed (crucial for num_workers > 0)
            if self.h5_file is None:
                self.h5_file = h5py.File(self.h5_path, 'r')

            # 1. Retrieve Optical Data (Values, Errors, Masks, Times)
            #    HDF5 structure: events/optical_data/...
            opt_val = torch.from_numpy(self.h5_file['events/optical_data/values'][opt_idx])
            opt_err = torch.from_numpy(self.h5_file['events/optical_data/errors'][opt_idx])
            opt_mask = torch.from_numpy(self.h5_file['events/optical_data/masks'][opt_idx])
            opt_time = torch.from_numpy(self.h5_file['events/optical_data/times'][opt_idx])
            opt_coords = torch.from_numpy(self.h5_file['events/optical_data/coordinates'][opt_idx])

            # 2. Retrieve Parent GW Index
            gw_idx = self.h5_file['events/optical_data/parent_gw_idx'][opt_idx]

            # 3. Retrieve Unique GW Data using gw_idx
            #    HDF5 structure: events/gw/...
            gw_scalar = torch.from_numpy(self.h5_file['events/gw_data/scalars'][gw_idx])
            gw_skymap = torch.from_numpy(self.h5_file['events/gw_data/skymaps'][gw_idx])
            if self.return_zero_time_mjd:
                if self.has_opt_zero_time_mjd_base:
                    opt_zero_time_mjd_base = torch.as_tensor(
                        self.h5_file["events/optical_data/zero_time_mjd_base"][opt_idx], dtype=torch.float32
                    )
                elif self.has_gw_event_time_mjd:
                    opt_zero_time_mjd_base = torch.as_tensor(
                        self.h5_file["events/gw_data/event_time_mjd"][gw_idx], dtype=torch.float32
                    )
                if self.has_opt_first_detection_mjd:
                    opt_first_detection_mjd = torch.as_tensor(
                        self.h5_file["events/optical_data/first_detection_mjd"][opt_idx], dtype=torch.float32
                    )
                elif self.has_opt_zero_time_mjd_base:
                    opt_first_detection_mjd = opt_zero_time_mjd_base.clone()
                else:
                    raise KeyError(
                        "Missing both 'events/optical_data/zero_time_mjd_base' and "
                        "'events/gw_data/event_time_mjd' while return_zero_time_mjd=True."
                    )
        else:
            # Data is already cached as tensors - direct indexing, no conversion needed
            opt_val = self.data_cache["opt_val"][opt_idx]
            opt_err = self.data_cache["opt_err"][opt_idx]
            opt_mask = self.data_cache["opt_mask"][opt_idx]
            opt_time = self.data_cache["opt_time"][opt_idx]
            opt_coords = self.data_cache["opt_coords"][opt_idx]
            gw_idx = self.data_cache["parent_gw_idx"][opt_idx]
            gw_scalar = self.data_cache["gw_scalar"][gw_idx]
            gw_skymap = self.data_cache["gw_skymap"][gw_idx]
            if self.return_zero_time_mjd:
                if "opt_zero_time_mjd_base" in self.data_cache:
                    opt_zero_time_mjd_base = self.data_cache["opt_zero_time_mjd_base"][opt_idx].to(torch.float32)
                elif "gw_event_time_mjd" in self.data_cache:
                    opt_zero_time_mjd_base = self.data_cache["gw_event_time_mjd"][gw_idx].to(torch.float32)
                else:
                    raise KeyError(
                        "Missing both cached opt_zero_time_mjd_base and gw_event_time_mjd while "
                        "return_zero_time_mjd=True."
                    )
                if "opt_first_detection_mjd" in self.data_cache:
                    opt_first_detection_mjd = self.data_cache["opt_first_detection_mjd"][opt_idx].to(torch.float32)
                elif "opt_zero_time_mjd_base" in self.data_cache:
                    opt_first_detection_mjd = opt_zero_time_mjd_base.clone()

        if self.opt_input_window_active:
            opt_time, opt_val, opt_mask, opt_err, _ = apply_runtime_input_window_torch(
                opt_time,
                opt_val,
                opt_mask,
                opt_err,
                None,
                window_start=self.opt_input_window_start,
                window_end=self.opt_input_window_end,
            )

        is_neg_gw = False
        if (
            neg_gw_local_idx is not None
            and self.use_neg_gw
            and int(neg_gw_local_idx) >= 0
        ):
            neg_gw_s, neg_gw_m = self.get_neg_gw_sample(int(neg_gw_local_idx))
            if neg_gw_s is not None and neg_gw_m is not None:
                gw_scalar = neg_gw_s
                gw_skymap = neg_gw_m
                is_neg_gw = True
        
        # Optional: Retrieve Negative Optical Data (non-KN or unrelated transient)
        if self.negative_h5_path is not None:
            neg_idx = self._sample_timeaware_negative_idx(int(gw_idx))
            neg_zero_time_mjd_base = None
            neg_zero_time_mjd_cls_base = None
            if self.neg_cache is None:
                if self.neg_file is None:
                    self.neg_file = h5py.File(self.negative_h5_path, 'r')
                neg_val = torch.from_numpy(self.neg_file[f"{self.negative_group}/values"][neg_idx])
                neg_err = torch.from_numpy(self.neg_file[f"{self.negative_group}/errors"][neg_idx])
                neg_mask = torch.from_numpy(self.neg_file[f"{self.negative_group}/masks"][neg_idx])
                neg_time = torch.from_numpy(self.neg_file[f"{self.negative_group}/times"][neg_idx])
                neg_coords = torch.from_numpy(self.neg_file[f"{self.negative_group}/coordinates"][neg_idx])
                if self.return_zero_time_mjd:
                    if not self.has_neg_zero_time_mjd_base:
                        raise KeyError(
                            f"Missing '{self.negative_group}/zero_time_mjd_base' in negative dataset "
                            "while return_zero_time_mjd=True."
                        )
                    neg_zero_time_mjd_base = torch.as_tensor(
                        self.neg_file[f"{self.negative_group}/zero_time_mjd_base"][neg_idx], dtype=torch.float32
                    )
                    if not self.has_neg_zero_time_mjd_cls_base:
                        raise KeyError(
                            f"Missing '{self.negative_group}/{self.nonkn_cls_base_field}' in negative dataset "
                            "while return_zero_time_mjd=True."
                        )
                    neg_zero_time_mjd_cls_base = torch.as_tensor(
                        self.neg_file[f"{self.negative_group}/{self.nonkn_cls_base_field}"][neg_idx], dtype=torch.float32
                    )
            else:
                # Data is already cached as tensors - direct indexing
                neg_val = self.neg_cache["neg_val"][neg_idx]
                neg_err = self.neg_cache["neg_err"][neg_idx]
                neg_mask = self.neg_cache["neg_mask"][neg_idx]
                neg_time = self.neg_cache["neg_time"][neg_idx]
                neg_coords = self.neg_cache["neg_coords"][neg_idx]
                if self.return_zero_time_mjd:
                    neg_zero_time_mjd_base = self.neg_cache["neg_zero_time_mjd_base"][neg_idx].to(torch.float32)
                    neg_zero_time_mjd_cls_base = self.neg_cache["neg_zero_time_mjd_cls_base"][neg_idx].to(torch.float32)

            if self.neg_input_window_active:
                neg_time, neg_val, neg_mask, neg_err, _ = apply_runtime_input_window_torch(
                    neg_time,
                    neg_val,
                    neg_mask,
                    neg_err,
                    None,
                    window_start=self.opt_input_window_start,
                    window_end=self.opt_input_window_end,
                )

            # Return tuple: (GW_Inputs, Optical_Inputs, Metadata, Negative_Optical_Inputs)
            # gw_idx is returned for masking the contrastive loss (handling same-source negatives)
            out = [
                gw_scalar, gw_skymap, opt_time, opt_val, opt_mask, opt_err, opt_coords, int(gw_idx),
                neg_time, neg_val, neg_mask, neg_err, neg_coords,
            ]
            if self.return_zero_time_mjd:
                out.extend([opt_zero_time_mjd_base, neg_zero_time_mjd_base,
                            neg_zero_time_mjd_cls_base, opt_first_detection_mjd])
            if neg_gw_local_idx is not None:
                out.append(is_neg_gw)
            return tuple(out)

        # Return tuple: (GW_Inputs, Optical_Inputs, Metadata)
        # gw_idx is returned for masking the contrastive loss (handling same-source negatives)
        out = [gw_scalar, gw_skymap, opt_time, opt_val, opt_mask, opt_err, opt_coords, int(gw_idx)]
        if self.return_zero_time_mjd:
            out.extend([opt_zero_time_mjd_base, opt_first_detection_mjd])
        if neg_gw_local_idx is not None:
            out.append(is_neg_gw)
        return tuple(out)

    def get_neg_gw_sample(self, local_idx: int = None):
        """
        Sample a random negative GW event (BNS without KN).

        Args:
            local_idx: Optional index into neg_gw_indices array. If None, random.

        Returns:
            Tuple of (gw_scalar, gw_skymap) as torch tensors.
        """
        if self.neg_gw_indices is None or len(self.neg_gw_indices) == 0:
            return None, None

        if local_idx is None:
            local_idx = np.random.randint(len(self.neg_gw_indices))

        if self.data_cache is not None and "neg_gw_scalar" in self.data_cache:
            # Data is already cached as tensors - clone to avoid in-place modification issues
            gw_scalar = self.data_cache["neg_gw_scalar"][local_idx].clone()
            gw_skymap = self.data_cache["neg_gw_skymap"][local_idx].clone()
        else:
            # Load from file
            if self.h5_file is None:
                self.h5_file = h5py.File(self.h5_path, 'r')
            actual_idx = self.neg_gw_indices[local_idx]
            gw_scalar = torch.from_numpy(self.h5_file['events/gw_data/scalars'][actual_idx])
            gw_skymap = torch.from_numpy(self.h5_file['events/gw_data/skymaps'][actual_idx])

        return gw_scalar, gw_skymap


class BalancedGWBatchedSampler(Sampler):
    """
    Custom Batch Sampler that ensures:
    1. Each batch contains 'batch_size' UNIQUE GW events.
    2. For each selected GW event, ONE light curve is randomly sampled.
    
    This prevents "false negatives" in contrastive learning where multiple LCs 
    from the same GW event appear in the same batch.
    """
    def __init__(self, gw_to_lc_map: dict, batch_size: int, steps_per_epoch: int):
        """
        Args:
            gw_to_lc_map: Dictionary mapping GW_ID -> [LC_ID_1, LC_ID_2, ...]
            batch_size: Number of unique GW events per batch.
            steps_per_epoch: Number of batches to yield per 'epoch'. (N_LC // batch_size)
        """
        self.gw_to_lc_map = gw_to_lc_map
        self.unique_gw_ids = list(gw_to_lc_map.keys())
        self.batch_size = batch_size
        self.steps_per_epoch = steps_per_epoch
        
        # Validation: Batch size cannot exceed total unique GW events
        if self.batch_size > len(self.unique_gw_ids):
            raise ValueError(f"Batch size ({batch_size}) > Unique GW events ({len(self.unique_gw_ids)}).")

    def __iter__(self) -> Iterator[List[int]]:
        for _ in range(self.steps_per_epoch):
            # 1. Sample unique GW IDs for this batch (without replacement)
            batch_gw_ids = np.random.choice(
                self.unique_gw_ids, 
                size=self.batch_size, 
                replace=False
            )
            
            batch_lc_indices = []
            
            # 2. For each GW ID, sample ONE random light curve index
            for gw_id in batch_gw_ids:
                possible_lcs = self.gw_to_lc_map[gw_id]
                chosen_lc = np.random.choice(possible_lcs)
                batch_lc_indices.append(chosen_lc)
                
            # Yield the list of optical indices for the DataLoader to fetch
            yield batch_lc_indices

    def __len__(self):
        return self.steps_per_epoch

class MultiPositiveGWBatchedSampler(Sampler):
    """
    Batch sampler for Supervised Contrastive Learning.
    Ensures multiple optical samples per GW event in each batch.

    Structure per batch:
    - Select (batch_size // samples_per_gw) unique GW events
    - For each GW, sample 'samples_per_gw' optical light curves
    - Total batch size = n_gw_per_batch * samples_per_gw
    """
    def __init__(
        self,
        gw_to_lc_map: dict,
        batch_size: int,
        samples_per_gw: int,
        steps_per_epoch: int,
        min_lc_per_gw: int = 2
    ):
        """
        Args:
            gw_to_lc_map: Dictionary mapping GW_ID -> [LC_ID_1, LC_ID_2, ...]
            batch_size: Total samples per batch
            samples_per_gw: Number of optical samples to draw per GW event
            steps_per_epoch: Number of batches per epoch
            min_lc_per_gw: Minimum light curves required for a GW to be eligible
        """
        self.gw_to_lc_map = gw_to_lc_map
        self.samples_per_gw = samples_per_gw
        self.steps_per_epoch = steps_per_epoch

        self.eligible_gw_ids = [
            gw_id for gw_id, lcs in gw_to_lc_map.items()
            if len(lcs) >= min_lc_per_gw
        ]

        self.n_gw_per_batch = batch_size // samples_per_gw

        if self.n_gw_per_batch < 2:
            raise ValueError(
                f"batch_size ({batch_size}) / samples_per_gw ({samples_per_gw}) "
                f"must be >= 2 for contrastive learning"
            )
        if self.n_gw_per_batch > len(self.eligible_gw_ids):
            raise ValueError(
                f"Not enough GW events with >= {min_lc_per_gw} light curves. "
                f"Need {self.n_gw_per_batch}, have {len(self.eligible_gw_ids)}"
            )

        print(
            f"MultiPositiveSampler: {len(self.eligible_gw_ids)} eligible GW events, "
            f"{self.n_gw_per_batch} GW/batch, {samples_per_gw} samples/GW"
        )

    def __iter__(self) -> Iterator[List[int]]:
        for _ in range(self.steps_per_epoch):
            batch_gw_ids = np.random.choice(
                self.eligible_gw_ids,
                size=self.n_gw_per_batch,
                replace=False
            )

            batch_lc_indices = []

            for gw_id in batch_gw_ids:
                possible_lcs = self.gw_to_lc_map[gw_id]
                replace = len(possible_lcs) < self.samples_per_gw
                chosen_lcs = np.random.choice(
                    possible_lcs,
                    size=self.samples_per_gw,
                    replace=replace
                )
                batch_lc_indices.extend(chosen_lcs.tolist())

            yield batch_lc_indices

    def __len__(self):
        return self.steps_per_epoch


class MixedGWBatchedSampler(Sampler):
    """
    Batch sampler that includes both positive and negative GW events.

    Structure per batch (batch_size=128, neg_gw_ratio=0.2):
    - 102 positive GW-optical pairs (80%)
    - 26 negative GW paired with random optical (20%)

    The sampler yields lists of (opt_idx, neg_gw_local_idx) tuples:
    - opt_idx: Optical sample index
    - neg_gw_local_idx: -1 means "use parent GW", >= 0 means "use negative GW at this local index"
    """
    def __init__(
        self,
        gw_to_lc_map: dict,
        neg_gw_indices: np.ndarray,
        batch_size: int,
        steps_per_epoch: int,
        neg_gw_ratio: float = 0.2,
        samples_per_gw: int = 1
    ):
        """
        Args:
            gw_to_lc_map: Dictionary mapping positive GW_ID -> [LC_ID_1, LC_ID_2, ...]
            neg_gw_indices: Array of local indices into dataset.neg_gw_indices
            batch_size: Total samples per batch
            steps_per_epoch: Number of batches per epoch
            neg_gw_ratio: Fraction of batch to fill with negative GW pairs
            samples_per_gw: Number of optical samples per GW (for SupCon compatibility)
        """
        self.gw_to_lc_map = gw_to_lc_map
        self.neg_gw_local_indices = np.array(neg_gw_indices, dtype=int)
        self.batch_size = batch_size
        self.steps_per_epoch = steps_per_epoch
        self.neg_gw_ratio = neg_gw_ratio
        self.samples_per_gw = samples_per_gw

        self.pos_gw_ids = list(gw_to_lc_map.keys())
        self.n_neg_per_batch = int(batch_size * neg_gw_ratio)
        self.n_pos_per_batch = batch_size - self.n_neg_per_batch

        # Collect all optical indices for negative pairing
        self.all_opt_indices = []
        for lcs in gw_to_lc_map.values():
            self.all_opt_indices.extend(lcs)
        self.all_opt_indices = np.array(self.all_opt_indices)

        # Adjust for SupCon mode
        if samples_per_gw > 1:
            self.n_gw_per_batch = self.n_pos_per_batch // samples_per_gw
            self.n_pos_per_batch = self.n_gw_per_batch * samples_per_gw
            self.n_neg_per_batch = self.batch_size - self.n_pos_per_batch
        else:
            self.n_gw_per_batch = self.n_pos_per_batch

        if self.n_gw_per_batch > len(self.pos_gw_ids):
            raise ValueError(
                f"Not enough positive GW events. Need {self.n_gw_per_batch}, have {len(self.pos_gw_ids)}"
            )
        if len(self.neg_gw_local_indices) == 0 and self.n_neg_per_batch > 0:
            raise ValueError("No negative GW indices provided for mixed sampling.")

        print(
            f"MixedGWBatchedSampler: {len(self.pos_gw_ids)} positive GW, "
            f"{len(self.neg_gw_local_indices)} negative GW, "
            f"{self.n_pos_per_batch} pos/batch, {self.n_neg_per_batch} neg/batch"
        )

    def __iter__(self) -> Iterator[list]:
        for _ in range(self.steps_per_epoch):
            batch_opt_indices = []
            batch_neg_gw_local_indices = []  # -1 for positive, local index for negative

            # 1. Sample positive GW-optical pairs
            if self.samples_per_gw > 1:
                # SupCon mode: multiple samples per GW
                batch_gw_ids = np.random.choice(
                    self.pos_gw_ids, self.n_gw_per_batch, replace=False
                )
                for gw_id in batch_gw_ids:
                    lcs = self.gw_to_lc_map[gw_id]
                    replace = len(lcs) < self.samples_per_gw
                    chosen_lcs = np.random.choice(lcs, self.samples_per_gw, replace=replace)
                    batch_opt_indices.extend(chosen_lcs.tolist())
                    batch_neg_gw_local_indices.extend([-1] * self.samples_per_gw)
            else:
                # Standard mode: one sample per GW
                batch_gw_ids = np.random.choice(
                    self.pos_gw_ids, self.n_pos_per_batch, replace=False
                )
                for gw_id in batch_gw_ids:
                    lcs = self.gw_to_lc_map[gw_id]
                    batch_opt_indices.append(np.random.choice(lcs))
                    batch_neg_gw_local_indices.append(-1)

            # 2. Sample negative GW pairs
            # Random optical indices paired with random negative GW
            if self.n_neg_per_batch > 0:
                opt_replace = self.n_neg_per_batch > len(self.all_opt_indices)
                neg_opt = np.random.choice(self.all_opt_indices, self.n_neg_per_batch, replace=opt_replace)
                neg_replace = self.n_neg_per_batch > len(self.neg_gw_local_indices)
                neg_gw_local = np.random.choice(
                    self.neg_gw_local_indices, self.n_neg_per_batch, replace=neg_replace
                )
                batch_opt_indices.extend(neg_opt.tolist())
                batch_neg_gw_local_indices.extend(neg_gw_local.tolist())

            batch_pairs = list(zip(batch_opt_indices, batch_neg_gw_local_indices))
            yield batch_pairs

    def __len__(self):
        return self.steps_per_epoch


class GWBatchedSampler(Sampler):
    """
    Batch sampler that iterates over GW IDs once per epoch.
    """
    def __init__(self, gw_to_lc_map: dict, batch_size: int, shuffle: bool = True, max_steps: int = None):
        self.gw_to_lc_map = gw_to_lc_map
        self.gw_ids = np.array(sorted(gw_to_lc_map.keys()))
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.max_steps = max_steps

        if self.batch_size > len(self.gw_ids):
            raise ValueError(f"Batch size ({batch_size}) > Unique GW events ({len(self.gw_ids)}).")

    def __iter__(self) -> Iterator[List[int]]:
        gw_ids = self.gw_ids.copy()
        if self.shuffle:
            np.random.shuffle(gw_ids)

        steps = 0
        for i in range(0, len(gw_ids), self.batch_size):
            if self.max_steps is not None and steps >= self.max_steps:
                break
            batch_gw_ids = gw_ids[i:i + self.batch_size]
            if len(batch_gw_ids) < self.batch_size:
                break

            batch_lc_indices = []
            for gw_id in batch_gw_ids:
                possible_lcs = self.gw_to_lc_map[gw_id]
                chosen_lc = np.random.choice(possible_lcs)
                batch_lc_indices.append(chosen_lc)

            yield batch_lc_indices
            steps += 1

    def __len__(self):
        total_steps = len(self.gw_ids) // self.batch_size
        if self.max_steps is None:
            return total_steps
        return min(total_steps, self.max_steps)

def split_gw_map(gw_to_lc_map: dict, val_split: float, seed: int):
    if val_split <= 0 or val_split >= 1:
        raise ValueError("val_split must be in (0, 1).")

    gw_ids = np.array(sorted(gw_to_lc_map.keys()))
    rng = np.random.default_rng(seed)
    rng.shuffle(gw_ids)

    val_size = max(1, int(len(gw_ids) * val_split))
    val_ids = set(gw_ids[:val_size])
    train_ids = set(gw_ids[val_size:])

    train_map = {gw_id: gw_to_lc_map[gw_id] for gw_id in train_ids}
    val_map = {gw_id: gw_to_lc_map[gw_id] for gw_id in val_ids}
    return train_map, val_map

def _build_dataloader(
    dataset: Dataset,
    sampler: Sampler,
    num_workers: int,
    pin_memory: bool,
    persistent_workers: bool,
    prefetch_factor: int
):
    if num_workers > 0:
        return DataLoader(
            dataset,
            batch_sampler=sampler,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
            prefetch_factor=prefetch_factor
        )
    return DataLoader(
        dataset,
        batch_sampler=sampler,
        num_workers=0,
        pin_memory=pin_memory
    )


def _build_configured_dataloader(
    dataset: Dataset,
    sampler: Sampler,
    num_workers: int,
    pin_memory: bool,
    persistent_workers: bool,
    prefetch_factor: int,
    *,
    cache_in_memory: bool,
    usage: str,
    label: str,
):
    policy = _resolve_noncache_loader_policy(
        cache_in_memory=cache_in_memory,
        usage=usage,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor,
    )
    _log_loader_runtime_policy(
        label=label,
        cache_in_memory=cache_in_memory,
        usage=usage,
        policy=policy,
    )
    return _build_dataloader(
        dataset,
        sampler,
        int(policy["num_workers"]),
        bool(policy["pin_memory"]),
        bool(policy["persistent_workers"]),
        int(policy["prefetch_factor"]),
    )


def create_training_dataloader(
    h5_path: str, 
    batch_size: int = 32, 
    steps_per_epoch: int = 1000, 
    num_workers: int = 4,
    pin_memory: bool = True,
    persistent_workers: bool = True,
    prefetch_factor: int = 4,
    negative_h5_path: str = None,
    negative_group: str = "events/optical_data",
    cache_in_memory: bool = False,
    return_zero_time_mjd: bool = False,
    nonkn_cls_base_field: str = "zero_time_mjd_cls_base",
    extra_negative_timeaware_enable: bool = False,
    extra_negative_timeaware_windows_days=None,
    extra_negative_timeaware_min_candidates: int = 1,
    extra_negative_timeaware_seed: int = 42,
    opt_input_window_start: Optional[float] = None,
    opt_input_window_end: Optional[float] = None,
    loader_usage: str = "train",
    loader_label: str = "Train DataLoader",
):
    """
    Factory function to initialize the Dataset, Sampler, and DataLoader.
    """
    if cache_in_memory and num_workers > 0:
        print("cache_in_memory=True with num_workers>0 may increase RAM usage.")
    # 1. Build Index Map (Once)
    gw_map = build_gw_to_lc_mapping(h5_path)
    
    # 2. Initialize Dataset
    dataset = RelationalHDF5Dataset(
        h5_path,
        negative_h5_path=negative_h5_path,
        negative_group=negative_group,
        cache_in_memory=cache_in_memory,
        return_zero_time_mjd=return_zero_time_mjd,
        nonkn_cls_base_field=nonkn_cls_base_field,
        extra_negative_timeaware_enable=extra_negative_timeaware_enable,
        extra_negative_timeaware_windows_days=extra_negative_timeaware_windows_days,
        extra_negative_timeaware_min_candidates=extra_negative_timeaware_min_candidates,
        extra_negative_timeaware_seed=extra_negative_timeaware_seed,
        opt_input_window_start=opt_input_window_start,
        opt_input_window_end=opt_input_window_end,
    )
    
    # 3. Initialize Custom Sampler
    # Note: 'steps_per_epoch' defines how many batches constitute one epoch loop
    sampler = BalancedGWBatchedSampler(
        gw_to_lc_map=gw_map,
        batch_size=batch_size,
        steps_per_epoch=steps_per_epoch
    )

    # 4. Initialize DataLoader
    # IMPORTANT: batch_sampler is used, so batch_size/shuffle/sampler/drop_last 
    # arguments in DataLoader constructor must not be provided.
    return _build_configured_dataloader(
        dataset,
        sampler,
        num_workers,
        pin_memory,
        persistent_workers,
        prefetch_factor,
        cache_in_memory=cache_in_memory,
        usage=loader_usage,
        label=loader_label,
    )

def create_train_val_dataloaders(
    h5_path: str,
    batch_size: int = 32,
    val_batch_size: int = None,
    steps_per_epoch: int = None,
    val_steps_per_epoch: int = None,
    val_split: float = 0.1,
    split_seed: int = 42,
    num_workers: int = 4,
    pin_memory: bool = True,
    persistent_workers: bool = True,
    prefetch_factor: int = 4,
    negative_h5_path: str = None,
    negative_group: str = "events/optical_data",
    cache_in_memory: bool = False,
    return_zero_time_mjd: bool = False,
    nonkn_cls_base_field: str = "zero_time_mjd_cls_base",
    extra_negative_timeaware_enable: bool = False,
    extra_negative_timeaware_windows_days=None,
    extra_negative_timeaware_min_candidates: int = 1,
    extra_negative_timeaware_seed: int = 42,
    opt_input_window_start: Optional[float] = None,
    opt_input_window_end: Optional[float] = None,
):
    if cache_in_memory and num_workers > 0:
        print("cache_in_memory=True with num_workers>0 may increase RAM usage.")

    gw_map = build_gw_to_lc_mapping(h5_path)
    train_map, val_map = split_gw_map(gw_map, val_split, split_seed)

    if steps_per_epoch is None:
        train_optical = sum(len(v) for v in train_map.values())
        steps_per_epoch = max(1, train_optical // batch_size)
    if val_batch_size is None:
        val_batch_size = batch_size
    val_batch_size = min(val_batch_size, len(val_map))
    if val_batch_size < 1:
        raise ValueError("val_batch_size must be >= 1.")
    if val_batch_size < batch_size:
        print(f"Validation batch size adjusted to {val_batch_size} based on unique GW events.")
    if val_steps_per_epoch is None:
        val_optical = sum(len(v) for v in val_map.values())
        val_steps_per_epoch = max(1, val_optical // val_batch_size)

    if cache_in_memory:
        shared_dataset = RelationalHDF5Dataset(
            h5_path,
            negative_h5_path=negative_h5_path,
            negative_group=negative_group,
            cache_in_memory=cache_in_memory,
            return_zero_time_mjd=return_zero_time_mjd,
            nonkn_cls_base_field=nonkn_cls_base_field,
            extra_negative_timeaware_enable=extra_negative_timeaware_enable,
            extra_negative_timeaware_windows_days=extra_negative_timeaware_windows_days,
            extra_negative_timeaware_min_candidates=extra_negative_timeaware_min_candidates,
            extra_negative_timeaware_seed=extra_negative_timeaware_seed,
            opt_input_window_start=opt_input_window_start,
            opt_input_window_end=opt_input_window_end,
        )
        train_dataset = shared_dataset
        val_dataset = shared_dataset
    else:
        train_dataset = RelationalHDF5Dataset(
            h5_path,
            negative_h5_path=negative_h5_path,
            negative_group=negative_group,
            cache_in_memory=cache_in_memory,
            return_zero_time_mjd=return_zero_time_mjd,
            nonkn_cls_base_field=nonkn_cls_base_field,
            extra_negative_timeaware_enable=extra_negative_timeaware_enable,
            extra_negative_timeaware_windows_days=extra_negative_timeaware_windows_days,
            extra_negative_timeaware_min_candidates=extra_negative_timeaware_min_candidates,
            extra_negative_timeaware_seed=extra_negative_timeaware_seed,
            opt_input_window_start=opt_input_window_start,
            opt_input_window_end=opt_input_window_end,
        )
        val_dataset = RelationalHDF5Dataset(
            h5_path,
            negative_h5_path=negative_h5_path,
            negative_group=negative_group,
            cache_in_memory=cache_in_memory,
            return_zero_time_mjd=return_zero_time_mjd,
            nonkn_cls_base_field=nonkn_cls_base_field,
            extra_negative_timeaware_enable=extra_negative_timeaware_enable,
            extra_negative_timeaware_windows_days=extra_negative_timeaware_windows_days,
            extra_negative_timeaware_min_candidates=extra_negative_timeaware_min_candidates,
            extra_negative_timeaware_seed=extra_negative_timeaware_seed,
            opt_input_window_start=opt_input_window_start,
            opt_input_window_end=opt_input_window_end,
        )

    train_sampler = BalancedGWBatchedSampler(
        gw_to_lc_map=train_map,
        batch_size=batch_size,
        steps_per_epoch=steps_per_epoch
    )
    val_sampler = BalancedGWBatchedSampler(
        gw_to_lc_map=val_map,
        batch_size=val_batch_size,
        steps_per_epoch=val_steps_per_epoch
    )

    train_loader = _build_configured_dataloader(
        train_dataset,
        train_sampler,
        num_workers,
        pin_memory,
        persistent_workers,
        prefetch_factor,
        cache_in_memory=cache_in_memory,
        usage="train",
        label="Train DataLoader",
    )
    val_loader = _build_configured_dataloader(
        val_dataset,
        val_sampler,
        num_workers,
        pin_memory,
        persistent_workers,
        prefetch_factor,
        cache_in_memory=cache_in_memory,
        usage="val",
        label="Validation DataLoader",
    )

    return train_loader, val_loader, steps_per_epoch, len(val_sampler)

def create_supcon_dataloaders(
    h5_path: str,
    batch_size: int = 128,
    samples_per_gw: int = 4,
    val_batch_size: int = None,
    steps_per_epoch: int = None,
    val_steps_per_epoch: int = None,
    val_split: float = 0.1,
    split_seed: int = 42,
    num_workers: int = 4,
    pin_memory: bool = True,
    persistent_workers: bool = True,
    prefetch_factor: int = 4,
    negative_h5_path: str = None,
    negative_group: str = "events/optical_data",
    cache_in_memory: bool = False,
    min_lc_per_gw: int = 2,
    return_zero_time_mjd: bool = False,
    nonkn_cls_base_field: str = "zero_time_mjd_cls_base",
    extra_negative_timeaware_enable: bool = False,
    extra_negative_timeaware_windows_days=None,
    extra_negative_timeaware_min_candidates: int = 1,
    extra_negative_timeaware_seed: int = 42,
    opt_input_window_start: Optional[float] = None,
    opt_input_window_end: Optional[float] = None,
):
    """
    Create dataloaders for Supervised Contrastive Learning.

    Each batch contains:
    - (batch_size // samples_per_gw) unique GW events
    - samples_per_gw optical samples per GW (positives for each other)
    """
    if cache_in_memory and num_workers > 0:
        print("cache_in_memory=True with num_workers>0 may increase RAM usage.")

    gw_map = build_gw_to_lc_mapping(h5_path)
    train_map, val_map = split_gw_map(gw_map, val_split, split_seed)

    if steps_per_epoch is None:
        # Calculate based on total optical samples, not GW events
        total_train_optical = sum(len(lcs) for lcs in train_map.values())
        steps_per_epoch = max(1, total_train_optical // batch_size)

    if val_batch_size is None:
        val_batch_size = batch_size
    if val_steps_per_epoch is None:
        # Calculate based on total optical samples, not GW events
        total_val_optical = sum(len(lcs) for lcs in val_map.values())
        val_steps_per_epoch = max(1, total_val_optical // val_batch_size)

    if cache_in_memory:
        shared_dataset = RelationalHDF5Dataset(
            h5_path,
            negative_h5_path=negative_h5_path,
            negative_group=negative_group,
            cache_in_memory=cache_in_memory,
            return_zero_time_mjd=return_zero_time_mjd,
            nonkn_cls_base_field=nonkn_cls_base_field,
            extra_negative_timeaware_enable=extra_negative_timeaware_enable,
            extra_negative_timeaware_windows_days=extra_negative_timeaware_windows_days,
            extra_negative_timeaware_min_candidates=extra_negative_timeaware_min_candidates,
            extra_negative_timeaware_seed=extra_negative_timeaware_seed,
            opt_input_window_start=opt_input_window_start,
            opt_input_window_end=opt_input_window_end,
        )
        train_dataset = shared_dataset
        val_dataset = shared_dataset
    else:
        train_dataset = RelationalHDF5Dataset(
            h5_path,
            negative_h5_path=negative_h5_path,
            negative_group=negative_group,
            cache_in_memory=cache_in_memory,
            return_zero_time_mjd=return_zero_time_mjd,
            nonkn_cls_base_field=nonkn_cls_base_field,
            extra_negative_timeaware_enable=extra_negative_timeaware_enable,
            extra_negative_timeaware_windows_days=extra_negative_timeaware_windows_days,
            extra_negative_timeaware_min_candidates=extra_negative_timeaware_min_candidates,
            extra_negative_timeaware_seed=extra_negative_timeaware_seed,
            opt_input_window_start=opt_input_window_start,
            opt_input_window_end=opt_input_window_end,
        )
        val_dataset = RelationalHDF5Dataset(
            h5_path,
            negative_h5_path=negative_h5_path,
            negative_group=negative_group,
            cache_in_memory=cache_in_memory,
            return_zero_time_mjd=return_zero_time_mjd,
            nonkn_cls_base_field=nonkn_cls_base_field,
            extra_negative_timeaware_enable=extra_negative_timeaware_enable,
            extra_negative_timeaware_windows_days=extra_negative_timeaware_windows_days,
            extra_negative_timeaware_min_candidates=extra_negative_timeaware_min_candidates,
            extra_negative_timeaware_seed=extra_negative_timeaware_seed,
            opt_input_window_start=opt_input_window_start,
            opt_input_window_end=opt_input_window_end,
        )

    train_sampler = MultiPositiveGWBatchedSampler(
        gw_to_lc_map=train_map,
        batch_size=batch_size,
        samples_per_gw=samples_per_gw,
        steps_per_epoch=steps_per_epoch,
        min_lc_per_gw=min_lc_per_gw
    )
    val_sampler = MultiPositiveGWBatchedSampler(
        gw_to_lc_map=val_map,
        batch_size=val_batch_size,
        samples_per_gw=samples_per_gw,
        steps_per_epoch=val_steps_per_epoch,
        min_lc_per_gw=min_lc_per_gw
    )

    train_loader = _build_configured_dataloader(
        train_dataset,
        train_sampler,
        num_workers,
        pin_memory,
        persistent_workers,
        prefetch_factor,
        cache_in_memory=cache_in_memory,
        usage="train",
        label="Train DataLoader",
    )
    val_loader = _build_configured_dataloader(
        val_dataset,
        val_sampler,
        num_workers,
        pin_memory,
        persistent_workers,
        prefetch_factor,
        cache_in_memory=cache_in_memory,
        usage="val",
        label="Validation DataLoader",
    )

    return train_loader, val_loader, steps_per_epoch, len(val_sampler)


def create_mixed_gw_dataloaders(
    h5_path: str,
    batch_size: int = 128,
    neg_gw_ratio: float = 0.2,
    samples_per_gw: int = 1,
    val_batch_size: int = None,
    steps_per_epoch: int = None,
    val_steps_per_epoch: int = None,
    val_split: float = 0.1,
    split_seed: int = 42,
    num_workers: int = 4,
    pin_memory: bool = True,
    persistent_workers: bool = True,
    prefetch_factor: int = 4,
    negative_h5_path: str = None,
    negative_group: str = "events/optical_data",
    cache_in_memory: bool = False,
    min_lc_per_gw: int = 1,
    return_zero_time_mjd: bool = False,
    nonkn_cls_base_field: str = "zero_time_mjd_cls_base",
    extra_negative_timeaware_enable: bool = False,
    extra_negative_timeaware_windows_days=None,
    extra_negative_timeaware_min_candidates: int = 1,
    extra_negative_timeaware_seed: int = 42,
    opt_input_window_start: Optional[float] = None,
    opt_input_window_end: Optional[float] = None,
):
    """
    Create dataloaders with mixed positive/negative GW sampling.

    Each batch contains:
    - (1 - neg_gw_ratio) * batch_size positive GW-optical pairs
    - neg_gw_ratio * batch_size negative GW paired with random optical

    Args:
        h5_path: Path to HDF5 file with has_kn field in gw_data
        batch_size: Total samples per batch
        neg_gw_ratio: Fraction of batch with negative GW (default 0.2 = 20%)
        samples_per_gw: Optical samples per GW for positive pairs (SupCon mode)
        val_batch_size: Validation batch size (default: same as batch_size)
        steps_per_epoch: Training steps per epoch (default: auto-calculated)
        val_steps_per_epoch: Validation steps per epoch (default: auto-calculated)
        val_split: Fraction of GW events for validation
        split_seed: Random seed for train/val split
        num_workers: DataLoader workers
        pin_memory: Pin memory for GPU transfer
        persistent_workers: Keep workers alive between epochs
        prefetch_factor: Prefetch batches per worker
        negative_h5_path: Path to negative optical samples (non-KN transients)
        negative_group: HDF5 group for negative optical data
        cache_in_memory: Cache datasets in memory
        min_lc_per_gw: Minimum light curves per GW event

    Returns:
        train_loader, val_loader, steps_per_epoch, val_steps
    """
    if cache_in_memory and num_workers > 0:
        print("cache_in_memory=True with num_workers>0 may increase RAM usage.")

    gw_map = build_gw_to_lc_mapping(h5_path)
    train_map, val_map = split_gw_map(gw_map, val_split, split_seed)

    # Create dataset with negative GW support
    if cache_in_memory:
        shared_dataset = RelationalHDF5Dataset(
            h5_path,
            negative_h5_path=negative_h5_path,
            negative_group=negative_group,
            cache_in_memory=cache_in_memory,
            use_neg_gw=True,
            return_zero_time_mjd=return_zero_time_mjd,
            nonkn_cls_base_field=nonkn_cls_base_field,
            extra_negative_timeaware_enable=extra_negative_timeaware_enable,
            extra_negative_timeaware_windows_days=extra_negative_timeaware_windows_days,
            extra_negative_timeaware_min_candidates=extra_negative_timeaware_min_candidates,
            extra_negative_timeaware_seed=extra_negative_timeaware_seed,
            opt_input_window_start=opt_input_window_start,
            opt_input_window_end=opt_input_window_end,
        )
        train_dataset = shared_dataset
        val_dataset = shared_dataset
    else:
        train_dataset = RelationalHDF5Dataset(
            h5_path,
            negative_h5_path=negative_h5_path,
            negative_group=negative_group,
            cache_in_memory=cache_in_memory,
            use_neg_gw=True,
            return_zero_time_mjd=return_zero_time_mjd,
            nonkn_cls_base_field=nonkn_cls_base_field,
            extra_negative_timeaware_enable=extra_negative_timeaware_enable,
            extra_negative_timeaware_windows_days=extra_negative_timeaware_windows_days,
            extra_negative_timeaware_min_candidates=extra_negative_timeaware_min_candidates,
            extra_negative_timeaware_seed=extra_negative_timeaware_seed,
            opt_input_window_start=opt_input_window_start,
            opt_input_window_end=opt_input_window_end,
        )
        val_dataset = RelationalHDF5Dataset(
            h5_path,
            negative_h5_path=negative_h5_path,
            negative_group=negative_group,
            cache_in_memory=cache_in_memory,
            use_neg_gw=True,
            return_zero_time_mjd=return_zero_time_mjd,
            nonkn_cls_base_field=nonkn_cls_base_field,
            extra_negative_timeaware_enable=extra_negative_timeaware_enable,
            extra_negative_timeaware_windows_days=extra_negative_timeaware_windows_days,
            extra_negative_timeaware_min_candidates=extra_negative_timeaware_min_candidates,
            extra_negative_timeaware_seed=extra_negative_timeaware_seed,
            opt_input_window_start=opt_input_window_start,
            opt_input_window_end=opt_input_window_end,
        )

    # Check that negative GW data is available
    if train_dataset.neg_gw_indices is None or len(train_dataset.neg_gw_indices) == 0:
        raise ValueError(
            f"No negative GW events found in {h5_path}. "
            f"Ensure the dataset has 'events/gw_data/has_kn' field."
        )

    # Calculate steps
    if steps_per_epoch is None:
        total_train_optical = sum(len(lcs) for lcs in train_map.values())
        steps_per_epoch = max(1, total_train_optical // batch_size)

    if val_batch_size is None:
        val_batch_size = batch_size
    if val_steps_per_epoch is None:
        total_val_optical = sum(len(lcs) for lcs in val_map.values())
        val_steps_per_epoch = max(1, total_val_optical // val_batch_size)

    # Split negative GW indices across train/val (local indices into dataset.neg_gw_indices)
    neg_local_indices = np.arange(len(train_dataset.neg_gw_indices))
    rng = np.random.default_rng(split_seed)
    rng.shuffle(neg_local_indices)
    val_neg_count = max(1, int(len(neg_local_indices) * val_split))
    if len(neg_local_indices) - val_neg_count < 1:
        val_neg_count = max(0, len(neg_local_indices) - 1)
    val_neg_local = neg_local_indices[:val_neg_count]
    train_neg_local = neg_local_indices[val_neg_count:]

    # Create mixed samplers
    train_sampler = MixedGWBatchedSampler(
        gw_to_lc_map=train_map,
        neg_gw_indices=train_neg_local,
        batch_size=batch_size,
        steps_per_epoch=steps_per_epoch,
        neg_gw_ratio=neg_gw_ratio,
        samples_per_gw=samples_per_gw
    )
    val_sampler = MixedGWBatchedSampler(
        gw_to_lc_map=val_map,
        neg_gw_indices=val_neg_local,
        batch_size=val_batch_size,
        steps_per_epoch=val_steps_per_epoch,
        neg_gw_ratio=neg_gw_ratio,
        samples_per_gw=samples_per_gw
    )

    train_loader = _build_configured_dataloader(
        train_dataset,
        train_sampler,
        num_workers,
        pin_memory,
        persistent_workers,
        prefetch_factor,
        cache_in_memory=cache_in_memory,
        usage="train",
        label="Train DataLoader",
    )
    val_loader = _build_configured_dataloader(
        val_dataset,
        val_sampler,
        num_workers,
        pin_memory,
        persistent_workers,
        prefetch_factor,
        cache_in_memory=cache_in_memory,
        usage="val",
        label="Validation DataLoader",
    )

    return train_loader, val_loader, steps_per_epoch, len(val_sampler)


class OpticalBinaryDataset(Dataset):
    """
    Optical-only binary dataset (KN vs non-KN).

    Positive samples come from:
        pos_h5_path/events/optical_data/*
    Negative samples come from:
        neg_h5_path/{neg_group}/*
    """
    DEFAULT_META_BINS_N_DET = np.asarray([3, 5, 8, 12, 20, 40, 80, 200], dtype=np.float64)
    DEFAULT_META_BINS_N_BANDS = np.asarray([1, 2, 3, 4, 5, 6], dtype=np.float64)
    DEFAULT_META_BINS_T_SPAN = np.asarray([0.0, 0.01, 0.05, 0.1, 0.2, 0.5, 1.0], dtype=np.float64)

    def __init__(
        self,
        pos_h5_path: str,
        neg_h5_path: str,
        neg_group: str = "ELASTICC2_TRAIN/optical_data",
        pos_indices: Optional[np.ndarray] = None,
        neg_indices: Optional[np.ndarray] = None,
        cache_in_memory: bool = False,
        load_meta_features: bool = True,
        return_prefix_aux: bool = False,
        runtime_input_window_start: Optional[float] = None,
        runtime_input_window_end: Optional[float] = None,
    ):
        super().__init__()
        self.pos_h5_path = pos_h5_path
        self.neg_h5_path = neg_h5_path
        self.neg_group = neg_group
        self.cache_in_memory = bool(cache_in_memory)

        self.pos_file = None
        self.neg_file = None
        self.pos_meta = {}
        self.neg_meta = {}
        self._meta_loaded = False
        self._load_meta_features = bool(load_meta_features)
        self.return_prefix_aux = bool(return_prefix_aux)
        self.runtime_input_window_start, self.runtime_input_window_end = _normalize_runtime_input_window(
            runtime_input_window_start,
            runtime_input_window_end,
        )
        self.pos_runtime_input_window_active = runtime_input_window_is_active_for_h5(
            self.pos_h5_path,
            self.runtime_input_window_start,
            self.runtime_input_window_end,
        )
        self.neg_runtime_input_window_active = runtime_input_window_is_active_for_h5(
            self.neg_h5_path,
            self.runtime_input_window_start,
            self.runtime_input_window_end,
        )

        with h5py.File(self.pos_h5_path, "r") as f:
            n_pos_total = int(f["events/optical_data/values"].shape[0])
            if self.return_prefix_aux and "slot_is_detection" not in f["events/optical_data"]:
                raise KeyError(
                    "Prefix auxiliary fields requested but 'events/optical_data/slot_is_detection' is missing "
                    f"in {self.pos_h5_path}"
                )

        with h5py.File(self.neg_h5_path, "r") as f:
            if self.neg_group not in f:
                raise KeyError(f"Negative group '{self.neg_group}' not found in {self.neg_h5_path}")
            n_neg_total = int(f[f"{self.neg_group}/values"].shape[0])
            if self.return_prefix_aux and "slot_is_detection" not in f[self.neg_group]:
                raise KeyError(
                    f"Prefix auxiliary fields requested but '{self.neg_group}/slot_is_detection' is missing "
                    f"in {self.neg_h5_path}"
                )

        self.pos_indices = (
            np.arange(n_pos_total, dtype=np.int64)
            if pos_indices is None
            else np.asarray(pos_indices, dtype=np.int64)
        )
        self.neg_indices = (
            np.arange(n_neg_total, dtype=np.int64)
            if neg_indices is None
            else np.asarray(neg_indices, dtype=np.int64)
        )

        self.n_pos = int(self.pos_indices.shape[0])
        self.n_neg = int(self.neg_indices.shape[0])
        if self.n_pos < 1 or self.n_neg < 1:
            raise ValueError(
                f"OpticalBinaryDataset requires both classes. got n_pos={self.n_pos}, n_neg={self.n_neg}"
            )

        if self._load_meta_features:
            self._ensure_meta_loaded()

        self.length = self.n_pos + self.n_neg

        if self.cache_in_memory:
            # Avoid caching multi-million samples by default; keep memory predictable.
            print("OpticalBinaryDataset: cache_in_memory is not supported, using lazy HDF5 loading.")
            self.cache_in_memory = False

    @staticmethod
    def _compute_meta_from_sample(mask_mat: np.ndarray, time_vec: np.ndarray) -> Tuple[int, int, float, int]:
        mask_bin = np.asarray(mask_mat, dtype=np.float32) > 0
        n_det = int(mask_bin.sum())
        band_hits = mask_bin.sum(axis=0) > 0
        n_bands = int(band_hits.sum())
        single_band_id = int(np.argmax(band_hits)) if n_bands == 1 else -1
        slot_valid = mask_bin.sum(axis=1) > 0
        if np.any(slot_valid):
            tv = np.asarray(time_vec, dtype=np.float32)[slot_valid]
            t_span = float(np.max(tv) - np.min(tv))
        else:
            t_span = 0.0
        return n_det, n_bands, t_span, single_band_id

    @classmethod
    def _load_or_compute_meta_arrays(
        cls,
        h5_file: h5py.File,
        group: str,
        chunk_size: int = 8192,
    ) -> dict:
        if group not in h5_file:
            raise KeyError(f"Group '{group}' not found in H5 file.")
        grp = h5_file[group]
        n_total = int(grp["values"].shape[0])
        if (
            "meta_n_obs" in grp
            and "meta_n_det_snr5" in grp
            and "meta_n_bands" in grp
            and "meta_t_span" in grp
            and "meta_single_band_id" in grp
        ):
            return {
                "n_obs": np.asarray(grp["meta_n_obs"][:], dtype=np.float32).reshape(-1),
                "n_det": np.asarray(grp["meta_n_det_snr5"][:], dtype=np.float32).reshape(-1),
                "n_bands": np.asarray(grp["meta_n_bands"][:], dtype=np.float32).reshape(-1),
                "t_span": np.asarray(grp["meta_t_span"][:], dtype=np.float32).reshape(-1),
                "single_band_id": np.asarray(grp["meta_single_band_id"][:], dtype=np.int16).reshape(-1),
            }
        if (
            "meta_n_det" in grp
            and "meta_n_bands" in grp
            and "meta_t_span" in grp
            and "meta_single_band_id" in grp
        ):
            return {
                "n_obs": np.asarray(grp["meta_n_det"][:], dtype=np.float32).reshape(-1),
                "n_det": np.asarray(grp["meta_n_det"][:], dtype=np.float32).reshape(-1),
                "n_bands": np.asarray(grp["meta_n_bands"][:], dtype=np.float32).reshape(-1),
                "t_span": np.asarray(grp["meta_t_span"][:], dtype=np.float32).reshape(-1),
                "single_band_id": np.asarray(grp["meta_single_band_id"][:], dtype=np.int16).reshape(-1),
            }

        ds_masks = grp["masks"]
        ds_times = grp["times"]
        ds_slot_is_detection = grp["slot_is_detection"] if "slot_is_detection" in grp else None
        n_obs = np.zeros((n_total,), dtype=np.float32)
        n_det = np.zeros((n_total,), dtype=np.float32)
        n_bands = np.zeros((n_total,), dtype=np.float32)
        t_span = np.zeros((n_total,), dtype=np.float32)
        single_band_id = np.full((n_total,), -1, dtype=np.int16)
        for s in range(0, n_total, int(chunk_size)):
            e = min(s + int(chunk_size), n_total)
            masks = np.asarray(ds_masks[s:e], dtype=np.float32)
            times = np.asarray(ds_times[s:e], dtype=np.float32)
            slot_det = (
                np.asarray(ds_slot_is_detection[s:e], dtype=np.float32)
                if ds_slot_is_detection is not None
                else None
            )
            for i in range(masks.shape[0]):
                nd_obs, nb, ts, sb = cls._compute_meta_from_sample(masks[i], times[i])
                idx = s + i
                n_obs[idx] = float(nd_obs)
                if slot_det is not None:
                    n_det[idx] = float(np.sum(np.asarray(slot_det[i], dtype=np.float32) > 0))
                else:
                    n_det[idx] = float(nd_obs)
                n_bands[idx] = float(nb)
                t_span[idx] = float(ts)
                single_band_id[idx] = int(sb)
        return {
            "n_obs": n_obs,
            "n_det": n_det,
            "n_bands": n_bands,
            "t_span": t_span,
            "single_band_id": single_band_id,
        }

    def _ensure_meta_loaded(self):
        if self._meta_loaded:
            return
        pos_meta_full = _load_or_compute_runtime_window_meta_arrays(
            self.pos_h5_path,
            "events/optical_data",
            window_start=self.runtime_input_window_start,
            window_end=self.runtime_input_window_end,
        )
        neg_meta_full = _load_or_compute_runtime_window_meta_arrays(
            self.neg_h5_path,
            self.neg_group,
            window_start=self.runtime_input_window_start,
            window_end=self.runtime_input_window_end,
        )
        self.pos_meta = {
            "n_obs": np.asarray(pos_meta_full["n_obs"][self.pos_indices], dtype=np.float32),
            "n_det": np.asarray(pos_meta_full["n_det"][self.pos_indices], dtype=np.float32),
            "n_bands": np.asarray(pos_meta_full["n_bands"][self.pos_indices], dtype=np.float32),
            "t_span": np.asarray(pos_meta_full["t_span"][self.pos_indices], dtype=np.float32),
            "single_band_id": np.asarray(pos_meta_full["single_band_id"][self.pos_indices], dtype=np.int16),
        }
        self.neg_meta = {
            "n_obs": np.asarray(neg_meta_full["n_obs"][self.neg_indices], dtype=np.float32),
            "n_det": np.asarray(neg_meta_full["n_det"][self.neg_indices], dtype=np.float32),
            "n_bands": np.asarray(neg_meta_full["n_bands"][self.neg_indices], dtype=np.float32),
            "t_span": np.asarray(neg_meta_full["t_span"][self.neg_indices], dtype=np.float32),
            "single_band_id": np.asarray(neg_meta_full["single_band_id"][self.neg_indices], dtype=np.int16),
        }
        self._meta_loaded = True

    def __len__(self):
        return self.length

    def _ensure_files_open(self):
        if self.pos_file is None:
            self.pos_file = h5py.File(self.pos_h5_path, "r")
        if self.neg_file is None:
            self.neg_file = h5py.File(self.neg_h5_path, "r")

    @staticmethod
    def _as_tensor(arr):
        return torch.from_numpy(arr).to(torch.float32)

    def _resolve_class_sample(self, label: int, class_local_idx: int):
        self._ensure_files_open()
        if int(label) == 1:
            if class_local_idx < 0 or class_local_idx >= self.n_pos:
                raise IndexError(f"Positive class index out of range: {class_local_idx}")
            return self.pos_file, "events/optical_data", int(self.pos_indices[class_local_idx]), 1.0
        if class_local_idx < 0 or class_local_idx >= self.n_neg:
            raise IndexError(f"Negative class index out of range: {class_local_idx}")
        return self.neg_file, self.neg_group, int(self.neg_indices[class_local_idx]), 0.0

    def get_sample_arrays_by_class_index(self, label: int, class_local_idx: int):
        f, grp, real_idx, target = self._resolve_class_sample(label, class_local_idx)
        opt_val = np.asarray(f[f"{grp}/values"][real_idx], dtype=np.float32)
        opt_err = np.asarray(f[f"{grp}/errors"][real_idx], dtype=np.float32)
        opt_mask = np.asarray(f[f"{grp}/masks"][real_idx], dtype=np.float32)
        opt_time = np.asarray(f[f"{grp}/times"][real_idx], dtype=np.float32)
        slot_is_detection = None
        if self.return_prefix_aux:
            slot_is_detection = np.asarray(f[f"{grp}/slot_is_detection"][real_idx], dtype=np.float32)
        runtime_active = self.pos_runtime_input_window_active if int(label) == 1 else self.neg_runtime_input_window_active
        if runtime_active:
            opt_time, opt_val, opt_mask, opt_err, slot_is_detection = apply_runtime_input_window_numpy(
                opt_time,
                opt_val,
                opt_mask,
                opt_err,
                slot_is_detection,
                window_start=self.runtime_input_window_start,
                window_end=self.runtime_input_window_end,
            )
        return opt_time, opt_val, opt_mask, opt_err, slot_is_detection, float(target)

    def __getitem__(self, idx):
        if isinstance(idx, np.ndarray):
            idx = int(idx.item())
        idx = int(idx)

        if idx < self.n_pos:
            opt_time, opt_val, opt_mask, opt_err, slot_is_detection, label = self.get_sample_arrays_by_class_index(
                1, idx
            )
        else:
            neg_local = idx - self.n_pos
            opt_time, opt_val, opt_mask, opt_err, slot_is_detection, label = self.get_sample_arrays_by_class_index(
                0, neg_local
            )

        target = torch.tensor(label, dtype=torch.float32)
        if self.return_prefix_aux:
            return (
                self._as_tensor(opt_time),
                self._as_tensor(opt_val),
                self._as_tensor(opt_mask),
                self._as_tensor(opt_err),
                target,
                self._as_tensor(slot_is_detection),
            )
        return (
            self._as_tensor(opt_time),
            self._as_tensor(opt_val),
            self._as_tensor(opt_mask),
            self._as_tensor(opt_err),
            target,
        )

    def __del__(self):
        if self.pos_file is not None:
            try:
                self.pos_file.close()
            except Exception:
                pass
        if self.neg_file is not None:
            try:
                self.neg_file.close()
            except Exception:
                pass

    def get_meta_for_sampling(self) -> dict:
        self._ensure_meta_loaded()
        return {
            "pos": self.pos_meta,
            "neg": self.neg_meta,
        }


class OpticalPrefixEvalDataset(Dataset):
    """
    Fixed-manifest prefix evaluation dataset built on top of OpticalBinaryDataset.
    """

    def __init__(
        self,
        base_dataset: OpticalBinaryDataset,
        manifest_rows: List[Dict[str, object]],
        prefix_min_det: int = 2,
    ):
        super().__init__()
        if not base_dataset.return_prefix_aux:
            raise ValueError("OpticalPrefixEvalDataset requires base_dataset.return_prefix_aux=True.")
        self.base_dataset = base_dataset
        self.prefix_min_det = int(prefix_min_det)
        self.manifest = pd.DataFrame(manifest_rows).reset_index(drop=True)
        if self.manifest.empty:
            raise ValueError("OpticalPrefixEvalDataset received an empty manifest.")

    def __len__(self):
        return int(len(self.manifest))

    def __getitem__(self, idx):
        row = self.manifest.iloc[int(idx)]
        label = int(row["label"])
        class_idx = int(row["base_idx"])
        target_k = int(row["prefix_det_target_k"])
        opt_time, opt_val, opt_mask, opt_err, slot_is_detection, target = self.base_dataset.get_sample_arrays_by_class_index(
            label, class_idx
        )
        opt_time, opt_val, opt_mask, opt_err, slot_is_detection, stats = apply_prefix_right_censoring_numpy(
            opt_t=opt_time,
            opt_v=opt_val,
            opt_mask=opt_mask,
            opt_err=opt_err,
            slot_is_detection=slot_is_detection,
            target_k=target_k,
            min_det=self.prefix_min_det,
        )
        return (
            torch.from_numpy(opt_time).to(torch.float32),
            torch.from_numpy(opt_val).to(torch.float32),
            torch.from_numpy(opt_mask).to(torch.float32),
            torch.from_numpy(opt_err).to(torch.float32),
            torch.tensor(target, dtype=torch.float32),
            torch.from_numpy(slot_is_detection).to(torch.float32),
            torch.tensor(int(stats["actual_target_k"]), dtype=torch.int64),
            torch.tensor(1 if bool(stats["is_terminal_prefix"]) else 0, dtype=torch.int64),
        )


def _group_has_meta_features(h5_path: str, group: str) -> bool:
    required_new = {"meta_n_obs", "meta_n_det_snr5", "meta_n_bands", "meta_t_span", "meta_single_band_id"}
    required_old = {"meta_n_det", "meta_n_bands", "meta_t_span", "meta_single_band_id"}
    with h5py.File(h5_path, "r") as f:
        if group not in f:
            return False
        keys = set(f[group].keys())
    return required_new.issubset(keys) or required_old.issubset(keys)


def _group_has_slot_is_detection(h5_path: str, group: str) -> bool:
    with h5py.File(h5_path, "r") as f:
        if group not in f:
            return False
        return "slot_is_detection" in f[group]


def _read_group_meta_array_for_indices(
    grp: h5py.Group,
    indices: np.ndarray,
    candidate_names: Sequence[str],
    dtype,
) -> Tuple[np.ndarray, str]:
    idx = np.asarray(indices, dtype=np.int64).reshape(-1)
    if idx.size == 0:
        return np.empty((0,), dtype=dtype), ""

    for name in candidate_names:
        if name not in grp:
            continue
        order = np.argsort(idx)
        sorted_idx = np.asarray(idx[order], dtype=np.int64)
        vals_sorted = np.asarray(grp[name][sorted_idx], dtype=dtype).reshape(-1)
        vals = np.empty_like(vals_sorted)
        vals[order] = vals_sorted
        return vals, str(name)

    raise KeyError(
        f"None of the requested meta arrays were found in group '{grp.name}': {list(candidate_names)}"
    )


def _filter_indices_by_meta_constraints(
    h5_path: str,
    group: str,
    indices: np.ndarray,
    n_det_min: Optional[int] = None,
    n_det_max: Optional[int] = None,
    n_bands_max: Optional[int] = None,
    t_span_max: Optional[float] = None,
    runtime_input_window_start: Optional[float] = None,
    runtime_input_window_end: Optional[float] = None,
) -> Tuple[np.ndarray, Dict[str, object]]:
    idx = np.asarray(indices, dtype=np.int64).reshape(-1)
    summary: Dict[str, object] = {
        "input_rows": int(idx.size),
        "output_rows": int(idx.size),
        "n_det_min": (None if n_det_min is None else int(n_det_min)),
        "n_det_max": (None if n_det_max is None else int(n_det_max)),
        "n_bands_max": (None if n_bands_max is None else int(n_bands_max)),
        "t_span_max": (None if t_span_max is None else float(t_span_max)),
        "runtime_input_window_start": (
            None if runtime_input_window_start is None else float(runtime_input_window_start)
        ),
        "runtime_input_window_end": (
            None if runtime_input_window_end is None else float(runtime_input_window_end)
        ),
        "applied": False,
    }
    if idx.size == 0:
        return idx, summary
    if (
        n_det_min is None
        and n_det_max is None
        and n_bands_max is None
        and t_span_max is None
    ):
        return idx, summary

    keep = np.ones((idx.shape[0],), dtype=bool)
    runtime_active = runtime_input_window_is_active_for_h5(
        h5_path,
        runtime_input_window_start,
        runtime_input_window_end,
    )
    if runtime_active:
        meta = _load_or_compute_runtime_window_meta_arrays(
            h5_path,
            group,
            window_start=runtime_input_window_start,
            window_end=runtime_input_window_end,
        )
        det_counts = np.asarray(meta["n_det"][idx], dtype=np.float32).reshape(-1)
        n_bands = np.asarray(meta["n_bands"][idx], dtype=np.float32).reshape(-1)
        t_span = np.asarray(meta["t_span"][idx], dtype=np.float32).reshape(-1)
        det_name = "runtime_window_n_det"
        n_bands_name = "runtime_window_n_bands"
        t_span_name = "runtime_window_t_span"
    else:
        with h5py.File(h5_path, "r") as f:
            if group not in f:
                raise KeyError(f"Group '{group}' not found in {h5_path}")
            grp = f[group]
            try:
                det_counts, det_name = _read_group_meta_array_for_indices(
                    grp,
                    idx,
                    ("meta_n_det_snr5", "meta_n_det"),
                    np.float32,
                )
                n_bands, n_bands_name = _read_group_meta_array_for_indices(
                    grp,
                    idx,
                    ("meta_n_bands",),
                    np.float32,
                )
                t_span, t_span_name = _read_group_meta_array_for_indices(
                    grp,
                    idx,
                    ("meta_t_span",),
                    np.float32,
                )
            except KeyError:
                meta = OpticalBinaryDataset._load_or_compute_meta_arrays(f, group)
                det_counts = np.asarray(meta["n_det"][idx], dtype=np.float32).reshape(-1)
                n_bands = np.asarray(meta["n_bands"][idx], dtype=np.float32).reshape(-1)
                t_span = np.asarray(meta["t_span"][idx], dtype=np.float32).reshape(-1)
                det_name = "computed_n_det"
                n_bands_name = "computed_n_bands"
                t_span_name = "computed_t_span"

    if n_det_min is not None:
        keep &= det_counts >= float(n_det_min)
    if n_det_max is not None:
        keep &= det_counts <= float(n_det_max)
    if n_bands_max is not None:
        keep &= n_bands <= float(n_bands_max)
    if t_span_max is not None:
        keep &= t_span <= float(t_span_max)

    out = np.asarray(idx[keep], dtype=np.int64)
    summary.update(
        {
            "output_rows": int(out.size),
            "applied": True,
            "det_source": det_name,
            "n_bands_source": n_bands_name,
            "t_span_source": t_span_name,
        }
    )
    return out, summary


def _filter_indices_by_min_detection_count(
    h5_path: str,
    group: str,
    indices: np.ndarray,
    prefix_min_det: int,
    runtime_input_window_start: Optional[float] = None,
    runtime_input_window_end: Optional[float] = None,
) -> np.ndarray:
    idx = np.asarray(indices, dtype=np.int64).reshape(-1)
    if idx.size == 0:
        return idx
    meta = _load_or_compute_runtime_window_meta_arrays(
        h5_path,
        group,
        window_start=runtime_input_window_start,
        window_end=runtime_input_window_end,
    )
    det_counts = np.asarray(meta["n_det"][idx], dtype=np.float32)
    keep = det_counts >= float(prefix_min_det)
    return idx[keep]


def _build_prefix_manifest_for_binary_dataset(
    base_dataset: OpticalBinaryDataset,
    det_support: Sequence[int],
    prefix_min_det: int,
    include_terminal: bool = True,
) -> List[Dict[str, object]]:
    manifest_rows: List[Dict[str, object]] = []
    for pos_idx in range(base_dataset.n_pos):
        opt_t, _, opt_mask, _, slot_is_detection, _ = base_dataset.get_sample_arrays_by_class_index(1, pos_idx)
        manifest_rows.extend(
            build_prefix_manifest_entries(
                base_idx=pos_idx,
                label=1,
                opt_t=opt_t,
                opt_mask=opt_mask,
                slot_is_detection=slot_is_detection,
                det_support=det_support,
                min_det=prefix_min_det,
                include_terminal=include_terminal,
            )
        )
    for neg_idx in range(base_dataset.n_neg):
        opt_t, _, opt_mask, _, slot_is_detection, _ = base_dataset.get_sample_arrays_by_class_index(0, neg_idx)
        manifest_rows.extend(
            build_prefix_manifest_entries(
                base_idx=neg_idx,
                label=0,
                opt_t=opt_t,
                opt_mask=opt_mask,
                slot_is_detection=slot_is_detection,
                det_support=det_support,
                min_det=prefix_min_det,
                include_terminal=include_terminal,
            )
        )
    return manifest_rows


class BalancedBinaryBatchSampler(Sampler):
    """
    Balanced batch sampler for binary classification.
    Each batch contains ~50% positives and ~50% negatives.
    """
    def __init__(
        self,
        n_pos: int,
        n_neg: int,
        batch_size: int,
        steps_per_epoch: int,
        seed: int = 42,
        shuffle: bool = True,
    ):
        self.n_pos = int(n_pos)
        self.n_neg = int(n_neg)
        self.batch_size = int(batch_size)
        self.steps_per_epoch = int(steps_per_epoch)
        self.shuffle = bool(shuffle)
        self.rng = np.random.default_rng(seed)

        if self.n_pos < 1 or self.n_neg < 1:
            raise ValueError(f"BalancedBinaryBatchSampler requires both classes. got n_pos={n_pos}, n_neg={n_neg}")
        if self.batch_size < 2:
            raise ValueError("batch_size must be >= 2 for balanced binary sampling.")

        self.pos_per_batch = self.batch_size // 2
        self.neg_per_batch = self.batch_size - self.pos_per_batch

    def __iter__(self):
        for _ in range(self.steps_per_epoch):
            pos_idx = self.rng.integers(0, self.n_pos, size=self.pos_per_batch, endpoint=False)
            neg_idx = self.rng.integers(0, self.n_neg, size=self.neg_per_batch, endpoint=False) + self.n_pos
            batch = np.concatenate([pos_idx, neg_idx], axis=0)
            if self.shuffle:
                self.rng.shuffle(batch)
            yield batch.tolist()

    def __len__(self):
        return self.steps_per_epoch


class MetaMatchedBinaryBatchSampler(Sampler):
    """
    Balanced binary sampler that aligns negative draws to positive meta bins.

    Each batch is still ~50/50 class-balanced. For each sampled positive index,
    it tries to sample a negative from the same (n_det, n_bands, t_span) bin.
    If no exact bin exists, it falls back to the nearest non-empty negative bin.
    """

    def __init__(
        self,
        pos_meta: dict,
        neg_meta: dict,
        batch_size: int,
        steps_per_epoch: int,
        seed: int = 42,
        shuffle: bool = True,
        bins_n_det: Sequence[float] = (3, 5, 8, 12, 20, 40, 80, 200),
        bins_n_bands: Sequence[float] = (1, 2, 3, 4, 5, 6),
        bins_t_span: Sequence[float] = (0.0, 0.01, 0.05, 0.1, 0.2, 0.5, 1.0),
        fallback: str = "nearest",
    ):
        self.shuffle = bool(shuffle)
        self.batch_size = int(batch_size)
        self.steps_per_epoch = int(steps_per_epoch)
        self.rng = np.random.default_rng(int(seed))
        self.fallback = str(fallback).strip().lower()

        if self.batch_size < 2:
            raise ValueError("batch_size must be >= 2 for binary sampling.")
        self.pos_per_batch = self.batch_size // 2
        self.neg_per_batch = self.batch_size - self.pos_per_batch

        self.pos_meta = pos_meta
        self.neg_meta = neg_meta
        self.n_pos = int(np.asarray(pos_meta["n_det"]).shape[0])
        self.n_neg = int(np.asarray(neg_meta["n_det"]).shape[0])
        if self.n_pos < 1 or self.n_neg < 1:
            raise ValueError(
                f"MetaMatchedBinaryBatchSampler requires both classes. got n_pos={self.n_pos}, n_neg={self.n_neg}"
            )

        self.edges_n_det = np.unique(np.asarray(list(bins_n_det), dtype=np.float64))
        self.edges_n_bands = np.unique(np.asarray(list(bins_n_bands), dtype=np.float64))
        self.edges_t_span = np.unique(np.asarray(list(bins_t_span), dtype=np.float64))
        if (
            self.edges_n_det.size == 0
            or self.edges_n_bands.size == 0
            or self.edges_t_span.size == 0
        ):
            raise ValueError("All meta bins must contain at least one edge.")

        self.n_det_bins = int(self.edges_n_det.size + 1)
        self.n_band_bins = int(self.edges_n_bands.size + 1)
        self.n_span_bins = int(self.edges_t_span.size + 1)
        self.n_joint_bins = int(self.n_det_bins * self.n_band_bins * self.n_span_bins)
        self._nearest_cache = {}

        self.pos_flat_bins = self._flat_bins_from_meta(self.pos_meta)
        self.neg_flat_bins = self._flat_bins_from_meta(self.neg_meta)
        self.neg_bins_to_indices = self._build_bin_index_map(self.neg_flat_bins)
        self.neg_nonempty_bins = np.asarray(
            sorted(self.neg_bins_to_indices.keys()), dtype=np.int64
        ).reshape(-1)
        if self.neg_nonempty_bins.size == 0:
            raise ValueError("No non-empty negative bins available for meta-matched sampling.")
        self.neg_nonempty_triplets = np.asarray(
            [self._unflatten_bin(int(b)) for b in self.neg_nonempty_bins], dtype=np.int64
        )

    def _flat_bins_from_meta(self, meta: dict) -> np.ndarray:
        n_det = np.asarray(meta["n_det"], dtype=np.float64).reshape(-1)
        n_bands = np.asarray(meta["n_bands"], dtype=np.float64).reshape(-1)
        t_span = np.asarray(meta["t_span"], dtype=np.float64).reshape(-1)
        d = np.searchsorted(self.edges_n_det, n_det, side="right").astype(np.int64)
        b = np.searchsorted(self.edges_n_bands, n_bands, side="right").astype(np.int64)
        t = np.searchsorted(self.edges_t_span, t_span, side="right").astype(np.int64)
        return ((d * self.n_band_bins) + b) * self.n_span_bins + t

    @staticmethod
    def _build_bin_index_map(flat_bins: np.ndarray) -> dict:
        mapping = {}
        for idx, b in enumerate(np.asarray(flat_bins, dtype=np.int64).tolist()):
            mapping.setdefault(int(b), []).append(int(idx))
        return {k: np.asarray(v, dtype=np.int64) for k, v in mapping.items()}

    def _unflatten_bin(self, flat_bin: int) -> Tuple[int, int, int]:
        x = int(flat_bin)
        d = x // (self.n_band_bins * self.n_span_bins)
        rem = x % (self.n_band_bins * self.n_span_bins)
        b = rem // self.n_span_bins
        t = rem % self.n_span_bins
        return int(d), int(b), int(t)

    def _nearest_nonempty_bin(self, flat_bin: int) -> int:
        key = int(flat_bin)
        cached = self._nearest_cache.get(key)
        if cached is not None:
            return int(cached)
        q = np.asarray(self._unflatten_bin(key), dtype=np.int64)
        d2 = np.sum((self.neg_nonempty_triplets - q[None, :]) ** 2, axis=1)
        best_idx = int(np.argmin(d2))
        best_bin = int(self.neg_nonempty_bins[best_idx])
        self._nearest_cache[key] = best_bin
        return best_bin

    def _sample_negative_local_index(self, target_flat_bin: int) -> int:
        if int(target_flat_bin) in self.neg_bins_to_indices:
            arr = self.neg_bins_to_indices[int(target_flat_bin)]
            return int(arr[self.rng.integers(0, arr.shape[0])])
        if self.fallback == "nearest":
            nearest_bin = self._nearest_nonempty_bin(int(target_flat_bin))
            arr = self.neg_bins_to_indices[nearest_bin]
            return int(arr[self.rng.integers(0, arr.shape[0])])
        return int(self.rng.integers(0, self.n_neg))

    def __iter__(self):
        for _ in range(self.steps_per_epoch):
            pos_local = self.rng.integers(0, self.n_pos, size=self.pos_per_batch, endpoint=False)
            neg_local = []
            for pidx in pos_local.tolist():
                pbin = int(self.pos_flat_bins[int(pidx)])
                neg_local.append(self._sample_negative_local_index(pbin))
            neg_local = np.asarray(neg_local, dtype=np.int64)

            if self.neg_per_batch > self.pos_per_batch:
                extra = self.rng.integers(0, self.n_neg, size=(self.neg_per_batch - self.pos_per_batch), endpoint=False)
                neg_local = np.concatenate([neg_local, extra], axis=0)
            elif self.neg_per_batch < self.pos_per_batch:
                neg_local = neg_local[: self.neg_per_batch]

            batch = np.concatenate([pos_local, (neg_local + self.n_pos)], axis=0)
            if self.shuffle:
                self.rng.shuffle(batch)
            yield batch.tolist()

    def __len__(self):
        return self.steps_per_epoch


def split_positive_optical_indices(
    pos_h5_path: str,
    val_split: float = 0.1,
    seed: int = 42
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Split positive optical indices by parent event index to avoid train/val leakage.
    Supports both the new optical-only field name 'parent_event_idx' and the
    legacy field name 'parent_gw_idx'.
    """
    if not (0 < val_split < 1):
        raise ValueError("val_split must be in (0, 1).")

    with h5py.File(pos_h5_path, "r") as f:
        if "events/optical_data/parent_event_idx" in f:
            parent_event_idx = f["events/optical_data/parent_event_idx"][:]
        elif "events/optical_data/parent_gw_idx" in f:
            parent_event_idx = f["events/optical_data/parent_gw_idx"][:]
        else:
            raise KeyError(
                "Positive optical H5 is missing both 'events/optical_data/parent_event_idx' "
                "and legacy 'events/optical_data/parent_gw_idx'."
            )

    unique_event = np.unique(parent_event_idx)
    rng = np.random.default_rng(seed)
    rng.shuffle(unique_event)

    n_val_event = max(1, int(len(unique_event) * val_split))
    val_event = unique_event[:n_val_event]

    is_val = np.isin(parent_event_idx, val_event)
    val_indices = np.where(is_val)[0].astype(np.int64)
    train_indices = np.where(~is_val)[0].astype(np.int64)
    return train_indices, val_indices


def split_negative_optical_indices(
    neg_h5_path: str,
    neg_group: str = "ELASTICC2_TRAIN/optical_data",
    val_split: float = 0.1,
    seed: int = 42
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Split negative optical indices randomly.
    """
    if not (0 < val_split < 1):
        raise ValueError("val_split must be in (0, 1).")

    with h5py.File(neg_h5_path, "r") as f:
        if neg_group not in f:
            raise KeyError(f"Negative group '{neg_group}' not found in {neg_h5_path}")
        n_total = int(f[f"{neg_group}/values"].shape[0])

    rng = np.random.default_rng(seed)
    perm = rng.permutation(n_total).astype(np.int64)

    n_val = max(1, int(n_total * val_split))
    if n_total - n_val < 1:
        n_val = max(0, n_total - 1)

    val_indices = perm[:n_val]
    train_indices = perm[n_val:]
    return train_indices, val_indices


def create_optical_binary_dataloaders(
    pos_h5_path: str,
    neg_h5_path: str,
    batch_size: int = 256,
    val_batch_size: int = None,
    steps_per_epoch: int = None,
    val_steps_per_epoch: int = None,
    val_split: float = 0.1,
    split_seed: int = 42,
    num_workers: int = 4,
    pin_memory: bool = True,
    persistent_workers: bool = True,
    prefetch_factor: int = 4,
    neg_group: str = "ELASTICC2_TRAIN/optical_data",
    cache_in_memory: bool = False,
    meta_matched_sampling: bool = True,
    meta_match_fallback: str = "nearest",
    meta_bins_n_det: Sequence[float] = (3, 5, 8, 12, 20, 40, 80, 200),
    meta_bins_n_bands: Sequence[float] = (1, 2, 3, 4, 5, 6),
    meta_bins_t_span: Sequence[float] = (0.0, 0.01, 0.05, 0.1, 0.2, 0.5, 1.0),
    prefix_train_enable: bool = False,
    prefix_min_det: int = 2,
    prefix_eval_det_support: Sequence[int] | str = DEFAULT_PREFIX_DET_SUPPORT,
    prefix_eval_include_terminal: bool = True,
    meta_filter_n_det_min: Optional[int] = None,
    meta_filter_n_det_max: Optional[int] = None,
    meta_filter_n_bands_max: Optional[int] = None,
    meta_filter_t_span_max: Optional[float] = None,
    meta_filter_relax_t_span_if_below_rows: Optional[int] = None,
    runtime_input_window_start: Optional[float] = None,
    runtime_input_window_end: Optional[float] = None,
):
    if cache_in_memory and num_workers > 0:
        print("cache_in_memory=True with num_workers>0 may increase RAM usage.")

    train_pos_idx, val_pos_idx = split_positive_optical_indices(
        pos_h5_path, val_split=val_split, seed=split_seed
    )
    train_neg_idx, val_neg_idx = split_negative_optical_indices(
        neg_h5_path, neg_group=neg_group, val_split=val_split, seed=split_seed
    )
    train_pos_idx_raw = np.asarray(train_pos_idx, dtype=np.int64)
    val_pos_idx_raw = np.asarray(val_pos_idx, dtype=np.int64)
    train_neg_idx_raw = np.asarray(train_neg_idx, dtype=np.int64)
    val_neg_idx_raw = np.asarray(val_neg_idx, dtype=np.int64)

    prefix_train_enable = bool(prefix_train_enable)
    prefix_min_det = int(prefix_min_det)
    prefix_det_support = parse_prefix_det_support(prefix_eval_det_support)
    runtime_input_window_start, runtime_input_window_end = _normalize_runtime_input_window(
        runtime_input_window_start,
        runtime_input_window_end,
    )

    meta_filter_n_det_min = None if meta_filter_n_det_min is None else int(meta_filter_n_det_min)
    meta_filter_n_det_max = None if meta_filter_n_det_max is None else int(meta_filter_n_det_max)
    meta_filter_n_bands_max = None if meta_filter_n_bands_max is None else int(meta_filter_n_bands_max)
    meta_filter_t_span_max = None if meta_filter_t_span_max is None else float(meta_filter_t_span_max)
    meta_filter_relax_t_span_if_below_rows = (
        None
        if meta_filter_relax_t_span_if_below_rows is None
        else int(meta_filter_relax_t_span_if_below_rows)
    )

    meta_filter_enabled = any(
        v is not None
        for v in (
            meta_filter_n_det_min,
            meta_filter_n_det_max,
            meta_filter_n_bands_max,
            meta_filter_t_span_max,
        )
    )
    if meta_filter_enabled:
        def _run_meta_filter_split(
            pos_idx: np.ndarray,
            neg_idx: np.ndarray,
            t_span_limit: Optional[float],
        ) -> Tuple[np.ndarray, np.ndarray, Dict[str, object]]:
            pos_out, pos_summary = _filter_indices_by_meta_constraints(
                pos_h5_path,
                "events/optical_data",
                pos_idx,
                n_det_min=meta_filter_n_det_min,
                n_det_max=meta_filter_n_det_max,
                n_bands_max=meta_filter_n_bands_max,
                t_span_max=t_span_limit,
                runtime_input_window_start=runtime_input_window_start,
                runtime_input_window_end=runtime_input_window_end,
            )
            neg_out, neg_summary = _filter_indices_by_meta_constraints(
                neg_h5_path,
                neg_group,
                neg_idx,
                n_det_min=meta_filter_n_det_min,
                n_det_max=meta_filter_n_det_max,
                n_bands_max=meta_filter_n_bands_max,
                t_span_max=t_span_limit,
                runtime_input_window_start=runtime_input_window_start,
                runtime_input_window_end=runtime_input_window_end,
            )
            return pos_out, neg_out, {
                "pos": pos_summary,
                "neg": neg_summary,
                "t_span_max": (None if t_span_limit is None else float(t_span_limit)),
            }

        train_pos_idx, train_neg_idx, train_meta_filter_summary = _run_meta_filter_split(
            train_pos_idx_raw,
            train_neg_idx_raw,
            meta_filter_t_span_max,
        )
        val_pos_idx, val_neg_idx, val_meta_filter_summary = _run_meta_filter_split(
            val_pos_idx_raw,
            val_neg_idx_raw,
            meta_filter_t_span_max,
        )
        relaxed_t_span_max = meta_filter_t_span_max
        relaxed = False
        if (
            meta_filter_t_span_max is not None
            and meta_filter_relax_t_span_if_below_rows is not None
            and train_pos_idx.size < int(meta_filter_relax_t_span_if_below_rows)
            and float(meta_filter_t_span_max) < 0.05
        ):
            relaxed = True
            relaxed_t_span_max = 0.05
            train_pos_idx, train_neg_idx, train_meta_filter_summary = _run_meta_filter_split(
                train_pos_idx_raw,
                train_neg_idx_raw,
                relaxed_t_span_max,
            )
            val_pos_idx, val_neg_idx, val_meta_filter_summary = _run_meta_filter_split(
                val_pos_idx_raw,
                val_neg_idx_raw,
                relaxed_t_span_max,
            )
        print(
            "Meta filter split summary: "
            f"train_pos={train_meta_filter_summary['pos']['output_rows']} "
            f"train_neg={train_meta_filter_summary['neg']['output_rows']} "
            f"val_pos={val_meta_filter_summary['pos']['output_rows']} "
            f"val_neg={val_meta_filter_summary['neg']['output_rows']} "
            f"| n_det=[{meta_filter_n_det_min},{meta_filter_n_det_max}] "
            f"| n_bands<={meta_filter_n_bands_max} "
            f"| t_span<={relaxed_t_span_max} "
            f"| relaxed={relaxed}"
        )
        if train_pos_idx.size == 0 or train_neg_idx.size == 0:
            raise ValueError("Meta filtering left an empty training split.")
        if val_pos_idx.size == 0 or val_neg_idx.size == 0:
            raise ValueError("Meta filtering left an empty validation split.")

    if prefix_train_enable:
        pos_group = "events/optical_data"
        if not _group_has_slot_is_detection(pos_h5_path, pos_group):
            raise ValueError(
                "prefix_train_enable=true requires slot_is_detection in the positive H5 dataset."
            )
        if not _group_has_slot_is_detection(neg_h5_path, neg_group):
            raise ValueError(
                "prefix_train_enable=true requires slot_is_detection in the negative H5 dataset."
            )
        train_pos_idx = _filter_indices_by_min_detection_count(
            pos_h5_path, pos_group, train_pos_idx, prefix_min_det,
            runtime_input_window_start=runtime_input_window_start,
            runtime_input_window_end=runtime_input_window_end,
        )
        val_pos_idx = _filter_indices_by_min_detection_count(
            pos_h5_path, pos_group, val_pos_idx, prefix_min_det,
            runtime_input_window_start=runtime_input_window_start,
            runtime_input_window_end=runtime_input_window_end,
        )
        train_neg_idx = _filter_indices_by_min_detection_count(
            neg_h5_path, neg_group, train_neg_idx, prefix_min_det,
            runtime_input_window_start=runtime_input_window_start,
            runtime_input_window_end=runtime_input_window_end,
        )
        val_neg_idx = _filter_indices_by_min_detection_count(
            neg_h5_path, neg_group, val_neg_idx, prefix_min_det,
            runtime_input_window_start=runtime_input_window_start,
            runtime_input_window_end=runtime_input_window_end,
        )
        if train_pos_idx.size == 0 or train_neg_idx.size == 0:
            raise ValueError(
                "prefix_train_enable=true left an empty training split after prefix_min_det filtering."
            )
        if val_pos_idx.size == 0 or val_neg_idx.size == 0:
            raise ValueError(
                "prefix_train_enable=true left an empty validation split after prefix_min_det filtering."
            )

    if val_batch_size is None:
        val_batch_size = batch_size

    if steps_per_epoch is None:
        steps_per_epoch = max(1, (2 * min(len(train_pos_idx), len(train_neg_idx))) // batch_size)

    if val_steps_per_epoch is None:
        val_steps_per_epoch = max(1, (2 * min(len(val_pos_idx), len(val_neg_idx))) // val_batch_size)

    meta_available = (
        _group_has_meta_features(pos_h5_path, "events/optical_data")
        and _group_has_meta_features(neg_h5_path, neg_group)
    )
    if bool(meta_matched_sampling) and (not meta_available):
        print(
            "Meta matched sampling requested but meta_* fields are missing in one or both H5 datasets. "
            "Falling back to BalancedBinaryBatchSampler."
        )
        meta_matched_sampling = False

    train_dataset = OpticalBinaryDataset(
        pos_h5_path=pos_h5_path,
        neg_h5_path=neg_h5_path,
        neg_group=neg_group,
        pos_indices=train_pos_idx,
        neg_indices=train_neg_idx,
        cache_in_memory=cache_in_memory,
        load_meta_features=bool(meta_matched_sampling),
        return_prefix_aux=prefix_train_enable,
        runtime_input_window_start=runtime_input_window_start,
        runtime_input_window_end=runtime_input_window_end,
    )

    if bool(meta_matched_sampling):
        train_meta = train_dataset.get_meta_for_sampling()
        train_sampler = MetaMatchedBinaryBatchSampler(
            pos_meta=train_meta["pos"],
            neg_meta=train_meta["neg"],
            batch_size=batch_size,
            steps_per_epoch=steps_per_epoch,
            seed=split_seed,
            shuffle=True,
            bins_n_det=meta_bins_n_det,
            bins_n_bands=meta_bins_n_bands,
            bins_t_span=meta_bins_t_span,
            fallback=meta_match_fallback,
        )
    else:
        train_sampler = BalancedBinaryBatchSampler(
            n_pos=train_dataset.n_pos,
            n_neg=train_dataset.n_neg,
            batch_size=batch_size,
            steps_per_epoch=steps_per_epoch,
            seed=split_seed,
            shuffle=True,
        )

    train_loader = _build_dataloader(
        train_dataset,
        train_sampler,
        num_workers,
        pin_memory,
        persistent_workers,
        prefetch_factor,
    )

    if prefix_train_enable:
        rng = np.random.default_rng(int(split_seed) + 17)
        target_val_base_per_class = max(1, (int(val_steps_per_epoch) * int(val_batch_size)) // 2)

        def _sample_eval_subset(idx: np.ndarray) -> np.ndarray:
            idx = np.asarray(idx, dtype=np.int64).reshape(-1)
            if idx.size <= target_val_base_per_class:
                return np.sort(idx.copy())
            picked = rng.choice(idx, size=int(target_val_base_per_class), replace=False)
            return np.sort(np.asarray(picked, dtype=np.int64))

        eval_pos_idx = _sample_eval_subset(val_pos_idx)
        eval_neg_idx = _sample_eval_subset(val_neg_idx)
        val_base_dataset = OpticalBinaryDataset(
            pos_h5_path=pos_h5_path,
            neg_h5_path=neg_h5_path,
            neg_group=neg_group,
            pos_indices=eval_pos_idx,
            neg_indices=eval_neg_idx,
            cache_in_memory=cache_in_memory,
            load_meta_features=False,
            return_prefix_aux=True,
            runtime_input_window_start=runtime_input_window_start,
            runtime_input_window_end=runtime_input_window_end,
        )
        manifest_rows = _build_prefix_manifest_for_binary_dataset(
            base_dataset=val_base_dataset,
            det_support=prefix_det_support,
            prefix_min_det=prefix_min_det,
            include_terminal=bool(prefix_eval_include_terminal),
        )
        if not manifest_rows:
            raise ValueError("Prefix validation manifest is empty after filtering by prefix_min_det.")
        val_dataset = OpticalPrefixEvalDataset(
            base_dataset=val_base_dataset,
            manifest_rows=manifest_rows,
            prefix_min_det=prefix_min_det,
        )
        print(
            "Prefix validation manifest: "
            f"n_base_pos={val_base_dataset.n_pos}, n_base_neg={val_base_dataset.n_neg}, "
            f"n_prefix_rows={len(val_dataset)}, det_support={prefix_det_support}, "
            f"include_terminal={bool(prefix_eval_include_terminal)}"
        )
        if num_workers > 0:
            val_loader = DataLoader(
                val_dataset,
                batch_size=val_batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=pin_memory,
                persistent_workers=persistent_workers,
                prefetch_factor=prefetch_factor,
            )
        else:
            val_loader = DataLoader(
                val_dataset,
                batch_size=val_batch_size,
                shuffle=False,
                num_workers=0,
                pin_memory=pin_memory,
            )
        return train_loader, val_loader, steps_per_epoch, len(val_loader)

    val_dataset = OpticalBinaryDataset(
        pos_h5_path=pos_h5_path,
        neg_h5_path=neg_h5_path,
        neg_group=neg_group,
        pos_indices=val_pos_idx,
        neg_indices=val_neg_idx,
        cache_in_memory=cache_in_memory,
        load_meta_features=bool(meta_matched_sampling),
        runtime_input_window_start=runtime_input_window_start,
        runtime_input_window_end=runtime_input_window_end,
    )

    if bool(meta_matched_sampling):
        val_meta = val_dataset.get_meta_for_sampling()
        val_sampler = MetaMatchedBinaryBatchSampler(
            pos_meta=val_meta["pos"],
            neg_meta=val_meta["neg"],
            batch_size=val_batch_size,
            steps_per_epoch=val_steps_per_epoch,
            seed=split_seed + 1,
            shuffle=True,
            bins_n_det=meta_bins_n_det,
            bins_n_bands=meta_bins_n_bands,
            bins_t_span=meta_bins_t_span,
            fallback=meta_match_fallback,
        )
    else:
        val_sampler = BalancedBinaryBatchSampler(
            n_pos=val_dataset.n_pos,
            n_neg=val_dataset.n_neg,
            batch_size=val_batch_size,
            steps_per_epoch=val_steps_per_epoch,
            seed=split_seed + 1,
            shuffle=True,
        )

    val_loader = _build_dataloader(
        val_dataset,
        val_sampler,
        num_workers,
        pin_memory,
        persistent_workers,
        prefetch_factor,
    )
    return train_loader, val_loader, steps_per_epoch, len(val_sampler)


# Constants
BAND_MAP = {'LSST-u': 0, 'LSST-g': 1, 'LSST-r': 2, 'LSST-i': 3, 'LSST-z': 4, 'LSST-Y': 5}
NEG_BAND_MAP = {'u': 0, 'g': 1, 'r': 2, 'i': 3, 'z': 4, 'Y': 5}
NUM_BANDS = 6
MAX_LC_LENGTH = 200  # Maximum length of light curves all band
LUPT_BAND_ORDER = ("u", "g", "r", "i", "z", "Y")
ASINH_MAG_FACTOR = 2.5 / np.log(10.0)

_ALL_BAND_MAP = {
    "LSST-u": 0,
    "LSST-g": 1,
    "LSST-r": 2,
    "LSST-i": 3,
    "LSST-z": 4,
    "LSST-Y": 5,
    "u": 0,
    "g": 1,
    "r": 2,
    "i": 3,
    "z": 4,
    "Y": 5,
    "y": 5,
}


def _normalize_band_token(band_raw) -> str:
    if isinstance(band_raw, bytes):
        band = band_raw.decode("utf-8", errors="ignore")
    else:
        band = str(band_raw)
    return band.strip()


def _band_index_from_raw(band_raw) -> Optional[int]:
    band = _normalize_band_token(band_raw)
    if band in _ALL_BAND_MAP:
        return int(_ALL_BAND_MAP[band])
    if band.startswith("LSST-"):
        tail = band.split("-", 1)[1]
        return int(_ALL_BAND_MAP[tail]) if tail in _ALL_BAND_MAP else None
    return None


def parse_lupt_m5_mag(text: str) -> np.ndarray:
    raw = str(text).strip()
    if raw == "":
        raise ValueError(
            "lupt_m5_mag is required and must contain 6 comma-separated finite values in order u,g,r,i,z,Y."
        )
    parts = [p.strip() for p in raw.split(",")]
    if len(parts) != NUM_BANDS:
        raise ValueError(
            f"lupt_m5_mag must provide exactly {NUM_BANDS} values in order u,g,r,i,z,Y; got {len(parts)}."
        )
    try:
        vals = np.asarray([float(p) for p in parts], dtype=np.float64)
    except ValueError as exc:
        raise ValueError("lupt_m5_mag contains non-numeric values.") from exc
    if not np.all(np.isfinite(vals)):
        raise ValueError("lupt_m5_mag values must be finite.")
    return vals


def build_luptitude_params(
    fluxcal_zp: float,
    psfflux_zp: float,
    lupt_k: float,
    lupt_m5_mag: np.ndarray,
) -> Tuple[float, np.ndarray, np.ndarray]:
    if not np.isfinite(fluxcal_zp) or not np.isfinite(psfflux_zp):
        raise ValueError("fluxcal_zp and psfflux_zp must be finite.")
    if not np.isfinite(lupt_k) or lupt_k <= 0:
        raise ValueError("lupt_k must be finite and > 0.")
    if lupt_m5_mag.shape != (NUM_BANDS,):
        raise ValueError(
            f"lupt_m5_mag must contain exactly {NUM_BANDS} values in order u,g,r,i,z,Y."
        )
    if not np.all(np.isfinite(lupt_m5_mag)):
        raise ValueError("lupt_m5_mag values must be finite.")

    fluxcal_to_psfflux_factor = 10.0 ** (0.4 * (float(psfflux_zp) - float(fluxcal_zp)))
    if not np.isfinite(fluxcal_to_psfflux_factor) or fluxcal_to_psfflux_factor <= 0:
        raise ValueError(
            f"Invalid FLUXCAL->psfFlux conversion factor computed from fluxcal_zp={fluxcal_zp}, psfflux_zp={psfflux_zp}."
        )

    lupt_f5sigma_njy = 10.0 ** ((float(psfflux_zp) - lupt_m5_mag.astype(np.float64, copy=False)) / 2.5)
    if np.any(lupt_f5sigma_njy <= 0) or not np.all(np.isfinite(lupt_f5sigma_njy)):
        raise ValueError("Derived lupt_f5sigma_njy values must be finite and > 0.")
    lupt_b_njy = float(lupt_k) * (lupt_f5sigma_njy / 5.0)
    if np.any(lupt_b_njy <= 0) or not np.all(np.isfinite(lupt_b_njy)):
        raise ValueError("Derived lupt_b_njy values must be finite and > 0.")
    return float(fluxcal_to_psfflux_factor), lupt_f5sigma_njy, lupt_b_njy


def _transform_fluxcal_to_luptitude(
    mjd: np.ndarray,
    fluxcal: np.ndarray,
    fluxcalerr: np.ndarray,
    flt: np.ndarray,
    fluxcal_to_psfflux_factor: float,
    psfflux_zp: float,
    lupt_b_njy: Sequence[float],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    lupt_b_arr = np.asarray(lupt_b_njy, dtype=np.float64)
    if lupt_b_arr.shape != (NUM_BANDS,):
        raise ValueError(f"lupt_b_njy must contain {NUM_BANDS} values in order u,g,r,i,z,Y.")

    band_idx_list: List[int] = []
    for band_raw in flt:
        idx = _band_index_from_raw(band_raw)
        band_idx_list.append(-1 if idx is None else int(idx))
    band_idx = np.asarray(band_idx_list, dtype=np.int64)

    base_valid = (
        np.isfinite(mjd)
        & np.isfinite(fluxcal)
        & np.isfinite(fluxcalerr)
        & (fluxcalerr > 0)
        & (band_idx >= 0)
    )
    if not np.any(base_valid):
        return (
            np.asarray([], dtype=np.float64),
            np.asarray([], dtype=np.float64),
            np.asarray([], dtype=np.float64),
            np.asarray([], dtype=flt.dtype),
        )

    idx_valid = np.nonzero(base_valid)[0]
    band_valid = band_idx[idx_valid]
    b_valid = lupt_b_arr[band_valid]

    f_psf = fluxcal[idx_valid] * float(fluxcal_to_psfflux_factor)
    sigma_psf = np.abs(fluxcalerr[idx_valid]) * float(fluxcal_to_psfflux_factor)
    m_lupt = float(psfflux_zp) - ASINH_MAG_FACTOR * (
        np.arcsinh(f_psf / (2.0 * b_valid)) + np.log(b_valid)
    )
    sigma_lupt = ASINH_MAG_FACTOR * sigma_psf / np.sqrt((f_psf * f_psf) + (2.0 * b_valid) ** 2)

    finite_valid = np.isfinite(m_lupt) & np.isfinite(sigma_lupt) & (sigma_lupt > 0)
    if not np.any(finite_valid):
        return (
            np.asarray([], dtype=np.float64),
            np.asarray([], dtype=np.float64),
            np.asarray([], dtype=np.float64),
            np.asarray([], dtype=flt.dtype),
        )

    keep = idx_valid[finite_valid]
    return (
        np.asarray(mjd[keep], dtype=np.float64),
        np.asarray(m_lupt[finite_valid], dtype=np.float64),
        np.asarray(sigma_lupt[finite_valid], dtype=np.float64),
        np.asarray(flt[keep]),
    )


def _transform_psfflux_to_luptitude(
    mjd: np.ndarray,
    psfflux: np.ndarray,
    psffluxerr: np.ndarray,
    flt: np.ndarray,
    psfflux_zp: float,
    lupt_b_njy: Sequence[float],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    lupt_b_arr = np.asarray(lupt_b_njy, dtype=np.float64)
    if lupt_b_arr.shape != (NUM_BANDS,):
        raise ValueError(f"lupt_b_njy must contain {NUM_BANDS} values in order u,g,r,i,z,Y.")

    band_idx_list: List[int] = []
    for band_raw in flt:
        idx = _band_index_from_raw(band_raw)
        band_idx_list.append(-1 if idx is None else int(idx))
    band_idx = np.asarray(band_idx_list, dtype=np.int64)

    base_valid = (
        np.isfinite(mjd)
        & np.isfinite(psfflux)
        & np.isfinite(psffluxerr)
        & (psffluxerr > 0)
        & (band_idx >= 0)
    )
    if not np.any(base_valid):
        return (
            np.asarray([], dtype=np.float64),
            np.asarray([], dtype=np.float64),
            np.asarray([], dtype=np.float64),
            np.asarray([], dtype=flt.dtype),
        )

    idx_valid = np.nonzero(base_valid)[0]
    band_valid = band_idx[idx_valid]
    b_valid = lupt_b_arr[band_valid]

    f_psf = np.asarray(psfflux[idx_valid], dtype=np.float64)
    sigma_psf = np.asarray(np.abs(psffluxerr[idx_valid]), dtype=np.float64)
    m_lupt = float(psfflux_zp) - ASINH_MAG_FACTOR * (
        np.arcsinh(f_psf / (2.0 * b_valid)) + np.log(b_valid)
    )
    sigma_lupt = ASINH_MAG_FACTOR * sigma_psf / np.sqrt((f_psf * f_psf) + (2.0 * b_valid) ** 2)

    finite_valid = np.isfinite(m_lupt) & np.isfinite(sigma_lupt) & (sigma_lupt > 0)
    if not np.any(finite_valid):
        return (
            np.asarray([], dtype=np.float64),
            np.asarray([], dtype=np.float64),
            np.asarray([], dtype=np.float64),
            np.asarray([], dtype=flt.dtype),
        )

    keep = idx_valid[finite_valid]
    return (
        np.asarray(mjd[keep], dtype=np.float64),
        np.asarray(m_lupt[finite_valid], dtype=np.float64),
        np.asarray(sigma_lupt[finite_valid], dtype=np.float64),
        np.asarray(flt[keep]),
    )


def _first_detection_index(
    flux: np.ndarray,
    fluxerr: np.ndarray,
    photflag: Optional[np.ndarray],
    snr_threshold: float = 5.0,
) -> Optional[int]:
    flux = np.asarray(flux, dtype=np.float64)
    fluxerr = np.asarray(fluxerr, dtype=np.float64)
    if flux.size == 0:
        return None

    valid = np.isfinite(flux) & np.isfinite(fluxerr) & (fluxerr > 0)
    if np.any(valid):
        snr = np.full(flux.shape, -np.inf, dtype=np.float64)
        snr[valid] = flux[valid] / fluxerr[valid]
        det_mask = snr > float(snr_threshold)
        if np.any(det_mask):
            return int(np.argmax(det_mask))

    if photflag is not None:
        flags = np.asarray(photflag, dtype=np.int64)
        if flags.shape == flux.shape:
            det_mask = (flags & (4096 | 1024)) != 0
            if np.any(det_mask):
                return int(np.argmax(det_mask))

    return None


def _safe_float(value: object) -> Optional[float]:
    try:
        out = float(value)
    except Exception:
        return None
    return float(out) if np.isfinite(out) else None


def _valid_head_detection_mjd(value: object) -> Optional[float]:
    out = _safe_float(value)
    if out is None:
        return None
    # SNANA commonly uses -9 and 1e6 sentinels for missing detection fields.
    if out <= 0.0 or out >= 900000.0:
        return None
    return float(out)


def _first_photflag_detection_mjd(mjd: np.ndarray, photflag: Optional[np.ndarray]) -> Optional[float]:
    if photflag is None:
        return None
    mjd_arr = np.asarray(mjd, dtype=np.float64)
    flags = np.asarray(photflag, dtype=np.int64)
    if mjd_arr.shape != flags.shape or mjd_arr.size == 0:
        return None
    det_mask = ((flags & (4096 | 1024)) != 0) & np.isfinite(mjd_arr)
    if not np.any(det_mask):
        return None
    return float(mjd_arr[int(np.argmax(det_mask))])


def resolve_first_detection_mjd(
    *,
    snr_mjd,
    snr_flux,
    snr_fluxerr,
    photflag_mjd=None,
    photflag=None,
    head_mjd_detect_first=None,
    snr_threshold: float = 5.0,
) -> Optional[float]:
    """Resolve first detection time using psfFlux SNR, PHOTFLAG, then HEAD fallback."""
    mjd_arr = np.asarray(snr_mjd, dtype=np.float64)
    flux_arr = np.asarray(snr_flux, dtype=np.float64)
    fluxerr_arr = np.asarray(snr_fluxerr, dtype=np.float64)
    if mjd_arr.shape == flux_arr.shape == fluxerr_arr.shape and mjd_arr.size > 0:
        det_idx = _first_detection_index(
            flux=flux_arr,
            fluxerr=fluxerr_arr,
            photflag=None,
            snr_threshold=float(snr_threshold),
        )
        if det_idx is not None and np.isfinite(mjd_arr[int(det_idx)]):
            return float(mjd_arr[int(det_idx)])

    if photflag_mjd is not None:
        out = _first_photflag_detection_mjd(np.asarray(photflag_mjd, dtype=np.float64), photflag)
        if out is not None:
            return out

    return _valid_head_detection_mjd(head_mjd_detect_first)

# Functions for parsing SNANA FITS files, and sampling MOC skymaps
def parse_snana_fits(
    event_id,
    sim_dir,
    sim_name="LSST_KN_BNS_AUG",
    fluxcal_to_psfflux_factor=None,
    psfflux_zp=31.4,
    lupt_b_njy=None,
    normalize_to_first_detection=False,
    snr_threshold=5.0,
):
    """
    Parses {event_id}_HEAD.fits and {event_id}_PHOT.fits.
    Extracts multiple light curve realizations for a single GW event.
    
    Args:
        event_id: String ID of the event.
        sim_dir: Directory containing FITS files.
        
    Returns:
        List of tuples: [(values, masks, times), ...]
        Returns empty list if files are missing.
    """
    if type(event_id) == str:
        event_id = int(float(event_id))
    head_path = os.path.join(sim_dir, f"{sim_name}_{event_id}",f"{sim_name}_{event_id}_HEAD.FITS")
    phot_path = os.path.join(sim_dir, f"{sim_name}_{event_id}",f"{sim_name}_{event_id}_PHOT.FITS")

    if not os.path.exists(head_path) or not os.path.exists(phot_path):
        print(f"Warning: FITS files not found for {event_id}")
        return []

    try:
        # open readme file and get MJD explode value
        with open(os.path.join(sim_dir, f"{sim_name}_{event_id}",f"{sim_name}_{event_id}.README")) as f:
            readme_lines = f.readlines()
            mjd_explode = readme_lines[27].split(":")[1].split()[0]
            mjd_explode = float(mjd_explode)
        # Open FITS files
        with fits.open(head_path) as hdul_head, fits.open(phot_path) as hdul_phot:
            # Usually data is in extension 1
            data_head = hdul_head[1].data
            data_phot = hdul_phot[1].data
            
            # Use columns directly (Astropy FITS columns are case-insensitive usually)
            # HEAD columns
            ptrobs_min = data_head['PTROBS_MIN']
            ptrobs_max = data_head['PTROBS_MAX']
            
            # PHOT columns
            mjd_all = data_phot['MJD']
            flux_all = data_phot['FLUXCAL']
            fluxerr_all = data_phot['FLUXCALERR'] # Optional usage
            flt_all = data_phot['BAND'] # Filters
            photflag_all = data_phot['PHOTFLAG'] if 'PHOTFLAG' in data_phot.columns.names else None
            head_columns = set(data_head.columns.names)

            extracted_lcs = []
            use_luptitude = (
                fluxcal_to_psfflux_factor is not None
                and lupt_b_njy is not None
            )
            
            # Iterate over each realization in HEAD
            for i in range(len(data_head)):
                # SNANA uses 1-based indexing for pointers, Python uses 0-based
                # Start index: value - 1
                # End index: value (exclusive in python slicing)
                start_idx = ptrobs_min[i] - 1
                end_idx = ptrobs_max[i]
                nobs = data_head['NOBS'][i]
                if (not use_luptitude) and nobs < 5:
                    # print(f"Warning: Light curve for event {event_id} realization {i} has less than 5 observations. Skipping.")
                    continue  # Skip light curves with less than 5 observations

                # get coordinates
                ra = data_head['RA'][i]
                dec = data_head['DEC'][i]
                coordinates = np.array([ra, dec], dtype=np.float32)
                
                # Slicing the PHOT data
                lc_mjd = mjd_all[start_idx : end_idx]
                lc_flux = flux_all[start_idx : end_idx]
                lc_fluxerr = fluxerr_all[start_idx : end_idx]
                lc_flt = flt_all[start_idx : end_idx]
                lc_photflag = photflag_all[start_idx : end_idx] if photflag_all is not None else None
                raw_lc_mjd = np.asarray(lc_mjd, dtype=np.float64)
                raw_lc_flux = np.asarray(lc_flux, dtype=np.float64)
                raw_lc_fluxerr = np.asarray(lc_fluxerr, dtype=np.float64)
                head_mjd_detect_first = data_head['MJD_DETECT_FIRST'][i] if 'MJD_DETECT_FIRST' in head_columns else None
                first_detection_mjd = None

                if use_luptitude:
                    merged_mjd, merged_psfflux, merged_psffluxerr, merged_flt = merge_photometry_psfflux(
                        mjd=np.asarray(lc_mjd, dtype=np.float64),
                        fluxcal=np.asarray(lc_flux, dtype=np.float64),
                        fluxcalerr=np.asarray(lc_fluxerr, dtype=np.float64),
                        flt=np.asarray(lc_flt),
                        fluxcal_to_psfflux_factor=float(fluxcal_to_psfflux_factor),
                    )
                    if len(merged_mjd) < 5:
                        continue
                    if normalize_to_first_detection:
                        first_detection_mjd = resolve_first_detection_mjd(
                            snr_mjd=merged_mjd,
                            snr_flux=merged_psfflux,
                            snr_fluxerr=merged_psffluxerr,
                            photflag_mjd=raw_lc_mjd,
                            photflag=lc_photflag,
                            head_mjd_detect_first=head_mjd_detect_first,
                            snr_threshold=float(snr_threshold),
                        )
                        if first_detection_mjd is None:
                            continue
                    lc_mjd, lc_flux, lc_fluxerr, lc_flt = _transform_psfflux_to_luptitude(
                        mjd=merged_mjd,
                        psfflux=merged_psfflux,
                        psffluxerr=merged_psffluxerr,
                        flt=np.asarray(merged_flt),
                        psfflux_zp=float(psfflux_zp),
                        lupt_b_njy=lupt_b_njy,
                    )
                    if len(lc_mjd) == 0:
                        continue
                else:
                    # Legacy behavior for callers not passing luptitude parameters.
                    std = np.std(lc_flux)
                    mean = np.mean(lc_flux)
                    lc_flux = (lc_flux - mean) / (std + 1e-8)
                    lc_fluxerr = lc_fluxerr / (std + 1e-8)
                
                # --- Format Conversion (to Tensor-ready numpy) ---
                val_mat = np.zeros((MAX_LC_LENGTH, NUM_BANDS), dtype=np.float32)    # Values matrix (flux)
                err_mat = np.zeros((MAX_LC_LENGTH, NUM_BANDS), dtype=np.float32)    # Errors matrix (flux errors)
                mask_mat = np.zeros((MAX_LC_LENGTH, NUM_BANDS), dtype=np.float32)
                time_vec = np.zeros((MAX_LC_LENGTH,), dtype=np.float32)

                if normalize_to_first_detection:
                    if first_detection_mjd is None:
                        first_detection_mjd = resolve_first_detection_mjd(
                            snr_mjd=raw_lc_mjd,
                            snr_flux=raw_lc_flux,
                            snr_fluxerr=raw_lc_fluxerr,
                            photflag_mjd=raw_lc_mjd,
                            photflag=lc_photflag,
                            head_mjd_detect_first=head_mjd_detect_first,
                            snr_threshold=float(snr_threshold),
                        )
                    if first_detection_mjd is None:
                        continue
                    rel_times = (np.asarray(lc_mjd, dtype=np.float64) - float(first_detection_mjd)) / 100.0
                else:
                    # 1. Time Normalization (Relative to BNS merger time)
                    if len(lc_mjd) > 0:
                        rel_times = (lc_mjd - mjd_explode) / 100  # Scale down to manageable range[-0.3, 0.6]
                    else:
                        continue # Skip empty light curves
                    first_detection_mjd = None

                # 2. Fill Matrices
                # Truncate if longer than MAX_LC_LENGTH
                seq_len = min(len(lc_mjd), MAX_LC_LENGTH)
                if seq_len <= 0:
                    continue
                if len(lc_mjd) > MAX_LC_LENGTH:
                    print(f"Warning: Light curve for event {event_id} exceeds MAX_LC_LENGTH. Truncating.")
                    # Keep the MAX_LC_LENGTH points with smallest absolute rel_times
                    sorted_indices = np.argsort(np.abs(rel_times))[:MAX_LC_LENGTH]
                    sorted_indices = np.sort(sorted_indices)  # Sort back to chronological order
                    lc_mjd = lc_mjd[sorted_indices]
                    lc_flux = lc_flux[sorted_indices]
                    lc_fluxerr = lc_fluxerr[sorted_indices]
                    lc_flt = lc_flt[sorted_indices]
                    rel_times = rel_times[sorted_indices]
                
                for t in range(seq_len):
                    b_idx = _band_index_from_raw(lc_flt[t])
                    if b_idx is None:
                        continue

                    val_mat[t, b_idx] = lc_flux[t]
                    err_mat[t, b_idx] = lc_fluxerr[t]
                    mask_mat[t, b_idx] = 1.0
                    time_vec[t] = rel_times[t]

                if not np.any(mask_mat):
                    continue

                if normalize_to_first_detection:
                    extracted_lcs.append((val_mat, err_mat, mask_mat, time_vec, coordinates, first_detection_mjd))
                else:
                    extracted_lcs.append((val_mat, err_mat, mask_mat, time_vec, coordinates))

            return extracted_lcs

    except Exception as e:
        print(f"Error processing FITS for {event_id}: {e}")
        return []

def sample_moc_skymap(map_file):
    """
    Convert UNIQ to sky coordinates and calculte pixel areas.
    Note: Do not include DISTNORM in the output. For inf values in DISTMU, relace with distmean and diststd from metadata.
    """
    
    # read moc skymap and metadata
    moc_map = read_sky_map(map_file, moc=True, distances=True)
    _, meta = read_sky_map(map_file, nest=True)

    # extract distance meta info
    dist_mean = meta.get('distmean', None)
    dist_std = meta.get('diststd', None)

    uniq = moc_map['UNIQ']
    probdensity = moc_map['PROBDENSITY']
    distmu = moc_map['DISTMU']
    distsigma = moc_map['DISTSIGMA']
    # distnorm = moc_map['DISTNORM']  # not used

    # 1) UNIQ -> order, ipix, nside
    order, ipix = uniq2nest(uniq)

    # 2) caculate pixel area
    dA = uniq2pixarea(uniq)
    # 3) calculate pixel probability
    dP = probdensity * dA
    # 4) calculate theta, phi
    xs  = np.zeros_like(ipix, dtype=np.float32)
    ys = np.zeros_like(ipix, dtype=np.float32)
    zs = np.zeros_like(ipix, dtype=np.float32)
    for k in np.unique(order):
        m = (order == k)
        this_ipix  = ipix[m]
        this_nside = 2 ** k
        theta, phi = hp.pix2ang(this_nside, this_ipix, nest=True)
        # ras[m]  = np.degrees(phi)
        # decs[m] = 90.0 - np.degrees(theta)
        xs[m] = np.sin(theta) * np.cos(phi)   # dec,[0, pi]
        ys[m] = np.sin(theta) * np.sin(phi)   # ra,[0, 2pi]
        zs[m] = np.cos(theta)
    
    # 5) return torch tensors
    gw_mocmap = torch.tensor(np.vstack([xs, ys, zs, dA, 100 * dP, distmu, distsigma]), dtype=torch.float32)   # [7, N_pixels], no distnorm

    # 6) process unnormal distance values
    inf_dist_mu = torch.where(torch.isinf(gw_mocmap[5]))[0]
    gw_mocmap[5, inf_dist_mu] = dist_mean  # set inf to mean value
    gw_mocmap[6, inf_dist_mu] = dist_std   # set inf to std value
    gw_mocmap[5,:] = gw_mocmap[5,:] / 1000.0  # scale down
    gw_mocmap[6,:] = gw_mocmap[6,:] / 1000.0 # scale down

    return gw_mocmap  # [7, N_pixels]


# Functions for parsing SNANA FITS files for negative samples
def parse_snana_fits_neg(
    head_path,
    phot_path,
    type="SN",
    fluxcal_to_psfflux_factor=None,
    psfflux_zp=31.4,
    lupt_b_njy=None,
):
    '''
    Parses HEAD.fits and PHOT.fits.
    Extracts multiple light curve realizations for a kind of transients.
    
    Args:
        head_path: Path to HEAD FITS file.
        phot_path: Path to PHOT FITS file.
        type: Type of transient ('SN', 'TDE', 'AGN', 'uLens', 'dwarf-nova')
        
    Returns:
        List of tuples: [(values, masks, times), ...]
        Returns empty list if files are missing.
    '''
    allowed_types = ['SN', 'TDE', 'AGN', 'uLens', 'dwarf-nova']
    if type not in allowed_types:
        raise ValueError(f"Type must be one of {allowed_types}")

    if not os.path.exists(head_path) or not os.path.exists(phot_path):
        print(f"Warning: FITS files not found for {head_path}")
        return []

    try:
        # Open FITS files
        with fits.open(head_path) as hdul_head, fits.open(phot_path) as hdul_phot:
            # Usually data is in extension 1
            data_head = hdul_head[1].data
            data_phot = hdul_phot[1].data
            
            # Use columns directly (Astropy FITS columns are case-insensitive usually)
            # HEAD columns
            
            ptrobs_min = data_head['PTROBS_MIN']
            ptrobs_max = data_head['PTROBS_MAX']
            
            # PHOT columns
            mjd_all = data_phot['MJD']
            flux_all = data_phot['FLUXCAL']
            fluxerr_all = data_phot['FLUXCALERR'] # Optional usage
            flt_all = data_phot['BAND'] # Filters
            photflag_all = data_phot['PHOTFLAG'] if 'PHOTFLAG' in data_phot.columns.names else None

            extracted_lcs = []
            use_luptitude = (
                fluxcal_to_psfflux_factor is not None
                and lupt_b_njy is not None
            )
            
            # Iterate over each realization in HEAD
            for i in range(len(data_head)):
                # SNANA uses 1-based indexing for pointers, Python uses 0-based
                # Start index: value - 1
                # End index: value (exclusive in python slicing)
                start_idx = ptrobs_min[i] - 1
                end_idx = ptrobs_max[i]
                nobs = data_head['NOBS'][i]
                if nobs < 5:
                    # print(f"Warning: Light curve for {head_path} realization {i} has less than 5 observations. Skipping.")
                    continue  # Skip light curves with less than 5 observations

                # get coordinates
                ra = data_head['RA'][i]
                dec = data_head['DEC'][i]
                coordinates = np.array([ra, dec], dtype=np.float32)
                
                # Slicing the PHOT data
                lc_mjd = mjd_all[start_idx : end_idx]
                lc_flux = flux_all[start_idx : end_idx]
                lc_fluxerr = fluxerr_all[start_idx : end_idx]
                lc_flt = flt_all[start_idx : end_idx]
                lc_photflag = (
                    np.asarray(photflag_all[start_idx:end_idx], dtype=np.int64)
                    if photflag_all is not None
                    else None
                )
                det_idx = _first_detection_index(
                    flux=np.asarray(lc_flux, dtype=np.float64),
                    fluxerr=np.asarray(lc_fluxerr, dtype=np.float64),
                    photflag=lc_photflag,
                    snr_threshold=5.0,
                )
                if det_idx is None:
                    continue
                t0_mjd = float(np.asarray(lc_mjd, dtype=np.float64)[det_idx])

                if use_luptitude:
                    lc_mjd, lc_flux, lc_fluxerr, lc_flt = _transform_fluxcal_to_luptitude(
                        mjd=np.asarray(lc_mjd, dtype=np.float64),
                        fluxcal=np.asarray(lc_flux, dtype=np.float64),
                        fluxcalerr=np.asarray(lc_fluxerr, dtype=np.float64),
                        flt=np.asarray(lc_flt),
                        fluxcal_to_psfflux_factor=float(fluxcal_to_psfflux_factor),
                        psfflux_zp=float(psfflux_zp),
                        lupt_b_njy=lupt_b_njy,
                    )
                    if len(lc_mjd) == 0:
                        continue
                else:
                    # Legacy behavior for callers not passing luptitude parameters.
                    std = np.std(lc_flux)
                    mean = np.mean(lc_flux)
                    lc_flux = (lc_flux - mean) / (std + 1e-8)
                    lc_fluxerr = lc_fluxerr / (std + 1e-8)
                
                # --- Format Conversion (to Tensor-ready numpy) ---
                val_mat = np.zeros((MAX_LC_LENGTH, NUM_BANDS), dtype=np.float32)    # Values matrix (flux)
                err_mat = np.zeros((MAX_LC_LENGTH, NUM_BANDS), dtype=np.float32)    # Errors matrix (flux errors)
                mask_mat = np.zeros((MAX_LC_LENGTH, NUM_BANDS), dtype=np.float32)
                time_vec = np.zeros((MAX_LC_LENGTH,), dtype=np.float32)
                
                # 1. Time normalization: optical-only rule (first detection as t0)
                if len(lc_mjd) > 0:
                    rel_times = (lc_mjd - t0_mjd) / 100  # first-detection anchored, scaled as before
                    time_mask = np.where((rel_times >= -0.3) & (rel_times <= 0.6))[0]
                    lc_mjd = lc_mjd[time_mask]
                    lc_flux = lc_flux[time_mask]
                    lc_fluxerr = lc_fluxerr[time_mask]
                    lc_flt = lc_flt[time_mask]
                    rel_times = rel_times[time_mask]
                else:
                    continue # Skip empty light curves

                # 2. Fill Matrices
                # Truncate if longer than MAX_LC_LENGTH
                seq_len = min(len(lc_mjd), MAX_LC_LENGTH)
                if seq_len <= 0:
                    continue
                if len(lc_mjd) > MAX_LC_LENGTH:
                    print(f"Warning: Light curve for realization {i} exceeds MAX_LC_LENGTH. Truncating.")
                    # Keep the MAX_LC_LENGTH points with smallest absolute rel_times
                    sorted_indices = np.argsort(np.abs(rel_times))[:MAX_LC_LENGTH]
                    sorted_indices = np.sort(sorted_indices)  # Sort back to chronological order
                    lc_mjd = lc_mjd[sorted_indices]
                    lc_flux = lc_flux[sorted_indices]
                    lc_fluxerr = lc_fluxerr[sorted_indices]
                    lc_flt = lc_flt[sorted_indices]
                    rel_times = rel_times[sorted_indices]
                
                for t in range(seq_len):
                    b_idx = _band_index_from_raw(lc_flt[t])
                    if b_idx is None:
                        continue

                    val_mat[t, b_idx] = lc_flux[t]
                    err_mat[t, b_idx] = lc_fluxerr[t]
                    mask_mat[t, b_idx] = 1.0
                    time_vec[t] = rel_times[t]

                if not np.any(mask_mat):
                    continue
                
                extracted_lcs.append((val_mat, err_mat, mask_mat, time_vec, coordinates))
                
            return extracted_lcs

    except Exception as e:
        print(f"Error processing FITS for {head_path}: {e}")
        return []

# Function to create negative dataset HDF5
def create_negative_dataset(
    sim_root,
    output_h5_path,
    buffer_limit=5000,
    fluxcal_zp=27.5,
    psfflux_zp=31.4,
    lupt_k=1.0,
    lupt_m5_mag="23.9,25.0,24.7,24.0,23.3,22.1",
):
    sim_root = Path(sim_root)
    lupt_m5_mag_arr = parse_lupt_m5_mag(lupt_m5_mag)
    fluxcal_to_psfflux_factor, lupt_f5sigma_njy, lupt_b_njy = build_luptitude_params(
        fluxcal_zp=float(fluxcal_zp),
        psfflux_zp=float(psfflux_zp),
        lupt_k=float(lupt_k),
        lupt_m5_mag=lupt_m5_mag_arr,
    )

    # Decompress FITS.gz files (exclude KN) and remove .gz
    gz_files = [p for p in sim_root.rglob("*.FITS.gz") if "ELASTICC2_TRAIN_02_KN_" not in p.as_posix()]
    if len(gz_files) > 0:
        for gz_path in tqdm(gz_files, desc="Decompressing FITS.gz"):
            subprocess.run(["gunzip", "-f", str(gz_path)], check=False)

    head_files = [p for p in sim_root.rglob("*_HEAD.FITS") if "ELASTICC2_TRAIN_02_KN_" not in p.as_posix()]
    head_files = sorted(head_files)
    if len(head_files) == 0:
        raise FileNotFoundError(f"No HEAD.FITS files found under {sim_root}")

    with h5py.File(output_h5_path, 'w') as f:
        grp_opt = f.create_group('ELASTICC2_TRAIN/optical_data')
        chunk_size = 1024
        ds_opt_vals = grp_opt.create_dataset(
            'values', (0, MAX_LC_LENGTH, NUM_BANDS),
            maxshape=(None, MAX_LC_LENGTH, NUM_BANDS),
            dtype='f4', chunks=(chunk_size, MAX_LC_LENGTH, NUM_BANDS)
        )
        ds_opt_errs = grp_opt.create_dataset(
            'errors', (0, MAX_LC_LENGTH, NUM_BANDS),
            maxshape=(None, MAX_LC_LENGTH, NUM_BANDS),
            dtype='f4', chunks=(chunk_size, MAX_LC_LENGTH, NUM_BANDS)
        )
        ds_opt_masks = grp_opt.create_dataset(
            'masks', (0, MAX_LC_LENGTH, NUM_BANDS),
            maxshape=(None, MAX_LC_LENGTH, NUM_BANDS),
            dtype='f4', chunks=(chunk_size, MAX_LC_LENGTH, NUM_BANDS)
        )
        ds_opt_times = grp_opt.create_dataset(
            'times', (0, MAX_LC_LENGTH),
            maxshape=(None, MAX_LC_LENGTH),
            dtype='f4', chunks=(chunk_size, MAX_LC_LENGTH)
        )
        ds_opt_coords = grp_opt.create_dataset(
            'coordinates', (0, 2),
            maxshape=(None, 2),
            dtype='f4', chunks=(chunk_size, 2)
        )
        dt_str = h5py.special_dtype(vlen=str)
        ds_opt_types = grp_opt.create_dataset(
            'types', (0,), maxshape=(None,), dtype=dt_str, chunks=(chunk_size,)
        )

        total_optical = 0
        opt_buffer_vals = []
        opt_buffer_errs = []
        opt_buffer_masks = []
        opt_buffer_times = []
        opt_buffer_coords = []
        opt_buffer_types = []

        def flush_buffer():
            nonlocal total_optical, opt_buffer_vals, opt_buffer_errs, opt_buffer_masks, opt_buffer_times, opt_buffer_coords, opt_buffer_types
            if len(opt_buffer_vals) == 0:
                return
            n_new = len(opt_buffer_vals)
            current_size = total_optical
            new_size = current_size + n_new

            ds_opt_vals.resize(new_size, axis=0)
            ds_opt_errs.resize(new_size, axis=0)
            ds_opt_masks.resize(new_size, axis=0)
            ds_opt_times.resize(new_size, axis=0)
            ds_opt_coords.resize(new_size, axis=0)
            ds_opt_types.resize(new_size, axis=0)

            ds_opt_vals[current_size:new_size] = np.array(opt_buffer_vals)
            ds_opt_errs[current_size:new_size] = np.array(opt_buffer_errs)
            ds_opt_masks[current_size:new_size] = np.array(opt_buffer_masks)
            ds_opt_times[current_size:new_size] = np.array(opt_buffer_times)
            ds_opt_coords[current_size:new_size] = np.array(opt_buffer_coords)
            ds_opt_types[current_size:new_size] = np.array(opt_buffer_types, dtype=object)

            total_optical += n_new
            opt_buffer_vals = []
            opt_buffer_errs = []
            opt_buffer_masks = []
            opt_buffer_times = []
            opt_buffer_coords = []
            opt_buffer_types = []

        for head_path in tqdm(head_files):
            name_lower = head_path.parent.name.lower()
            if 'kn' in name_lower:
                continue
            if 'agn' in name_lower:
                transient_type = 'AGN'
            elif 'tde' in name_lower:
                transient_type = 'TDE'
            elif 'ulens' in name_lower:
                transient_type = 'uLens'
            elif 'dwarf-nova' in name_lower:
                transient_type = 'dwarf-nova'
            elif 'sn' in name_lower or 'slsn' in name_lower or 'pisn' in name_lower:
                transient_type = 'SN'
            else:
                print(f"Skipping unsupported type folder: {head_path.parent.name}")
                continue

            phot_path = head_path.with_name(head_path.name.replace('_HEAD.FITS', '_PHOT.FITS'))
            if not phot_path.exists():
                print(f"Missing PHOT file for {head_path.name}")
                continue

            lcs = parse_snana_fits_neg(
                str(head_path),
                str(phot_path),
                type=transient_type,
                fluxcal_to_psfflux_factor=float(fluxcal_to_psfflux_factor),
                psfflux_zp=float(psfflux_zp),
                lupt_b_njy=lupt_b_njy,
            )
            for (vals, errs, masks, times, coordinates) in lcs:
                opt_buffer_vals.append(vals)
                opt_buffer_errs.append(errs)
                opt_buffer_masks.append(masks)
                opt_buffer_times.append(times)
                opt_buffer_coords.append(coordinates)
                opt_buffer_types.append(transient_type)

            if len(opt_buffer_vals) >= buffer_limit:
                flush_buffer()

        flush_buffer()
        f.attrs['n_total_optical'] = total_optical
        f.attrs["photometry_representation"] = "luptitude"
        f.attrs["flux_input_column"] = "FLUXCAL"
        f.attrs["fluxerr_input_column"] = "FLUXCALERR"
        f.attrs["fluxcal_zp"] = float(fluxcal_zp)
        f.attrs["psfflux_zp"] = float(psfflux_zp)
        f.attrs["fluxcal_to_psfflux_factor"] = float(fluxcal_to_psfflux_factor)
        f.attrs["lupt_k"] = float(lupt_k)
        f.attrs["lupt_band_order"] = ",".join(LUPT_BAND_ORDER)
        f.attrs["lupt_m5_mag"] = np.asarray(lupt_m5_mag_arr, dtype=np.float64)
        f.attrs["lupt_f5sigma_njy"] = np.asarray(lupt_f5sigma_njy, dtype=np.float64)
        f.attrs["lupt_b_njy"] = np.asarray(lupt_b_njy, dtype=np.float64)
        f.attrs["values_semantics"] = "luptitude"
        f.attrs["errors_semantics"] = "luptitude_sigma"
        print(f"Total negative light curves: {total_optical}")
        print(f"Saved to: {output_h5_path}")
