import h5py
import numpy as np
import pandas as pd
from astropy.io import fits
from tqdm import tqdm
import os
import healpy as hp
from ligo.skymap.io.fits import read_sky_map
from ligo.skymap.moc import uniq2nest, uniq2pixarea
import warnings
warnings.filterwarnings("ignore", "Wswiglal-redir-stdio")
import torch
from collections import defaultdict
from typing import List, Iterator, Optional, Tuple
from torch.utils.data import Dataset, DataLoader, Sampler
from pathlib import Path
import subprocess


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

        # Open file temporarily to get dataset length
        with h5py.File(h5_path, 'r') as f:
            self.length = f['events/optical_data/values'].shape[0]
            self.has_opt_zero_time_mjd_base = "events/optical_data/zero_time_mjd_base" in f
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
        lefts = np.searchsorted(times, anchor - np.asarray(windows, dtype=np.float64), side="left")
        rights = np.searchsorted(times, anchor + np.asarray(windows, dtype=np.float64), side="right")

        eligible_levels = []
        level_counts = []
        for i in range(len(windows)):
            outer_l = int(lefts[i])
            outer_r = int(rights[i])
            if i == 0:
                count = max(0, outer_r - outer_l)
            else:
                inner_l = int(lefts[i - 1])
                inner_r = int(rights[i - 1])
                left_count = max(0, inner_l - outer_l)
                right_count = max(0, outer_r - inner_r)
                count = left_count + right_count
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
        if picked == 0:
            count = max(0, outer_r - outer_l)
            if count <= 0:
                return self._sample_uniform_neg_idx()
            sorted_idx = outer_l + int(np.random.randint(0, count))
            return int(sorted_to_orig[sorted_idx])

        inner_l = int(lefts[picked - 1])
        inner_r = int(rights[picked - 1])
        left_count = max(0, inner_l - outer_l)
        right_count = max(0, outer_r - inner_r)
        count = left_count + right_count
        if count <= 0:
            return self._sample_uniform_neg_idx()

        draw = int(np.random.randint(0, count))
        if draw < left_count:
            sorted_idx = outer_l + draw
        else:
            sorted_idx = inner_r + (draw - left_count)
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

            # Return tuple: (GW_Inputs, Optical_Inputs, Metadata, Negative_Optical_Inputs)
            # gw_idx is returned for masking the contrastive loss (handling same-source negatives)
            out = [
                gw_scalar, gw_skymap, opt_time, opt_val, opt_mask, opt_err, opt_coords, int(gw_idx),
                neg_time, neg_val, neg_mask, neg_err, neg_coords,
            ]
            if self.return_zero_time_mjd:
                out.extend([opt_zero_time_mjd_base, neg_zero_time_mjd_base, neg_zero_time_mjd_cls_base])
            if neg_gw_local_idx is not None:
                out.append(is_neg_gw)
            return tuple(out)

        # Return tuple: (GW_Inputs, Optical_Inputs, Metadata)
        # gw_idx is returned for masking the contrastive loss (handling same-source negatives)
        out = [gw_scalar, gw_skymap, opt_time, opt_val, opt_mask, opt_err, opt_coords, int(gw_idx)]
        if self.return_zero_time_mjd:
            out.append(opt_zero_time_mjd_base)
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

def     _build_dataloader(
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
    return _build_dataloader(
        dataset,
        sampler,
        num_workers,
        pin_memory,
        persistent_workers,
        prefetch_factor
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

    train_loader = _build_dataloader(
        train_dataset,
        train_sampler,
        num_workers,
        pin_memory,
        persistent_workers,
        prefetch_factor
    )
    val_loader = _build_dataloader(
        val_dataset,
        val_sampler,
        num_workers,
        pin_memory,
        persistent_workers,
        prefetch_factor
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

    train_loader = _build_dataloader(
        train_dataset,
        train_sampler,
        num_workers,
        pin_memory,
        persistent_workers,
        prefetch_factor
    )
    val_loader = _build_dataloader(
        val_dataset,
        val_sampler,
        num_workers,
        pin_memory,
        persistent_workers,
        prefetch_factor
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

    train_loader = _build_dataloader(
        train_dataset,
        train_sampler,
        num_workers,
        pin_memory,
        persistent_workers,
        prefetch_factor
    )
    val_loader = _build_dataloader(
        val_dataset,
        val_sampler,
        num_workers,
        pin_memory,
        persistent_workers,
        prefetch_factor
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
    def __init__(
        self,
        pos_h5_path: str,
        neg_h5_path: str,
        neg_group: str = "ELASTICC2_TRAIN/optical_data",
        pos_indices: Optional[np.ndarray] = None,
        neg_indices: Optional[np.ndarray] = None,
        cache_in_memory: bool = False,
    ):
        super().__init__()
        self.pos_h5_path = pos_h5_path
        self.neg_h5_path = neg_h5_path
        self.neg_group = neg_group
        self.cache_in_memory = bool(cache_in_memory)

        self.pos_file = None
        self.neg_file = None

        with h5py.File(self.pos_h5_path, "r") as f:
            n_pos_total = int(f["events/optical_data/values"].shape[0])

        with h5py.File(self.neg_h5_path, "r") as f:
            if self.neg_group not in f:
                raise KeyError(f"Negative group '{self.neg_group}' not found in {self.neg_h5_path}")
            n_neg_total = int(f[f"{self.neg_group}/values"].shape[0])

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

        self.length = self.n_pos + self.n_neg

        if self.cache_in_memory:
            # Avoid caching multi-million samples by default; keep memory predictable.
            print("OpticalBinaryDataset: cache_in_memory is not supported, using lazy HDF5 loading.")
            self.cache_in_memory = False

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

    def __getitem__(self, idx):
        if isinstance(idx, np.ndarray):
            idx = int(idx.item())
        idx = int(idx)
        self._ensure_files_open()

        if idx < self.n_pos:
            real_idx = int(self.pos_indices[idx])
            grp = "events/optical_data"
            f = self.pos_file
            label = 1.0
        else:
            real_idx = int(self.neg_indices[idx - self.n_pos])
            grp = self.neg_group
            f = self.neg_file
            label = 0.0

        opt_val = self._as_tensor(f[f"{grp}/values"][real_idx])
        opt_err = self._as_tensor(f[f"{grp}/errors"][real_idx])
        opt_mask = self._as_tensor(f[f"{grp}/masks"][real_idx])
        opt_time = self._as_tensor(f[f"{grp}/times"][real_idx])

        target = torch.tensor(label, dtype=torch.float32)
        return opt_time, opt_val, opt_mask, opt_err, target

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


def split_positive_optical_indices(
    pos_h5_path: str,
    val_split: float = 0.1,
    seed: int = 42
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Split positive optical indices by parent GW event to avoid train/val leakage.
    """
    if not (0 < val_split < 1):
        raise ValueError("val_split must be in (0, 1).")

    with h5py.File(pos_h5_path, "r") as f:
        parent_gw_idx = f["events/optical_data/parent_gw_idx"][:]

    unique_gw = np.unique(parent_gw_idx)
    rng = np.random.default_rng(seed)
    rng.shuffle(unique_gw)

    n_val_gw = max(1, int(len(unique_gw) * val_split))
    val_gw = unique_gw[:n_val_gw]

    is_val = np.isin(parent_gw_idx, val_gw)
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
):
    if cache_in_memory and num_workers > 0:
        print("cache_in_memory=True with num_workers>0 may increase RAM usage.")

    train_pos_idx, val_pos_idx = split_positive_optical_indices(
        pos_h5_path, val_split=val_split, seed=split_seed
    )
    train_neg_idx, val_neg_idx = split_negative_optical_indices(
        neg_h5_path, neg_group=neg_group, val_split=val_split, seed=split_seed
    )

    if val_batch_size is None:
        val_batch_size = batch_size

    if steps_per_epoch is None:
        steps_per_epoch = max(1, (2 * min(len(train_pos_idx), len(train_neg_idx))) // batch_size)

    if val_steps_per_epoch is None:
        val_steps_per_epoch = max(1, (2 * min(len(val_pos_idx), len(val_neg_idx))) // val_batch_size)

    train_dataset = OpticalBinaryDataset(
        pos_h5_path=pos_h5_path,
        neg_h5_path=neg_h5_path,
        neg_group=neg_group,
        pos_indices=train_pos_idx,
        neg_indices=train_neg_idx,
        cache_in_memory=cache_in_memory,
    )
    val_dataset = OpticalBinaryDataset(
        pos_h5_path=pos_h5_path,
        neg_h5_path=neg_h5_path,
        neg_group=neg_group,
        pos_indices=val_pos_idx,
        neg_indices=val_neg_idx,
        cache_in_memory=cache_in_memory,
    )

    train_sampler = BalancedBinaryBatchSampler(
        n_pos=train_dataset.n_pos,
        n_neg=train_dataset.n_neg,
        batch_size=batch_size,
        steps_per_epoch=steps_per_epoch,
        seed=split_seed,
        shuffle=True,
    )
    val_sampler = BalancedBinaryBatchSampler(
        n_pos=val_dataset.n_pos,
        n_neg=val_dataset.n_neg,
        batch_size=val_batch_size,
        steps_per_epoch=val_steps_per_epoch,
        seed=split_seed + 1,
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

# Functions for parsing SNANA FITS files, and sampling MOC skymaps
def parse_snana_fits(event_id, sim_dir, sim_name="LSST_KN_BNS_AUG"):
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

            extracted_lcs = []
            
            # Iterate over each realization in HEAD
            for i in range(len(data_head)):
                # SNANA uses 1-based indexing for pointers, Python uses 0-based
                # Start index: value - 1
                # End index: value (exclusive in python slicing)
                start_idx = ptrobs_min[i] - 1
                end_idx = ptrobs_max[i]
                nobs = data_head['NOBS'][i]
                if nobs < 5:
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

                # Normalization
                std = np.std(lc_flux)
                mean = np.mean(lc_flux)
                lc_flux = (lc_flux - mean) / (std + 1e-8)
                lc_fluxerr = lc_fluxerr / (std + 1e-8)
                
                # --- Format Conversion (to Tensor-ready numpy) ---
                val_mat = np.zeros((MAX_LC_LENGTH, NUM_BANDS), dtype=np.float32)    # Values matrix (flux)
                err_mat = np.zeros((MAX_LC_LENGTH, NUM_BANDS), dtype=np.float32)    # Errors matrix (flux errors)
                mask_mat = np.zeros((MAX_LC_LENGTH, NUM_BANDS), dtype=np.float32)
                time_vec = np.zeros((MAX_LC_LENGTH,), dtype=np.float32)
                
                # 1. Time Normalization (Relative to BNS merger time)
                if len(lc_mjd) > 0:
                    rel_times = (lc_mjd - mjd_explode) / 100  # Scale down to manageable range[-0.3, 0.6]
                else:
                    continue # Skip empty light curves

                # 2. Fill Matrices
                # Truncate if longer than MAX_LC_LENGTH
                seq_len = min(len(lc_mjd), MAX_LC_LENGTH)
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
                    band_char = lc_flt[t].strip() # Remove whitespace
                    if band_char in BAND_MAP:
                        b_idx = BAND_MAP[band_char]
                        
                        val_mat[t, b_idx] = lc_flux[t]
                        err_mat[t, b_idx] = lc_fluxerr[t]
                        mask_mat[t, b_idx] = 1.0
                        time_vec[t] = rel_times[t]
                
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
def parse_snana_fits_neg(head_path, phot_path, type="SN"):
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

            extracted_lcs = []
            
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

                # Normalization
                std = np.std(lc_flux)
                mean = np.mean(lc_flux)
                lc_flux = (lc_flux - mean) / (std + 1e-8)
                lc_fluxerr = lc_fluxerr / (std + 1e-8)
                
                # --- Format Conversion (to Tensor-ready numpy) ---
                val_mat = np.zeros((MAX_LC_LENGTH, NUM_BANDS), dtype=np.float32)    # Values matrix (flux)
                err_mat = np.zeros((MAX_LC_LENGTH, NUM_BANDS), dtype=np.float32)    # Errors matrix (flux errors)
                mask_mat = np.zeros((MAX_LC_LENGTH, NUM_BANDS), dtype=np.float32)
                time_vec = np.zeros((MAX_LC_LENGTH,), dtype=np.float32)
                
                # 1. Time Normalization (Relative to pseudo-explosion time)
                if len(lc_mjd) > 0:
                    use_peak = type in ['SN', 'uLens', 'dwarf-nova', 'TDE']
                    if use_peak and 'PEAKMJD' in data_head.columns.names:
                        pesudo_mjd_explode = data_head['PEAKMJD'][i] - np.random.uniform(0.5, 5.0)
                    else:
                        pesudo_mjd_explode = np.random.uniform(lc_mjd.min(), lc_mjd.max())
                    rel_times = (lc_mjd - pesudo_mjd_explode) / 100  # Scale down to manageable range[-0.3, 0.6]
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
                    band_char = lc_flt[t].strip() # Remove whitespace
                    if band_char in NEG_BAND_MAP:
                        b_idx = NEG_BAND_MAP[band_char]
                        
                        val_mat[t, b_idx] = lc_flux[t]
                        err_mat[t, b_idx] = lc_fluxerr[t]
                        mask_mat[t, b_idx] = 1.0
                        time_vec[t] = rel_times[t]
                
                extracted_lcs.append((val_mat, err_mat, mask_mat, time_vec, coordinates))
                
            return extracted_lcs

    except Exception as e:
        print(f"Error processing FITS for {head_path}: {e}")
        return []

# Function to create negative dataset HDF5
def create_negative_dataset(sim_root, output_h5_path, buffer_limit=5000):
    sim_root = Path(sim_root)

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

            lcs = parse_snana_fits_neg(str(head_path), str(phot_path), type=transient_type)
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
        print(f"Total negative light curves: {total_optical}")
        print(f"Saved to: {output_h5_path}")
