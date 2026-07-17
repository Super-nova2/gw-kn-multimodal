import os
import importlib.util
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import h5py
import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.time import Time
from tqdm import tqdm

# Load required functions/constants from Model/data_loader.py.
_DATA_LOADER_PATH = Path(__file__).resolve().parents[3] / "data_loader.py"
_spec = importlib.util.spec_from_file_location("data_loader", _DATA_LOADER_PATH)
_data_loader = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_data_loader)  # type: ignore

parse_snana_fits = _data_loader.parse_snana_fits
sample_moc_skymap = _data_loader.sample_moc_skymap
MAX_LC_LENGTH = _data_loader.MAX_LC_LENGTH
NUM_BANDS = _data_loader.NUM_BANDS

GW_PARAM_COLUMNS = [
    "mass1_detector",
    "mass2_detector",
    "spin1z",
    "spin2z",
    "inclination",
    "distmean",
    "diststd",
]

MJD_TIME_COLUMN_CANDIDATES = (
    "mjd_time",
    "mjd",
    "mjd_event",
    "event_mjd",
    "merger_mjd",
    "trigger_mjd",
)
GPS_TIME_COLUMN_CANDIDATES = (
    "gps_time",
    "gps",
    "gps_event_time",
    "event_gps_time",
    "trigger_time_gps",
)


@dataclass
class EventSelection:
    gw_df: pd.DataFrame
    event_to_row: Dict[int, int]
    pos_event_ids: np.ndarray
    neg_event_ids: np.ndarray
    neg_type_by_event: Dict[int, int]
    mej_by_event: Dict[int, float]
    n_neg_type1: int
    n_neg_type2: int
    n_missing_skymap: int
    n_filtered_non_success_mej_pos: int
    n_invalid_mej: int
    nsbh_mej_col_resolved: Optional[str] = None
    success_ids_mej_pos: Optional[Set[int]] = None


def _load_success_ids(success_ids_path: str) -> Set[int]:
    with open(success_ids_path, "r", encoding="utf-8") as f:
        return {int(line.strip()) for line in f if line.strip()}


def _load_gw_catalog(
    full_catalog_path: str,
    success_ids_path: Optional[str] = None,
    apply_success_filter: bool = True,
) -> pd.DataFrame:
    gw_df = pd.read_csv(full_catalog_path)
    if "simulation_id" not in gw_df.columns:
        raise ValueError(f"'simulation_id' column not found in {full_catalog_path}")

    gw_df = gw_df.copy()
    gw_df["simulation_id"] = gw_df["simulation_id"].astype(int)

    if apply_success_filter and success_ids_path:
        success_ids = _load_success_ids(success_ids_path)
        gw_df = gw_df.loc[gw_df["simulation_id"].isin(success_ids)].copy()
        gw_df.reset_index(drop=True, inplace=True)
        print(f"Filtered GW catalog by success ids: {len(gw_df)} events")

    dup_count = int(gw_df["simulation_id"].duplicated().sum())
    if dup_count > 0:
        print(f"WARNING: found {dup_count} duplicate simulation_id rows. Keeping first occurrence.")
        gw_df = gw_df.drop_duplicates(subset=["simulation_id"], keep="first").copy()
        gw_df.reset_index(drop=True, inplace=True)

    return gw_df


def _normalize_max_count(value: Optional[int]) -> Optional[int]:
    if value is None:
        return None
    if int(value) <= 0:
        return None
    return int(value)


def _sample_event_ids(
    event_ids: np.ndarray,
    max_count: Optional[int],
    rng: np.random.Generator,
) -> np.ndarray:
    if max_count is None or len(event_ids) <= max_count:
        return event_ids
    chosen = rng.choice(event_ids, size=max_count, replace=False)
    return np.sort(chosen.astype(np.int64))


def _resolve_nsbh_mej_col(gw_df: pd.DataFrame, requested_col: str) -> str:
    if requested_col in gw_df.columns:
        return requested_col
    if requested_col == "mej_tot" and "mej_total" in gw_df.columns:
        return "mej_total"
    if "mej_tot" in gw_df.columns:
        return "mej_tot"
    if "mej_total" in gw_df.columns:
        return "mej_total"
    raise ValueError(
        f"NSBH catalog missing mej column. Tried '{requested_col}', 'mej_tot', 'mej_total'."
    )


def _find_column_case_insensitive(df: pd.DataFrame, candidates: Tuple[str, ...]) -> Optional[str]:
    col_by_lower = {str(col).lower(): str(col) for col in df.columns}
    for cand in candidates:
        found = col_by_lower.get(cand.lower())
        if found is not None:
            return found
    return None


def _gps_to_mjd(gps_values: np.ndarray) -> np.ndarray:
    out = np.full(gps_values.shape, np.nan, dtype=np.float64)
    finite_mask = np.isfinite(gps_values)
    if not np.any(finite_mask):
        return out
    out[finite_mask] = Time(gps_values[finite_mask], format="gps", scale="utc").mjd.astype(np.float64)
    return out


def _extract_event_time_mjd(
    gw_df: pd.DataFrame,
) -> Tuple[np.ndarray, str, bool, int]:
    """
    Resolve GW event time in MJD.

    Priority:
      1. `mjd_*`-style columns
      2. `gps_*`-style columns converted to MJD
      3. fallback to all-NaN if no usable column exists
    """
    event_time_col = _find_column_case_insensitive(gw_df, MJD_TIME_COLUMN_CANDIDATES)
    used_gps_fallback = False

    if event_time_col is not None:
        event_time_mjd = pd.to_numeric(gw_df[event_time_col], errors="coerce").to_numpy(np.float64)
    else:
        gps_col = _find_column_case_insensitive(gw_df, GPS_TIME_COLUMN_CANDIDATES)
        if gps_col is not None:
            gps_values = pd.to_numeric(gw_df[gps_col], errors="coerce").to_numpy(np.float64)
            event_time_mjd = _gps_to_mjd(gps_values)
            event_time_col = gps_col
            used_gps_fallback = True
        else:
            event_time_mjd = np.full((len(gw_df),), np.nan, dtype=np.float64)
            event_time_col = "MISSING"

    n_invalid = int((~np.isfinite(event_time_mjd)).sum())
    return event_time_mjd, event_time_col, used_gps_fallback, n_invalid


def _scan_has_optical(
    event_ids: np.ndarray,
    skymap_dir: str,
    sim_root: str,
    sim_name: str,
    scan_desc: str,
) -> Tuple[np.ndarray, np.ndarray, int]:
    pos_ids: List[int] = []
    neg_ids: List[int] = []
    missing_skymap = 0

    for event_id in tqdm(
        event_ids,
        total=len(event_ids),
        desc=scan_desc,
        mininterval=0.5,
        miniters=200,
    ):
        event_id = int(event_id)
        skymap_path = os.path.join(skymap_dir, f"{event_id}.fits")
        if not os.path.exists(skymap_path):
            missing_skymap += 1
            continue

        head_path = os.path.join(
            sim_root,
            f"{sim_name}_{event_id}",
            f"{sim_name}_{event_id}_HEAD.FITS",
        )

        has_optical = False
        if os.path.exists(head_path) and os.path.getsize(head_path) > 0:
            try:
                nrows = fits.getheader(head_path, 1).get("NAXIS2", 0)
                has_optical = bool(nrows and int(nrows) > 0)
            except Exception:
                has_optical = False

        if has_optical:
            pos_ids.append(event_id)
        else:
            neg_ids.append(event_id)

    return (
        np.asarray(pos_ids, dtype=np.int64),
        np.asarray(neg_ids, dtype=np.int64),
        missing_skymap,
    )


def _scan_skymap_only(
    event_ids: np.ndarray,
    skymap_dir: str,
    scan_desc: str,
) -> Tuple[np.ndarray, int]:
    valid_ids: List[int] = []
    missing_skymap = 0

    for event_id in tqdm(
        event_ids,
        total=len(event_ids),
        desc=scan_desc,
        mininterval=0.5,
        miniters=200,
    ):
        event_id = int(event_id)
        skymap_path = os.path.join(skymap_dir, f"{event_id}.fits")
        if not os.path.exists(skymap_path):
            missing_skymap += 1
            continue
        valid_ids.append(event_id)

    return np.asarray(valid_ids, dtype=np.int64), missing_skymap


def _cap_nsbh_neg_total(
    type1_ids: np.ndarray,
    type2_ids: np.ndarray,
    max_total: Optional[int],
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    if max_total is None:
        return type1_ids, type2_ids

    total = len(type1_ids) + len(type2_ids)
    if total <= max_total:
        return type1_ids, type2_ids

    all_ids = np.concatenate([type1_ids, type2_ids], axis=0)
    all_types = np.concatenate(
        [np.ones(len(type1_ids), dtype=np.int8), np.full(len(type2_ids), 2, dtype=np.int8)],
        axis=0,
    )

    chosen_idx = rng.choice(total, size=max_total, replace=False)
    chosen_ids = all_ids[chosen_idx]
    chosen_types = all_types[chosen_idx]

    kept_type1 = np.sort(chosen_ids[chosen_types == 1].astype(np.int64))
    kept_type2 = np.sort(chosen_ids[chosen_types == 2].astype(np.int64))
    return kept_type1, kept_type2


def build_event_selection(
    full_catalog_path: str,
    skymap_dir: str,
    sim_root: str,
    sim_name: str = "LSST_KN_BNS_AUG",
    success_ids_path: Optional[str] = None,
    source_type: str = "bns",
    nsbh_mej_col: str = "mej_tot",
    nsbh_type1_threshold: float = 0.0,
    nsbh_require_success_for_mej_pos: bool = True,
    max_neg_gw: Optional[int] = None,
    nsbh_max_neg_type1_gw: Optional[int] = None,
    nsbh_max_neg_type2_gw: Optional[int] = None,
    seed: Optional[int] = None,
) -> EventSelection:
    source_type = source_type.strip().lower()
    if source_type not in {"bns", "nsbh"}:
        raise ValueError(f"source_type must be 'bns' or 'nsbh', got: {source_type}")

    rng = np.random.default_rng(seed)

    if source_type == "bns":
        gw_df = _load_gw_catalog(
            full_catalog_path,
            success_ids_path=success_ids_path,
            apply_success_filter=True,
        )
        sim_ids = gw_df["simulation_id"].to_numpy(np.int64)

        pos_ids, neg_ids, missing_skymap = _scan_has_optical(
            event_ids=sim_ids,
            skymap_dir=skymap_dir,
            sim_root=sim_root,
            sim_name=sim_name,
            scan_desc="Scanning BNS events",
        )

        neg_ids = _sample_event_ids(neg_ids, _normalize_max_count(max_neg_gw), rng)

        event_to_row = {int(eid): i for i, eid in enumerate(sim_ids)}
        return EventSelection(
            gw_df=gw_df,
            event_to_row=event_to_row,
            pos_event_ids=pos_ids,
            neg_event_ids=neg_ids,
            neg_type_by_event={int(eid): 2 for eid in neg_ids.tolist()},
            mej_by_event={int(eid): np.nan for eid in sim_ids.tolist()},
            n_neg_type1=0,
            n_neg_type2=len(neg_ids),
            n_missing_skymap=missing_skymap,
            n_filtered_non_success_mej_pos=0,
            n_invalid_mej=0,
        )

    # NSBH branch
    gw_df = _load_gw_catalog(
        full_catalog_path,
        success_ids_path=None,
        apply_success_filter=False,
    )

    sim_ids = gw_df["simulation_id"].to_numpy(np.int64)
    event_to_row = {int(eid): i for i, eid in enumerate(sim_ids)}

    mej_col = _resolve_nsbh_mej_col(gw_df, nsbh_mej_col)
    mej_values = pd.to_numeric(gw_df[mej_col], errors="coerce").to_numpy(np.float64)

    finite_mej = np.isfinite(mej_values)
    n_invalid_mej = int((~finite_mej).sum())

    type1_mask = finite_mej & (mej_values <= float(nsbh_type1_threshold))
    mej_pos_mask = finite_mej & (mej_values > float(nsbh_type1_threshold))

    type1_ids_all = sim_ids[type1_mask]
    mej_pos_ids_all = sim_ids[mej_pos_mask]

    success_ids: Optional[Set[int]] = None
    mej_pos_ids_filtered = mej_pos_ids_all
    n_filtered_non_success_mej_pos = 0

    if nsbh_require_success_for_mej_pos:
        if not success_ids_path:
            raise ValueError(
                "NSBH requires success filtering for mej>threshold, but --success_ids_path is empty."
            )
        success_ids = _load_success_ids(success_ids_path)
        keep_mask = np.fromiter(
            (int(eid) in success_ids for eid in mej_pos_ids_all),
            dtype=bool,
            count=len(mej_pos_ids_all),
        )
        mej_pos_ids_filtered = mej_pos_ids_all[keep_mask]
        n_filtered_non_success_mej_pos = int(len(mej_pos_ids_all) - len(mej_pos_ids_filtered))

    pos_ids, type2_ids, missing_skymap_mej_pos = _scan_has_optical(
        event_ids=mej_pos_ids_filtered,
        skymap_dir=skymap_dir,
        sim_root=sim_root,
        sim_name=sim_name,
        scan_desc="Scanning NSBH mej>threshold success-eligible events",
    )

    type1_ids, missing_skymap_type1 = _scan_skymap_only(
        event_ids=type1_ids_all,
        skymap_dir=skymap_dir,
        scan_desc="Scanning NSBH type1 negative candidates (mej<=threshold)",
    )

    sampled_type1 = _sample_event_ids(
        type1_ids,
        _normalize_max_count(nsbh_max_neg_type1_gw),
        rng,
    )
    sampled_type2 = _sample_event_ids(
        type2_ids,
        _normalize_max_count(nsbh_max_neg_type2_gw),
        rng,
    )

    sampled_type1, sampled_type2 = _cap_nsbh_neg_total(
        sampled_type1,
        sampled_type2,
        _normalize_max_count(max_neg_gw),
        rng,
    )

    neg_ids = np.concatenate([sampled_type1, sampled_type2], axis=0)
    neg_type_by_event: Dict[int, int] = {int(eid): 1 for eid in sampled_type1.tolist()}
    neg_type_by_event.update({int(eid): 2 for eid in sampled_type2.tolist()})

    mej_by_event = {
        int(sim_ids[i]): float(mej_values[i]) if np.isfinite(mej_values[i]) else np.nan
        for i in range(len(sim_ids))
    }

    print(
        "\n[NSBH] "
        f"catalog={len(sim_ids)} "
        f"mej_col={mej_col} "
        f"invalid_mej={n_invalid_mej} "
        f"type1_candidates={len(type1_ids_all)} "
        f"mej_pos_candidates={len(mej_pos_ids_all)} "
        f"filtered_non_success_mej_pos={n_filtered_non_success_mej_pos} "
        f"selected_pos={len(pos_ids)} "
        f"selected_neg_type1={len(sampled_type1)} "
        f"selected_neg_type2={len(sampled_type2)} "
        f"missing_skymap={missing_skymap_mej_pos + missing_skymap_type1}"
    )

    return EventSelection(
        gw_df=gw_df,
        event_to_row=event_to_row,
        pos_event_ids=pos_ids,
        neg_event_ids=neg_ids,
        neg_type_by_event=neg_type_by_event,
        mej_by_event=mej_by_event,
        n_neg_type1=len(sampled_type1),
        n_neg_type2=len(sampled_type2),
        n_missing_skymap=missing_skymap_mej_pos + missing_skymap_type1,
        n_filtered_non_success_mej_pos=n_filtered_non_success_mej_pos,
        n_invalid_mej=n_invalid_mej,
        nsbh_mej_col_resolved=mej_col,
        success_ids_mej_pos=success_ids,
    )


def build_pos_neg_indices(
    full_catalog_path: str,
    skymap_dir: str,
    sim_root: str,
    sim_name: str = "LSST_KN_BNS_AUG",
    success_ids_path: Optional[str] = None,
) -> Tuple[List[int], List[int]]:
    """
    Backward-compatible wrapper for old notebook usage.
    Returns row indices from the filtered catalog (BNS flow).
    """
    selection = build_event_selection(
        full_catalog_path=full_catalog_path,
        skymap_dir=skymap_dir,
        sim_root=sim_root,
        sim_name=sim_name,
        success_ids_path=success_ids_path,
        source_type="bns",
    )

    pos_indices = [selection.event_to_row[int(eid)] for eid in selection.pos_event_ids.tolist()]
    neg_indices = [selection.event_to_row[int(eid)] for eid in selection.neg_event_ids.tolist()]

    print(f"\nTotal BNS events in catalog: {len(selection.gw_df)}")
    print(f"Positive GW (with KN): {len(pos_indices)}")
    print(f"Negative GW (no KN but has skymap): {len(neg_indices)}")
    return pos_indices, neg_indices


def create_dataset_with_neg_gw_fast(
    full_catalog_path: str,
    skymap_dir: str,
    fits_dir: str,
    output_h5_path: str,
    sim_name: str = "LSST_KN_BNS_AUG",
    max_neg_gw: Optional[int] = None,
    buffer_limit: int = 10000,
    success_ids_path: Optional[str] = None,
    max_lc_per_gw: Optional[int] = 1000,
    seed: Optional[int] = None,
    source_type: str = "bns",
    nsbh_mej_col: str = "mej_tot",
    nsbh_type1_threshold: float = 0.0,
    nsbh_require_success_for_mej_pos: bool = True,
    nsbh_max_neg_type1_gw: Optional[int] = None,
    nsbh_max_neg_type2_gw: Optional[int] = None,
):
    if buffer_limit <= 0:
        raise ValueError("buffer_limit must be positive")

    source_type = source_type.strip().lower()
    if source_type not in {"bns", "nsbh"}:
        raise ValueError(f"source_type must be 'bns' or 'nsbh', got: {source_type}")

    selection = build_event_selection(
        full_catalog_path=full_catalog_path,
        skymap_dir=skymap_dir,
        sim_root=fits_dir,
        sim_name=sim_name,
        success_ids_path=success_ids_path,
        source_type=source_type,
        nsbh_mej_col=nsbh_mej_col,
        nsbh_type1_threshold=nsbh_type1_threshold,
        nsbh_require_success_for_mej_pos=nsbh_require_success_for_mej_pos,
        max_neg_gw=max_neg_gw,
        nsbh_max_neg_type1_gw=nsbh_max_neg_type1_gw,
        nsbh_max_neg_type2_gw=nsbh_max_neg_type2_gw,
        seed=seed,
    )

    gw_df = selection.gw_df.copy()
    missing_cols = [col for col in GW_PARAM_COLUMNS if col not in gw_df.columns]
    if missing_cols:
        raise ValueError(f"Missing required GW columns in catalog: {missing_cols}")

    event_time_mjd, event_time_col, event_time_from_gps, n_invalid_event_time_catalog = _extract_event_time_mjd(
        gw_df
    )
    gw_df["inclination"] = np.cos(gw_df["inclination"])
    gw_df["distmean"] = gw_df["distmean"] / 1000.0
    gw_df["diststd"] = gw_df["diststd"] / 1000.0

    gw_params = gw_df[GW_PARAM_COLUMNS].to_numpy(np.float32)
    rng = np.random.default_rng(seed)

    pos_event_ids = selection.pos_event_ids
    neg_event_ids = selection.neg_event_ids
    n_expected_gw = len(pos_event_ids) + len(neg_event_ids)

    print("\nCreating dataset with:")
    print(f"  Source type: {source_type}")
    print(f"  Positive GW (with KN): {len(pos_event_ids)}")
    print(f"  Negative GW (no KN): {len(neg_event_ids)}")
    if source_type == "nsbh":
        print(
            "  NSBH neg types: "
            f"type1={selection.n_neg_type1}, type2={selection.n_neg_type2}"
        )
    print(
        "  Event time source: "
        f"{event_time_col} (gps_fallback={int(event_time_from_gps)}), "
        f"invalid_in_catalog={n_invalid_event_time_catalog}"
    )
    print(f"  Expected total GW events: {n_expected_gw}")

    dt_str = h5py.string_dtype(encoding="utf-8")
    gw_chunk = max(1, min(256, max(1, n_expected_gw)))
    skymap_chunk = (1, 7, 19200)

    os.makedirs(os.path.dirname(os.path.abspath(output_h5_path)), exist_ok=True)
    with h5py.File(output_h5_path, "w") as f:
        grp_gw = f.create_group("events/gw_data")
        ds_gw_scalars = grp_gw.create_dataset(
            "scalars",
            (0, 7),
            maxshape=(None, 7),
            dtype="f4",
            chunks=(gw_chunk, 7),
        )
        ds_gw_skymaps = grp_gw.create_dataset(
            "skymaps",
            (0, 7, 19200),
            maxshape=(None, 7, 19200),
            dtype="f4",
            chunks=skymap_chunk,
        )
        ds_gw_ids = grp_gw.create_dataset(
            "ids",
            (0,),
            maxshape=(None,),
            dtype=dt_str,
            chunks=(gw_chunk,),
        )
        ds_gw_has_kn = grp_gw.create_dataset(
            "has_kn",
            (0,),
            maxshape=(None,),
            dtype="i4",
            chunks=(gw_chunk,),
        )
        ds_gw_neg_type = grp_gw.create_dataset(
            "neg_type",
            (0,),
            maxshape=(None,),
            dtype="i4",
            chunks=(gw_chunk,),
        )
        ds_gw_mej_tot = grp_gw.create_dataset(
            "mej_tot",
            (0,),
            maxshape=(None,),
            dtype="f4",
            chunks=(gw_chunk,),
        )
        ds_gw_event_time_mjd = grp_gw.create_dataset(
            "event_time_mjd",
            (0,),
            maxshape=(None,),
            dtype="f8",
            chunks=(gw_chunk,),
        )

        grp_opt = f.create_group("events/optical_data")
        chunk_size = 1024
        ds_opt_vals = grp_opt.create_dataset(
            "values",
            (0, MAX_LC_LENGTH, NUM_BANDS),
            maxshape=(None, MAX_LC_LENGTH, NUM_BANDS),
            dtype="f4",
            chunks=(chunk_size, MAX_LC_LENGTH, NUM_BANDS),
        )
        ds_opt_errs = grp_opt.create_dataset(
            "errors",
            (0, MAX_LC_LENGTH, NUM_BANDS),
            maxshape=(None, MAX_LC_LENGTH, NUM_BANDS),
            dtype="f4",
            chunks=(chunk_size, MAX_LC_LENGTH, NUM_BANDS),
        )
        ds_opt_masks = grp_opt.create_dataset(
            "masks",
            (0, MAX_LC_LENGTH, NUM_BANDS),
            maxshape=(None, MAX_LC_LENGTH, NUM_BANDS),
            dtype="f4",
            chunks=(chunk_size, MAX_LC_LENGTH, NUM_BANDS),
        )
        ds_opt_times = grp_opt.create_dataset(
            "times",
            (0, MAX_LC_LENGTH),
            maxshape=(None, MAX_LC_LENGTH),
            dtype="f4",
            chunks=(chunk_size, MAX_LC_LENGTH),
        )
        ds_opt_zero_time_mjd_base = grp_opt.create_dataset(
            "zero_time_mjd_base",
            (0,),
            maxshape=(None,),
            dtype="f8",
            chunks=(chunk_size,),
        )
        ds_opt_coordinates = grp_opt.create_dataset(
            "coordinates",
            (0, 2),
            maxshape=(None, 2),
            dtype="f4",
            chunks=(chunk_size, 2),
        )
        ds_parent_idx = grp_opt.create_dataset(
            "parent_gw_idx",
            (0,),
            maxshape=(None,),
            dtype="i4",
            chunks=(chunk_size,),
        )

        gw_count = 0
        n_written_pos = 0
        n_written_neg = 0
        n_written_neg_type1 = 0
        n_written_neg_type2 = 0
        n_invalid_event_time_written = 0

        n_drop_empty_lc = 0
        n_drop_skymap = 0

        total_optical_count = 0
        opt_buffer_vals: List[np.ndarray] = []
        opt_buffer_errs: List[np.ndarray] = []
        opt_buffer_masks: List[np.ndarray] = []
        opt_buffer_times: List[np.ndarray] = []
        opt_buffer_zero_time_mjd_base: List[float] = []
        opt_buffer_p_idx: List[int] = []
        opt_buffer_coordinates: List[np.ndarray] = []
        written_id_set: Set[str] = set()

        def resize_gw(new_size: int) -> None:
            ds_gw_scalars.resize(new_size, axis=0)
            ds_gw_skymaps.resize(new_size, axis=0)
            ds_gw_ids.resize(new_size, axis=0)
            ds_gw_has_kn.resize(new_size, axis=0)
            ds_gw_neg_type.resize(new_size, axis=0)
            ds_gw_mej_tot.resize(new_size, axis=0)
            ds_gw_event_time_mjd.resize(new_size, axis=0)

        def append_gw(
            scalar: np.ndarray,
            skymap: np.ndarray,
            event_id: int,
            has_kn: int,
            neg_type: int,
            mej_tot: float,
            event_time_mjd: float,
        ) -> int:
            nonlocal gw_count
            resize_gw(gw_count + 1)
            ds_gw_scalars[gw_count] = scalar
            ds_gw_skymaps[gw_count] = skymap
            ds_gw_ids[gw_count] = str(event_id)
            ds_gw_has_kn[gw_count] = int(has_kn)
            ds_gw_neg_type[gw_count] = int(neg_type)
            ds_gw_mej_tot[gw_count] = np.float32(mej_tot)
            ds_gw_event_time_mjd[gw_count] = np.float64(event_time_mjd)
            gw_idx = gw_count
            gw_count += 1
            return gw_idx

        def flush_buffer() -> None:
            nonlocal total_optical_count
            if not opt_buffer_vals:
                return

            n_new = len(opt_buffer_vals)
            current_size = total_optical_count
            new_size = current_size + n_new

            ds_opt_vals.resize(new_size, axis=0)
            ds_opt_errs.resize(new_size, axis=0)
            ds_opt_masks.resize(new_size, axis=0)
            ds_opt_times.resize(new_size, axis=0)
            ds_opt_zero_time_mjd_base.resize(new_size, axis=0)
            ds_parent_idx.resize(new_size, axis=0)
            ds_opt_coordinates.resize(new_size, axis=0)

            ds_opt_vals[current_size:new_size] = np.asarray(opt_buffer_vals)
            ds_opt_errs[current_size:new_size] = np.asarray(opt_buffer_errs)
            ds_opt_masks[current_size:new_size] = np.asarray(opt_buffer_masks)
            ds_opt_times[current_size:new_size] = np.asarray(opt_buffer_times)
            ds_opt_zero_time_mjd_base[current_size:new_size] = np.asarray(
                opt_buffer_zero_time_mjd_base, dtype=np.float64
            )
            ds_parent_idx[current_size:new_size] = np.asarray(opt_buffer_p_idx, dtype=np.int32)
            ds_opt_coordinates[current_size:new_size] = np.asarray(opt_buffer_coordinates)

            total_optical_count += n_new
            opt_buffer_vals.clear()
            opt_buffer_errs.clear()
            opt_buffer_masks.clear()
            opt_buffer_times.clear()
            opt_buffer_zero_time_mjd_base.clear()
            opt_buffer_p_idx.clear()
            opt_buffer_coordinates.clear()

        print("\nProcessing positive GW events (with KN)...")
        max_lc = _normalize_max_count(max_lc_per_gw)
        for event_id in tqdm(pos_event_ids, desc="Positive GW", mininterval=0.5, miniters=200):
            event_id = int(event_id)
            row_idx = selection.event_to_row.get(event_id)
            if row_idx is None:
                continue

            lcs = parse_snana_fits(event_id, sim_dir=fits_dir, sim_name=sim_name)
            if max_lc is not None and len(lcs) > max_lc:
                keep_idx = rng.choice(len(lcs), size=max_lc, replace=False)
                lcs = [lcs[int(i)] for i in keep_idx]

            if len(lcs) == 0:
                n_drop_empty_lc += 1
                continue

            skymap_path = os.path.join(skymap_dir, f"{event_id}.fits")
            try:
                skymap = sample_moc_skymap(skymap_path).numpy()
            except Exception:
                n_drop_skymap += 1
                continue

            gw_id = str(event_id)
            if gw_id in written_id_set:
                continue
            written_id_set.add(gw_id)

            mej_val = float(selection.mej_by_event.get(event_id, np.nan))
            event_time_val = float(event_time_mjd[row_idx])
            if (
                source_type == "nsbh"
                and nsbh_require_success_for_mej_pos
                and np.isfinite(mej_val)
                and mej_val > nsbh_type1_threshold
                and selection.success_ids_mej_pos is not None
                and event_id not in selection.success_ids_mej_pos
            ):
                raise RuntimeError(
                    f"NSBH mej>threshold event {event_id} passed into output but not in success ids."
                )

            gw_idx = append_gw(
                scalar=gw_params[row_idx],
                skymap=skymap,
                event_id=event_id,
                has_kn=1,
                neg_type=0,
                mej_tot=mej_val,
                event_time_mjd=event_time_val,
            )
            n_written_pos += 1
            if not np.isfinite(event_time_val):
                n_invalid_event_time_written += 1

            for vals, errs, masks, times, coordinates in lcs:
                opt_buffer_vals.append(vals)
                opt_buffer_errs.append(errs)
                opt_buffer_masks.append(masks)
                opt_buffer_times.append(times)
                opt_buffer_zero_time_mjd_base.append(event_time_val)
                opt_buffer_p_idx.append(gw_idx)
                opt_buffer_coordinates.append(coordinates)

            if len(opt_buffer_vals) >= buffer_limit:
                flush_buffer()

        flush_buffer()

        print("\nProcessing negative GW events (no KN)...")
        for event_id in tqdm(neg_event_ids, desc="Negative GW", mininterval=0.5, miniters=200):
            event_id = int(event_id)
            row_idx = selection.event_to_row.get(event_id)
            if row_idx is None:
                continue

            skymap_path = os.path.join(skymap_dir, f"{event_id}.fits")
            try:
                skymap = sample_moc_skymap(skymap_path).numpy()
            except Exception:
                n_drop_skymap += 1
                continue

            gw_id = str(event_id)
            if gw_id in written_id_set:
                continue
            written_id_set.add(gw_id)

            neg_type = int(selection.neg_type_by_event.get(event_id, 2))
            mej_val = float(selection.mej_by_event.get(event_id, np.nan))
            event_time_val = float(event_time_mjd[row_idx])
            if (
                source_type == "nsbh"
                and nsbh_require_success_for_mej_pos
                and np.isfinite(mej_val)
                and mej_val > nsbh_type1_threshold
                and selection.success_ids_mej_pos is not None
                and event_id not in selection.success_ids_mej_pos
            ):
                raise RuntimeError(
                    f"NSBH mej>threshold event {event_id} passed into output but not in success ids."
                )

            append_gw(
                scalar=gw_params[row_idx],
                skymap=skymap,
                event_id=event_id,
                has_kn=0,
                neg_type=neg_type,
                mej_tot=mej_val,
                event_time_mjd=event_time_val,
            )
            n_written_neg += 1
            if not np.isfinite(event_time_val):
                n_invalid_event_time_written += 1
            if neg_type == 1:
                n_written_neg_type1 += 1
            elif neg_type == 2:
                n_written_neg_type2 += 1

        f.attrs["source_type"] = source_type
        f.attrs["n_pos_gw"] = int(n_written_pos)
        f.attrs["n_neg_gw"] = int(n_written_neg)
        f.attrs["n_total_gw"] = int(gw_count)
        f.attrs["n_total_optical"] = int(total_optical_count)

        f.attrs["n_neg_type1_gw"] = int(n_written_neg_type1)
        f.attrs["n_neg_type2_gw"] = int(n_written_neg_type2)

        f.attrs["n_missing_skymap_scan"] = int(selection.n_missing_skymap)
        f.attrs["n_dropped_empty_lc_pos"] = int(n_drop_empty_lc)
        f.attrs["n_dropped_skymap_write"] = int(n_drop_skymap)

        f.attrs["n_filtered_non_success_mej_pos"] = int(selection.n_filtered_non_success_mej_pos)
        f.attrs["n_invalid_mej"] = int(selection.n_invalid_mej)
        f.attrs["nsbh_mej_col"] = selection.nsbh_mej_col_resolved or nsbh_mej_col
        f.attrs["nsbh_type1_threshold"] = float(nsbh_type1_threshold)
        f.attrs["nsbh_require_success_for_mej_pos"] = int(nsbh_require_success_for_mej_pos)
        f.attrs["event_time_col"] = event_time_col
        f.attrs["event_time_from_gps"] = int(event_time_from_gps)
        f.attrs["n_invalid_event_time_catalog"] = int(n_invalid_event_time_catalog)
        f.attrs["n_invalid_event_time_written"] = int(n_invalid_event_time_written)
        f.attrs["time_zero_base_semantics"] = "optical zero_time_mjd_base stores parent GW event_time_mjd"
        f.attrs["time_unit"] = "mjd_days"
        f.attrs["runtime_offset_applied"] = 1

        print("\nProcessing Complete.")
        print(f"  Source type: {source_type}")
        print(f"  Positive GW Events: {n_written_pos}")
        print(f"  Negative GW Events: {n_written_neg}")
        print(f"  Total GW Events: {gw_count}")
        print(f"  Total Light Curves: {total_optical_count}")
        print(f"  Negative type1/type2: {n_written_neg_type1}/{n_written_neg_type2}")
        print(
            "  Event time source: "
            f"{event_time_col} (gps_fallback={int(event_time_from_gps)})"
        )
        print(
            "  Invalid event_time_mjd (catalog/written): "
            f"{n_invalid_event_time_catalog}/{n_invalid_event_time_written}"
        )
        if source_type == "nsbh":
            print(
                "  NSBH filtered mej>threshold non-success events: "
                f"{selection.n_filtered_non_success_mej_pos}"
            )
        print(f"  Saved to: {output_h5_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Create GW dataset with negative events (fast).")
    parser.add_argument("--full_catalog_path", required=True)
    parser.add_argument("--skymap_dir", required=True)
    parser.add_argument("--sim_root", required=True)
    parser.add_argument("--output_h5_path", required=True)
    parser.add_argument("--sim_name", default="LSST_KN_BNS_AUG")
    parser.add_argument("--success_ids_path", default=None)
    parser.add_argument("--max_neg_gw", type=int, default=None)
    parser.add_argument("--buffer_limit", type=int, default=10000)
    parser.add_argument("--max_lc_per_gw", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=None)

    parser.add_argument("--source_type", choices=["bns", "nsbh"], default="bns")
    parser.add_argument("--nsbh_mej_col", default="mej_tot")
    parser.add_argument("--nsbh_type1_threshold", type=float, default=0.0)
    parser.add_argument(
        "--nsbh_require_success_for_mej_pos",
        type=int,
        choices=[0, 1],
        default=1,
        help="1: require mej>threshold NSBH events in success ids; 0: disable success filter.",
    )
    parser.add_argument("--nsbh_max_neg_type1_gw", type=int, default=None)
    parser.add_argument("--nsbh_max_neg_type2_gw", type=int, default=None)

    args = parser.parse_args()

    create_dataset_with_neg_gw_fast(
        full_catalog_path=args.full_catalog_path,
        skymap_dir=args.skymap_dir,
        fits_dir=args.sim_root,
        output_h5_path=args.output_h5_path,
        sim_name=args.sim_name,
        success_ids_path=args.success_ids_path,
        max_neg_gw=args.max_neg_gw,
        buffer_limit=args.buffer_limit,
        max_lc_per_gw=args.max_lc_per_gw,
        seed=args.seed,
        source_type=args.source_type,
        nsbh_mej_col=args.nsbh_mej_col,
        nsbh_type1_threshold=args.nsbh_type1_threshold,
        nsbh_require_success_for_mej_pos=bool(args.nsbh_require_success_for_mej_pos),
        nsbh_max_neg_type1_gw=args.nsbh_max_neg_type1_gw,
        nsbh_max_neg_type2_gw=args.nsbh_max_neg_type2_gw,
    )
