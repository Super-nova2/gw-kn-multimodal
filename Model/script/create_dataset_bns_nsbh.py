import os
import importlib.util
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

import h5py
import numpy as np
import pandas as pd
from astropy.io import fits
from tqdm import tqdm

# Load required functions/constants from data_loader.py (not a standard package due to '+' in path)
_DATA_LOADER_PATH = "/fred/oz016/bgao_kn/ML+GW+KN/Model/data_loader.py"
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


@dataclass
class SourceConfig:
    tag: str
    full_catalog_path: str
    skymap_dir: str
    sim_root: str
    sim_name: str
    success_ids_path: Optional[str] = None
    max_lc_per_gw: Optional[int] = 1000
    max_neg_gw: Optional[int] = None
    max_pos_gw: Optional[int] = None
    # NSBH-only controls
    mej_col: str = "mej_tot"
    type1_threshold: float = 0.0
    require_success_for_mej_pos: bool = False
    max_neg_type1_gw: Optional[int] = None
    max_neg_type2_gw: Optional[int] = None


@dataclass
class SourcePrepared:
    cfg: SourceConfig
    gw_df: pd.DataFrame
    gw_params: np.ndarray
    sim_ids: np.ndarray
    event_to_row: Dict[int, int]
    pos_event_ids: np.ndarray
    neg_event_ids: np.ndarray
    neg_type_by_event: Dict[int, int]
    mej_by_event: Dict[int, float]
    n_missing_skymap: int
    n_filtered_non_success_mej_pos: int = 0
    nsbh_mej_col_resolved: Optional[str] = None
    success_ids_mej_pos: Optional[Set[int]] = None


def _load_success_ids(success_ids_path: str) -> Set[int]:
    with open(success_ids_path, "r", encoding="utf-8") as f:
        return {int(line.strip()) for line in f if line.strip()}


def _load_gw_catalog(
    full_catalog_path: str,
    success_ids_path: Optional[str] = None,
) -> pd.DataFrame:
    gw_df = pd.read_csv(full_catalog_path)
    if "simulation_id" not in gw_df.columns:
        raise ValueError(f"'simulation_id' column not found in {full_catalog_path}")

    gw_df = gw_df.copy()
    gw_df["simulation_id"] = gw_df["simulation_id"].astype(int)

    if success_ids_path:
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


def _resolve_nsbh_mej_col(gw_df: pd.DataFrame, requested_col: str) -> str:
    if requested_col in gw_df.columns:
        return requested_col
    if requested_col == "mej_tot" and "mej_total" in gw_df.columns:
        return "mej_total"
    # Extra fallback for robustness if caller passes unexpected name.
    if "mej_tot" in gw_df.columns:
        return "mej_tot"
    if "mej_total" in gw_df.columns:
        return "mej_total"
    raise ValueError(
        f"NSBH catalog missing mej column. Tried '{requested_col}', 'mej_tot', 'mej_total'."
    )


def _scan_source_pos_neg(
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


def _scan_ids_with_skymap_only(
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


def _sample_event_ids(
    event_ids: np.ndarray,
    max_count: Optional[int],
    rng: np.random.Generator,
) -> np.ndarray:
    if max_count is None or len(event_ids) <= max_count:
        return event_ids
    chosen = rng.choice(event_ids, size=max_count, replace=False)
    return np.sort(chosen.astype(np.int64))


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


def _prepare_bns_source(
    cfg: SourceConfig,
    dataset_mode: str,
    rng: np.random.Generator,
) -> SourcePrepared:
    gw_df = _load_gw_catalog(cfg.full_catalog_path, cfg.success_ids_path)

    missing_cols = [col for col in GW_PARAM_COLUMNS if col not in gw_df.columns]
    if missing_cols:
        raise ValueError(
            f"{cfg.tag}: missing required columns {missing_cols} in {cfg.full_catalog_path}"
        )

    gw_df = gw_df.copy()
    gw_df["inclination"] = np.cos(gw_df["inclination"])
    gw_df["distmean"] = gw_df["distmean"] / 1000.0
    gw_df["diststd"] = gw_df["diststd"] / 1000.0

    sim_ids = gw_df["simulation_id"].to_numpy(np.int64)

    pos_ids, neg_ids, missing_skymap = _scan_source_pos_neg(
        event_ids=sim_ids,
        skymap_dir=cfg.skymap_dir,
        sim_root=cfg.sim_root,
        sim_name=cfg.sim_name,
        scan_desc=f"Scanning {cfg.tag.upper()} events",
    )

    max_pos = _normalize_max_count(cfg.max_pos_gw)
    max_neg = _normalize_max_count(cfg.max_neg_gw)

    pos_ids = _sample_event_ids(pos_ids, max_pos, rng)
    if dataset_mode == "test":
        neg_ids = _sample_event_ids(neg_ids, max_neg, rng)
    else:
        neg_ids = np.empty((0,), dtype=np.int64)

    event_to_row: Dict[int, int] = {int(eid): i for i, eid in enumerate(sim_ids)}
    gw_params = gw_df[GW_PARAM_COLUMNS].to_numpy(np.float32)
    neg_type_by_event = {int(eid): 2 for eid in neg_ids.tolist()}
    mej_by_event = {int(eid): np.nan for eid in sim_ids.tolist()}

    print(
        f"\n[{cfg.tag}] catalog={len(sim_ids)} "
        f"missing_skymap={missing_skymap} "
        f"selected_pos={len(pos_ids)} selected_neg={len(neg_ids)}"
    )

    return SourcePrepared(
        cfg=cfg,
        gw_df=gw_df,
        gw_params=gw_params,
        sim_ids=sim_ids,
        event_to_row=event_to_row,
        pos_event_ids=pos_ids,
        neg_event_ids=neg_ids,
        neg_type_by_event=neg_type_by_event,
        mej_by_event=mej_by_event,
        n_missing_skymap=missing_skymap,
    )


def _prepare_nsbh_source(
    cfg: SourceConfig,
    dataset_mode: str,
    rng: np.random.Generator,
) -> SourcePrepared:
    gw_df = _load_gw_catalog(cfg.full_catalog_path, success_ids_path=None)

    missing_cols = [col for col in GW_PARAM_COLUMNS if col not in gw_df.columns]
    if missing_cols:
        raise ValueError(
            f"{cfg.tag}: missing required columns {missing_cols} in {cfg.full_catalog_path}"
        )

    gw_df = gw_df.copy()
    gw_df["inclination"] = np.cos(gw_df["inclination"])
    gw_df["distmean"] = gw_df["distmean"] / 1000.0
    gw_df["diststd"] = gw_df["diststd"] / 1000.0

    mej_col = _resolve_nsbh_mej_col(gw_df, cfg.mej_col)
    mej_values = pd.to_numeric(gw_df[mej_col], errors="coerce").to_numpy(np.float64)

    sim_ids = gw_df["simulation_id"].to_numpy(np.int64)
    event_to_row: Dict[int, int] = {int(eid): i for i, eid in enumerate(sim_ids)}
    gw_params = gw_df[GW_PARAM_COLUMNS].to_numpy(np.float32)
    mej_by_event = {
        int(sim_ids[i]): float(mej_values[i]) if np.isfinite(mej_values[i]) else np.nan
        for i in range(len(sim_ids))
    }

    type1_mask = np.isfinite(mej_values) & (mej_values <= float(cfg.type1_threshold))
    mej_pos_mask = np.isfinite(mej_values) & (mej_values > float(cfg.type1_threshold))
    type1_ids_all = sim_ids[type1_mask]
    mej_pos_ids_all = sim_ids[mej_pos_mask]

    success_ids: Optional[Set[int]] = None
    mej_pos_ids_filtered = mej_pos_ids_all
    n_filtered_non_success = 0
    if cfg.require_success_for_mej_pos:
        if not cfg.success_ids_path:
            raise ValueError(
                "NSBH requires success filtering for mej>threshold, but --nsbh_success_ids_path is empty."
            )
        success_ids = _load_success_ids(cfg.success_ids_path)
        keep_mask = np.fromiter(
            (int(eid) in success_ids for eid in mej_pos_ids_all),
            dtype=bool,
            count=len(mej_pos_ids_all),
        )
        mej_pos_ids_filtered = mej_pos_ids_all[keep_mask]
        n_filtered_non_success = int(len(mej_pos_ids_all) - len(mej_pos_ids_filtered))

    pos_ids, type2_ids, missing_skymap_mej_pos = _scan_source_pos_neg(
        event_ids=mej_pos_ids_filtered,
        skymap_dir=cfg.skymap_dir,
        sim_root=cfg.sim_root,
        sim_name=cfg.sim_name,
        scan_desc="Scanning NSBH mej>threshold success-eligible events",
    )

    type1_ids = np.empty((0,), dtype=np.int64)
    missing_skymap_type1 = 0
    if dataset_mode == "test":
        type1_ids, missing_skymap_type1 = _scan_ids_with_skymap_only(
            event_ids=type1_ids_all,
            skymap_dir=cfg.skymap_dir,
            scan_desc="Scanning NSBH type1 negative candidates (mej<=threshold)",
        )

    max_pos = _normalize_max_count(cfg.max_pos_gw)
    pos_ids = _sample_event_ids(pos_ids, max_pos, rng)

    sampled_type1 = np.empty((0,), dtype=np.int64)
    sampled_type2 = np.empty((0,), dtype=np.int64)
    if dataset_mode == "test":
        sampled_type1 = _sample_event_ids(
            type1_ids,
            _normalize_max_count(cfg.max_neg_type1_gw),
            rng,
        )
        sampled_type2 = _sample_event_ids(
            type2_ids,
            _normalize_max_count(cfg.max_neg_type2_gw),
            rng,
        )
        sampled_type1, sampled_type2 = _cap_nsbh_neg_total(
            sampled_type1,
            sampled_type2,
            _normalize_max_count(cfg.max_neg_gw),
            rng,
        )

    neg_ids = np.concatenate([sampled_type1, sampled_type2], axis=0)
    neg_type_by_event: Dict[int, int] = {int(eid): 1 for eid in sampled_type1.tolist()}
    neg_type_by_event.update({int(eid): 2 for eid in sampled_type2.tolist()})

    print(
        f"\n[{cfg.tag}] catalog={len(sim_ids)} "
        f"mej_col={mej_col} "
        f"type1_candidates={len(type1_ids_all)} "
        f"mej_pos_candidates={len(mej_pos_ids_all)} "
        f"filtered_non_success_mej_pos={n_filtered_non_success} "
        f"selected_pos={len(pos_ids)} "
        f"selected_neg_type1={len(sampled_type1)} "
        f"selected_neg_type2={len(sampled_type2)} "
        f"missing_skymap={missing_skymap_mej_pos + missing_skymap_type1}"
    )

    return SourcePrepared(
        cfg=cfg,
        gw_df=gw_df,
        gw_params=gw_params,
        sim_ids=sim_ids,
        event_to_row=event_to_row,
        pos_event_ids=pos_ids,
        neg_event_ids=neg_ids,
        neg_type_by_event=neg_type_by_event,
        mej_by_event=mej_by_event,
        n_missing_skymap=missing_skymap_mej_pos + missing_skymap_type1,
        n_filtered_non_success_mej_pos=n_filtered_non_success,
        nsbh_mej_col_resolved=mej_col,
        success_ids_mej_pos=success_ids,
    )


def create_dataset_with_neg_gw_bns_nsbh_fast(
    output_h5_path: str,
    dataset_mode: str,
    bns_cfg: SourceConfig,
    nsbh_cfg: SourceConfig,
    buffer_limit: int = 10000,
    seed: int = 42,
):
    dataset_mode = dataset_mode.strip().lower()
    if dataset_mode not in {"train", "test"}:
        raise ValueError(f"dataset_mode must be 'train' or 'test', got: {dataset_mode}")
    if buffer_limit <= 0:
        raise ValueError("buffer_limit must be positive")

    rng = np.random.default_rng(seed)
    bns_rng = np.random.default_rng(int(rng.integers(0, 2**31 - 1)))
    nsbh_rng = np.random.default_rng(int(rng.integers(0, 2**31 - 1)))

    prepared: Dict[str, SourcePrepared] = {
        "bns": _prepare_bns_source(bns_cfg, dataset_mode, bns_rng),
        "nsbh": _prepare_nsbh_source(nsbh_cfg, dataset_mode, nsbh_rng),
    }

    pos_events: List[Tuple[str, int]] = (
        [("bns", int(eid)) for eid in prepared["bns"].pos_event_ids.tolist()]
        + [("nsbh", int(eid)) for eid in prepared["nsbh"].pos_event_ids.tolist()]
    )
    neg_events: List[Tuple[str, int]] = []
    if dataset_mode == "test":
        neg_events = (
            [("bns", int(eid)) for eid in prepared["bns"].neg_event_ids.tolist()]
            + [("nsbh", int(eid)) for eid in prepared["nsbh"].neg_event_ids.tolist()]
        )

    n_expected_gw = len(pos_events) + len(neg_events)
    print(
        f"\nCreating combined dataset ({dataset_mode} mode): "
        f"expected_pos={len(pos_events)} expected_neg={len(neg_events)} "
        f"expected_total={n_expected_gw}"
    )

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
        ds_gw_source_type = grp_gw.create_dataset(
            "source_type",
            (0,),
            maxshape=(None,),
            dtype=dt_str,
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
        total_optical_count = 0
        written_id_set: Set[str] = set()

        source_counts = {
            "bns": {
                "pos": 0,
                "neg": 0,
                "neg_type1": 0,
                "neg_type2": 0,
                "drop_empty_lc": 0,
                "drop_skymap": 0,
            },
            "nsbh": {
                "pos": 0,
                "neg": 0,
                "neg_type1": 0,
                "neg_type2": 0,
                "drop_empty_lc": 0,
                "drop_skymap": 0,
            },
        }

        opt_buffer_vals: List[np.ndarray] = []
        opt_buffer_errs: List[np.ndarray] = []
        opt_buffer_masks: List[np.ndarray] = []
        opt_buffer_times: List[np.ndarray] = []
        opt_buffer_parent_idx: List[int] = []
        opt_buffer_coordinates: List[np.ndarray] = []

        def resize_gw(new_size: int) -> None:
            ds_gw_scalars.resize(new_size, axis=0)
            ds_gw_skymaps.resize(new_size, axis=0)
            ds_gw_ids.resize(new_size, axis=0)
            ds_gw_has_kn.resize(new_size, axis=0)
            ds_gw_neg_type.resize(new_size, axis=0)
            ds_gw_mej_tot.resize(new_size, axis=0)
            ds_gw_source_type.resize(new_size, axis=0)

        def append_gw(
            scalar: np.ndarray,
            skymap: np.ndarray,
            gw_id: str,
            has_kn: int,
            neg_type: int,
            mej_tot: float,
            source_type: str,
        ) -> int:
            nonlocal gw_count
            resize_gw(gw_count + 1)
            ds_gw_scalars[gw_count] = scalar
            ds_gw_skymaps[gw_count] = skymap
            ds_gw_ids[gw_count] = gw_id
            ds_gw_has_kn[gw_count] = int(has_kn)
            ds_gw_neg_type[gw_count] = int(neg_type)
            ds_gw_mej_tot[gw_count] = np.float32(mej_tot)
            ds_gw_source_type[gw_count] = source_type
            gw_idx = gw_count
            gw_count += 1
            return gw_idx

        def flush_opt_buffer() -> None:
            nonlocal total_optical_count
            if not opt_buffer_vals:
                return
            n_new = len(opt_buffer_vals)
            cur = total_optical_count
            new_size = cur + n_new

            ds_opt_vals.resize(new_size, axis=0)
            ds_opt_errs.resize(new_size, axis=0)
            ds_opt_masks.resize(new_size, axis=0)
            ds_opt_times.resize(new_size, axis=0)
            ds_opt_coordinates.resize(new_size, axis=0)
            ds_parent_idx.resize(new_size, axis=0)

            ds_opt_vals[cur:new_size] = np.asarray(opt_buffer_vals)
            ds_opt_errs[cur:new_size] = np.asarray(opt_buffer_errs)
            ds_opt_masks[cur:new_size] = np.asarray(opt_buffer_masks)
            ds_opt_times[cur:new_size] = np.asarray(opt_buffer_times)
            ds_opt_coordinates[cur:new_size] = np.asarray(opt_buffer_coordinates)
            ds_parent_idx[cur:new_size] = np.asarray(opt_buffer_parent_idx, dtype=np.int32)

            total_optical_count += n_new
            opt_buffer_vals.clear()
            opt_buffer_errs.clear()
            opt_buffer_masks.clear()
            opt_buffer_times.clear()
            opt_buffer_coordinates.clear()
            opt_buffer_parent_idx.clear()

        print("\nWriting positive GW events...")
        for tag, event_id in tqdm(pos_events, desc="Positive GW", mininterval=0.5, miniters=100):
            src = prepared[tag]
            row_idx = src.event_to_row.get(event_id)
            if row_idx is None:
                continue

            lcs = parse_snana_fits(
                event_id=event_id,
                sim_dir=src.cfg.sim_root,
                sim_name=src.cfg.sim_name,
            )
            max_lc = _normalize_max_count(src.cfg.max_lc_per_gw)
            if max_lc is not None and len(lcs) > max_lc:
                keep_idx = rng.choice(len(lcs), size=max_lc, replace=False)
                lcs = [lcs[int(i)] for i in keep_idx]

            if len(lcs) == 0:
                source_counts[tag]["drop_empty_lc"] += 1
                continue

            skymap_path = os.path.join(src.cfg.skymap_dir, f"{event_id}.fits")
            try:
                skymap = sample_moc_skymap(skymap_path).numpy()
            except Exception:
                source_counts[tag]["drop_skymap"] += 1
                continue

            gw_id = f"{tag}_{event_id}"
            if gw_id in written_id_set:
                # Keep strict uniqueness guarantee for ids.
                continue
            written_id_set.add(gw_id)

            mej_val = float(src.mej_by_event.get(event_id, np.nan))
            if (
                tag == "nsbh"
                and nsbh_cfg.require_success_for_mej_pos
                and np.isfinite(mej_val)
                and mej_val > nsbh_cfg.type1_threshold
                and src.success_ids_mej_pos is not None
                and event_id not in src.success_ids_mej_pos
            ):
                raise RuntimeError(
                    f"NSBH mej>threshold event {event_id} passed into output but not in success ids."
                )

            gw_idx = append_gw(
                scalar=src.gw_params[row_idx],
                skymap=skymap,
                gw_id=gw_id,
                has_kn=1,
                neg_type=0,
                mej_tot=mej_val,
                source_type=tag,
            )
            source_counts[tag]["pos"] += 1

            for vals, errs, masks, times, coordinates in lcs:
                opt_buffer_vals.append(vals)
                opt_buffer_errs.append(errs)
                opt_buffer_masks.append(masks)
                opt_buffer_times.append(times)
                opt_buffer_coordinates.append(coordinates)
                opt_buffer_parent_idx.append(gw_idx)

            if len(opt_buffer_vals) >= buffer_limit:
                flush_opt_buffer()

        flush_opt_buffer()

        if dataset_mode == "test":
            print("\nWriting negative GW events...")
            for tag, event_id in tqdm(neg_events, desc="Negative GW", mininterval=0.5, miniters=100):
                src = prepared[tag]
                row_idx = src.event_to_row.get(event_id)
                if row_idx is None:
                    continue

                skymap_path = os.path.join(src.cfg.skymap_dir, f"{event_id}.fits")
                try:
                    skymap = sample_moc_skymap(skymap_path).numpy()
                except Exception:
                    source_counts[tag]["drop_skymap"] += 1
                    continue

                gw_id = f"{tag}_{event_id}"
                if gw_id in written_id_set:
                    continue
                written_id_set.add(gw_id)

                neg_type = int(src.neg_type_by_event.get(event_id, 2))
                mej_val = float(src.mej_by_event.get(event_id, np.nan))
                if (
                    tag == "nsbh"
                    and nsbh_cfg.require_success_for_mej_pos
                    and np.isfinite(mej_val)
                    and mej_val > nsbh_cfg.type1_threshold
                    and src.success_ids_mej_pos is not None
                    and event_id not in src.success_ids_mej_pos
                ):
                    raise RuntimeError(
                        f"NSBH mej>threshold event {event_id} passed into output but not in success ids."
                    )

                append_gw(
                    scalar=src.gw_params[row_idx],
                    skymap=skymap,
                    gw_id=gw_id,
                    has_kn=0,
                    neg_type=neg_type,
                    mej_tot=mej_val,
                    source_type=tag,
                )
                source_counts[tag]["neg"] += 1
                if neg_type == 1:
                    source_counts[tag]["neg_type1"] += 1
                elif neg_type == 2:
                    source_counts[tag]["neg_type2"] += 1

        # Ensure GW arrays are exactly used size (already true, but explicit for safety).
        resize_gw(gw_count)

        n_pos_bns = int(source_counts["bns"]["pos"])
        n_pos_nsbh = int(source_counts["nsbh"]["pos"])
        n_neg_bns = int(source_counts["bns"]["neg"])
        n_neg_nsbh = int(source_counts["nsbh"]["neg"])
        n_neg_type1_nsbh = int(source_counts["nsbh"]["neg_type1"])
        n_neg_type2_nsbh = int(source_counts["nsbh"]["neg_type2"])
        n_pos = n_pos_bns + n_pos_nsbh
        n_neg = n_neg_bns + n_neg_nsbh

        f.attrs["dataset_mode"] = dataset_mode
        f.attrs["n_pos_gw_bns"] = n_pos_bns
        f.attrs["n_pos_gw_nsbh"] = n_pos_nsbh
        f.attrs["n_neg_gw_bns"] = n_neg_bns
        f.attrs["n_neg_gw_nsbh"] = n_neg_nsbh
        f.attrs["n_pos_gw"] = n_pos
        f.attrs["n_neg_gw"] = n_neg
        f.attrs["n_neg_type1_gw_nsbh"] = n_neg_type1_nsbh
        f.attrs["n_neg_type2_gw_nsbh"] = n_neg_type2_nsbh
        f.attrs["n_total_gw"] = int(gw_count)
        f.attrs["n_total_optical"] = int(total_optical_count)

        f.attrs["dropped_empty_lc_bns"] = int(source_counts["bns"]["drop_empty_lc"])
        f.attrs["dropped_empty_lc_nsbh"] = int(source_counts["nsbh"]["drop_empty_lc"])
        f.attrs["dropped_skymap_bns"] = int(source_counts["bns"]["drop_skymap"])
        f.attrs["dropped_skymap_nsbh"] = int(source_counts["nsbh"]["drop_skymap"])
        f.attrs["nsbh_mej_col"] = prepared["nsbh"].nsbh_mej_col_resolved or nsbh_cfg.mej_col
        f.attrs["nsbh_type1_threshold"] = float(nsbh_cfg.type1_threshold)
        f.attrs["nsbh_require_success_for_mej_pos"] = int(nsbh_cfg.require_success_for_mej_pos)
        f.attrs["n_filtered_non_success_mej_pos_nsbh"] = int(
            prepared["nsbh"].n_filtered_non_success_mej_pos
        )

        print("\nProcessing complete.")
        print(f"  Mode: {dataset_mode}")
        print(
            f"  Pos GW: total={n_pos} (bns={n_pos_bns}, nsbh={n_pos_nsbh})"
        )
        print(
            f"  Neg GW: total={n_neg} (bns={n_neg_bns}, nsbh={n_neg_nsbh})"
        )
        print(
            f"  NSBH neg types: type1={n_neg_type1_nsbh}, type2={n_neg_type2_nsbh}"
        )
        print(
            "  NSBH filtered mej>threshold non-success events: "
            f"{prepared['nsbh'].n_filtered_non_success_mej_pos}"
        )
        print(f"  Total GW events written: {gw_count}")
        print(f"  Total optical light curves: {total_optical_count}")
        print(
            "  Dropped positive GW due to empty/invalid optical: "
            f"bns={source_counts['bns']['drop_empty_lc']}, "
            f"nsbh={source_counts['nsbh']['drop_empty_lc']}"
        )
        print(f"  Saved to: {output_h5_path}")


def _build_arg_parser():
    import argparse

    p = argparse.ArgumentParser(
        description="Create combined BNS+NSBH GW dataset with optional negative GW events."
    )

    # Common
    p.add_argument("--output_h5_path", required=True)
    p.add_argument("--dataset_mode", choices=["train", "test"], default="train")
    p.add_argument("--buffer_limit", type=int, default=10000)
    p.add_argument("--seed", type=int, default=42)

    # BNS source
    p.add_argument("--bns_full_catalog_path", required=True)
    p.add_argument("--bns_skymap_dir", required=True)
    p.add_argument("--bns_sim_root", required=True)
    p.add_argument("--bns_sim_name", required=True)
    p.add_argument("--bns_success_ids_path", default=None)
    p.add_argument("--bns_max_lc_per_gw", type=int, default=1000)
    p.add_argument("--bns_max_neg_gw", type=int, default=None)
    p.add_argument("--bns_max_pos_gw", type=int, default=None)

    # NSBH source
    p.add_argument("--nsbh_full_catalog_path", required=True)
    p.add_argument("--nsbh_skymap_dir", required=True)
    p.add_argument("--nsbh_sim_root", required=True)
    p.add_argument("--nsbh_sim_name", required=True)
    p.add_argument("--nsbh_success_ids_path", default=None)
    p.add_argument("--nsbh_max_lc_per_gw", type=int, default=1000)
    p.add_argument("--nsbh_max_neg_gw", type=int, default=None)
    p.add_argument("--nsbh_max_pos_gw", type=int, default=None)
    p.add_argument("--nsbh_mej_col", default="mej_tot")
    p.add_argument("--nsbh_type1_threshold", type=float, default=0.0)
    p.add_argument(
        "--nsbh_require_success_for_mej_pos",
        type=int,
        choices=[0, 1],
        default=1,
        help="1: require mej>threshold NSBH events to be in success ids; 0: disable this filter.",
    )
    p.add_argument("--nsbh_max_neg_type1_gw", type=int, default=None)
    p.add_argument("--nsbh_max_neg_type2_gw", type=int, default=None)

    return p


if __name__ == "__main__":
    args = _build_arg_parser().parse_args()

    bns_cfg = SourceConfig(
        tag="bns",
        full_catalog_path=args.bns_full_catalog_path,
        skymap_dir=args.bns_skymap_dir,
        sim_root=args.bns_sim_root,
        sim_name=args.bns_sim_name,
        success_ids_path=args.bns_success_ids_path,
        max_lc_per_gw=args.bns_max_lc_per_gw,
        max_neg_gw=args.bns_max_neg_gw,
        max_pos_gw=args.bns_max_pos_gw,
    )
    nsbh_cfg = SourceConfig(
        tag="nsbh",
        full_catalog_path=args.nsbh_full_catalog_path,
        skymap_dir=args.nsbh_skymap_dir,
        sim_root=args.nsbh_sim_root,
        sim_name=args.nsbh_sim_name,
        success_ids_path=args.nsbh_success_ids_path,
        max_lc_per_gw=args.nsbh_max_lc_per_gw,
        max_neg_gw=args.nsbh_max_neg_gw,
        max_pos_gw=args.nsbh_max_pos_gw,
        mej_col=args.nsbh_mej_col,
        type1_threshold=args.nsbh_type1_threshold,
        require_success_for_mej_pos=bool(args.nsbh_require_success_for_mej_pos),
        max_neg_type1_gw=args.nsbh_max_neg_type1_gw,
        max_neg_type2_gw=args.nsbh_max_neg_type2_gw,
    )

    create_dataset_with_neg_gw_bns_nsbh_fast(
        output_h5_path=args.output_h5_path,
        dataset_mode=args.dataset_mode,
        bns_cfg=bns_cfg,
        nsbh_cfg=nsbh_cfg,
        buffer_limit=args.buffer_limit,
        seed=args.seed,
    )
