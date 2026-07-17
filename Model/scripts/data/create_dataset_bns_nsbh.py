import os
import importlib.util
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import h5py
import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.time import Time
from tqdm import tqdm

# Load required functions/constants from data_loader.py.
_DATA_LOADER_PATH = Path(__file__).resolve().parents[2] / "data_loader.py"
_spec = importlib.util.spec_from_file_location("data_loader", _DATA_LOADER_PATH)
_data_loader = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_data_loader)  # type: ignore

parse_snana_fits = _data_loader.parse_snana_fits
sample_moc_skymap = _data_loader.sample_moc_skymap
MAX_LC_LENGTH = _data_loader.MAX_LC_LENGTH
NUM_BANDS = _data_loader.NUM_BANDS
MERGE_WINDOW_HOURS = _data_loader.MERGE_WINDOW_HOURS
MERGE_MODE = _data_loader.MERGE_MODE
MERGE_FLUX_DOMAIN = _data_loader.MERGE_FLUX_DOMAIN

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
LUPT_BAND_ORDER = ("u", "g", "r", "i", "z", "Y")
FIRST_DETECTION_POLICY = "psfflux_snr5_then_photflag_then_head_mjd_detect_first"
FIRST_DETECTION_SNR_DOMAIN = "merged_psfflux"
SCALAR_COLUMN_NAMES = "mass1_detector,mass2_detector,spin1z,spin2z,costheta,distmean_gpc,diststd_gpc"
DEFAULT_MTAN_SNR_S0 = 3.0
DEFAULT_MTAN_SNR_BETA = 1.0
DEFAULT_MTAN_SNR_CLIP_MIN = -8.0
DEFAULT_MTAN_SNR_CLIP_MAX = 20.0
DEFAULT_MTAN_SNR_EPS = 1e-9


def parse_lupt_m5_mag(text: str) -> np.ndarray:
    raw = str(text).strip()
    if raw == "":
        raise ValueError(
            "--lupt_m5_mag is required and must contain 6 comma-separated finite values in order u,g,r,i,z,Y."
        )
    parts = [p.strip() for p in raw.split(",")]
    if len(parts) != NUM_BANDS:
        raise ValueError(
            f"--lupt_m5_mag must provide exactly {NUM_BANDS} values in order u,g,r,i,z,Y; got {len(parts)}."
        )
    try:
        vals = np.asarray([float(p) for p in parts], dtype=np.float64)
    except ValueError as exc:
        raise ValueError("--lupt_m5_mag contains non-numeric values.") from exc
    if not np.all(np.isfinite(vals)):
        raise ValueError("--lupt_m5_mag values must be finite.")
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


def write_luptitude_metadata_attrs(
    h5_obj,
    fluxcal_zp: float,
    psfflux_zp: float,
    fluxcal_to_psfflux_factor: float,
    lupt_k: float,
    lupt_m5_mag: np.ndarray,
    lupt_f5sigma_njy: np.ndarray,
    lupt_b_njy: np.ndarray,
) -> None:
    h5_obj.attrs["photometry_representation"] = "luptitude"
    h5_obj.attrs["flux_input_column"] = "FLUXCAL"
    h5_obj.attrs["fluxerr_input_column"] = "FLUXCALERR"
    h5_obj.attrs["fluxcal_zp"] = float(fluxcal_zp)
    h5_obj.attrs["psfflux_zp"] = float(psfflux_zp)
    h5_obj.attrs["fluxcal_to_psfflux_factor"] = float(fluxcal_to_psfflux_factor)
    h5_obj.attrs["lupt_k"] = float(lupt_k)
    h5_obj.attrs["lupt_band_order"] = ",".join(LUPT_BAND_ORDER)
    h5_obj.attrs["lupt_m5_mag"] = np.asarray(lupt_m5_mag, dtype=np.float64)
    h5_obj.attrs["lupt_f5sigma_njy"] = np.asarray(lupt_f5sigma_njy, dtype=np.float64)
    h5_obj.attrs["lupt_b_njy"] = np.asarray(lupt_b_njy, dtype=np.float64)
    h5_obj.attrs["values_semantics"] = "luptitude"
    h5_obj.attrs["errors_semantics"] = "luptitude_sigma"
    h5_obj.attrs["lightcurve_merge_window_hours"] = float(MERGE_WINDOW_HOURS)
    h5_obj.attrs["lightcurve_merge_mode"] = MERGE_MODE
    h5_obj.attrs["lightcurve_merge_flux_domain"] = MERGE_FLUX_DOMAIN
    h5_obj.attrs["mtan_snr_s0"] = float(DEFAULT_MTAN_SNR_S0)
    h5_obj.attrs["mtan_snr_beta"] = float(DEFAULT_MTAN_SNR_BETA)
    h5_obj.attrs["mtan_snr_clip_min"] = float(DEFAULT_MTAN_SNR_CLIP_MIN)
    h5_obj.attrs["mtan_snr_clip_max"] = float(DEFAULT_MTAN_SNR_CLIP_MAX)
    h5_obj.attrs["mtan_snr_eps"] = float(DEFAULT_MTAN_SNR_EPS)
    h5_obj.attrs["mtan_snr_source"] = "flux_snr_from_psfflux"


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
    event_time_mjd: np.ndarray
    event_time_col: str
    event_time_from_gps: bool
    n_invalid_event_time: int
    n_missing_skymap: int
    n_filtered_non_success_mej_pos: int = 0
    nsbh_mej_col_resolved: Optional[str] = None
    success_ids_mej_pos: Optional[Set[int]] = None


@dataclass(frozen=True)
class EventProcessTask:
    tag: str
    event_id: int
    sim_root: str
    sim_name: str
    skymap_dir: str
    include_lightcurves: bool
    fluxcal_to_psfflux_factor: float
    psfflux_zp: float
    lupt_b_njy: Tuple[float, ...]


@dataclass
class EventProcessResult:
    tag: str
    event_id: int
    include_lightcurves: bool
    status: str
    lcs: Optional[List[Tuple]] = None  # 5-tuple normally; 6-tuple when first_detection mode
    skymap: Optional[np.ndarray] = None


def _build_event_process_task(
    src: SourcePrepared,
    event_id: int,
    include_lightcurves: bool,
    fluxcal_to_psfflux_factor: float,
    psfflux_zp: float,
    lupt_b_njy: np.ndarray,
) -> EventProcessTask:
    return EventProcessTask(
        tag=str(src.cfg.tag),
        event_id=int(event_id),
        sim_root=str(src.cfg.sim_root),
        sim_name=str(src.cfg.sim_name),
        skymap_dir=str(src.cfg.skymap_dir),
        include_lightcurves=bool(include_lightcurves),
        fluxcal_to_psfflux_factor=float(fluxcal_to_psfflux_factor),
        psfflux_zp=float(psfflux_zp),
        lupt_b_njy=tuple(float(x) for x in np.asarray(lupt_b_njy, dtype=np.float64).tolist()),
    )


def _sampled_skymap_to_numpy(skymap_obj) -> np.ndarray:
    arr = skymap_obj
    if hasattr(arr, "detach"):
        arr = arr.detach()
    if hasattr(arr, "cpu"):
        arr = arr.cpu()
    if hasattr(arr, "numpy"):
        arr = arr.numpy()
    return np.asarray(arr, dtype=np.float32)


def _process_event_task(task: EventProcessTask) -> EventProcessResult:
    lcs: Optional[List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]]] = None
    if task.include_lightcurves:
        lcs = parse_snana_fits(
            event_id=int(task.event_id),
            sim_dir=task.sim_root,
            sim_name=task.sim_name,
            fluxcal_to_psfflux_factor=float(task.fluxcal_to_psfflux_factor),
            psfflux_zp=float(task.psfflux_zp),
            lupt_b_njy=np.asarray(task.lupt_b_njy, dtype=np.float64),
            normalize_to_first_detection=True,
        )
        if len(lcs) == 0:
            return EventProcessResult(
                tag=task.tag,
                event_id=int(task.event_id),
                include_lightcurves=True,
                status="empty_lc",
                lcs=[],
                skymap=None,
            )

    skymap_path = os.path.join(task.skymap_dir, f"{int(task.event_id)}.fits")
    try:
        skymap = _sampled_skymap_to_numpy(sample_moc_skymap(skymap_path))
    except Exception:
        return EventProcessResult(
            tag=task.tag,
            event_id=int(task.event_id),
            include_lightcurves=bool(task.include_lightcurves),
            status="missing_skymap",
            lcs=lcs,
            skymap=None,
        )

    return EventProcessResult(
        tag=task.tag,
        event_id=int(task.event_id),
        include_lightcurves=bool(task.include_lightcurves),
        status="ok",
        lcs=lcs,
        skymap=skymap,
    )


def _iter_event_task_results(
    tasks: List[EventProcessTask],
    num_workers: int,
    desc: str,
):
    if not tasks:
        return

    effective_workers = max(1, min(int(num_workers), len(tasks)))
    if effective_workers <= 1:
        iterator = (_process_event_task(task) for task in tasks)
        for result in tqdm(iterator, total=len(tasks), desc=desc, mininterval=0.5, miniters=100):
            yield result
        return

    # chunksize = max(1, len(tasks) // (effective_workers * 8))
    chunksize = 10
    executor_kwargs = {"max_workers": effective_workers}
    if os.name != "nt":
        try:
            executor_kwargs["mp_context"] = mp.get_context("fork")
        except ValueError:
            pass

    print(f"[{desc}] using ordered process pool with {effective_workers} workers (chunksize={chunksize})")
    with ProcessPoolExecutor(**executor_kwargs) as executor:
        iterator = executor.map(_process_event_task, tasks, chunksize=chunksize)
        for result in tqdm(iterator, total=len(tasks), desc=desc, mininterval=0.5, miniters=100):
            yield result


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


def _shuffle_event_ids(
    event_ids: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    if len(event_ids) <= 1:
        return event_ids
    return rng.permutation(event_ids).astype(np.int64, copy=False)


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
    event_time_mjd, event_time_col, event_time_from_gps, n_invalid_event_time = _extract_event_time_mjd(
        gw_df
    )
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

    max_neg = _normalize_max_count(cfg.max_neg_gw)

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
        f"event_time_col={event_time_col} "
        f"event_time_from_gps={int(event_time_from_gps)} "
        f"invalid_event_time={n_invalid_event_time} "
        f"missing_skymap={missing_skymap} "
        f"positive_candidates={len(pos_ids)} selected_neg={len(neg_ids)}"
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
        event_time_mjd=event_time_mjd,
        event_time_col=event_time_col,
        event_time_from_gps=event_time_from_gps,
        n_invalid_event_time=n_invalid_event_time,
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
    event_time_mjd, event_time_col, event_time_from_gps, n_invalid_event_time = _extract_event_time_mjd(
        gw_df
    )
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
        f"event_time_col={event_time_col} "
        f"event_time_from_gps={int(event_time_from_gps)} "
        f"invalid_event_time={n_invalid_event_time} "
        f"mej_col={mej_col} "
        f"type1_candidates={len(type1_ids_all)} "
        f"mej_pos_candidates={len(mej_pos_ids_all)} "
        f"filtered_non_success_mej_pos={n_filtered_non_success} "
        f"positive_candidates={len(pos_ids)} "
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
        event_time_mjd=event_time_mjd,
        event_time_col=event_time_col,
        event_time_from_gps=event_time_from_gps,
        n_invalid_event_time=n_invalid_event_time,
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
    num_workers: int = 1,
    seed: int = 42,
    fluxcal_zp: float = 27.5,
    psfflux_zp: float = 31.4,
    lupt_k: float = 1.0,
    lupt_m5_mag: Optional[np.ndarray] = None,
    lupt_f5sigma_njy: Optional[np.ndarray] = None,
    fluxcal_to_psfflux_factor: float = 1.0,
    lupt_b_njy: Optional[np.ndarray] = None,
):
    dataset_mode = dataset_mode.strip().lower()
    if dataset_mode not in {"train", "test"}:
        raise ValueError(f"dataset_mode must be 'train' or 'test', got: {dataset_mode}")
    if buffer_limit <= 0:
        raise ValueError("buffer_limit must be positive")
    if int(num_workers) <= 0:
        raise ValueError("num_workers must be positive")
    if not np.isfinite(fluxcal_to_psfflux_factor) or fluxcal_to_psfflux_factor <= 0:
        raise ValueError("fluxcal_to_psfflux_factor must be finite and > 0.")
    if lupt_m5_mag is None or np.asarray(lupt_m5_mag).shape != (NUM_BANDS,):
        raise ValueError(f"lupt_m5_mag must contain {NUM_BANDS} values in order u,g,r,i,z,Y.")
    if lupt_f5sigma_njy is None or np.asarray(lupt_f5sigma_njy).shape != (NUM_BANDS,):
        raise ValueError(f"lupt_f5sigma_njy must contain {NUM_BANDS} values in order u,g,r,i,z,Y.")
    if lupt_b_njy is None or np.asarray(lupt_b_njy).shape != (NUM_BANDS,):
        raise ValueError(f"lupt_b_njy must contain {NUM_BANDS} values in order u,g,r,i,z,Y.")

    rng = np.random.default_rng(seed)
    bns_rng = np.random.default_rng(int(rng.integers(0, 2**31 - 1)))
    nsbh_rng = np.random.default_rng(int(rng.integers(0, 2**31 - 1)))
    bns_pos_rng = np.random.default_rng(int(rng.integers(0, 2**31 - 1)))
    nsbh_pos_rng = np.random.default_rng(int(rng.integers(0, 2**31 - 1)))
    lc_rng = np.random.default_rng(int(rng.integers(0, 2**31 - 1)))

    prepared: Dict[str, SourcePrepared] = {
        "bns": _prepare_bns_source(bns_cfg, dataset_mode, bns_rng),
        "nsbh": _prepare_nsbh_source(nsbh_cfg, dataset_mode, nsbh_rng),
    }

    pos_limits = {
        "bns": _normalize_max_count(bns_cfg.max_pos_gw),
        "nsbh": _normalize_max_count(nsbh_cfg.max_pos_gw),
    }
    pos_order_rngs = {"bns": bns_pos_rng, "nsbh": nsbh_pos_rng}
    pos_candidate_counts = {
        "bns": int(len(prepared["bns"].pos_event_ids)),
        "nsbh": int(len(prepared["nsbh"].pos_event_ids)),
    }
    target_pos_counts = {
        tag: (
            pos_candidate_counts[tag]
            if pos_limits[tag] is None
            else min(pos_candidate_counts[tag], int(pos_limits[tag]))
        )
        for tag in ("bns", "nsbh")
    }
    neg_events: List[Tuple[str, int]] = []
    if dataset_mode == "test":
        neg_events = (
            [("bns", int(eid)) for eid in prepared["bns"].neg_event_ids.tolist()]
            + [("nsbh", int(eid)) for eid in prepared["nsbh"].neg_event_ids.tolist()]
        )

    n_positive_candidates = int(pos_candidate_counts["bns"] + pos_candidate_counts["nsbh"])
    n_target_pos = int(target_pos_counts["bns"] + target_pos_counts["nsbh"])
    n_expected_gw = n_target_pos + len(neg_events)
    print(
        f"\nCreating combined dataset ({dataset_mode} mode): "
        f"positive_candidates={n_positive_candidates} "
        f"target_pos={n_target_pos} expected_neg={len(neg_events)} "
        f"expected_total={n_expected_gw} num_workers={int(num_workers)}"
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
        ds_gw_event_time_mjd = grp_gw.create_dataset(
            "event_time_mjd",
            (0,),
            maxshape=(None,),
            dtype="f8",
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
        ds_opt_zero_time_mjd_base = grp_opt.create_dataset(
            "zero_time_mjd_base",
            (0,),
            maxshape=(None,),
            dtype="f8",
            chunks=(chunk_size,),
        )
        ds_opt_first_detection_mjd = grp_opt.create_dataset(
            "first_detection_mjd",
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
                "invalid_event_time_written": 0,
            },
            "nsbh": {
                "pos": 0,
                "neg": 0,
                "neg_type1": 0,
                "neg_type2": 0,
                "drop_empty_lc": 0,
                "drop_skymap": 0,
                "invalid_event_time_written": 0,
            },
        }

        opt_buffer_vals: List[np.ndarray] = []
        opt_buffer_errs: List[np.ndarray] = []
        opt_buffer_masks: List[np.ndarray] = []
        opt_buffer_times: List[np.ndarray] = []
        opt_buffer_zero_time_mjd_base: List[float] = []
        opt_buffer_first_detection_mjd: List[float] = []
        opt_buffer_parent_idx: List[int] = []
        opt_buffer_coordinates: List[np.ndarray] = []

        def resize_gw(new_size: int) -> None:
            ds_gw_scalars.resize(new_size, axis=0)
            ds_gw_skymaps.resize(new_size, axis=0)
            ds_gw_ids.resize(new_size, axis=0)
            ds_gw_has_kn.resize(new_size, axis=0)
            ds_gw_neg_type.resize(new_size, axis=0)
            ds_gw_mej_tot.resize(new_size, axis=0)
            ds_gw_event_time_mjd.resize(new_size, axis=0)
            ds_gw_source_type.resize(new_size, axis=0)

        def append_gw(
            scalar: np.ndarray,
            skymap: np.ndarray,
            gw_id: str,
            has_kn: int,
            neg_type: int,
            mej_tot: float,
            event_time_mjd: float,
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
            ds_gw_event_time_mjd[gw_count] = np.float64(event_time_mjd)
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
            ds_opt_zero_time_mjd_base.resize(new_size, axis=0)
            ds_opt_first_detection_mjd.resize(new_size, axis=0)
            ds_opt_coordinates.resize(new_size, axis=0)
            ds_parent_idx.resize(new_size, axis=0)

            ds_opt_vals[cur:new_size] = np.asarray(opt_buffer_vals)
            ds_opt_errs[cur:new_size] = np.asarray(opt_buffer_errs)
            ds_opt_masks[cur:new_size] = np.asarray(opt_buffer_masks)
            ds_opt_times[cur:new_size] = np.asarray(opt_buffer_times)
            ds_opt_zero_time_mjd_base[cur:new_size] = np.asarray(
                opt_buffer_zero_time_mjd_base, dtype=np.float64
            )
            ds_opt_first_detection_mjd[cur:new_size] = np.asarray(
                opt_buffer_first_detection_mjd, dtype=np.float64
            )
            ds_opt_coordinates[cur:new_size] = np.asarray(opt_buffer_coordinates)
            ds_parent_idx[cur:new_size] = np.asarray(opt_buffer_parent_idx, dtype=np.int32)

            total_optical_count += n_new
            opt_buffer_vals.clear()
            opt_buffer_errs.clear()
            opt_buffer_masks.clear()
            opt_buffer_times.clear()
            opt_buffer_zero_time_mjd_base.clear()
            opt_buffer_first_detection_mjd.clear()
            opt_buffer_coordinates.clear()
            opt_buffer_parent_idx.clear()

        print("\nWriting positive GW events...")
        pos_task_batch_size = max(1, int(num_workers) * 20)
        for tag in ("bns", "nsbh"):
            src = prepared[tag]
            limit = pos_limits[tag]
            candidate_ids = np.asarray(src.pos_event_ids, dtype=np.int64)
            if limit is not None:
                candidate_ids = _shuffle_event_ids(candidate_ids, pos_order_rngs[tag])
            task_batch_size = pos_task_batch_size if limit is not None else max(1, len(candidate_ids))

            for batch_start in range(0, len(candidate_ids), task_batch_size):
                if limit is not None and source_counts[tag]["pos"] >= limit:
                    break

                batch_records: List[Tuple[str, int, int]] = []
                batch_tasks: List[EventProcessTask] = []
                for event_id_np in candidate_ids[batch_start: batch_start + task_batch_size]:
                    event_id = int(event_id_np)
                    row_idx = src.event_to_row.get(event_id)
                    if row_idx is None:
                        continue
                    batch_records.append((tag, event_id, int(row_idx)))
                    batch_tasks.append(
                        _build_event_process_task(
                            src=src,
                            event_id=event_id,
                            include_lightcurves=True,
                            fluxcal_to_psfflux_factor=float(fluxcal_to_psfflux_factor),
                            psfflux_zp=float(psfflux_zp),
                            lupt_b_njy=np.asarray(lupt_b_njy, dtype=np.float64),
                        )
                    )

                for (rec_tag, event_id, row_idx), task_result in zip(
                    batch_records,
                    _iter_event_task_results(
                        batch_tasks,
                        num_workers=int(num_workers),
                        desc=f"Positive GW {tag.upper()}",
                    ),
                ):
                    if task_result.tag != rec_tag or int(task_result.event_id) != int(event_id):
                        raise RuntimeError(
                            f"Mismatched positive task result ordering: expected {rec_tag}_{event_id}, got {task_result.tag}_{task_result.event_id}"
                        )

                    if limit is not None and source_counts[tag]["pos"] >= limit:
                        continue

                    lcs = list(task_result.lcs or [])
                    max_lc = _normalize_max_count(src.cfg.max_lc_per_gw)
                    if max_lc is not None and len(lcs) > max_lc:
                        keep_idx = lc_rng.choice(len(lcs), size=max_lc, replace=False)
                        lcs = [lcs[int(i)] for i in keep_idx]

                    if len(lcs) == 0:
                        source_counts[tag]["drop_empty_lc"] += 1
                        continue
                    if task_result.status == "missing_skymap" or task_result.skymap is None:
                        source_counts[tag]["drop_skymap"] += 1
                        continue
                    if task_result.status != "ok":
                        raise RuntimeError(
                            f"Unexpected positive task status for {tag}_{event_id}: {task_result.status}"
                        )

                    skymap = np.asarray(task_result.skymap, dtype=np.float32)
                    gw_id = f"{tag}_{event_id}"
                    if gw_id in written_id_set:
                        continue
                    written_id_set.add(gw_id)

                    mej_val = float(src.mej_by_event.get(event_id, np.nan))
                    event_time_val = float(src.event_time_mjd[row_idx])
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
                        event_time_mjd=event_time_val,
                        source_type=tag,
                    )
                    source_counts[tag]["pos"] += 1
                    if not np.isfinite(event_time_val):
                        source_counts[tag]["invalid_event_time_written"] += 1

                    for vals, errs, masks, times, coordinates, first_detection_mjd in lcs:
                        opt_buffer_vals.append(vals)
                        opt_buffer_errs.append(errs)
                        opt_buffer_masks.append(masks)
                        opt_buffer_times.append(times)
                        opt_buffer_zero_time_mjd_base.append(float(first_detection_mjd))
                        opt_buffer_first_detection_mjd.append(float(first_detection_mjd))
                        opt_buffer_coordinates.append(coordinates)
                        opt_buffer_parent_idx.append(gw_idx)

                    if len(opt_buffer_vals) >= buffer_limit:
                        flush_opt_buffer()

        flush_opt_buffer()

        if dataset_mode == "test":
            neg_event_records: List[Tuple[str, int, int]] = []
            neg_tasks: List[EventProcessTask] = []
            for tag, event_id in neg_events:
                src = prepared[tag]
                row_idx = src.event_to_row.get(event_id)
                if row_idx is None:
                    continue
                neg_event_records.append((tag, int(event_id), int(row_idx)))
                neg_tasks.append(
                    _build_event_process_task(
                        src=src,
                        event_id=int(event_id),
                        include_lightcurves=False,
                        fluxcal_to_psfflux_factor=float(fluxcal_to_psfflux_factor),
                        psfflux_zp=float(psfflux_zp),
                        lupt_b_njy=np.asarray(lupt_b_njy, dtype=np.float64),
                    )
                )

            print("\nWriting negative GW events...")
            for (tag, event_id, row_idx), task_result in zip(
                neg_event_records,
                _iter_event_task_results(neg_tasks, num_workers=int(num_workers), desc="Negative GW"),
            ):
                if task_result.tag != tag or int(task_result.event_id) != int(event_id):
                    raise RuntimeError(
                        f"Mismatched negative task result ordering: expected {tag}_{event_id}, got {task_result.tag}_{task_result.event_id}"
                    )
                if task_result.status == "missing_skymap" or task_result.skymap is None:
                    source_counts[tag]["drop_skymap"] += 1
                    continue
                if task_result.status != "ok":
                    raise RuntimeError(
                        f"Unexpected negative task status for {tag}_{event_id}: {task_result.status}"
                    )

                src = prepared[tag]
                skymap = np.asarray(task_result.skymap, dtype=np.float32)
                gw_id = f"{tag}_{event_id}"
                if gw_id in written_id_set:
                    continue
                written_id_set.add(gw_id)

                neg_type = int(src.neg_type_by_event.get(event_id, 2))
                mej_val = float(src.mej_by_event.get(event_id, np.nan))
                event_time_val = float(src.event_time_mjd[row_idx])
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
                    event_time_mjd=event_time_val,
                    source_type=tag,
                )
                source_counts[tag]["neg"] += 1
                if not np.isfinite(event_time_val):
                    source_counts[tag]["invalid_event_time_written"] += 1
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
        f.attrs["preprocess_num_workers"] = int(num_workers)

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
        f.attrs["event_time_col_bns"] = prepared["bns"].event_time_col
        f.attrs["event_time_col_nsbh"] = prepared["nsbh"].event_time_col
        f.attrs["event_time_from_gps_bns"] = int(prepared["bns"].event_time_from_gps)
        f.attrs["event_time_from_gps_nsbh"] = int(prepared["nsbh"].event_time_from_gps)
        f.attrs["n_invalid_event_time_bns_catalog"] = int(prepared["bns"].n_invalid_event_time)
        f.attrs["n_invalid_event_time_nsbh_catalog"] = int(prepared["nsbh"].n_invalid_event_time)
        f.attrs["n_invalid_event_time_bns_written"] = int(
            source_counts["bns"]["invalid_event_time_written"]
        )
        f.attrs["n_invalid_event_time_nsbh_written"] = int(
            source_counts["nsbh"]["invalid_event_time_written"]
        )
        f.attrs["n_invalid_event_time_written_total"] = int(
            source_counts["bns"]["invalid_event_time_written"]
            + source_counts["nsbh"]["invalid_event_time_written"]
        )
        f.attrs["requested_max_pos_gw_bns"] = int(pos_limits["bns"] or 0)
        f.attrs["requested_max_pos_gw_nsbh"] = int(pos_limits["nsbh"] or 0)
        f.attrs["positive_candidates_bns"] = int(pos_candidate_counts["bns"])
        f.attrs["positive_candidates_nsbh"] = int(pos_candidate_counts["nsbh"])
        f.attrs["time_zero_base_semantics"] = "optical zero_time_mjd_base stores first_detection_mjd"
        f.attrs["first_detection_policy"] = FIRST_DETECTION_POLICY
        f.attrs["first_detection_snr_domain"] = FIRST_DETECTION_SNR_DOMAIN
        f.attrs["first_detection_snr_threshold"] = 5.0
        f.attrs["scalar_column_names"] = SCALAR_COLUMN_NAMES
        f.attrs["time_unit"] = "mjd_days"
        f.attrs["runtime_offset_applied"] = 0
        f.attrs["min_nobs_stage"] = "post_merge"
        write_luptitude_metadata_attrs(
            h5_obj=f,
            fluxcal_zp=float(fluxcal_zp),
            psfflux_zp=float(psfflux_zp),
            fluxcal_to_psfflux_factor=float(fluxcal_to_psfflux_factor),
            lupt_k=float(lupt_k),
            lupt_m5_mag=np.asarray(lupt_m5_mag, dtype=np.float64),
            lupt_f5sigma_njy=np.asarray(lupt_f5sigma_njy, dtype=np.float64),
            lupt_b_njy=np.asarray(lupt_b_njy, dtype=np.float64),
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
        print(
            "  Event time source (bns/nsbh): "
            f"{prepared['bns'].event_time_col}/{prepared['nsbh'].event_time_col} "
            f"(gps_fallback={int(prepared['bns'].event_time_from_gps)}/"
            f"{int(prepared['nsbh'].event_time_from_gps)})"
        )
        print(
            "  Invalid event_time_mjd (catalog bns/nsbh): "
            f"{prepared['bns'].n_invalid_event_time}/{prepared['nsbh'].n_invalid_event_time}"
        )
        print(
            "  Invalid event_time_mjd written (bns/nsbh): "
            f"{source_counts['bns']['invalid_event_time_written']}/"
            f"{source_counts['nsbh']['invalid_event_time_written']}"
        )
        print(f"  Total GW events written: {gw_count}")
        print(f"  Total optical light curves: {total_optical_count}")
        print(
            "  Dropped positive GW due to empty/invalid optical: "
            f"bns={source_counts['bns']['drop_empty_lc']}, "
            f"nsbh={source_counts['nsbh']['drop_empty_lc']}"
        )
        for tag in ("bns", "nsbh"):
            limit = pos_limits[tag]
            if limit is not None and source_counts[tag]["pos"] < limit:
                print(
                    f"  WARNING: requested {limit} {tag.upper()} positive GW, "
                    f"but only wrote {source_counts[tag]['pos']} after filtering all "
                    f"{pos_candidate_counts[tag]} candidates."
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
    p.add_argument("--num_workers", type=int, default=1, help="Parallel worker count for per-event preprocessing.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--fluxcal_zp", type=float, default=27.5)
    p.add_argument("--psfflux_zp", type=float, default=31.4)
    p.add_argument("--lupt_k", type=float, default=1.0)
    p.add_argument(
        "--lupt_m5_mag",
        type=str,
        default="23.9,25.0,24.7,24.0,23.3,22.1",
        help="Comma-separated 6 Rubin single-exposure m5 values (AB mag) in order u,g,r,i,z,Y.",
    )

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
    lupt_m5_mag = parse_lupt_m5_mag(args.lupt_m5_mag)
    fluxcal_to_psfflux_factor, lupt_f5sigma_njy, lupt_b_njy = build_luptitude_params(
        fluxcal_zp=float(args.fluxcal_zp),
        psfflux_zp=float(args.psfflux_zp),
        lupt_k=float(args.lupt_k),
        lupt_m5_mag=lupt_m5_mag,
    )

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
        num_workers=int(args.num_workers),
        seed=args.seed,
        fluxcal_zp=float(args.fluxcal_zp),
        psfflux_zp=float(args.psfflux_zp),
        lupt_k=float(args.lupt_k),
        lupt_m5_mag=lupt_m5_mag,
        lupt_f5sigma_njy=lupt_f5sigma_njy,
        fluxcal_to_psfflux_factor=float(fluxcal_to_psfflux_factor),
        lupt_b_njy=lupt_b_njy,
    )
