#!/usr/bin/env python3
"""Build a MAGIKS HDF5 for fixed-physics GW170817A LSST scenarios."""

from __future__ import annotations

import argparse
import importlib.util
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import h5py
import numpy as np
import pandas as pd
import torch
from astropy.io import fits
from ligo.skymap.io.fits import read_sky_map

REPO_ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = REPO_ROOT / "Model"
if str(MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(MODEL_DIR))

_DATA_LOADER_PATH = MODEL_DIR / "data_loader.py"
_spec = importlib.util.spec_from_file_location("data_loader", _DATA_LOADER_PATH)
_data_loader = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_data_loader)  # type: ignore[union-attr]

from generate_gw170817a_lsst_docs import (  # noqa: E402
    DEFAULT_GENVERSION,
    DEFAULT_POSTERIOR_DATASET,
    DEFAULT_POSTERIOR_H5,
    DEFAULT_SKYMAP,
    GW170817A_DIR,
    HOST_DISTANCE_MPC,
    HOST_REDSHIFT_OBSERVED,
    PHYSICAL_EVENT_UID,
    REAL_TRIGGER_MJD,
)
from retrieval_gallery import compute_credible_levels_single_gw  # noqa: E402

MAX_LC_LENGTH = int(_data_loader.MAX_LC_LENGTH)
NUM_BANDS = int(_data_loader.NUM_BANDS)
DEFAULT_OUTPUT_H5 = "/fred/oz016/bgao_kn/data/ALBEF_dataset/gw170817a_lsst_scenarios.h5"
DEFAULT_MANIFEST = str(
    GW170817A_DIR / "lsst_scenario_experiment" / "gw170817a_lsst_manifest.csv"
)
DEFAULT_COORDINATE_DIR = str(GW170817A_DIR / "lsst_scenario_experiment" / "COORDINATES")
DEFAULT_SNANA_SIM_DIR = "/fred/oz016/bgao_kn/SNANA/SNDATA_ROOT/SIM/gw170817a_scenarios"
FIRST_DETECTION_POLICY = "psfflux_snr5_then_photflag_then_head_mjd_detect_first"
FIRST_DETECTION_SNR_DOMAIN = "merged_psfflux"
SCALAR_COLUMN_NAMES = (
    "mass1_detector,mass2_detector,spin1z,spin2z," "costheta,distmean_gpc,diststd_gpc"
)
GW_SCALAR_SOURCE = "posterior_median_with_derived_aligned_spins"
SPIN_POLICY = "median_of_spin_times_costilt"
GALLERY_TASK_DEFINITION = (
    "Each gallery contains 1 positive KN counterpart and non-KN optical distractors."
)
LUPT_BAND_ORDER = ("u", "g", "r", "i", "z", "Y")


@dataclass(frozen=True)
class GwScalarReference:
    mass1_detector: float
    mass2_detector: float
    spin1z: float
    spin2z: float
    costheta: float
    distmean_mpc: float
    diststd_mpc: float
    posterior_h5: str
    posterior_dataset: str
    skymap_path: str
    source: str = GW_SCALAR_SOURCE

    def scalar_vector(self) -> np.ndarray:
        return np.asarray(
            [
                self.mass1_detector,
                self.mass2_detector,
                self.spin1z,
                self.spin2z,
                self.costheta,
                self.distmean_mpc / 1000.0,
                self.diststd_mpc / 1000.0,
            ],
            dtype=np.float32,
        )


@dataclass(frozen=True)
class FormattedEventRecord:
    sim_event_id: int
    scenario_id: int
    coordinate_id: int
    libid: int
    values: np.ndarray
    errors: np.ndarray
    masks: np.ndarray
    times: np.ndarray
    coordinates: np.ndarray
    event_time_mjd: float
    sim_explosion_mjd: float
    first_detection_mjd: float
    scalar: np.ndarray
    skymap: np.ndarray
    event_uid: str
    simulation_id: int
    sample_class: str
    mej_dynamic: float
    mej_wind: float
    is_true_position: bool
    posterior_probability: float
    skymap_credible_level: float
    too_nobs: int
    too_mode: str
    n_observations: int


def _load_manifest(path: str | Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {
        "sim_event_id",
        "scenario_id",
        "trigger_mjd",
        "optical_candidate_count",
        "luminosity_distance",
        "redshift",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Scenario manifest missing required columns: {missing}")
    if frame["scenario_id"].astype(int).duplicated().any():
        raise ValueError("Scenario manifest must contain one row per scenario_id.")
    return frame.sort_values("scenario_id", kind="stable").reset_index(drop=True)


def generated_counts_from_manifest(path: str | Path) -> dict[int, int]:
    frame = _load_manifest(path)
    return {
        int(row.scenario_id): int(row.optical_candidate_count)
        for row in frame.itertuples(index=False)
    }


def _finite_median(values: np.ndarray, *, field_name: str) -> float:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        raise ValueError(f"Posterior field {field_name!r} has no finite values.")
    return float(np.median(finite))


def _load_real_skymap(
    skymap_path: str | Path,
) -> tuple[np.ndarray, float, float]:
    skymap = (
        _data_loader.sample_moc_skymap(str(skymap_path))
        .detach()
        .cpu()
        .numpy()
        .astype(np.float32)
    )
    if skymap.shape != (7, 19200):
        raise ValueError(f"Expected sky map shape (7, 19200), got {skymap.shape}.")
    _probability, metadata = read_sky_map(str(skymap_path), nest=True)
    distmean = float(metadata.get("distmean"))
    diststd = float(metadata.get("diststd"))
    if not np.isfinite(distmean) or distmean <= 0:
        raise ValueError(f"Invalid distmean in {skymap_path}: {distmean}")
    if not np.isfinite(diststd) or diststd <= 0:
        raise ValueError(f"Invalid diststd in {skymap_path}: {diststd}")
    return skymap, distmean, diststd


def load_posterior_scalar_reference(
    posterior_h5: str | Path = DEFAULT_POSTERIOR_H5,
    posterior_dataset: str = DEFAULT_POSTERIOR_DATASET,
    skymap_path: str | Path = DEFAULT_SKYMAP,
) -> GwScalarReference:
    posterior_path = Path(posterior_h5)
    with h5py.File(posterior_path, "r") as handle:
        if posterior_dataset not in handle:
            raise KeyError(
                f"Posterior dataset {posterior_dataset!r} not found in {posterior_path}"
            )
        posterior = handle[posterior_dataset]
        required = {
            "m1_detector_frame_Msun",
            "m2_detector_frame_Msun",
            "spin1",
            "spin2",
            "costilt1",
            "costilt2",
            "costheta_jn",
        }
        missing = sorted(required - set(posterior.dtype.names or ()))
        if missing:
            raise ValueError(f"Posterior dataset missing required fields: {missing}")
        spin1z = np.asarray(posterior["spin1"][:], dtype=np.float64) * np.asarray(
            posterior["costilt1"][:], dtype=np.float64
        )
        spin2z = np.asarray(posterior["spin2"][:], dtype=np.float64) * np.asarray(
            posterior["costilt2"][:], dtype=np.float64
        )
        values = {
            "mass1_detector": _finite_median(
                posterior["m1_detector_frame_Msun"][:],
                field_name="m1_detector_frame_Msun",
            ),
            "mass2_detector": _finite_median(
                posterior["m2_detector_frame_Msun"][:],
                field_name="m2_detector_frame_Msun",
            ),
            "spin1z": _finite_median(spin1z, field_name="spin1z"),
            "spin2z": _finite_median(spin2z, field_name="spin2z"),
            "costheta": _finite_median(
                posterior["costheta_jn"][:], field_name="costheta_jn"
            ),
        }
    _skymap, distmean, diststd = _load_real_skymap(skymap_path)
    return GwScalarReference(
        **values,
        distmean_mpc=distmean,
        diststd_mpc=diststd,
        posterior_h5=str(posterior_path),
        posterior_dataset=str(posterior_dataset),
        skymap_path=str(Path(skymap_path)),
    )


def _finite_float(value: object, default: float = float("nan")) -> float:
    try:
        result = float(value)
    except Exception:
        return float(default)
    return result if np.isfinite(result) else float(default)


def _head_value(row, names: Sequence[str], default: float = float("nan")) -> float:
    available = set(getattr(row, "array").names)
    for name in names:
        if name in available:
            result = _finite_float(row[name])
            if np.isfinite(result):
                return result
    return float(default)


def _format_lightcurve(
    *,
    mjd: np.ndarray,
    fluxcal: np.ndarray,
    fluxcalerr: np.ndarray,
    band: np.ndarray,
    fluxcal_to_psfflux_factor: float,
    psfflux_zp: float,
    lupt_b_njy: Sequence[float],
    min_nobs: int,
    photflag: np.ndarray | None = None,
    head_mjd_detect_first: float | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float] | None:
    merged_mjd, merged_flux, merged_error, merged_band = (
        _data_loader.merge_photometry_psfflux(
            mjd=np.asarray(mjd, dtype=np.float64),
            fluxcal=np.asarray(fluxcal, dtype=np.float64),
            fluxcalerr=np.asarray(fluxcalerr, dtype=np.float64),
            flt=np.asarray(band),
            fluxcal_to_psfflux_factor=float(fluxcal_to_psfflux_factor),
        )
    )
    if len(merged_mjd) < int(min_nobs):
        return None
    first_detection_mjd = _data_loader.resolve_first_detection_mjd(
        snr_mjd=merged_mjd,
        snr_flux=merged_flux,
        snr_fluxerr=merged_error,
        photflag_mjd=np.asarray(mjd, dtype=np.float64),
        photflag=photflag,
        head_mjd_detect_first=head_mjd_detect_first,
        snr_threshold=5.0,
    )
    if first_detection_mjd is None:
        return None
    lc_mjd, lc_value, lc_error, lc_band = _data_loader._transform_psfflux_to_luptitude(
        mjd=merged_mjd,
        psfflux=merged_flux,
        psffluxerr=merged_error,
        flt=np.asarray(merged_band),
        psfflux_zp=float(psfflux_zp),
        lupt_b_njy=lupt_b_njy,
    )
    if len(lc_mjd) < int(min_nobs):
        return None
    relative_time = (
        np.asarray(lc_mjd, dtype=np.float64) - float(first_detection_mjd)
    ) / 100.0
    if relative_time.size > MAX_LC_LENGTH:
        keep = np.sort(np.argsort(np.abs(relative_time))[:MAX_LC_LENGTH])
        relative_time = relative_time[keep]
        lc_value = lc_value[keep]
        lc_error = lc_error[keep]
        lc_band = lc_band[keep]

    values = np.zeros((MAX_LC_LENGTH, NUM_BANDS), dtype=np.float32)
    errors = np.zeros((MAX_LC_LENGTH, NUM_BANDS), dtype=np.float32)
    masks = np.zeros((MAX_LC_LENGTH, NUM_BANDS), dtype=np.float32)
    times = np.zeros((MAX_LC_LENGTH,), dtype=np.float32)
    for index, (time_value, value, error, raw_band) in enumerate(
        zip(relative_time, lc_value, lc_error, lc_band)
    ):
        band_index = _data_loader._band_index_from_raw(raw_band)
        if band_index is None:
            continue
        values[index, int(band_index)] = np.float32(value)
        errors[index, int(band_index)] = np.float32(error)
        masks[index, int(band_index)] = 1.0
        times[index] = np.float32(time_value)
    if not np.any(masks > 0):
        return None
    return values, errors, masks, times, float(first_detection_mjd)


def iter_snana_fits(sim_dir: str | Path, genversion: str = DEFAULT_GENVERSION):
    root = Path(sim_dir)
    heads = sorted(list(root.rglob("*_HEAD.FITS")) + list(root.rglob("*_HEAD.FITS.gz")))
    if not heads:
        for suffix in (".FITS", ".FITS.gz"):
            head = root / f"{genversion}_HEAD{suffix}"
            phot = root / f"{genversion}_PHOT{suffix}"
            if head.exists() and phot.exists():
                yield head, phot
                return
        raise FileNotFoundError(
            f"Could not find HEAD/PHOT FITS under {root} for genversion={genversion}"
        )
    for head in heads:
        suffix = "_HEAD.FITS.gz" if str(head).endswith(".gz") else "_HEAD.FITS"
        replacement = "_PHOT.FITS.gz" if str(head).endswith(".gz") else "_PHOT.FITS"
        phot = head.with_name(head.name.replace(suffix, replacement))
        if phot.exists():
            yield head, phot


def simulation_id_from_head_path(head_path: str | Path) -> int:
    match = re.search(r"_(\d+)_HEAD\.FITS(?:\.gz)?$", Path(head_path).name)
    if match is None:
        raise ValueError(f"Cannot determine simulation_id from HEAD path: {head_path}")
    return int(match.group(1))


def load_true_libid(coordinate_dir: str | Path, simulation_id: int) -> int:
    frame = pd.read_csv(Path(coordinate_dir) / f"{int(simulation_id)}.csv")
    true_mask = frame["is_true_position"].astype(str).str.lower().isin(("true", "1"))
    true_rows = frame.loc[true_mask]
    if len(true_rows) != 1:
        raise ValueError(
            f"Coordinate manifest must contain exactly one true position; "
            f"found {len(true_rows)}"
        )
    return int(true_rows.iloc[0]["libid"])


def load_coordinate_lookup(
    coordinate_dir: str | Path, simulation_id: int
) -> dict[int, dict[str, object]]:
    path = Path(coordinate_dir) / f"{int(simulation_id)}.csv"
    if not path.is_file():
        raise FileNotFoundError(f"Coordinate manifest not found: {path}")
    frame = pd.read_csv(path)
    required = {
        "simulation_id",
        "sample_index",
        "libid",
        "ra",
        "dec",
        "is_true_position",
        "posterior_probability",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Coordinate manifest {path} missing columns: {missing}")
    frame = frame.loc[
        (frame["simulation_id"].astype(int) == int(simulation_id))
        & np.isfinite(pd.to_numeric(frame["libid"], errors="coerce"))
    ].copy()
    frame["libid"] = frame["libid"].astype(int)
    if frame["libid"].duplicated().any():
        duplicates = sorted(frame.loc[frame["libid"].duplicated(), "libid"].unique())
        raise ValueError(
            f"Coordinate manifest {path} has duplicate LIBIDs: {duplicates[:10]}"
        )
    return {int(row.libid): row._asdict() for row in frame.itertuples(index=False)}


def _bool_value(value: object) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes"}


def _coordinate_credible_levels(
    skymap: np.ndarray, lookup: Mapping[int, Mapping[str, object]]
) -> dict[int, float]:
    libids = sorted(int(value) for value in lookup)
    coordinates = torch.tensor(
        [[float(lookup[libid]["ra"]), float(lookup[libid]["dec"])] for libid in libids],
        dtype=torch.float32,
    )
    levels = compute_credible_levels_single_gw(
        torch.from_numpy(np.asarray(skymap, dtype=np.float32)), coordinates
    )
    return {
        libid: float(level)
        for libid, level in zip(libids, levels.detach().cpu().tolist())
    }


def format_snana_fits_to_records(
    *,
    sim_dir: str | Path,
    manifest_path: str | Path,
    coordinate_dir: str | Path = DEFAULT_COORDINATE_DIR,
    skymap_path: str | Path,
    genversion: str = DEFAULT_GENVERSION,
    fluxcal_zp: float = 27.5,
    psfflux_zp: float = 31.4,
    lupt_k: float = 1.0,
    lupt_m5_mag: str = "23.9,25.0,24.7,24.0,23.3,22.1",
    min_nobs: int = 5,
    scalar_reference: GwScalarReference | None = None,
) -> list[FormattedEventRecord]:
    manifest = _load_manifest(manifest_path)
    manifest_lookup = {
        int(row.sim_event_id): row._asdict() for row in manifest.itertuples(index=False)
    }
    skymap, _distmean, _diststd = _load_real_skymap(skymap_path)
    reference = scalar_reference or load_posterior_scalar_reference(
        skymap_path=skymap_path
    )
    scalar = reference.scalar_vector()
    m5 = _data_loader.parse_lupt_m5_mag(str(lupt_m5_mag))
    flux_factor, _f5, lupt_b_njy = _data_loader.build_luptitude_params(
        fluxcal_zp=float(fluxcal_zp),
        psfflux_zp=float(psfflux_zp),
        lupt_k=float(lupt_k),
        lupt_m5_mag=m5,
    )

    records: list[FormattedEventRecord] = []
    for head_path, phot_path in iter_snana_fits(sim_dir, genversion=genversion):
        simulation_id = simulation_id_from_head_path(head_path)
        manifest_row = manifest_lookup.get(simulation_id)
        if manifest_row is None:
            continue
        coordinate_lookup = load_coordinate_lookup(coordinate_dir, simulation_id)
        credible_lookup = _coordinate_credible_levels(skymap, coordinate_lookup)
        with fits.open(head_path) as head_hdul, fits.open(phot_path) as phot_hdul:
            head = head_hdul[1].data
            phot = phot_hdul[1].data
            phot_columns = set(phot.columns.names)
            seen_libids: set[int] = set()
            for row in head:
                libid = int(row["SIM_LIBID"])
                if libid in seen_libids or libid not in coordinate_lookup:
                    continue
                seen_libids.add(libid)
                start = int(row["PTROBS_MIN"]) - 1
                end = int(row["PTROBS_MAX"])
                sim_explosion_mjd = _head_value(
                    row, ("SIM_MJD_EXPLODE", "SIM_PEAKMJD", "PEAKMJD")
                )
                if not np.isfinite(sim_explosion_mjd):
                    continue
                photflag = (
                    np.asarray(phot["PHOTFLAG"][start:end], dtype=np.int64)
                    if "PHOTFLAG" in phot_columns
                    else None
                )
                formatted = _format_lightcurve(
                    mjd=np.asarray(phot["MJD"][start:end], dtype=np.float64),
                    fluxcal=np.asarray(phot["FLUXCAL"][start:end], dtype=np.float64),
                    fluxcalerr=np.asarray(
                        phot["FLUXCALERR"][start:end], dtype=np.float64
                    ),
                    band=np.asarray(phot["BAND"][start:end]),
                    photflag=photflag,
                    head_mjd_detect_first=_head_value(
                        row, ("MJD_DETECT_FIRST",), default=float("nan")
                    ),
                    fluxcal_to_psfflux_factor=float(flux_factor),
                    psfflux_zp=float(psfflux_zp),
                    lupt_b_njy=lupt_b_njy,
                    min_nobs=int(min_nobs),
                )
                if formatted is None:
                    continue
                values, errors, masks, times, first_detection_mjd = formatted
                coordinate = coordinate_lookup[libid]
                scenario_id = int(manifest_row["scenario_id"])
                coordinate_id = int(coordinate["sample_index"])
                records.append(
                    FormattedEventRecord(
                        sim_event_id=int(simulation_id),
                        scenario_id=scenario_id,
                        coordinate_id=coordinate_id,
                        libid=libid,
                        values=values,
                        errors=errors,
                        masks=masks,
                        times=times,
                        coordinates=np.asarray(
                            [float(coordinate["ra"]), float(coordinate["dec"])],
                            dtype=np.float32,
                        ),
                        event_time_mjd=float(manifest_row["trigger_mjd"]),
                        sim_explosion_mjd=float(sim_explosion_mjd),
                        first_detection_mjd=float(first_detection_mjd),
                        scalar=scalar,
                        skymap=skymap,
                        event_uid=f"gw170817a_scenario_{scenario_id:02d}",
                        simulation_id=int(simulation_id),
                        sample_class="positive",
                        mej_dynamic=float(manifest_row.get("mej_dynamic", 0.016)),
                        mej_wind=float(manifest_row.get("mej_wind", 0.024)),
                        is_true_position=_bool_value(coordinate["is_true_position"]),
                        posterior_probability=float(
                            coordinate["posterior_probability"]
                        ),
                        skymap_credible_level=float(credible_lookup[libid]),
                        too_nobs=int(coordinate.get("too_nobs", 0)),
                        too_mode=str(coordinate.get("too_mode", "unknown")),
                        n_observations=int(np.sum(masks > 0)),
                    )
                )
    return records


def select_complete_coordinate_panel(
    records: Sequence[FormattedEventRecord],
    *,
    target_per_scenario: int,
    expected_scenario_ids: Sequence[int],
    seed: int = 170817,
) -> list[FormattedEventRecord]:
    """Select one identical coordinate panel that is usable in every scenario."""
    target = int(target_per_scenario)
    scenario_ids = sorted(int(value) for value in expected_scenario_ids)
    if target < 1:
        raise ValueError("target_per_scenario must be >= 1")
    if not scenario_ids:
        raise ValueError("expected_scenario_ids must be non-empty")

    by_key: dict[tuple[int, int], FormattedEventRecord] = {}
    coordinates_by_scenario: dict[int, set[int]] = {
        scenario_id: set() for scenario_id in scenario_ids
    }
    for record in records:
        key = (int(record.scenario_id), int(record.coordinate_id))
        if key in by_key:
            raise ValueError(f"Duplicate usable record for scenario/coordinate {key}")
        by_key[key] = record
        if int(record.scenario_id) in coordinates_by_scenario:
            coordinates_by_scenario[int(record.scenario_id)].add(
                int(record.coordinate_id)
            )

    common = set.intersection(
        *(coordinates_by_scenario[scenario_id] for scenario_id in scenario_ids)
    )
    true_common = sorted(
        coordinate_id
        for coordinate_id in common
        if all(
            by_key[(scenario_id, coordinate_id)].is_true_position
            for scenario_id in scenario_ids
        )
    )
    counts = {
        scenario_id: len(coordinates_by_scenario[scenario_id])
        for scenario_id in scenario_ids
    }
    if len(true_common) != 1:
        raise ValueError(
            "Complete panel must contain exactly one true coordinate; "
            f"found {true_common}. Usable counts: {counts}"
        )
    if len(common) < target:
        raise ValueError(
            f"Only {len(common)} coordinate IDs are usable in all scenarios; "
            f"need {target}. Per-scenario usable counts: {counts}"
        )

    true_id = true_common[0]
    alternatives = np.asarray(sorted(common - {true_id}), dtype=np.int64)
    rng = np.random.default_rng(int(seed))
    chosen = {int(true_id)}
    if target > 1:
        selected = rng.choice(alternatives, size=target - 1, replace=False)
        chosen.update(int(value) for value in selected.tolist())

    return [
        by_key[(scenario_id, coordinate_id)]
        for scenario_id in scenario_ids
        for coordinate_id in sorted(chosen)
    ]


def _array_or_empty(
    records: Sequence[FormattedEventRecord],
    attribute: str,
    shape: tuple[int, ...],
    dtype,
) -> np.ndarray:
    if not records:
        return np.empty((0, *shape), dtype=dtype)
    return np.asarray(
        [getattr(record, attribute) for record in records], dtype=dtype
    ).reshape(len(records), *shape)


def _single_manifest_value(manifest: pd.DataFrame, field: str, default: float) -> float:
    if field not in manifest:
        return float(default)
    values = np.asarray(manifest[field], dtype=np.float64)
    unique = np.unique(values[np.isfinite(values)])
    if unique.size != 1:
        raise ValueError(f"Manifest field {field!r} must be constant; got {unique}")
    return float(unique[0])


def write_retrieval_h5(
    output_path: str | Path,
    records: Sequence[FormattedEventRecord],
    *,
    manifest: pd.DataFrame,
    generated_per_scenario: Mapping[int, int] | None = None,
    fluxcal_zp: float = 27.5,
    psfflux_zp: float = 31.4,
    lupt_k: float = 1.0,
    lupt_m5_mag: str | Sequence[float] = "23.9,25.0,24.7,24.0,23.3,22.1",
    scalar_reference: GwScalarReference,
) -> None:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    scenario_records: dict[int, FormattedEventRecord] = {}
    for record in records:
        scenario_records.setdefault(int(record.scenario_id), record)
    gw_records = [
        scenario_records[scenario_id] for scenario_id in sorted(scenario_records)
    ]
    parent_index = {
        int(record.scenario_id): index for index, record in enumerate(gw_records)
    }
    parent_gw_idx = np.asarray(
        [parent_index[int(record.scenario_id)] for record in records],
        dtype=np.int32,
    )

    n_gw = len(gw_records)
    n_optical = len(records)
    gw_chunk = max(1, min(256, n_gw or 1))
    optical_chunk = max(1, min(1024, n_optical or 1))
    string_dtype = h5py.string_dtype(encoding="utf-8")
    m5 = (
        _data_loader.parse_lupt_m5_mag(lupt_m5_mag)
        if isinstance(lupt_m5_mag, str)
        else np.asarray(lupt_m5_mag, dtype=np.float64).reshape(-1)
    )
    flux_factor, f5, lupt_b_njy = _data_loader.build_luptitude_params(
        fluxcal_zp=float(fluxcal_zp),
        psfflux_zp=float(psfflux_zp),
        lupt_k=float(lupt_k),
        lupt_m5_mag=m5,
    )

    with h5py.File(output, "w") as handle:
        gw = handle.create_group("events/gw_data")
        gw.create_dataset(
            "scalars",
            data=_array_or_empty(gw_records, "scalar", (7,), np.float32),
            chunks=(gw_chunk, 7),
            maxshape=(None, 7),
        )
        gw.create_dataset(
            "skymaps",
            data=_array_or_empty(gw_records, "skymap", (7, 19200), np.float32),
            chunks=(1, 7, 19200),
            maxshape=(None, 7, 19200),
        )
        gw.create_dataset(
            "ids",
            data=np.asarray([record.event_uid for record in gw_records], dtype=object),
            dtype=string_dtype,
        )
        gw.create_dataset(
            "event_uid",
            data=np.asarray([record.event_uid for record in gw_records], dtype=object),
            dtype=string_dtype,
        )
        gw.create_dataset(
            "physical_event_uid",
            data=np.asarray([PHYSICAL_EVENT_UID] * n_gw, dtype=object),
            dtype=string_dtype,
        )
        gw.create_dataset(
            "scenario_id",
            data=np.asarray(
                [record.scenario_id for record in gw_records], dtype=np.int16
            ),
        )
        gw.create_dataset(
            "simulation_id",
            data=np.asarray(
                [record.simulation_id for record in gw_records], dtype=np.int64
            ),
        )
        gw.create_dataset(
            "sample_class",
            data=np.asarray(["positive"] * n_gw, dtype=object),
            dtype=string_dtype,
        )
        gw.create_dataset("has_kn", data=np.ones(n_gw, dtype=np.int32))
        gw.create_dataset("neg_type", data=np.zeros(n_gw, dtype=np.int32))
        gw.create_dataset(
            "mej_dynamic",
            data=np.asarray(
                [record.mej_dynamic for record in gw_records], dtype=np.float32
            ),
        )
        gw.create_dataset(
            "mej_wind",
            data=np.asarray(
                [record.mej_wind for record in gw_records], dtype=np.float32
            ),
        )
        gw.create_dataset(
            "mej_tot",
            data=np.asarray(
                [record.mej_dynamic + record.mej_wind for record in gw_records],
                dtype=np.float32,
            ),
        )
        gw.create_dataset(
            "event_time_mjd",
            data=np.asarray(
                [record.event_time_mjd for record in gw_records], dtype=np.float64
            ),
        )
        gw.create_dataset(
            "source_type",
            data=np.asarray(["gw170817a"] * n_gw, dtype=object),
            dtype=string_dtype,
        )

        optical = handle.create_group("events/optical_data")
        for name, shape, dtype in (
            ("values", (MAX_LC_LENGTH, NUM_BANDS), np.float32),
            ("errors", (MAX_LC_LENGTH, NUM_BANDS), np.float32),
            ("masks", (MAX_LC_LENGTH, NUM_BANDS), np.float32),
            ("times", (MAX_LC_LENGTH,), np.float32),
            ("coordinates", (2,), np.float32),
        ):
            optical.create_dataset(
                name,
                data=_array_or_empty(records, name, shape, dtype),
                chunks=(optical_chunk, *shape),
                maxshape=(None, *shape),
            )
        optical.create_dataset("parent_gw_idx", data=parent_gw_idx)
        optical.create_dataset(
            "zero_time_mjd_base",
            data=np.asarray(
                [record.first_detection_mjd for record in records],
                dtype=np.float64,
            ),
        )
        optical.create_dataset(
            "first_detection_mjd",
            data=np.asarray(
                [record.first_detection_mjd for record in records],
                dtype=np.float64,
            ),
        )
        numeric_fields = (
            ("scenario_id", np.int16),
            ("coordinate_id", np.int32),
            ("libid", np.int32),
            ("is_true_position", np.bool_),
            ("posterior_probability", np.float64),
            ("skymap_credible_level", np.float32),
            ("too_nobs", np.int16),
            ("n_observations", np.int16),
            ("sim_explosion_mjd", np.float64),
        )
        for name, dtype in numeric_fields:
            optical.create_dataset(
                name,
                data=np.asarray(
                    [getattr(record, name) for record in records], dtype=dtype
                ),
            )
        optical.create_dataset(
            "too_mode",
            data=np.asarray([record.too_mode for record in records], dtype=object),
            dtype=string_dtype,
        )

        handle.attrs["dataset_mode"] = "gw170817a_lsst_scenarios_v1"
        handle.attrs["physical_event_uid"] = PHYSICAL_EVENT_UID
        handle.attrs["n_total_gw"] = n_gw
        handle.attrs["n_pos_gw"] = n_gw
        handle.attrs["n_neg_gw"] = 0
        handle.attrs["n_total_optical"] = n_optical
        handle.attrs["n_scenarios"] = n_gw
        handle.attrs["n_coordinates_per_scenario"] = int(n_optical // max(n_gw, 1))
        handle.attrs["real_trigger_mjd"] = _single_manifest_value(
            manifest, "real_trigger_mjd", REAL_TRIGGER_MJD
        )
        handle.attrs["scenario_trigger_mjds"] = np.asarray(
            manifest["trigger_mjd"], dtype=np.float64
        )
        handle.attrs["scenario_epoch_policy"] = "annual_sidereal_shift_within_opsim"
        handle.attrs["coordinate_panel_policy"] = (
            "shared_complete_true_plus_unique_90pct_posterior"
        )
        handle.attrs["host_distance_mpc"] = _single_manifest_value(
            manifest, "luminosity_distance", HOST_DISTANCE_MPC
        )
        handle.attrs["host_redshift_observed"] = _single_manifest_value(
            manifest, "host_redshift_observed", HOST_REDSHIFT_OBSERVED
        )
        handle.attrs["snana_redshift"] = _single_manifest_value(
            manifest, "redshift", float("nan")
        )
        handle.attrs["photometry_representation"] = "luptitude"
        handle.attrs["flux_input_column"] = "FLUXCAL"
        handle.attrs["fluxerr_input_column"] = "FLUXCALERR"
        handle.attrs["fluxcal_zp"] = float(fluxcal_zp)
        handle.attrs["psfflux_zp"] = float(psfflux_zp)
        handle.attrs["fluxcal_to_psfflux_factor"] = float(flux_factor)
        handle.attrs["lupt_k"] = float(lupt_k)
        handle.attrs["lupt_m5_mag"] = np.asarray(m5, dtype=np.float64)
        handle.attrs["lupt_f5sigma_njy"] = np.asarray(f5, dtype=np.float64)
        handle.attrs["lupt_b_njy"] = np.asarray(lupt_b_njy, dtype=np.float64)
        handle.attrs["values_semantics"] = "luptitude"
        handle.attrs["errors_semantics"] = "luptitude_sigma"
        handle.attrs["lupt_band_order"] = ",".join(LUPT_BAND_ORDER)
        handle.attrs["lightcurve_merge_window_hours"] = float(
            _data_loader.MERGE_WINDOW_HOURS
        )
        handle.attrs["lightcurve_merge_mode"] = str(_data_loader.MERGE_MODE)
        handle.attrs["lightcurve_merge_flux_domain"] = str(
            _data_loader.MERGE_FLUX_DOMAIN
        )
        handle.attrs["time_unit"] = "mjd_days"
        handle.attrs["time_zero_base_semantics"] = (
            "optical zero_time_mjd_base stores first_detection_mjd"
        )
        handle.attrs["gw_distance_policy"] = "unmodified_bayestar_no_virgo"
        handle.attrs["kn_distance_policy"] = "fixed_host_distance_40p7_mpc"
        handle.attrs["first_detection_policy"] = FIRST_DETECTION_POLICY
        handle.attrs["first_detection_snr_domain"] = FIRST_DETECTION_SNR_DOMAIN
        handle.attrs["scalar_column_names"] = SCALAR_COLUMN_NAMES
        handle.attrs["mass_scaling_policy"] = "none"
        handle.attrs["gw_scalar_source"] = scalar_reference.source
        handle.attrs["gw_scalar_posterior_h5"] = scalar_reference.posterior_h5
        handle.attrs["gw_scalar_posterior_dataset"] = scalar_reference.posterior_dataset
        handle.attrs["gw_skymap_path"] = scalar_reference.skymap_path
        handle.attrs["posterior_mass1_detector_median"] = (
            scalar_reference.mass1_detector
        )
        handle.attrs["posterior_mass2_detector_median"] = (
            scalar_reference.mass2_detector
        )
        handle.attrs["posterior_spin1z_median"] = scalar_reference.spin1z
        handle.attrs["posterior_spin2z_median"] = scalar_reference.spin2z
        handle.attrs["posterior_costheta_median"] = scalar_reference.costheta
        handle.attrs["bayestar_distmean_mpc"] = scalar_reference.distmean_mpc
        handle.attrs["bayestar_diststd_mpc"] = scalar_reference.diststd_mpc
        handle.attrs["spin_policy"] = SPIN_POLICY
        handle.attrs["gallery_task_definition"] = GALLERY_TASK_DEFINITION

        generated = dict(generated_per_scenario or {})
        for scenario_id in sorted(scenario_records):
            handle.attrs[f"n_generated_scenario_{scenario_id:02d}"] = int(
                generated.get(scenario_id, 0)
            )
            handle.attrs[f"n_kept_scenario_{scenario_id:02d}"] = int(
                sum(int(record.scenario_id) == scenario_id for record in records)
            )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sim-dir", default=DEFAULT_SNANA_SIM_DIR)
    parser.add_argument("--genversion", default=DEFAULT_GENVERSION)
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST)
    parser.add_argument("--coordinate-dir", default=DEFAULT_COORDINATE_DIR)
    parser.add_argument("--skymap", default=str(DEFAULT_SKYMAP))
    parser.add_argument("--output-h5", default=DEFAULT_OUTPUT_H5)
    parser.add_argument("--min-nobs", type=int, default=5)
    parser.add_argument("--target-per-scenario", type=int, default=50)
    parser.add_argument("--selection-seed", type=int, default=170817)
    parser.add_argument("--fluxcal-zp", type=float, default=27.5)
    parser.add_argument("--psfflux-zp", type=float, default=31.4)
    parser.add_argument("--lupt-k", type=float, default=1.0)
    parser.add_argument("--lupt-m5-mag", default="23.9,25.0,24.7,24.0,23.3,22.1")
    parser.add_argument("--posterior-h5", default=str(DEFAULT_POSTERIOR_H5))
    parser.add_argument("--posterior-dataset", default=DEFAULT_POSTERIOR_DATASET)
    args = parser.parse_args(argv)

    scalar_reference = load_posterior_scalar_reference(
        args.posterior_h5, args.posterior_dataset, args.skymap
    )
    candidates = format_snana_fits_to_records(
        sim_dir=args.sim_dir,
        genversion=args.genversion,
        manifest_path=args.manifest,
        coordinate_dir=args.coordinate_dir,
        skymap_path=args.skymap,
        fluxcal_zp=args.fluxcal_zp,
        psfflux_zp=args.psfflux_zp,
        lupt_k=args.lupt_k,
        lupt_m5_mag=args.lupt_m5_mag,
        min_nobs=args.min_nobs,
        scalar_reference=scalar_reference,
    )
    manifest = _load_manifest(args.manifest)
    scenario_ids = manifest["scenario_id"].astype(int).tolist()
    records = select_complete_coordinate_panel(
        candidates,
        target_per_scenario=args.target_per_scenario,
        expected_scenario_ids=scenario_ids,
        seed=args.selection_seed,
    )
    write_retrieval_h5(
        args.output_h5,
        records,
        manifest=manifest,
        generated_per_scenario=generated_counts_from_manifest(args.manifest),
        fluxcal_zp=args.fluxcal_zp,
        psfflux_zp=args.psfflux_zp,
        lupt_k=args.lupt_k,
        lupt_m5_mag=args.lupt_m5_mag,
        scalar_reference=scalar_reference,
    )
    print(
        f"Selected {len(records)} of {len(candidates)} usable light curves "
        f"across {len(scenario_ids)} scenarios and wrote {args.output_h5}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
