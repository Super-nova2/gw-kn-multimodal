#!/usr/bin/env python3
"""Build a MAGIKS-compatible retrieval HDF5 for the GW170817A LSST experiment."""

from __future__ import annotations

import argparse
import importlib.util
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Mapping, Sequence

import h5py
import numpy as np
import pandas as pd
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
    DEFAULT_REDSHIFTS,
    GW170817A_DIR,
    luminosity_distance_mpc,
    rescale_scalar_distance_fields,
    rescale_skymap_distance_channels,
)


MAX_LC_LENGTH = int(_data_loader.MAX_LC_LENGTH)
NUM_BANDS = int(_data_loader.NUM_BANDS)
DEFAULT_OUTPUT_H5 = "/fred/oz016/bgao_kn/data/ALBEF_dataset/gw170817a_lsst_redshift_test.h5"
DEFAULT_MANIFEST = str(GW170817A_DIR / "lsst_redshift_experiment" / "gw170817a_lsst_manifest.csv")
DEFAULT_SNANA_SIM_DIR = "/fred/oz016/bgao_kn/SNANA/SNDATA_ROOT/SIM/gw170817a"
DEFAULT_SKYMAP = str(GW170817A_DIR / "bayestar_no_virgo.fits")
DEFAULT_POSTERIOR_H5 = str(GW170817A_DIR / "GW170817_GWTC-1.hdf5")
DEFAULT_POSTERIOR_DATASET = "IMRPhenomPv2NRT_lowSpin_posterior"
REFERENCE_REDSHIFT = 0.01
FIRST_DETECTION_POLICY = "psfflux_snr5_then_photflag_then_head_mjd_detect_first"
FIRST_DETECTION_SNR_DOMAIN = "merged_psfflux"
SCALAR_COLUMN_NAMES = "mass1_detector,mass2_detector,spin1z,spin2z,costheta,distmean_gpc,diststd_gpc"
MASS_SCALING_POLICY = "detector_frame_scaled_relative_to_zref_0.01"
GW_SCALAR_SOURCE = "posterior_median"
SPIN_POLICY = "fixed_zero"
GALLERY_TASK_DEFINITION = "Each gallery contains 1 positive KN counterpart and non-KN optical distractors."
LUPT_BAND_ORDER = ("u", "g", "r", "i", "z", "Y")


@dataclass(frozen=True)
class GwScalarReference:
    mass1_detector: float
    mass2_detector: float
    costheta: float
    posterior_h5: str
    posterior_dataset: str
    spin1z: float = 0.0
    spin2z: float = 0.0
    source: str = GW_SCALAR_SOURCE

    def scalar_prefix(self) -> np.ndarray:
        return np.asarray(
            [
                self.mass1_detector,
                self.mass2_detector,
                self.spin1z,
                self.spin2z,
                self.costheta,
                0.0,
                0.0,
            ],
            dtype=np.float32,
        )


@dataclass(frozen=True)
class FormattedEventRecord:
    sim_event_id: int
    values: np.ndarray
    errors: np.ndarray
    masks: np.ndarray
    times: np.ndarray
    coordinates: np.ndarray
    event_time_mjd: float
    first_detection_mjd: float
    redshift: float
    redshift_bin: int
    credible_level: float
    scalar: np.ndarray
    skymap: np.ndarray
    event_uid: str
    simulation_id: int
    sample_class: str
    mej_dynamic: float
    mej_wind: float


def _load_manifest(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"sim_event_id", "redshift", "redshift_bin", "skymap_credible_level"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Manifest missing required columns: {missing}")
    return df


def generated_counts_from_manifest(manifest_path: str | Path) -> Dict[float, int]:
    df = _load_manifest(manifest_path)
    counts = df["redshift"].value_counts().sort_index()
    return {float(redshift): int(count) for redshift, count in counts.items()}


def _finite_median(values: np.ndarray, *, field_name: str) -> float:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        raise ValueError(f"Posterior field {field_name!r} has no finite values.")
    return float(np.median(finite))


def load_posterior_scalar_reference(
    posterior_h5: str | Path = DEFAULT_POSTERIOR_H5,
    posterior_dataset: str = DEFAULT_POSTERIOR_DATASET,
) -> GwScalarReference:
    posterior_path = Path(posterior_h5)
    required_fields = {
        "m1_detector_frame_Msun": "mass1_detector",
        "m2_detector_frame_Msun": "mass2_detector",
        "costheta_jn": "costheta",
    }
    with h5py.File(posterior_path, "r") as f:
        if posterior_dataset not in f:
            raise KeyError(f"Posterior dataset {posterior_dataset!r} not found in {posterior_path}")
        posterior = f[posterior_dataset]
        dtype_names = set(posterior.dtype.names or ())
        missing = sorted(set(required_fields) - dtype_names)
        if missing:
            raise ValueError(
                f"Posterior dataset {posterior_dataset!r} missing required fields: {missing}"
            )

        medians = {
            output_name: _finite_median(posterior[field_name][:], field_name=field_name)
            for field_name, output_name in required_fields.items()
        }

    return GwScalarReference(
        mass1_detector=medians["mass1_detector"],
        mass2_detector=medians["mass2_detector"],
        costheta=medians["costheta"],
        posterior_h5=str(posterior_path),
        posterior_dataset=str(posterior_dataset),
    )


def _load_base_skymap(skymap_path: str | Path) -> tuple[np.ndarray, float, float]:
    skymap = _data_loader.sample_moc_skymap(str(skymap_path)).detach().cpu().numpy().astype(np.float32)
    _nest_map, meta = read_sky_map(str(skymap_path), nest=True)
    distmean = float(meta.get("distmean"))
    diststd = float(meta.get("diststd"))
    if not np.isfinite(distmean) or distmean <= 0:
        raise ValueError(f"Invalid distmean in {skymap_path}: {distmean}")
    if not np.isfinite(diststd) or diststd <= 0:
        raise ValueError(f"Invalid diststd in {skymap_path}: {diststd}")
    return skymap, distmean, diststd


def scale_detector_frame_masses(
    scalar: np.ndarray,
    *,
    redshift: float,
    reference_redshift: float = REFERENCE_REDSHIFT,
) -> np.ndarray:
    out = np.asarray(scalar, dtype=np.float32).copy()
    mass_scale = (1.0 + float(redshift)) / (1.0 + float(reference_redshift))
    out[0] = np.float32(out[0] * mass_scale)
    out[1] = np.float32(out[1] * mass_scale)
    return out


def _scaled_gw_inputs(
    *,
    base_skymap: np.ndarray,
    reference_distance_mpc: float,
    reference_distance_std_mpc: float,
    redshift: float,
    scalar_prefix: Sequence[float],
) -> tuple[np.ndarray, np.ndarray]:
    target_distance = luminosity_distance_mpc(float(redshift))
    target_std = float(reference_distance_std_mpc) * target_distance / float(reference_distance_mpc)
    scalar_prefix = scale_detector_frame_masses(
        np.asarray(scalar_prefix, dtype=np.float32),
        redshift=float(redshift),
    )
    scalar = rescale_scalar_distance_fields(
        scalar_prefix,
        target_distance_mpc=target_distance,
        target_distance_std_mpc=target_std,
    )
    skymap = rescale_skymap_distance_channels(
        base_skymap,
        target_distance_mpc=target_distance,
        reference_distance_mpc=float(reference_distance_mpc),
    )
    return scalar.astype(np.float32, copy=False), skymap.astype(np.float32, copy=False)


def _finite_float(value: object, default: float = float("nan")) -> float:
    try:
        out = float(value)
    except Exception:
        return float(default)
    return out if np.isfinite(out) else float(default)


def _head_value(row, names: Sequence[str], default: float = float("nan")) -> float:
    available = set(getattr(row, "array").names)
    for name in names:
        if name in available:
            out = _finite_float(row[name])
            if np.isfinite(out):
                return out
    return float(default)


def _format_lightcurve(
    *,
    mjd: np.ndarray,
    fluxcal: np.ndarray,
    fluxcalerr: np.ndarray,
    band: np.ndarray,
    event_time_mjd: float,
    fluxcal_to_psfflux_factor: float,
    psfflux_zp: float,
    lupt_b_njy: Sequence[float],
    min_nobs: int,
    photflag: np.ndarray | None = None,
    head_mjd_detect_first: float | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float] | None:
    merged_mjd, merged_psfflux, merged_psffluxerr, merged_band = _data_loader.merge_photometry_psfflux(
        mjd=np.asarray(mjd, dtype=np.float64),
        fluxcal=np.asarray(fluxcal, dtype=np.float64),
        fluxcalerr=np.asarray(fluxcalerr, dtype=np.float64),
        flt=np.asarray(band),
        fluxcal_to_psfflux_factor=float(fluxcal_to_psfflux_factor),
    )
    if len(merged_mjd) < int(min_nobs):
        return None

    first_detection_mjd = _data_loader.resolve_first_detection_mjd(
        snr_mjd=merged_mjd,
        snr_flux=merged_psfflux,
        snr_fluxerr=merged_psffluxerr,
        photflag_mjd=np.asarray(mjd, dtype=np.float64),
        photflag=photflag,
        head_mjd_detect_first=head_mjd_detect_first,
        snr_threshold=5.0,
    )
    if first_detection_mjd is None:
        return None

    lc_mjd, lc_val, lc_err, lc_band = _data_loader._transform_psfflux_to_luptitude(
        mjd=merged_mjd,
        psfflux=merged_psfflux,
        psffluxerr=merged_psffluxerr,
        flt=np.asarray(merged_band),
        psfflux_zp=float(psfflux_zp),
        lupt_b_njy=lupt_b_njy,
    )
    if len(lc_mjd) < int(min_nobs):
        return None

    rel_times = (np.asarray(lc_mjd, dtype=np.float64) - first_detection_mjd) / 100.0
    if rel_times.size > MAX_LC_LENGTH:
        keep = np.sort(np.argsort(np.abs(rel_times))[:MAX_LC_LENGTH])
        rel_times = rel_times[keep]
        lc_val = lc_val[keep]
        lc_err = lc_err[keep]
        lc_band = lc_band[keep]

    values = np.zeros((MAX_LC_LENGTH, NUM_BANDS), dtype=np.float32)
    errors = np.zeros((MAX_LC_LENGTH, NUM_BANDS), dtype=np.float32)
    masks = np.zeros((MAX_LC_LENGTH, NUM_BANDS), dtype=np.float32)
    times = np.zeros((MAX_LC_LENGTH,), dtype=np.float32)
    for i, (time_value, val, err, raw_band) in enumerate(zip(rel_times, lc_val, lc_err, lc_band)):
        band_idx = _data_loader._band_index_from_raw(raw_band)
        if band_idx is None:
            continue
        values[i, int(band_idx)] = np.float32(val)
        errors[i, int(band_idx)] = np.float32(err)
        masks[i, int(band_idx)] = 1.0
        times[i] = np.float32(time_value)
    if not np.any(masks > 0):
        return None
    return values, errors, masks, times, first_detection_mjd


def iter_snana_fits(sim_dir: str | Path, genversion: str = DEFAULT_GENVERSION):
    """Yield (HEAD, PHOT) FITS pairs from either the legacy combined layout or the
    current kn_simulation per-event layout."""
    root = Path(sim_dir)
    heads = sorted(
        list(root.rglob("*_HEAD.FITS")) + list(root.rglob("*_HEAD.FITS.gz"))
    )
    if not heads:
        # Legacy combined layout fallback.
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
        if str(head).endswith(".gz"):
            phot = head.with_name(head.name.replace("_HEAD.FITS.gz", "_PHOT.FITS.gz"))
        else:
            phot = head.with_name(head.name.replace("_HEAD.FITS", "_PHOT.FITS"))
        if phot.exists():
            yield head, phot


def format_snana_fits_to_records(
    *,
    sim_dir: str | Path,
    manifest_path: str | Path,
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
    manifest_lookup = {int(row.sim_event_id): row._asdict() for row in manifest.itertuples(index=False)}
    base_skymap, reference_distance, reference_std = _load_base_skymap(skymap_path)
    scalar_prefix = (scalar_reference or load_posterior_scalar_reference()).scalar_prefix()
    m5 = _data_loader.parse_lupt_m5_mag(str(lupt_m5_mag))
    fluxcal_to_psfflux_factor, _f5, lupt_b_njy = _data_loader.build_luptitude_params(
        fluxcal_zp=float(fluxcal_zp),
        psfflux_zp=float(psfflux_zp),
        lupt_k=float(lupt_k),
        lupt_m5_mag=m5,
    )

    records: list[FormattedEventRecord] = []
    for head_path, phot_path in iter_snana_fits(sim_dir, genversion=genversion):
        with fits.open(head_path) as hdul_head, fits.open(phot_path) as hdul_phot:
            head = hdul_head[1].data
            phot = hdul_phot[1].data
            phot_columns = set(phot.columns.names)
            for row in head:
                sim_event_id = int(_head_value(row, ("SIM_LIBID", "LIBID"), default=-1))
                if sim_event_id < 0:
                    sim_event_id = int(float(str(row["SNID"]).strip()))
                manifest_row = manifest_lookup.get(sim_event_id)
                if manifest_row is None:
                    continue

                start = int(row["PTROBS_MIN"]) - 1
                end = int(row["PTROBS_MAX"])
                event_time_mjd = _head_value(row, ("SIM_MJD_EXPLODE", "SIM_PEAKMJD", "PEAKMJD"))
                if not np.isfinite(event_time_mjd):
                    continue
                head_mjd_detect_first = _head_value(row, ("MJD_DETECT_FIRST",), default=float("nan"))
                photflag = np.asarray(phot["PHOTFLAG"][start:end], dtype=np.int64) if "PHOTFLAG" in phot_columns else None
                formatted = _format_lightcurve(
                    mjd=np.asarray(phot["MJD"][start:end], dtype=np.float64),
                    fluxcal=np.asarray(phot["FLUXCAL"][start:end], dtype=np.float64),
                    fluxcalerr=np.asarray(phot["FLUXCALERR"][start:end], dtype=np.float64),
                    band=np.asarray(phot["BAND"][start:end]),
                    photflag=photflag,
                    head_mjd_detect_first=head_mjd_detect_first,
                    event_time_mjd=float(event_time_mjd),
                    fluxcal_to_psfflux_factor=float(fluxcal_to_psfflux_factor),
                    psfflux_zp=float(psfflux_zp),
                    lupt_b_njy=lupt_b_njy,
                    min_nobs=int(min_nobs),
                )
                if formatted is None:
                    continue
                redshift = float(manifest_row["redshift"])
                scalar, skymap = _scaled_gw_inputs(
                    base_skymap=base_skymap,
                    reference_distance_mpc=reference_distance,
                    reference_distance_std_mpc=reference_std,
                    redshift=redshift,
                    scalar_prefix=scalar_prefix,
                )
                values, errors, masks, times, first_detection_mjd = formatted
                records.append(
                    FormattedEventRecord(
                        sim_event_id=sim_event_id,
                        values=values,
                        errors=errors,
                        masks=masks,
                        times=times,
                        coordinates=np.asarray([float(row["RA"]), float(row["DEC"])], dtype=np.float32),
                        event_time_mjd=float(event_time_mjd),
                        first_detection_mjd=float(first_detection_mjd),
                        redshift=redshift,
                        redshift_bin=int(manifest_row["redshift_bin"]),
                        credible_level=float(manifest_row["skymap_credible_level"]),
                        scalar=scalar,
                        skymap=skymap,
                        event_uid=f"gw170817a_{sim_event_id}",
                        simulation_id=int(sim_event_id),
                        sample_class=str(manifest_row.get("sample_class", "positive")),
                        mej_dynamic=float(manifest_row.get("mej_dynamic", 0.016)),
                        mej_wind=float(manifest_row.get("mej_wind", 0.024)),
                    )
                )
    return records


def _array_or_empty(records: Sequence[FormattedEventRecord], attr: str, shape: tuple[int, ...], dtype) -> np.ndarray:
    if not records:
        return np.empty((0, *shape), dtype=dtype)
    return np.asarray([getattr(record, attr) for record in records], dtype=dtype).reshape(len(records), *shape)


def write_retrieval_h5(
    output_path: str | Path,
    records: Sequence[FormattedEventRecord],
    *,
    generated_per_redshift: Mapping[float, int] | None = None,
    fluxcal_zp: float = 27.5,
    psfflux_zp: float = 31.4,
    lupt_k: float = 1.0,
    lupt_m5_mag: str | Sequence[float] = "23.9,25.0,24.7,24.0,23.3,22.1",
    scalar_reference: GwScalarReference | None = None,
) -> None:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    scalar_reference = scalar_reference or load_posterior_scalar_reference()
    n = int(len(records))
    gw_chunk = max(1, min(256, n or 1))
    opt_chunk = max(1, min(1024, n or 1))
    dt_str = h5py.string_dtype(encoding="utf-8")
    if isinstance(lupt_m5_mag, str):
        m5 = _data_loader.parse_lupt_m5_mag(str(lupt_m5_mag))
    else:
        m5 = np.asarray(lupt_m5_mag, dtype=np.float64).reshape(-1)
    fluxcal_to_psfflux_factor, lupt_f5sigma_njy, lupt_b_njy = _data_loader.build_luptitude_params(
        fluxcal_zp=float(fluxcal_zp),
        psfflux_zp=float(psfflux_zp),
        lupt_k=float(lupt_k),
        lupt_m5_mag=m5,
    )

    with h5py.File(output, "w") as f:
        gw = f.create_group("events/gw_data")
        gw.create_dataset("scalars", data=_array_or_empty(records, "scalar", (7,), np.float32), chunks=(gw_chunk, 7), maxshape=(None, 7))
        gw.create_dataset("skymaps", data=_array_or_empty(records, "skymap", (7, 19200), np.float32), chunks=(1, 7, 19200), maxshape=(None, 7, 19200))
        gw.create_dataset("ids", data=np.asarray([f"gw170817a_{r.sim_event_id}" for r in records], dtype=object), dtype=dt_str, chunks=(gw_chunk,), maxshape=(None,))
        gw.create_dataset("event_uid", data=np.asarray([r.event_uid for r in records], dtype=object), dtype=dt_str, chunks=(gw_chunk,), maxshape=(None,))
        gw.create_dataset("simulation_id", data=np.asarray([r.simulation_id for r in records], dtype=np.int64), chunks=(gw_chunk,), maxshape=(None,))
        gw.create_dataset("sample_class", data=np.asarray([r.sample_class for r in records], dtype=object), dtype=dt_str, chunks=(gw_chunk,), maxshape=(None,))
        gw.create_dataset("mej_dynamic", data=np.asarray([r.mej_dynamic for r in records], dtype=np.float32), chunks=(gw_chunk,), maxshape=(None,))
        gw.create_dataset("mej_wind", data=np.asarray([r.mej_wind for r in records], dtype=np.float32), chunks=(gw_chunk,), maxshape=(None,))
        gw.create_dataset("has_kn", data=np.ones((n,), dtype=np.int32), chunks=(gw_chunk,), maxshape=(None,))
        gw.create_dataset("neg_type", data=np.zeros((n,), dtype=np.int32), chunks=(gw_chunk,), maxshape=(None,))
        gw.create_dataset("mej_tot", data=np.full((n,), 0.04, dtype=np.float32), chunks=(gw_chunk,), maxshape=(None,))
        gw.create_dataset("event_time_mjd", data=np.asarray([r.event_time_mjd for r in records], dtype=np.float64), chunks=(gw_chunk,), maxshape=(None,))
        gw.create_dataset("source_type", data=np.asarray(["gw170817a"] * n, dtype=object), dtype=dt_str, chunks=(gw_chunk,), maxshape=(None,))
        gw.create_dataset("redshift", data=np.asarray([r.redshift for r in records], dtype=np.float64), chunks=(gw_chunk,), maxshape=(None,))
        gw.create_dataset("redshift_bin", data=np.asarray([r.redshift_bin for r in records], dtype=np.int16), chunks=(gw_chunk,), maxshape=(None,))
        gw.create_dataset("credible_level", data=np.asarray([r.credible_level for r in records], dtype=np.float32), chunks=(gw_chunk,), maxshape=(None,))

        opt = f.create_group("events/optical_data")
        opt.create_dataset("values", data=_array_or_empty(records, "values", (MAX_LC_LENGTH, NUM_BANDS), np.float32), chunks=(opt_chunk, MAX_LC_LENGTH, NUM_BANDS), maxshape=(None, MAX_LC_LENGTH, NUM_BANDS))
        opt.create_dataset("errors", data=_array_or_empty(records, "errors", (MAX_LC_LENGTH, NUM_BANDS), np.float32), chunks=(opt_chunk, MAX_LC_LENGTH, NUM_BANDS), maxshape=(None, MAX_LC_LENGTH, NUM_BANDS))
        opt.create_dataset("masks", data=_array_or_empty(records, "masks", (MAX_LC_LENGTH, NUM_BANDS), np.float32), chunks=(opt_chunk, MAX_LC_LENGTH, NUM_BANDS), maxshape=(None, MAX_LC_LENGTH, NUM_BANDS))
        opt.create_dataset("times", data=_array_or_empty(records, "times", (MAX_LC_LENGTH,), np.float32), chunks=(opt_chunk, MAX_LC_LENGTH), maxshape=(None, MAX_LC_LENGTH))
        opt.create_dataset("zero_time_mjd_base", data=np.asarray([r.first_detection_mjd for r in records], dtype=np.float64), chunks=(opt_chunk,), maxshape=(None,))
        opt.create_dataset("first_detection_mjd", data=np.asarray([r.first_detection_mjd for r in records], dtype=np.float64), chunks=(opt_chunk,), maxshape=(None,))
        opt.create_dataset("coordinates", data=_array_or_empty(records, "coordinates", (2,), np.float32), chunks=(opt_chunk, 2), maxshape=(None, 2))
        opt.create_dataset("parent_gw_idx", data=np.arange(n, dtype=np.int32), chunks=(opt_chunk,), maxshape=(None,))

        f.attrs["dataset_mode"] = "gw170817a_lsst_redshift_balanced_firstdet_v2"
        f.attrs["n_total_gw"] = n
        f.attrs["n_pos_gw"] = n
        f.attrs["n_neg_gw"] = 0
        f.attrs["n_total_optical"] = n
        f.attrs["photometry_representation"] = "luptitude"
        f.attrs["flux_input_column"] = "FLUXCAL"
        f.attrs["fluxerr_input_column"] = "FLUXCALERR"
        f.attrs["fluxcal_zp"] = float(fluxcal_zp)
        f.attrs["psfflux_zp"] = float(psfflux_zp)
        f.attrs["fluxcal_to_psfflux_factor"] = float(fluxcal_to_psfflux_factor)
        f.attrs["lupt_k"] = float(lupt_k)
        f.attrs["lupt_m5_mag"] = np.asarray(m5, dtype=np.float64)
        f.attrs["lupt_f5sigma_njy"] = np.asarray(lupt_f5sigma_njy, dtype=np.float64)
        f.attrs["lupt_b_njy"] = np.asarray(lupt_b_njy, dtype=np.float64)
        f.attrs["values_semantics"] = "luptitude"
        f.attrs["errors_semantics"] = "luptitude_sigma"
        f.attrs["lupt_band_order"] = ",".join(LUPT_BAND_ORDER)
        f.attrs["lightcurve_merge_window_hours"] = float(_data_loader.MERGE_WINDOW_HOURS)
        f.attrs["lightcurve_merge_mode"] = str(_data_loader.MERGE_MODE)
        f.attrs["lightcurve_merge_flux_domain"] = str(_data_loader.MERGE_FLUX_DOMAIN)
        f.attrs["time_unit"] = "mjd_days"
        f.attrs["time_zero_base_semantics"] = "optical zero_time_mjd_base stores first_detection_mjd"
        f.attrs["gw_distance_policy"] = "bayestar distance channels rescaled to redshift luminosity distance"
        f.attrs["first_detection_policy"] = FIRST_DETECTION_POLICY
        f.attrs["first_detection_snr_domain"] = FIRST_DETECTION_SNR_DOMAIN
        f.attrs["scalar_column_names"] = SCALAR_COLUMN_NAMES
        f.attrs["mass_scaling_policy"] = MASS_SCALING_POLICY
        f.attrs["gw_scalar_source"] = scalar_reference.source
        f.attrs["gw_scalar_posterior_h5"] = scalar_reference.posterior_h5
        f.attrs["gw_scalar_posterior_dataset"] = scalar_reference.posterior_dataset
        f.attrs["posterior_mass1_detector_median"] = float(scalar_reference.mass1_detector)
        f.attrs["posterior_mass2_detector_median"] = float(scalar_reference.mass2_detector)
        f.attrs["posterior_costheta_median"] = float(scalar_reference.costheta)
        f.attrs["spin_policy"] = SPIN_POLICY
        f.attrs["reference_detector_mass1"] = float(scalar_reference.mass1_detector)
        f.attrs["reference_detector_mass2"] = float(scalar_reference.mass2_detector)
        f.attrs["reference_redshift"] = float(REFERENCE_REDSHIFT)
        f.attrs["gallery_task_definition"] = GALLERY_TASK_DEFINITION

        kept_by_z: Dict[float, int] = {}
        for record in records:
            key = round(float(record.redshift), 4)
            kept_by_z[key] = kept_by_z.get(key, 0) + 1
        for redshift, generated_count in dict(generated_per_redshift or {}).items():
            key = round(float(redshift), 4)
            f.attrs[f"n_generated_z{key:.4f}"] = int(generated_count)
            f.attrs[f"n_kept_z{key:.4f}"] = int(kept_by_z.get(key, 0))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build GW170817A LSST retrieval HDF5 from SNANA FITS output.")
    parser.add_argument("--sim-dir", default=DEFAULT_SNANA_SIM_DIR)
    parser.add_argument("--genversion", default=DEFAULT_GENVERSION)
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST)
    parser.add_argument("--skymap", default=DEFAULT_SKYMAP)
    parser.add_argument("--output-h5", default=DEFAULT_OUTPUT_H5)
    parser.add_argument("--min-nobs", type=int, default=5)
    parser.add_argument("--fluxcal-zp", type=float, default=27.5)
    parser.add_argument("--psfflux-zp", type=float, default=31.4)
    parser.add_argument("--lupt-k", type=float, default=1.0)
    parser.add_argument("--lupt-m5-mag", default="23.9,25.0,24.7,24.0,23.3,22.1")
    parser.add_argument("--posterior-h5", default=DEFAULT_POSTERIOR_H5)
    parser.add_argument("--posterior-dataset", default=DEFAULT_POSTERIOR_DATASET)
    args = parser.parse_args(argv)

    scalar_reference = load_posterior_scalar_reference(args.posterior_h5, args.posterior_dataset)
    records = format_snana_fits_to_records(
        sim_dir=args.sim_dir,
        genversion=args.genversion,
        manifest_path=args.manifest,
        skymap_path=args.skymap,
        fluxcal_zp=float(args.fluxcal_zp),
        psfflux_zp=float(args.psfflux_zp),
        lupt_k=float(args.lupt_k),
        lupt_m5_mag=str(args.lupt_m5_mag),
        min_nobs=int(args.min_nobs),
        scalar_reference=scalar_reference,
    )
    write_retrieval_h5(
        args.output_h5,
        records,
        generated_per_redshift=generated_counts_from_manifest(args.manifest),
        fluxcal_zp=float(args.fluxcal_zp),
        psfflux_zp=float(args.psfflux_zp),
        lupt_k=float(args.lupt_k),
        lupt_m5_mag=str(args.lupt_m5_mag),
        scalar_reference=scalar_reference,
    )
    print(f"Wrote {len(records)} formatted GW170817A records to {args.output_h5}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
