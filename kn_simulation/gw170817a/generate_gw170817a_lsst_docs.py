#!/usr/bin/env python3
"""Generate GW170817A LSST/SNANA manifest, SIMLIB, and INPUT files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

import numpy as np
import pandas as pd


BASE_DIR = Path("/fred/oz016/bgao_kn")
GW170817A_DIR = BASE_DIR / "data" / "GW_real_events" / "GW_data" / "GW170817A"
DEFAULT_SKYMAP = GW170817A_DIR / "bayestar_no_virgo.fits"
DEFAULT_OUTPUT_DIR = GW170817A_DIR / "lsst_redshift_experiment"
DEFAULT_OPSIM_DB = BASE_DIR / "data" / "rubin_sim" / "baseline" / "baseline_v5.0.1_10yrs.db"
DEFAULT_REDSHIFTS = (0.01, 0.02, 0.03, 0.05, 0.08, 0.12, 0.16, 0.22)
DEFAULT_N_PER_REDSHIFT = 200
DEFAULT_GENVERSION = "LSST_KN_GW170817A_REDSHIFT_GRID"
SIMGEN_DUMP_VAR_GROUPS = (
    ("CID", "LIBID", "GENTYPE", "SNTYPE", "ZCMB", "RA", "DECL", "PEAKMJD", "PEAKMJD_SMEAR", "MJD_TRIGGER"),
    ("MJD_DETECT_FIRST", "MJD_DETECT_LAST", "NOBS", "NEPOCH", "MWEBV", "MU", "SNRMAX", "SNRMAX2", "SNRMAX3"),
    ("SNRMAX_u", "SNRMAX_g", "SNRMAX_r", "SNRMAX_i", "SNRMAX_z", "SNRMAX_Y"),
    ("TIME_ABOVE_SNRMIN", "CUTMASK", "SIM_SEARCHEFF_MASK"),
)


def parse_redshifts(value: str | Sequence[float]) -> list[float]:
    if isinstance(value, str):
        values = [float(part.strip()) for part in value.split(",") if part.strip()]
    else:
        values = [float(item) for item in value]
    if not values or any((not np.isfinite(z)) or z <= 0 for z in values):
        raise ValueError(f"Redshifts must be finite positive values: {values}")
    return values


def parse_n_per_redshift(value: str | int | Sequence[int], redshifts: Sequence[float]) -> list[int]:
    if isinstance(value, str):
        counts = [int(part.strip()) for part in value.split(",") if part.strip()]
    elif isinstance(value, (int, np.integer)):
        counts = [int(value)]
    else:
        counts = [int(item) for item in value]

    n_redshifts = len(parse_redshifts(redshifts))
    if len(counts) not in (1, n_redshifts):
        raise ValueError(
            f"n_per_redshift must contain either 1 value or {n_redshifts} values; got {len(counts)}."
        )
    if any(count <= 0 for count in counts):
        raise ValueError(f"n_per_redshift values must be positive integers: {counts}")
    if len(counts) == 1:
        counts = counts * n_redshifts
    return counts


def luminosity_distance_mpc(redshift: float) -> float:
    from astropy import units as u
    from astropy.cosmology import Planck15 as cosmo

    return float(cosmo.luminosity_distance(float(redshift)).to(u.Mpc).value)


def _moc_probability(moc_map: Any) -> np.ndarray:
    from ligo.skymap.moc import uniq2pixarea

    return np.asarray(moc_map["PROBDENSITY"], dtype=np.float64) * np.asarray(
        uniq2pixarea(moc_map["UNIQ"]), dtype=np.float64
    )


def summarize_moc_skymap(skymap_path: str | Path) -> Dict[str, Any]:
    from ligo.skymap.io.fits import read_sky_map

    moc_map = read_sky_map(str(skymap_path), moc=True, distances=True)
    probability = _moc_probability(moc_map)
    return {
        "path": str(Path(skymap_path).expanduser()),
        "n_pixels": int(len(moc_map)),
        "has_distance": all(col in moc_map.colnames for col in ("DISTMU", "DISTSIGMA", "DISTNORM")),
        "probability_sum": float(np.sum(probability)),
    }


def load_probability_pixels_from_moc(
    skymap_path: str | Path,
    *,
    credible_level_max: float = 0.9,
) -> Dict[str, np.ndarray]:
    import healpy as hp
    import ligo.skymap.postprocess as postprocess
    from ligo.skymap.io.fits import read_sky_map
    from ligo.skymap.moc import uniq2nest

    moc_map = read_sky_map(str(skymap_path), moc=True, distances=True)
    probability = _moc_probability(moc_map)
    probdensity = np.asarray(moc_map["PROBDENSITY"], dtype=np.float64)
    credible = np.asarray(postprocess.find_greedy_credible_levels(probability, probdensity), dtype=np.float64)
    credible = np.clip(credible, 0.0, 1.0)

    keep = np.isfinite(probability) & np.isfinite(credible) & (credible <= float(credible_level_max))
    if not np.any(keep):
        raise ValueError(f"No pixels pass credible_level_max={credible_level_max}")

    order, ipix = uniq2nest(np.asarray(moc_map["UNIQ"]))
    ra = np.empty(len(order), dtype=np.float64)
    dec = np.empty(len(order), dtype=np.float64)
    for level in np.unique(order):
        mask = order == level
        theta, phi = hp.pix2ang(2 ** int(level), ipix[mask], nest=True)
        ra[mask] = np.degrees(phi)
        dec[mask] = 90.0 - np.degrees(theta)

    return {
        "ra": ra[keep],
        "dec": dec[keep],
        "probability": probability[keep],
        "credible_level": credible[keep],
        "pixel_index": np.arange(len(order), dtype=np.int64)[keep],
    }


def _normalized_weights(probability: Sequence[float]) -> np.ndarray:
    weights = np.asarray(probability, dtype=np.float64).reshape(-1)
    if weights.size == 0:
        raise ValueError("At least one probability value is required.")
    weights = np.where(np.isfinite(weights) & (weights > 0), weights, 0.0)
    total = float(np.sum(weights))
    if total <= 0:
        return np.full(weights.shape, 1.0 / float(weights.size), dtype=np.float64)
    return weights / total


def build_manifest_from_probability_pixels(
    *,
    ra: Sequence[float],
    dec: Sequence[float],
    probability: Sequence[float],
    credible_level: Sequence[float],
    redshifts: Sequence[float] = DEFAULT_REDSHIFTS,
    n_per_redshift: str | int | Sequence[int] = DEFAULT_N_PER_REDSHIFT,
    seed: int = 170817,
    pixel_index: Sequence[int] | None = None,
) -> pd.DataFrame:
    ra_arr = np.asarray(ra, dtype=np.float64).reshape(-1)
    dec_arr = np.asarray(dec, dtype=np.float64).reshape(-1)
    prob_arr = np.asarray(probability, dtype=np.float64).reshape(-1)
    cred_arr = np.asarray(credible_level, dtype=np.float64).reshape(-1)
    if not (ra_arr.shape == dec_arr.shape == prob_arr.shape == cred_arr.shape):
        raise ValueError("ra, dec, probability, and credible_level must have the same shape.")
    pix_arr = np.arange(ra_arr.size, dtype=np.int64) if pixel_index is None else np.asarray(pixel_index, dtype=np.int64)
    if pix_arr.shape != ra_arr.shape:
        raise ValueError("pixel_index must have the same shape as ra.")

    redshift_values = parse_redshifts(redshifts)
    counts = parse_n_per_redshift(n_per_redshift, redshift_values)

    rng = np.random.default_rng(int(seed))
    weights = _normalized_weights(prob_arr)
    rows: list[dict[str, Any]] = []
    sim_event_id = 0
    for redshift_bin, (redshift, n_each) in enumerate(zip(redshift_values, counts)):
        selected = rng.choice(ra_arr.size, size=n_each, replace=True, p=weights)
        target_distance = luminosity_distance_mpc(float(redshift))
        for local_idx, idx in enumerate(selected.tolist()):
            rows.append(
                {
                    "sim_event_id": int(sim_event_id),
                    "redshift_bin": int(redshift_bin),
                    "redshift": float(redshift),
                    "ra": float(ra_arr[idx]),
                    "dec": float(dec_arr[idx]),
                    "skymap_credible_level": float(cred_arr[idx]),
                    "skymap_pixel_index": int(pix_arr[idx]),
                    "target_distance_mpc": float(target_distance),
                    "local_index_in_redshift": int(local_idx),
                    "seed": int(seed),
                }
            )
            sim_event_id += 1
    return pd.DataFrame(rows)


def build_manifest_from_skymap(
    skymap_path: str | Path,
    *,
    credible_level_max: float = 0.9,
    redshifts: Sequence[float] = DEFAULT_REDSHIFTS,
    n_per_redshift: str | int | Sequence[int] = DEFAULT_N_PER_REDSHIFT,
    seed: int = 170817,
) -> pd.DataFrame:
    pixels = load_probability_pixels_from_moc(skymap_path, credible_level_max=credible_level_max)
    return build_manifest_from_probability_pixels(
        ra=pixels["ra"],
        dec=pixels["dec"],
        probability=pixels["probability"],
        credible_level=pixels["credible_level"],
        pixel_index=pixels["pixel_index"],
        redshifts=redshifts,
        n_per_redshift=n_per_redshift,
        seed=seed,
    )


def rescale_skymap_distance_channels(
    skymap: np.ndarray,
    *,
    target_distance_mpc: float,
    reference_distance_mpc: float,
) -> np.ndarray:
    arr = np.asarray(skymap, dtype=np.float32).copy()
    if arr.ndim != 2 or arr.shape[0] != 7:
        raise ValueError(f"Expected skymap shape [7, N], got {arr.shape}")
    reference = float(reference_distance_mpc)
    target = float(target_distance_mpc)
    if not np.isfinite(reference) or reference <= 0:
        raise ValueError("reference_distance_mpc must be finite and positive.")
    if not np.isfinite(target) or target <= 0:
        raise ValueError("target_distance_mpc must be finite and positive.")
    scale = np.float32(target / reference)
    arr[5, :] *= scale
    arr[6, :] *= scale
    return arr


def rescale_scalar_distance_fields(
    scalar: Sequence[float],
    *,
    target_distance_mpc: float,
    target_distance_std_mpc: float,
) -> np.ndarray:
    arr = np.asarray(scalar, dtype=np.float32).copy().reshape(-1)
    if arr.shape != (7,):
        raise ValueError(f"Expected scalar shape [7], got {arr.shape}")
    arr[5] = np.float32(float(target_distance_mpc) / 1000.0)
    arr[6] = np.float32(float(target_distance_std_mpc) / 1000.0)
    return arr


def write_snana_input(
    output_path: str | Path,
    *,
    simlib_path: str | Path,
    genversion: str,
    n_lc: int,
    seed: int = 170817,
    peakmjd_start: float = 61000.0,
    peakmjd_end: float = 64500.0,
) -> None:
    dump_var_count = sum(len(group) for group in SIMGEN_DUMP_VAR_GROUPS)
    dump_var_lines = "\n".join("  " + " ".join(group) for group in SIMGEN_DUMP_VAR_GROUPS)
    text = f"""# GW170817A LSST redshift-grid retrieval experiment
SIMLIB_NREPEAT: 1
NGENTOT_LC: {int(n_lc)}
GENVERSION: {genversion}
GENMODEL: $SNDATA_ROOT/models/SIMSED/SIMSED.KN-BULLA19/SIMSED.BULLA-BNS-M3-3COMP
DNDZ: POWERLAW 1.0E-6 0.0
GENFILTERS: ugrizY
SIMSED_PARAM: COSTHETA
GENPEAK_COSTHETA: 0.7959
GENRANGE_COSTHETA: 0.7959 0.7959
GENSIGMA_COSTHETA: 0 0
SIMSED_PARAM: MEJDYN
GENPEAK_MEJDYN: 0.016
GENRANGE_MEJDYN: 0.016 0.016
GENSIGMA_MEJDYN: 0 0
SIMSED_PARAM: MEJWIND
GENPEAK_MEJWIND: 0.024
GENRANGE_MEJWIND: 0.024 0.024
GENSIGMA_MEJWIND: 0 0
SIMSED_PARAM: PHI
GENPEAK_PHI: 30
GENRANGE_PHI: 30 30
GENSIGMA_PHI: 0 0
GENRANGE_PEAKMJD: {float(peakmjd_start):.4f} {float(peakmjd_end):.4f}
GENRANGE_REDSHIFT: 0.009 0.22
USE_SIMLIB_REDSHIFT: 1
SOLID_ANGLE: 6.233
OPT_MWEBV: 1
OPT_MWCOLORLAW: 99
GENRANGE_TREST: -30 60
GENSOURCE: RANDOM
RANSEED: {int(seed)}
SIMLIB_FILE: {Path(simlib_path).expanduser().resolve()}
KCOR_FILE: $SNDATA_ROOT/kcor/LSST/baseline_1.9/kcor_LSST.fits
SEARCHEFF_PIPELINE_LOGIC_FILE: $SNDATA_ROOT/models/searcheff/SEARCHEFF_PIPELINE_LOGIC.DAT
SEARCHEFF_PIPELINE_FILE: $SNDATA_ROOT/models/searcheff/SEARCHEFF_PIPELINE_LSST_Heaviside.DAT
APPLY_SEARCHEFF_OPT: 0
APPLY_CUTWIN_OPT: 1
CUTWIN_NEPOCH: 2 +5
NEWMJD_DIF: 7.0000e-03
SIMSED_USE_BINARY: 1
SIMSED_PATH_BINARY: /fred/oz016/bgao_kn/SNANA/SNDATA_ROOT/models/SIMSED/BINARY
FORMAT_MASK: 32
SIMGEN_DUMP: {dump_var_count}
{dump_var_lines}
"""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def write_simlib_from_manifest(
    manifest: pd.DataFrame,
    *,
    opsim_db: str | Path,
    output_dir: str | Path,
    sim_name: str,
    nside: int = 256,
) -> Path:
    import opsimsummaryv2 as opsim

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    survey = opsim.OpSimSurvey(str(opsim_db))
    survey.compute_hp_rep(nside=int(nside), minVisits=1, maxVisits=10000)
    survey.sample_coordinates(
        manifest["ra"].to_numpy(np.float64),
        manifest["dec"].to_numpy(np.float64),
        manifest["redshift"].to_numpy(np.float64),
        nsides=int(nside),
        is_deg=True,
    )
    sim = opsim.sim_io.SNANA_Simlib(survey, out_path=str(out_dir), file_suffix=f"_{sim_name}")
    sim.write_SIMLIB()
    simlib_path = out_dir / f"{Path(opsim_db).stem}_{sim_name}.SIMLIB"
    if not simlib_path.exists():
        raise FileNotFoundError(f"Expected SIMLIB was not created: {simlib_path}")
    return simlib_path


def write_manifest_metadata(path: str | Path, payload: Mapping[str, Any]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(dict(payload), indent=2, sort_keys=True), encoding="utf-8")


def _finite_median(values, *, field_name):
    import numpy as np

    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        raise ValueError(f"Posterior field {field_name!r} has no finite values.")
    return float(np.median(finite))


def load_posterior_medians(posterior_h5, posterior_dataset="IMRPhenomPv2NRT_lowSpin_posterior"):
    """Return detector-frame mass and viewing-angle medians from GWTC-1."""
    import h5py

    required = {
        "m1_detector_frame_Msun": "mass1_detector_ref",
        "m2_detector_frame_Msun": "mass2_detector_ref",
        "costheta_jn": "viewing_costheta",
    }
    with h5py.File(posterior_h5, "r") as f:
        if posterior_dataset not in f:
            raise KeyError(f"Posterior dataset {posterior_dataset!r} not found in {posterior_h5}")
        posterior = f[posterior_dataset]
        dtype_names = set(posterior.dtype.names or ())
        missing = sorted(set(required) - dtype_names)
        if missing:
            raise ValueError(f"Posterior dataset missing required fields: {missing}")
        medians = {
            output_name: _finite_median(posterior[field_name][:], field_name=field_name)
            for field_name, output_name in required.items()
        }
    return medians


def _skymap_meta_distances(sky_map):
    import numpy as np

    distmean = sky_map.meta.get("distmean")
    diststd = sky_map.meta.get("diststd")
    if distmean is None or not np.isfinite(float(distmean)) or float(distmean) <= 0:
        distmean = float(np.median(np.asarray(sky_map["DISTMU"], dtype=np.float64)))
    if diststd is None or not np.isfinite(float(diststd)) or float(diststd) <= 0:
        diststd = float(np.std(np.asarray(sky_map["DISTMU"], dtype=np.float64)))
    return float(distmean), float(diststd)


def write_rescaled_skymap(sky_map, output_path, redshift):
    """Write a MOC FITS file with distance channels rescaled to a redshift."""
    import numpy as np
    from astropy.table import Table
    from ligo.skymap.io.fits import write_sky_map

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    reference_distance, reference_std = _skymap_meta_distances(sky_map)
    target_distance = luminosity_distance_mpc(float(redshift))
    scale = target_distance / reference_distance
    target_std = reference_std * scale

    table = Table(
        {
            "UNIQ": np.asarray(sky_map["UNIQ"]),
            "PROBDENSITY": np.asarray(sky_map["PROBDENSITY"]),
            "DISTMU": np.asarray(sky_map["DISTMU"], dtype=np.float64) * scale,
            "DISTSIGMA": np.asarray(sky_map["DISTSIGMA"], dtype=np.float64) * scale,
            "DISTNORM": np.asarray(sky_map["DISTNORM"]),
        }
    )
    table.meta.update(dict(sky_map.meta))
    table.meta["distmean"] = float(target_distance)
    table.meta["diststd"] = float(target_std)
    write_sky_map(str(output_path), table, moc=True)


def build_prepared_catalog(
    *,
    skymap_path,
    posterior_h5,
    posterior_dataset="IMRPhenomPv2NRT_lowSpin_posterior",
    redshifts=DEFAULT_REDSHIFTS,
    n_per_redshift=DEFAULT_N_PER_REDSHIFT,
    output_dir=DEFAULT_OUTPUT_DIR,
    skymap_dir=None,
    seed=170817,
    network_snr=32.4,
    trigger_mjd=62500.0,
    mej_dynamic=0.016,
    mej_wind=0.024,
    phi_deg=30.0,
):
    """Build the kn_simulation prepared catalog and per-redshift MOC skymaps.

    GW170817 is treated as a bright source: network_snr is fixed above the ToO
    threshold and the real angular localization is retained at every redshift.
    """
    import numpy as np
    from ligo.skymap.io.fits import read_sky_map

    if skymap_dir is None:
        skymap_dir = Path(output_dir) / "skymaps"
    output_dir = Path(output_dir)
    skymap_dir = Path(skymap_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    skymap_dir.mkdir(parents=True, exist_ok=True)

    sky_map = read_sky_map(str(skymap_path), moc=True, distances=True)
    pixels = load_probability_pixels_from_moc(skymap_path, credible_level_max=0.9)
    weights = np.asarray(pixels["probability"], dtype=np.float64)
    weights = np.where(np.isfinite(weights) & (weights > 0), weights, 0.0)
    total = float(weights.sum())
    if total <= 0:
        raise ValueError("No positive-probability pixels in the sky map.")
    weights = weights / total

    posterior = load_posterior_medians(posterior_h5, posterior_dataset)
    redshift_values = parse_redshifts(redshifts)
    counts = parse_n_per_redshift(n_per_redshift, redshift_values)

    rows = []
    for redshift_bin, (redshift, n_each) in enumerate(zip(redshift_values, counts)):
        skymap_file = skymap_dir / f"gw170817a_z{redshift:.4f}.fits"
        write_rescaled_skymap(sky_map, skymap_file, redshift)
        # One GW parent per redshift. Optical positions are sampled later;
        # n_each is the candidate light-curve budget, not a GW event count.
        rng = np.random.default_rng(
            np.random.SeedSequence([int(seed), int(redshift_bin)])
        )
        idx = int(rng.choice(len(pixels["ra"]), p=weights))
        distance = luminosity_distance_mpc(float(redshift))
        sim_id = int(redshift_bin)
        sequence = np.random.SeedSequence([int(seed), sim_id])
        snana_seq, coord_seq = sequence.spawn(2)
        snana_seed = int(snana_seq.generate_state(1, dtype=np.uint32)[0]) % 2_000_000_000 + 1
        coordinate_seed = int(coord_seq.generate_state(1, dtype=np.uint32)[0]) % 2_000_000_000 + 1
        rows.append(
            {
                "simulation_id": sim_id,
                "sim_event_id": sim_id,
                "redshift_bin": int(redshift_bin),
                "redshift": float(redshift),
                "optical_candidate_count": int(n_each),
                "luminosity_distance": float(distance),
                "target_distance_mpc": float(distance),
                "ra_deg": float(pixels["ra"][idx]),
                "dec_deg": float(pixels["dec"][idx]),
                "ra": float(pixels["ra"][idx]),
                "dec": float(pixels["dec"][idx]),
                "skymap_credible_level": float(pixels["credible_level"][idx]),
                "skymap_pixel_index": int(pixels["pixel_index"][idx]),
                "skymap_path": str(skymap_file),
                "network_snr": float(network_snr),
                "trigger_mjd": float(trigger_mjd),
                "viewing_costheta": float(abs(posterior["viewing_costheta"])),
                "phi_deg": float(phi_deg),
                "mej_dynamic": float(mej_dynamic),
                "mej_wind": float(mej_wind),
                "mej_total": float(mej_dynamic + mej_wind),
                "mass1_detector_ref": float(posterior["mass1_detector_ref"]),
                "mass2_detector_ref": float(posterior["mass2_detector_ref"]),
                "snana_seed": int(snana_seed),
                "coordinate_seed": int(coordinate_seed),
                "seed": int(seed),
            }
        )
    return rows


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate GW170817A LSST redshift-grid SNANA inputs.")
    parser.add_argument("--skymap", default=str(DEFAULT_SKYMAP))
    parser.add_argument("--opsim-db", default=str(DEFAULT_OPSIM_DB))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--genversion", default=DEFAULT_GENVERSION)
    parser.add_argument("--redshifts", default=",".join(str(z) for z in DEFAULT_REDSHIFTS))
    parser.add_argument("--n-per-redshift", default=str(DEFAULT_N_PER_REDSHIFT))
    parser.add_argument("--credible-level-max", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=170817)
    parser.add_argument("--posterior-h5", default=str(GW170817A_DIR / "GW170817_GWTC-1.hdf5"))
    parser.add_argument("--posterior-dataset", default="IMRPhenomPv2NRT_lowSpin_posterior")
    parser.add_argument("--skymap-dir", default=str(BASE_DIR / "data" / "skymap" / "gw170817a"))
    parser.add_argument("--network-snr", type=float, default=32.4)
    parser.add_argument("--trigger-mjd", type=float, default=62500.0)
    parser.add_argument("--mej-dynamic", type=float, default=0.016)
    parser.add_argument("--mej-wind", type=float, default=0.024)
    parser.add_argument("--phi-deg", type=float, default=30.0)
    parser.add_argument("--prepared-catalog", action="store_true",
                        help="Write the kn_simulation prepared catalog and per-redshift MOC skymaps instead of legacy SIMLIB/INPUT.")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    redshifts = parse_redshifts(args.redshifts)
    n_per_redshift = parse_n_per_redshift(args.n_per_redshift, redshifts)
    manifest_path = out_dir / "gw170817a_lsst_manifest.csv"
    if args.prepared_catalog:
        rows = build_prepared_catalog(
            skymap_path=args.skymap,
            posterior_h5=args.posterior_h5,
            posterior_dataset=args.posterior_dataset,
            redshifts=redshifts,
            n_per_redshift=n_per_redshift,
            output_dir=out_dir,
            skymap_dir=args.skymap_dir,
            seed=int(args.seed),
            network_snr=float(args.network_snr),
            trigger_mjd=float(args.trigger_mjd),
            mej_dynamic=float(args.mej_dynamic),
            mej_wind=float(args.mej_wind),
            phi_deg=float(args.phi_deg),
        )
        catalog_path = out_dir / "gw170817a_prepared_catalog.csv"
        ids_path = out_dir / "simulation_ids.txt"
        frame = pd.DataFrame(rows)
        frame.to_csv(catalog_path, index=False)
        frame.to_csv(manifest_path, index=False)
        ids_path.write_text(
            "\n".join(str(int(v)) for v in frame["simulation_id"]) + "\n",
            encoding="utf-8",
        )
        write_manifest_metadata(
            out_dir / "gw170817a_lsst_manifest.meta.json",
            {
                "skymap": str(Path(args.skymap).expanduser().resolve()),
                "redshifts": redshifts,
                "n_gw_parents": int(len(frame)),
                "n_optical_candidates": int(sum(n_per_redshift)),
                "optical_candidate_count_by_redshift": {
                    f"{float(redshift):.4f}": int(count)
                    for redshift, count in zip(redshifts, n_per_redshift)
                },
                "seed": int(args.seed),
            },
        )
        print(f"Prepared catalog written: {catalog_path}")
        print(f"Prepared skymaps written under: {args.skymap_dir}")
        print(
            f"GW parents: {len(frame)}; "
            f"optical candidate budget: {sum(n_per_redshift)}"
        )
        return 0

    manifest = build_manifest_from_skymap(
        args.skymap,
        credible_level_max=float(args.credible_level_max),
        redshifts=redshifts,
        n_per_redshift=n_per_redshift,
        seed=int(args.seed),
    )
    manifest.to_csv(manifest_path, index=False)
    write_manifest_metadata(
        out_dir / "gw170817a_lsst_manifest.meta.json",
        {
            "skymap": str(Path(args.skymap).expanduser().resolve()),
            "opsim_db": str(Path(args.opsim_db).expanduser().resolve()),
            "genversion": str(args.genversion),
            "credible_level_max": float(args.credible_level_max),
            "redshifts": redshifts,
            "n_per_redshift": n_per_redshift,
            "n_per_redshift_by_redshift": {
                f"{float(redshift):.4f}": int(count) for redshift, count in zip(redshifts, n_per_redshift)
            },
            "n_manifest_rows": int(len(manifest)),
            "seed": int(args.seed),
        },
    )
    if args.dry_run:
        print(f"Manifest written: {manifest_path}")
        return 0

    simlib_path = write_simlib_from_manifest(
        manifest,
        opsim_db=args.opsim_db,
        output_dir=out_dir / "SIMLIB",
        sim_name=str(args.genversion),
    )
    input_path = out_dir / "SIM_INPUT" / f"SIMGEN_{args.genversion}.INPUT"
    write_snana_input(
        input_path,
        simlib_path=simlib_path,
        genversion=str(args.genversion),
        n_lc=len(manifest),
        seed=int(args.seed),
    )
    print(f"Manifest written: {manifest_path}")
    print(f"SIMLIB written: {simlib_path}")
    print(f"SNANA INPUT written: {input_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
