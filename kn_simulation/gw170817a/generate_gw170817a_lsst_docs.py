#!/usr/bin/env python3
"""Prepare fixed-physics GW170817A counterfactual LSST observing scenarios."""

from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path
from typing import Any, Mapping, Sequence

import h5py
import numpy as np
import pandas as pd
from astropy import units as u
from astropy.cosmology import Planck15, z_at_value
from ligo.skymap.io.fits import read_sky_map
from ligo.skymap.moc import uniq2pixarea

BASE_DIR = Path("/fred/oz016/bgao_kn")
GW170817A_DIR = BASE_DIR / "data" / "GW_real_events" / "GW_data" / "GW170817A"
DEFAULT_SKYMAP = GW170817A_DIR / "bayestar_no_virgo.fits"
DEFAULT_POSTERIOR_H5 = GW170817A_DIR / "GW170817_GWTC-1.hdf5"
DEFAULT_POSTERIOR_DATASET = "IMRPhenomPv2NRT_lowSpin_posterior"
DEFAULT_OUTPUT_DIR = GW170817A_DIR / "lsst_scenario_experiment"
DEFAULT_OPSIM_DB = (
    BASE_DIR / "data" / "rubin_sim" / "baseline_v5.1" / "baseline_v5.1.1_10yrs.db"
)
DEFAULT_GENVERSION = "LSST_KN_GW170817A_SCENARIOS"

PHYSICAL_EVENT_UID = "GW170817A"
REAL_TRIGGER_MJD = 57982.528523
SIDEREAL_YEAR_DAYS = 365.256363004
TRUE_RA_DEG = 197.450374
TRUE_DEC_DEG = -23.381495
HOST_DISTANCE_MPC = 40.7
HOST_REDSHIFT_OBSERVED = 0.009783
DEFAULT_N_SCENARIOS = 10
DEFAULT_CANDIDATE_COORDINATES = 80


def _finite_median(values: Sequence[float], *, field_name: str) -> float:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        raise ValueError(f"Posterior field {field_name!r} has no finite values.")
    return float(np.median(finite))


def summarize_moc_skymap(skymap_path: str | Path) -> dict[str, Any]:
    """Return provenance and normalization checks for the real 3D sky map."""
    moc_map = read_sky_map(str(skymap_path), moc=True, distances=True)
    probability = np.asarray(moc_map["PROBDENSITY"], dtype=np.float64) * np.asarray(
        uniq2pixarea(moc_map["UNIQ"]), dtype=np.float64
    )
    distmean, diststd = skymap_distance_summary(moc_map)
    return {
        "path": str(Path(skymap_path).expanduser().resolve()),
        "n_pixels": int(len(moc_map)),
        "has_distance": all(
            column in moc_map.colnames for column in ("DISTMU", "DISTSIGMA", "DISTNORM")
        ),
        "probability_sum": float(np.sum(probability)),
        "distmean_mpc": float(distmean),
        "diststd_mpc": float(diststd),
    }


def skymap_distance_summary(moc_map) -> tuple[float, float]:
    """Read the global BAYESTAR distance mean and standard deviation."""
    distmean = moc_map.meta.get("distmean")
    diststd = moc_map.meta.get("diststd")
    if distmean is None or not np.isfinite(float(distmean)) or float(distmean) <= 0:
        raise ValueError("GW170817A sky map is missing a finite positive distmean.")
    if diststd is None or not np.isfinite(float(diststd)) or float(diststd) <= 0:
        raise ValueError("GW170817A sky map is missing a finite positive diststd.")
    return float(distmean), float(diststd)


def load_posterior_medians(
    posterior_h5: str | Path,
    posterior_dataset: str = DEFAULT_POSTERIOR_DATASET,
) -> dict[str, float]:
    """Load detector masses, aligned spins, and signed inclination cosine."""
    path = Path(posterior_h5)
    with h5py.File(path, "r") as handle:
        if posterior_dataset not in handle:
            raise KeyError(
                f"Posterior dataset {posterior_dataset!r} not found in {path}"
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
        return {
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


def simulation_redshift_for_distance(distance_mpc: float) -> float:
    """Return the Planck15 redshift whose luminosity distance matches the host."""
    distance = float(distance_mpc)
    if not np.isfinite(distance) or distance <= 0:
        raise ValueError("distance_mpc must be finite and positive.")
    return float(z_at_value(Planck15.luminosity_distance, distance * u.Mpc))


def opsim_mjd_bounds(path: str | Path) -> tuple[float, float]:
    """Read the observation time range without modifying the OpSim database."""
    uri = f"file:{Path(path).expanduser().resolve()}?mode=ro"
    with sqlite3.connect(uri, uri=True) as connection:
        row = connection.execute(
            "SELECT MIN(observationStartMJD), MAX(observationStartMJD) "
            "FROM observations"
        ).fetchone()
    if row is None or row[0] is None or row[1] is None:
        raise ValueError(f"Could not read observationStartMJD bounds from {path}")
    return float(row[0]), float(row[1])


def annual_scenario_trigger_mjds(
    *,
    real_trigger_mjd: float,
    n_scenarios: int,
    opsim_min_mjd: float,
    opsim_max_mjd: float,
    sidereal_year_days: float = SIDEREAL_YEAR_DAYS,
    followup_days: float = 4.0,
) -> list[float]:
    """Select the earliest annual sidereal shifts fully covered by OpSim."""
    if int(n_scenarios) < 1:
        raise ValueError("n_scenarios must be >= 1")
    shifts: list[float] = []
    k = int(
        np.ceil((float(opsim_min_mjd) - float(real_trigger_mjd)) / sidereal_year_days)
    )
    while len(shifts) < int(n_scenarios):
        trigger = float(real_trigger_mjd) + k * float(sidereal_year_days)
        if trigger + float(followup_days) > float(opsim_max_mjd):
            break
        if trigger >= float(opsim_min_mjd):
            shifts.append(trigger)
        k += 1
    if len(shifts) != int(n_scenarios):
        raise ValueError(
            f"OpSim range [{opsim_min_mjd}, {opsim_max_mjd}] supports only "
            f"{len(shifts)} annual scenarios; requested {n_scenarios}."
        )
    return shifts


def _scenario_seeds(seed: int, scenario_id: int) -> tuple[int, int]:
    sequence = np.random.SeedSequence([int(seed), int(scenario_id)])
    snana_seq, _unused_coordinate_seq = sequence.spawn(2)
    snana_seed = (
        int(snana_seq.generate_state(1, dtype=np.uint32)[0]) % 2_000_000_000 + 1
    )
    shared_coordinate_seq = np.random.SeedSequence([int(seed), 0x170817])
    coordinate_seed = (
        int(shared_coordinate_seq.generate_state(1, dtype=np.uint32)[0]) % 2_000_000_000
        + 1
    )
    return snana_seed, coordinate_seed


def build_scenario_catalog(
    *,
    skymap_path: str | Path,
    posterior_h5: str | Path,
    opsim_db: str | Path,
    posterior_dataset: str = DEFAULT_POSTERIOR_DATASET,
    n_scenarios: int = DEFAULT_N_SCENARIOS,
    candidate_coordinates: int = DEFAULT_CANDIDATE_COORDINATES,
    real_trigger_mjd: float = REAL_TRIGGER_MJD,
    host_distance_mpc: float = HOST_DISTANCE_MPC,
    host_redshift_observed: float = HOST_REDSHIFT_OBSERVED,
    true_ra_deg: float = TRUE_RA_DEG,
    true_dec_deg: float = TRUE_DEC_DEG,
    seed: int = 170817,
    network_snr: float = 32.4,
    mej_dynamic: float = 0.016,
    mej_wind: float = 0.024,
    phi_deg: float = 30.0,
) -> list[dict[str, Any]]:
    """Build one physical event with several counterfactual observing epochs."""
    if int(candidate_coordinates) < 1:
        raise ValueError("candidate_coordinates must be >= 1")
    posterior = load_posterior_medians(posterior_h5, posterior_dataset)
    sky_summary = summarize_moc_skymap(skymap_path)
    opsim_min, opsim_max = opsim_mjd_bounds(opsim_db)
    triggers = annual_scenario_trigger_mjds(
        real_trigger_mjd=float(real_trigger_mjd),
        n_scenarios=int(n_scenarios),
        opsim_min_mjd=opsim_min,
        opsim_max_mjd=opsim_max,
    )
    simulation_redshift = simulation_redshift_for_distance(host_distance_mpc)

    rows: list[dict[str, Any]] = []
    for scenario_id, trigger_mjd in enumerate(triggers):
        snana_seed, coordinate_seed = _scenario_seeds(seed, scenario_id)
        rows.append(
            {
                "simulation_id": int(scenario_id),
                "sim_event_id": int(scenario_id),
                "scenario_id": int(scenario_id),
                "physical_event_uid": PHYSICAL_EVENT_UID,
                "optical_candidate_count": int(candidate_coordinates),
                "luminosity_distance": float(host_distance_mpc),
                "target_distance_mpc": float(host_distance_mpc),
                "redshift": float(simulation_redshift),
                "host_redshift_observed": float(host_redshift_observed),
                "ra_deg": float(true_ra_deg),
                "dec_deg": float(true_dec_deg),
                "ra": float(true_ra_deg),
                "dec": float(true_dec_deg),
                "skymap_path": str(Path(skymap_path).expanduser().resolve()),
                "skymap_distmean_mpc": float(sky_summary["distmean_mpc"]),
                "skymap_diststd_mpc": float(sky_summary["diststd_mpc"]),
                "network_snr": float(network_snr),
                "trigger_mjd": float(trigger_mjd),
                "real_trigger_mjd": float(real_trigger_mjd),
                "viewing_costheta": float(abs(posterior["costheta"])),
                "posterior_costheta": float(posterior["costheta"]),
                "phi_deg": float(phi_deg),
                "mej_dynamic": float(mej_dynamic),
                "mej_wind": float(mej_wind),
                "mej_total": float(mej_dynamic + mej_wind),
                "mass1_detector": float(posterior["mass1_detector"]),
                "mass2_detector": float(posterior["mass2_detector"]),
                "spin1z": float(posterior["spin1z"]),
                "spin2z": float(posterior["spin2z"]),
                "snana_h0": float(Planck15.H0.value),
                "snana_omega_matter": float(Planck15.Om0),
                "snana_omega_lambda": float(Planck15.Ode0),
                "snana_seed": int(snana_seed),
                "coordinate_seed": int(coordinate_seed),
                "seed": int(seed),
            }
        )
    return rows


def write_manifest_metadata(path: str | Path, payload: Mapping[str, Any]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(dict(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skymap", default=str(DEFAULT_SKYMAP))
    parser.add_argument("--posterior-h5", default=str(DEFAULT_POSTERIOR_H5))
    parser.add_argument("--posterior-dataset", default=DEFAULT_POSTERIOR_DATASET)
    parser.add_argument("--opsim-db", default=str(DEFAULT_OPSIM_DB))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--n-scenarios", type=int, default=DEFAULT_N_SCENARIOS)
    parser.add_argument(
        "--candidate-coordinates", type=int, default=DEFAULT_CANDIDATE_COORDINATES
    )
    parser.add_argument("--real-trigger-mjd", type=float, default=REAL_TRIGGER_MJD)
    parser.add_argument("--host-distance-mpc", type=float, default=HOST_DISTANCE_MPC)
    parser.add_argument(
        "--host-redshift-observed", type=float, default=HOST_REDSHIFT_OBSERVED
    )
    parser.add_argument("--true-ra-deg", type=float, default=TRUE_RA_DEG)
    parser.add_argument("--true-dec-deg", type=float, default=TRUE_DEC_DEG)
    parser.add_argument("--seed", type=int, default=170817)
    parser.add_argument("--network-snr", type=float, default=32.4)
    parser.add_argument("--mej-dynamic", type=float, default=0.016)
    parser.add_argument("--mej-wind", type=float, default=0.024)
    parser.add_argument("--phi-deg", type=float, default=30.0)
    args = parser.parse_args(argv)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = build_scenario_catalog(
        skymap_path=args.skymap,
        posterior_h5=args.posterior_h5,
        posterior_dataset=args.posterior_dataset,
        opsim_db=args.opsim_db,
        n_scenarios=args.n_scenarios,
        candidate_coordinates=args.candidate_coordinates,
        real_trigger_mjd=args.real_trigger_mjd,
        host_distance_mpc=args.host_distance_mpc,
        host_redshift_observed=args.host_redshift_observed,
        true_ra_deg=args.true_ra_deg,
        true_dec_deg=args.true_dec_deg,
        seed=args.seed,
        network_snr=args.network_snr,
        mej_dynamic=args.mej_dynamic,
        mej_wind=args.mej_wind,
        phi_deg=args.phi_deg,
    )
    frame = pd.DataFrame(rows)
    catalog_path = output_dir / "gw170817a_prepared_catalog.csv"
    manifest_path = output_dir / "gw170817a_lsst_manifest.csv"
    ids_path = output_dir / "simulation_ids.txt"
    frame.to_csv(catalog_path, index=False)
    frame.to_csv(manifest_path, index=False)
    ids_path.write_text(
        "\n".join(str(int(value)) for value in frame["simulation_id"]) + "\n",
        encoding="utf-8",
    )
    write_manifest_metadata(
        output_dir / "gw170817a_lsst_manifest.meta.json",
        {
            "schema_version": "gw170817a_lsst_scenarios_v1",
            "physical_event_uid": PHYSICAL_EVENT_UID,
            "skymap": str(Path(args.skymap).expanduser().resolve()),
            "posterior_h5": str(Path(args.posterior_h5).expanduser().resolve()),
            "posterior_dataset": str(args.posterior_dataset),
            "opsim_db": str(Path(args.opsim_db).expanduser().resolve()),
            "real_trigger_mjd": float(args.real_trigger_mjd),
            "scenario_trigger_mjds": [
                float(value) for value in frame["trigger_mjd"].tolist()
            ],
            "scenario_epoch_policy": "annual_sidereal_shift_within_opsim",
            "n_scenarios": int(len(frame)),
            "candidate_coordinates_per_scenario": int(args.candidate_coordinates),
            "coordinate_panel_policy": "shared_true_plus_unique_90pct_posterior",
            "host_distance_mpc": float(args.host_distance_mpc),
            "host_redshift_observed": float(args.host_redshift_observed),
            "snana_redshift": float(frame["redshift"].iloc[0]),
            "seed": int(args.seed),
        },
    )
    print(f"Prepared scenario catalog: {catalog_path}")
    print(
        f"Scenarios: {len(frame)}; candidate light curves: "
        f"{len(frame) * int(args.candidate_coordinates)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
