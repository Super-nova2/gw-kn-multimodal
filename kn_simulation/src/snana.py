"""Generate SNANA SIMLIB and input files for GW-associated kilonovae."""

from __future__ import annotations

import argparse
import os
import re
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
from astropy import cosmology as astropy_cosmology
from astropy import units as u
from astropy.cosmology import z_at_value
from catalog import OPTICAL_MODEL_RANGES
from ligo.skymap.io.fits import read_sky_map
from rubin_too import (
    AugmentedSNANASimlib,
    ToOConditionLibrary,
    build_event_schedule,
    classify_event,
    credible_area_deg2,
    filter_unrelated_too_observations,
    load_too_config,
    materialize_sample_visits,
    maximum_posterior_coordinate,
    probability_rank_tiles,
    sample_coordinates_with_indices,
)
from sky_sampling import (
    build_fixed_distance_coordinate_samples,
    build_posterior_coordinate_samples,
    build_test_coordinate_samples,
)

SCRIPT_DIR = Path(__file__).resolve().parent
PIPELINE_DIR = SCRIPT_DIR.parent
_BASE_DIR = os.environ.get("BASE_DIR", "/fred/oz016/bgao_kn")
PREPARED_CATALOG_COLUMNS = {
    "simulation_id",
    "network_snr",
    "trigger_mjd",
    "ra_deg",
    "dec_deg",
    "luminosity_distance",
    "redshift",
    "viewing_costheta",
    "phi_deg",
    "mej_dynamic",
    "mej_wind",
    "snana_seed",
    "coordinate_seed",
}
PREPARED_NUMERIC_COLUMNS = PREPARED_CATALOG_COLUMNS


def get_nlibid(simlib_file):
    """Read NLIBID from a generated SIMLIB file."""
    with open(simlib_file, encoding="utf-8") as handle:
        for line in handle:
            if line.startswith("NLIBID:"):
                return int(line.split()[1])
    raise ValueError(f"NLIBID not found in {simlib_file}")


def _event_row(injections, sim_id):
    rows = injections.loc[injections["simulation_id"] == sim_id]
    if len(rows) != 1:
        raise ValueError(
            f"Expected exactly one row for simulation_id={sim_id}, found {len(rows)}"
        )
    return rows.iloc[0]


def _prepared_ejecta(row, gw_type):
    if gw_type not in OPTICAL_MODEL_RANGES:
        raise ValueError(f"Unsupported GW type: {gw_type}")
    mej_dyn = float(row["mej_dynamic"])
    mej_wind = float(row["mej_wind"])
    dynamic_range = OPTICAL_MODEL_RANGES[gw_type]["mej_dynamic"]
    wind_range = OPTICAL_MODEL_RANGES[gw_type]["mej_wind"]
    if not dynamic_range[0] <= mej_dyn <= dynamic_range[1]:
        raise ValueError(
            f"simulation_id={int(row['simulation_id'])} has mej_dynamic "
            f"outside {dynamic_range}"
        )
    if not wind_range[0] <= mej_wind <= wind_range[1]:
        raise ValueError(
            f"simulation_id={int(row['simulation_id'])} has mej_wind "
            f"outside {wind_range}"
        )
    return mej_dyn, mej_wind


def gen_input(injections, text, sim_id, gw_type="bns", sndata_sim_dir=None):
    """Generate SIMGEN input content using one prepared event row."""
    row = _event_row(injections, sim_id)
    mej_dyn, mej_wind = _prepared_ejecta(row, gw_type)

    replacements = {
        r"^(MJD_EXPLODE:\s*)\S+.*$": rf"\1 {float(row['trigger_mjd'])}",
        r"^(GENPEAK_COSTHETA:\s*)\S+.*$": (rf"\1 {abs(float(row['viewing_costheta']))}"),
        r"^(GENPEAK_MEJDYN:\s*)\S+.*$": rf"\1 {mej_dyn}",
        r"^(GENPEAK_MEJWIND:\s*)\S+.*$": rf"\1 {mej_wind}",
    }
    if gw_type == "bns":
        replacements[r"^(GENPEAK_PHI:\s*)\S+.*$"] = rf"\1 {float(row['phi_deg'])}"
    if sndata_sim_dir is not None:
        if re.search(r"^PATH_SNDATA_SIM:", text, flags=re.MULTILINE):
            replacements[r"^(PATH_SNDATA_SIM:\s*)\S+.*$"] = rf"\1 {sndata_sim_dir}"
        else:
            text = re.sub(
                r"^(GENVERSION:.*)$",
                rf"\1\nPATH_SNDATA_SIM: {sndata_sim_dir}",
                text,
                count=1,
                flags=re.MULTILINE,
            )
    for pattern, replacement in replacements.items():
        text = re.sub(pattern, replacement, text, flags=re.MULTILINE)
    return text


def _coordinate_samples(args, injections, sky_map, sim_id):
    row = _event_row(injections, sim_id)
    seed = int(row["coordinate_seed"])
    samples_per_event = int(row.get("optical_candidate_count", args.samples_per_event))
    if samples_per_event < 1:
        raise ValueError(
            f"simulation_id={sim_id} has invalid optical_candidate_count={samples_per_event}"
        )
    if args.coordinate_mode == "posterior_3d":
        return build_posterior_coordinate_samples(
            sky_map,
            samples_per_event=samples_per_event,
            level=args.level,
            nside=args.sampling_nside,
            seed=seed,
        )

    if args.coordinate_mode == "posterior_test":
        return build_test_coordinate_samples(
            sky_map,
            true_ra=float(row["ra_deg"]),
            true_dec=float(row["dec_deg"]),
            true_distance_mpc=float(row["luminosity_distance"]),
            samples_per_event=samples_per_event,
            level=args.level,
            nside=args.sampling_nside,
            seed=seed,
        )

    if args.coordinate_mode == "posterior_fixed_distance":
        return build_fixed_distance_coordinate_samples(
            sky_map,
            distance_mpc=float(row["luminosity_distance"]),
            samples_per_event=samples_per_event,
            level=args.level,
            nside=args.sampling_nside,
            seed=seed,
        )

    raise ValueError(f"Unsupported production coordinate mode: {args.coordinate_mode}")


def _coordinate_manifest_frame(
    sim_id,
    ra,
    dec,
    distance,
    posterior_probability,
    is_true_position,
    *,
    redshift=None,
    libid=None,
    in_baseline_footprint=None,
    too_tile_index=None,
    too_nobs=None,
    too_mode=None,
):
    data = {
        "simulation_id": int(sim_id),
        "sample_index": np.arange(len(ra), dtype=int),
        "ra": ra,
        "dec": dec,
        "distance_mpc": distance,
        "posterior_probability": posterior_probability,
        "is_true_position": is_true_position,
    }
    optional_columns = {
        "redshift": redshift,
        "libid": libid,
        "in_baseline_footprint": in_baseline_footprint,
        "too_tile_index": too_tile_index,
        "too_nobs": too_nobs,
        "too_mode": too_mode,
    }
    for name, values in optional_columns.items():
        if values is None:
            continue
        if np.isscalar(values):
            data[name] = np.full(len(ra), values)
        else:
            if len(values) != len(ra):
                raise ValueError(f"Manifest column {name} has the wrong length")
            data[name] = values
    return pd.DataFrame(data)


def _validate_requested_network_snr(catalog, sim_ids):
    missing = sorted(PREPARED_CATALOG_COLUMNS - set(catalog.columns))
    if missing:
        raise ValueError(f"KN catalog is missing required columns: {missing}")
    if catalog["simulation_id"].duplicated().any():
        duplicate = catalog.loc[
            catalog["simulation_id"].duplicated(), "simulation_id"
        ].tolist()
        raise ValueError(
            f"KN catalog has duplicate simulation_id values: {duplicate[:10]}"
        )
    if len(sim_ids) != len(set(sim_ids)):
        raise ValueError("Requested simulation_id values must be unique")

    result = {}
    for sim_id in sim_ids:
        row = _event_row(catalog, sim_id)
        for column in PREPARED_NUMERIC_COLUMNS:
            value = float(row[column])
            if not np.isfinite(value):
                raise ValueError(f"simulation_id={sim_id} has invalid {column}")
        network_snr = float(row["network_snr"])
        if network_snr < 0:
            raise ValueError(f"simulation_id={sim_id} has invalid network_snr")
        result[int(sim_id)] = network_snr
    return result


def _remove_event_products(simlib_file, input_file):
    simlib_file.unlink(missing_ok=True)
    input_file.unlink(missing_ok=True)


def build_parser():
    parser = argparse.ArgumentParser(
        description="Generate SNANA SIMLIB and input files for GW events."
    )
    parser.add_argument("--GW_type", choices=("bns", "nsbh"), default="bns")
    parser.add_argument(
        "--skymap_path",
        default=f"{_BASE_DIR}/data/bns_skymap/",
        help="Directory containing <simulation_id>.fits sky maps",
    )
    parser.add_argument("--sim_name", default="LSST_KN_BNS")
    parser.add_argument("--sim_ids", nargs="+", type=int, required=False)
    parser.add_argument(
        "--sim-id-file",
        type=Path,
        default=None,
        help="Optional file with one simulation_id per line (alternative to --sim_ids).",
    )
    parser.add_argument(
        "--GW_catalog",
        required=True,
        help="Prepared kn_catalog.csv produced by prepare_kn_catalog.py",
    )
    parser.add_argument(
        "--Opsim",
        default=(
            f"{_BASE_DIR}/data/rubin_sim/baseline_v5.1/" "baseline_v5.1.1_10yrs.db"
        ),
    )
    parser.add_argument("--level", type=float, default=0.9)
    parser.add_argument("--outdir", default="./data/")
    parser.add_argument(
        "--template_input",
        default=str(PIPELINE_DIR / "templates" / "bns.input"),
    )
    parser.add_argument(
        "--sndata-sim-dir",
        default=None,
        help="Parent SNANA SIM output directory (default: $SNDATA_ROOT/SIM)",
    )
    parser.add_argument(
        "--coordinate_mode",
        choices=("posterior_3d", "posterior_test", "posterior_fixed_distance"),
        default="posterior_3d",
    )
    parser.add_argument("--samples_per_event", type=int, default=64)
    parser.add_argument("--sampling_nside", type=int, default=256)
    parser.add_argument("--cosmology", default="Planck15")
    parser.add_argument(
        "--too_config",
        default=str(PIPELINE_DIR / "config" / "rubin_too_2024.yaml"),
        help="Rubin GW ToO strategy YAML configuration",
    )
    return parser


def main(argv=None):
    import opsimsummaryv2 as opsim

    args = build_parser().parse_args(argv)
    outdir = Path(args.outdir)
    input_dir = outdir / "SIM_INPUT"
    simlib_dir = outdir / "SIMLIB"
    coordinate_dir = outdir / "COORDINATES"
    input_dir.mkdir(parents=True, exist_ok=True)
    simlib_dir.mkdir(parents=True, exist_ok=True)
    coordinate_dir.mkdir(parents=True, exist_ok=True)
    if args.sndata_sim_dir is not None:
        Path(args.sndata_sim_dir).mkdir(parents=True, exist_ok=True)
        # SNANA requires $SNDATA_ROOT/SIM/PATH_SNDATA_SIM.LIST to exist when
        # PATH_SNDATA_SIM is supplied; create it if the run has not yet done so.
        sndata_root = os.environ.get("SNDATA_ROOT")
        if sndata_root:
            path_sndata_sim_list = Path(sndata_root) / "SIM" / "PATH_SNDATA_SIM.LIST"
        else:
            path_sndata_sim_list = Path(args.sndata_sim_dir).parent / "PATH_SNDATA_SIM.LIST"
        path_sndata_sim_list.parent.mkdir(parents=True, exist_ok=True)
        path_sndata_sim_list.touch(exist_ok=True)
    opsim_stem = Path(args.Opsim).stem
    catalog = pd.read_csv(args.GW_catalog)
    if args.sim_id_file is not None:
        with open(args.sim_id_file, "r", encoding="utf-8") as handle:
            sim_ids = [int(line.strip()) for line in handle if line.strip()]
        if not sim_ids:
            raise ValueError(f"sim-id file is empty: {args.sim_id_file}")
    else:
        sim_ids = args.sim_ids
    network_snr_by_id = _validate_requested_network_snr(catalog, sim_ids)
    cosmology = getattr(astropy_cosmology, args.cosmology, None)
    if cosmology is None or not hasattr(cosmology, "luminosity_distance"):
        raise ValueError(f"Unknown Astropy cosmology: {args.cosmology}")
    template_text = Path(args.template_input).read_text(encoding="utf-8")
    too_config = load_too_config(args.too_config)

    survey = opsim.OpSimSurvey(args.Opsim)
    removed_existing_too = filter_unrelated_too_observations(survey)
    print(f"Removed {removed_existing_too} unrelated built-in OpSim ToO visits")
    survey.compute_hp_rep(
        nside=args.sampling_nside,
        minVisits=1,
        maxVisits=10000,
    )
    condition_library = None

    artifacts = {}
    for sim_id in sim_ids:
        print(f"\nProcessing simulation ID: {sim_id}")
        sim_id = int(sim_id)
        network_snr = network_snr_by_id[sim_id]
        row = _event_row(catalog, sim_id)
        simlib_file = simlib_dir / (f"{opsim_stem}_{args.sim_name}_{sim_id}.SIMLIB")
        input_file = input_dir / f"SIMGEN_{args.sim_name}_{sim_id}.INPUT"
        coordinate_file = coordinate_dir / f"{sim_id}.csv"
        _remove_event_products(simlib_file, input_file)
        coordinate_file.unlink(missing_ok=True)
        coordinate_frame = None

        plan = {
            "simulation_id": sim_id,
            "status": "pending",
            "network_snr": network_snr,
            "trigger_mjd": float(row["trigger_mjd"]),
            "cosmology": args.cosmology,
            "mej_dynamic": float(row["mej_dynamic"]),
            "mej_wind": float(row["mej_wind"]),
            "snana_seed": int(row["snana_seed"]),
            "coordinate_seed": int(row["coordinate_seed"]),
            "area90_deg2": None,
            "area90_source": "skymap",
            "requested_mode": None,
            "mode": None,
            "reason": None,
            "strategy": None,
            "strategy_version": too_config["version"],
            "strategy_source_url": too_config["source_url"],
            "existing_opsim_too_removed": removed_existing_too,
            "n_tiles": 0,
            "night0_branch": None,
            "probability_coverage_by_night": {},
            "visibility_windows": [],
            "scan_intervals": [],
            "event_schedule_visits": 0,
            "too_visits_generated": 0,
            "too_visits_written": 0,
            "removed_baseline_visits": 0,
            "nlibid": 0,
        }

        try:
            row_skymap = row.get("skymap_path")
            if isinstance(row_skymap, str) and row_skymap.strip():
                sky_map_path = Path(row_skymap)
            else:
                sky_map_path = Path(args.skymap_path) / f"{sim_id}.fits"
            sky_map = read_sky_map(sky_map_path, moc=True)
            print(
                "Distance mean:",
                sky_map.meta.get("distmean"),
                "Mpc\nDistance std:",
                sky_map.meta.get("diststd"),
                "Mpc",
            )
            area90_deg2 = credible_area_deg2(
                sky_map,
                level=float(too_config["credible_level"]),
            )
            decision = classify_event(network_snr, area90_deg2, too_config)
            plan.update(
                area90_deg2=area90_deg2,
                requested_mode=decision.mode,
                mode=decision.mode,
                reason=decision.reason,
                strategy=decision.strategy,
            )

            ra, dec, distance, probability, is_true = _coordinate_samples(
                args,
                catalog,
                sky_map,
                sim_id,
            )
            if len(ra) == 0:
                raise ValueError(f"No coordinates generated for simulation_id={sim_id}")
            ra = np.asarray(ra, dtype=float)
            dec = np.asarray(dec, dtype=float)
            distance = np.asarray(distance, dtype=float)
            redshift = np.asarray(
                z_at_value(cosmology.luminosity_distance, distance * u.Mpc).value,
                dtype=float,
            )
            if redshift.ndim == 0:
                redshift = np.full(len(ra), float(redshift), dtype=float)
            true_mask = np.asarray(is_true, dtype=bool)
            if args.coordinate_mode == "posterior_fixed_distance":
                redshift[:] = float(row["redshift"])
            elif np.any(true_mask):
                redshift[true_mask] = float(row["redshift"])

            too_tile_index = np.full(len(ra), -1, dtype=int)
            tile_probability = np.empty(0, dtype=float)
            if decision.mode == "baseline_plus_too":
                too_tile_index, tile_probability = probability_rank_tiles(
                    sky_map,
                    ra,
                    dec,
                    area90_deg2=area90_deg2,
                    level=float(too_config["credible_level"]),
                    nside=args.sampling_nside,
                    effective_fov_deg2=float(too_config["effective_fov_deg2"]),
                )
                plan["n_tiles"] = len(tile_probability)

            in_footprint, libid = sample_coordinates_with_indices(
                survey,
                ra,
                dec,
                redshift,
                nside=args.sampling_nside,
            )
            print(
                f"Generated {len(ra)} sky coordinates; "
                f"{int(in_footprint.sum())} are in the WFD/DDF footprint"
            )

            too_observations = pd.DataFrame()
            busy_intervals = []
            effective_mode = decision.mode
            effective_reason = decision.reason
            if decision.mode == "baseline_plus_too":
                eligible = in_footprint & (too_tile_index >= 0)
                if not np.any(eligible):
                    effective_mode = "baseline"
                    effective_reason = "no_too_tiles_in_baseline_footprint"
                else:
                    anchor_ra, anchor_dec = maximum_posterior_coordinate(sky_map)
                    schedule = build_event_schedule(
                        trigger_mjd=float(row["trigger_mjd"]),
                        anchor_ra_deg=anchor_ra,
                        anchor_dec_deg=anchor_dec,
                        strategy_name=decision.strategy,
                        area90_deg2=area90_deg2,
                        tile_probability=tile_probability,
                        config=too_config,
                    )
                    plan.update(
                        n_tiles=schedule.n_tiles,
                        night0_branch=schedule.night0_branch,
                        probability_coverage_by_night=(
                            schedule.probability_coverage_by_night
                        ),
                        visibility_windows=schedule.visibility.to_dict(
                            orient="records"
                        ),
                        scan_intervals=schedule.scan_intervals,
                        event_schedule_visits=len(schedule.observations),
                    )
                    if schedule.observations.empty:
                        effective_mode = "baseline"
                        effective_reason = "too_not_observable"
                    else:
                        samples = pd.DataFrame(
                            {
                                "sample_index": np.arange(len(ra), dtype=int),
                                "ra": ra,
                                "dec": dec,
                                "tile_index": too_tile_index,
                            }
                        ).loc[eligible]
                        if condition_library is None:
                            condition_library = ToOConditionLibrary.from_database(
                                args.Opsim,
                                too_config["conditions"],
                            )
                        too_observations = materialize_sample_visits(
                            schedule,
                            samples,
                            condition_library,
                            too_config,
                        )
                        if too_observations.empty:
                            effective_mode = "baseline"
                            effective_reason = "too_no_visible_sample_visits"
                        else:
                            busy_intervals = schedule.busy_intervals

            plan.update(mode=effective_mode, reason=effective_reason)
            sim = AugmentedSNANASimlib(
                survey,
                out_path=str(simlib_dir),
                file_suffix=f"_{args.sim_name}_{sim_id}",
                too_observations=too_observations,
                busy_intervals=busy_intervals,
                NOTES={
                    "GW simulation_id": sim_id,
                    "GW network SNR": network_snr,
                    "GW area90_deg2": area90_deg2,
                    "observation mode": effective_mode,
                    "Rubin ToO strategy": decision.strategy or "none",
                },
            )
            sim.write_SIMLIB()
            nlibid = get_nlibid(simlib_file)
            print(f"NLIBID for simulation ID {sim_id}: {nlibid}")

            too_nobs = np.zeros(len(ra), dtype=int)
            if not too_observations.empty:
                counts = too_observations.groupby("sample_index").size()
                too_nobs[counts.index.to_numpy(dtype=int)] = counts.to_numpy(dtype=int)
            coordinate_frame = _coordinate_manifest_frame(
                sim_id,
                ra,
                dec,
                distance,
                probability,
                is_true,
                redshift=redshift,
                libid=libid,
                in_baseline_footprint=in_footprint,
                too_tile_index=too_tile_index,
                too_nobs=too_nobs,
                too_mode=effective_mode,
            )

            text = re.sub(
                r"^(NGENTOT_LC:\s*)\S+.*$",
                rf"\1 {nlibid}",
                template_text,
                flags=re.MULTILINE,
            )
            text = re.sub(
                r"^(GENVERSION:\s*)\S+.*$",
                rf"\1{args.sim_name}_{sim_id}",
                text,
                flags=re.MULTILINE,
            )
            text = re.sub(
                r"^(SIMLIB_FILE:\s*)\S+.*$",
                rf"\1{simlib_file}",
                text,
                flags=re.MULTILINE,
            )
            text = re.sub(
                r"^(RANSEED:\s*)\S+.*$",
                rf"\1 {int(row['snana_seed'])}",
                text,
                flags=re.MULTILINE,
            )
            text = gen_input(
                catalog,
                text,
                sim_id,
                gw_type=args.GW_type,
                sndata_sim_dir=args.sndata_sim_dir,
            )
            if args.coordinate_mode == "posterior_fixed_distance":
                parent_redshift = float(row["redshift"])
                redshift_min = max(1.0e-5, parent_redshift - 1.0e-4)
                redshift_max = parent_redshift + 1.0e-4
                text = re.sub(
                    r"^(GENRANGE_REDSHIFT:\s*)\S+\s+\S+.*$",
                    rf"\1 {redshift_min:.6f} {redshift_max:.6f}",
                    text,
                    flags=re.MULTILINE,
                )
            input_file.write_text(text, encoding="utf-8")
            coordinate_frame.to_csv(coordinate_file, index=False)

            plan.update(
                status="generated",
                nlibid=nlibid,
                too_visits_generated=len(too_observations),
                too_visits_written=sim.too_visits_written,
                removed_baseline_visits=sim.removed_baseline_visits,
            )
            artifacts[sim_id] = {
                "plan": plan,
                "coordinates": coordinate_frame,
            }
        except Exception as error:  # noqa: BLE001 - isolate failures by event
            _remove_event_products(simlib_file, input_file)
            coordinate_file.unlink(missing_ok=True)
            plan.update(
                status="failed",
                error_type=type(error).__name__,
                error=str(error),
            )
            artifacts[sim_id] = {
                "plan": plan,
                "coordinates": coordinate_frame,
            }
            print(f"Failed simulation ID {sim_id}: {error}")
            traceback.print_exc()
    return artifacts


if __name__ == "__main__":
    main()
