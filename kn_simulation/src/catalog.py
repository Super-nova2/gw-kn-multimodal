"""Prepare GWSamplegen event catalogs for Rubin/SNANA kilonova simulation."""

from __future__ import annotations

import hashlib
import json
import shutil
import sqlite3
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import pandas as pd
from astropy.time import Time

SCHEMA_VERSION = "gwsamplegen-kn-v3"
DEFAULT_MJD_MIN = 61_000.0
DEFAULT_MJD_MAX = 64_500.0

REQUIRED_COLUMNS = (
    "simulation_id",
    "mass1_source",
    "mass2_source",
    "mass1_detector",
    "mass2_detector",
    "spin1z",
    "spin2z",
    "redshift",
    "luminosity_distance",
    "ra",
    "dec",
    "theta_jn",
    "psi",
    "phase",
    "geocent_time",
    "mej_dynamic",
    "mej_wind",
    "mej_total",
    "recovered_mass1_detector",
    "recovered_mass2_detector",
    "recovered_spin1z",
    "recovered_spin2z",
    "network_snr",
    "optimal_network_snr",
    "online_ifos",
    "skymap",
)

NUMERIC_COLUMNS = tuple(
    column for column in REQUIRED_COLUMNS if column not in {"online_ifos", "skymap"}
)

OPTICAL_MODEL_RANGES = {
    "bns": {
        "mej_dynamic": (0.001, 0.02),
        "mej_wind": (0.01, 0.13),
        "viewing_costheta": (0.0, 1.0),
        "phi_deg": (15.0, 75.0),
    },
    "nsbh": {
        "mej_dynamic": (0.01, 0.09),
        "mej_wind": (0.01, 0.09),
        "viewing_costheta": (0.0, 1.0),
        "phi_deg": (30.0, 30.0),
    },
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _opsim_mjd_bounds(path: Path) -> tuple[float, float]:
    if not path.is_file():
        raise FileNotFoundError(f"OpSim database does not exist: {path}")
    with sqlite3.connect(f"file:{path}?mode=ro", uri=True) as connection:
        row = connection.execute(
            "SELECT MIN(observationStartMJD), MAX(observationStartMJD) "
            "FROM observations"
        ).fetchone()
    if row is None or row[0] is None or row[1] is None:
        raise ValueError(f"OpSim database has no observations: {path}")
    return float(row[0]), float(row[1])


def _require_finite(frame: pd.DataFrame, columns: Iterable[str]) -> None:
    for column in columns:
        values = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float)
        if not np.all(np.isfinite(values)):
            bad = frame.loc[~np.isfinite(values), "simulation_id"].tolist()[:10]
            raise ValueError(
                f"column {column} contains non-finite values for simulation_id={bad}"
            )
        frame[column] = values


def _validate_ids(frame: pd.DataFrame) -> None:
    values = frame["simulation_id"].to_numpy(dtype=float)
    if np.any(values < 0) or not np.all(values == np.floor(values)):
        raise ValueError("simulation_id values must be non-negative integers")
    frame["simulation_id"] = values.astype(np.int64)
    duplicate = frame.loc[frame["simulation_id"].duplicated(), "simulation_id"].tolist()
    if duplicate:
        raise ValueError(f"duplicate simulation_id values: {duplicate[:10]}")


def _validate_online_ifos(frame: pd.DataFrame) -> None:
    allowed = {"H1", "L1", "V1"}
    for sim_id, value in frame[["simulation_id", "online_ifos"]].itertuples(
        index=False, name=None
    ):
        text = str(value)
        if not text or len(text) % 2:
            raise ValueError(f"invalid online_ifos for simulation_id={sim_id}: {text}")
        detectors = [text[index : index + 2] for index in range(0, len(text), 2)]
        if not set(detectors) <= allowed or len(detectors) != len(set(detectors)):
            raise ValueError(f"invalid online_ifos for simulation_id={sim_id}: {text}")


def _validate_physics(frame: pd.DataFrame, source: str) -> None:
    if np.any(frame["mass1_source"] < frame["mass2_source"]):
        raise ValueError(
            "source-frame masses must satisfy mass1_source >= mass2_source"
        )
    if np.any(frame[["mass1_source", "mass2_source"]] <= 0):
        raise ValueError("source-frame masses must be positive")
    if source == "bns":
        if np.any(frame[["mass1_source", "mass2_source"]] > 2.05):
            raise ValueError("BNS source-frame masses must not exceed 2.05 Msun")
    elif np.any(frame["mass1_source"] < 2.05) or np.any(frame["mass2_source"] > 2.05):
        raise ValueError("NSBH source-frame masses do not match the BH/NS convention")

    if np.any(np.abs(frame[["spin1z", "spin2z"]]) > 1):
        raise ValueError("dimensionless spins must lie in [-1, 1]")
    if np.any(frame["luminosity_distance"] <= 0) or np.any(frame["redshift"] < 0):
        raise ValueError("distance must be positive and redshift must be non-negative")
    if np.any((frame["ra"] < 0) | (frame["ra"] >= 2 * np.pi)):
        raise ValueError("ra must be in radians within [0, 2*pi)")
    if np.any(np.abs(frame["dec"]) > np.pi / 2):
        raise ValueError("dec must be in radians within [-pi/2, pi/2]")
    if np.any((frame["theta_jn"] < 0) | (frame["theta_jn"] > np.pi)):
        raise ValueError("theta_jn must be in radians within [0, pi]")

    expected_total = frame["mej_dynamic"] + frame["mej_wind"]
    if not np.allclose(frame["mej_total"], expected_total, rtol=1e-8, atol=1e-12):
        raise ValueError("mej_total is inconsistent with mej_dynamic + mej_wind")
    if np.any(frame["network_snr"] < 0):
        raise ValueError("network_snr must be non-negative")


def _normalize_neg_type(frame: pd.DataFrame) -> None:
    """Validate the physical double-zero type-1 label in-place."""
    components = frame[["mej_dynamic", "mej_wind"]]
    if np.any(components < 0):
        raise ValueError("physical ejecta component masses must be non-negative")
    both_zero = (frame["mej_dynamic"] == 0.0) & (frame["mej_wind"] == 0.0)
    if "neg_type" not in frame:
        frame["neg_type"] = both_zero.astype(np.int8)
        return
    values = pd.to_numeric(frame["neg_type"], errors="coerce").to_numpy()
    if not np.all(np.isfinite(values)) or not np.all(np.isin(values, (0, 1))):
        raise ValueError("input GW catalog neg_type must contain only 0 or 1")
    values = values.astype(np.int8)
    expected = both_zero.to_numpy(dtype=np.int8)
    if not np.array_equal(values, expected):
        bad = frame.loc[values != expected, "simulation_id"].tolist()[:10]
        raise ValueError(
            "neg_type is inconsistent with physical double-zero ejecta for "
            f"simulation_id={bad}"
        )
    frame["neg_type"] = values


def _filter_to_optical_model_ejecta_range(
    frame: pd.DataFrame,
    source: str,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Keep events inside both closed ejecta ranges without modifying values."""

    ranges = OPTICAL_MODEL_RANGES[source]
    dynamic_min, dynamic_max = ranges["mej_dynamic"]
    wind_min, wind_max = ranges["mej_wind"]
    dynamic_in_range = frame["mej_dynamic"].between(
        dynamic_min, dynamic_max, inclusive="both"
    )
    wind_in_range = frame["mej_wind"].between(wind_min, wind_max, inclusive="both")
    type1 = frame["neg_type"] == 1
    keep = (~type1) & dynamic_in_range & wind_in_range
    statistics = {
        "model_range_filtered_rows": int((~keep).sum()),
        "mej_dynamic_out_of_range_rows": int((~dynamic_in_range).sum()),
        "mej_wind_out_of_range_rows": int((~wind_in_range).sum()),
        "input_type1_rows": int(type1.sum()),
        "excluded_type1_no_ejecta_rows": int(type1.sum()),
        "snana_input_rows": int(keep.sum()),
    }
    filtered = frame.loc[keep].copy().reset_index(drop=True)
    if filtered.empty:
        raise ValueError(
            f"no {source.upper()} events remain inside the optical-model ejecta "
            "parameter ranges"
        )
    return filtered, statistics


def _validate_generated_model_parameters(frame: pd.DataFrame, source: str) -> None:
    """Assert that deterministically generated angular parameters are in range."""

    for column in ("viewing_costheta", "phi_deg"):
        lower, upper = OPTICAL_MODEL_RANGES[source][column]
        if np.any((frame[column] < lower) | (frame[column] > upper)):
            raise RuntimeError(
                f"generated {column} lies outside optical-model range "
                f"[{lower}, {upper}]"
            )


def _event_random_values(
    simulation_ids: np.ndarray,
    *,
    source: str,
    split: str,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    source_code = {"bns": 1, "nsbh": 2}[source]
    split_code = {"train": 1, "test": 2}[split]
    phi = np.empty(len(simulation_ids), dtype=float)
    snana_seed = np.empty(len(simulation_ids), dtype=np.int64)
    coordinate_seed = np.empty(len(simulation_ids), dtype=np.int64)
    for index, sim_id in enumerate(simulation_ids):
        sequence = np.random.SeedSequence(
            [int(seed), source_code, split_code, int(sim_id)]
        )
        phi_sequence, snana_sequence, coordinate_sequence = sequence.spawn(3)
        if source == "bns":
            phi_min, phi_max = OPTICAL_MODEL_RANGES[source]["phi_deg"]
            phi[index] = np.random.default_rng(phi_sequence).uniform(phi_min, phi_max)
        else:
            phi[index] = OPTICAL_MODEL_RANGES[source]["phi_deg"][0]
        snana_seed[index] = (
            int(snana_sequence.generate_state(1, dtype=np.uint32)[0]) % 2_000_000_000
            + 1
        )
        coordinate_seed[index] = (
            int(coordinate_sequence.generate_state(1, dtype=np.uint32)[0])
            % 2_000_000_000
            + 1
        )
    return phi, snana_seed, coordinate_seed


def _validate_skymaps(frame: pd.DataFrame, skymap_dir: Path) -> None:
    if not skymap_dir.is_dir():
        raise FileNotFoundError(f"skymap directory does not exist: {skymap_dir}")
    missing = []
    mismatched = []
    for sim_id, relative in frame[["simulation_id", "skymap"]].itertuples(
        index=False, name=None
    ):
        expected_name = f"{int(sim_id)}.fits"
        if Path(str(relative)).name != expected_name:
            mismatched.append((int(sim_id), str(relative)))
        if not (skymap_dir / expected_name).is_file():
            missing.append(int(sim_id))
    if mismatched:
        raise ValueError(
            f"catalog skymap names do not match simulation_id: {mismatched[:10]}"
        )
    if missing:
        raise FileNotFoundError(
            f"missing skymaps for simulation_id={missing[:10]} "
            f"({len(missing)} missing in total)"
        )


def prepare_catalog(
    catalog: pd.DataFrame,
    *,
    source: str,
    split: str,
    seed: int,
    skymap_dir: Path,
    opsim_db: Path,
    mjd_min: float = DEFAULT_MJD_MIN,
    mjd_max: float = DEFAULT_MJD_MAX,
) -> tuple[pd.DataFrame, dict]:
    """Validate and enrich one complete-event GWSamplegen catalog."""

    source = source.lower()
    split = split.lower()
    if source not in {"bns", "nsbh"}:
        raise ValueError("source must be bns or nsbh")
    if split not in {"train", "test"}:
        raise ValueError("split must be train or test")
    if not np.isfinite(mjd_min) or not np.isfinite(mjd_max) or mjd_min >= mjd_max:
        raise ValueError("MJD bounds must be finite and increasing")

    missing_columns = sorted(set(REQUIRED_COLUMNS) - set(catalog.columns))
    if missing_columns:
        raise ValueError(f"catalog is missing required columns: {missing_columns}")
    if catalog.empty:
        raise ValueError("catalog contains no complete GW events")

    prepared = catalog.copy()
    _require_finite(prepared, NUMERIC_COLUMNS)
    _validate_ids(prepared)
    prepared = prepared.sort_values("simulation_id", kind="stable").reset_index(
        drop=True
    )
    _validate_online_ifos(prepared)
    _validate_physics(prepared, source)
    _normalize_neg_type(prepared)
    prepared, filter_statistics = _filter_to_optical_model_ejecta_range(
        prepared, source
    )
    _validate_skymaps(prepared, Path(skymap_dir))

    trigger_mjd = np.asarray(
        Time(prepared["geocent_time"].to_numpy(dtype=float), format="gps").mjd,
        dtype=float,
    )
    opsim_min, opsim_max = _opsim_mjd_bounds(Path(opsim_db))
    if np.any((trigger_mjd < mjd_min) | (trigger_mjd > mjd_max)):
        observed = (float(trigger_mjd.min()), float(trigger_mjd.max()))
        raise ValueError(
            f"GW trigger MJD range {observed} lies outside requested "
            f"[{mjd_min}, {mjd_max}]"
        )
    if np.any((trigger_mjd < opsim_min) | (trigger_mjd > opsim_max)):
        raise ValueError(
            f"GW trigger times lie outside OpSim coverage [{opsim_min}, {opsim_max}]"
        )

    phi, snana_seed, coordinate_seed = _event_random_values(
        prepared["simulation_id"].to_numpy(),
        source=source,
        split=split,
        seed=int(seed),
    )
    prepared["trigger_mjd"] = trigger_mjd
    prepared["ra_deg"] = np.degrees(prepared["ra"])
    prepared["dec_deg"] = np.degrees(prepared["dec"])
    prepared["viewing_costheta"] = np.abs(np.cos(prepared["theta_jn"]))
    prepared["phi_deg"] = phi
    prepared["snana_seed"] = snana_seed
    prepared["coordinate_seed"] = coordinate_seed
    _validate_generated_model_parameters(prepared, source)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "source": source,
        "split": split,
        "seed": int(seed),
        "input_rows": len(catalog),
        "output_rows": len(prepared),
        "network_snr_min": float(prepared["network_snr"].min()),
        "network_snr_max": float(prepared["network_snr"].max()),
        "requested_mjd_min": float(mjd_min),
        "requested_mjd_max": float(mjd_max),
        "trigger_mjd_min": float(trigger_mjd.min()),
        "trigger_mjd_max": float(trigger_mjd.max()),
        "opsim_mjd_min": opsim_min,
        "opsim_mjd_max": opsim_max,
        **filter_statistics,
        "optical_model_parameter_ranges": {
            name: list(bounds) for name, bounds in OPTICAL_MODEL_RANGES[source].items()
        },
        "skymap_directory": str(Path(skymap_dir).resolve()),
        "opsim_database": str(Path(opsim_db).resolve()),
    }
    return prepared, manifest


def write_prepared_catalog(
    input_path: Path,
    output_path: Path,
    *,
    source: str,
    split: str,
    seed: int,
    skymap_dir: Path,
    opsim_db: Path,
    mjd_min: float = DEFAULT_MJD_MIN,
    mjd_max: float = DEFAULT_MJD_MAX,
    manifest_path: Path | None = None,
    overwrite: bool = False,
) -> dict:
    """Prepare a catalog and atomically write its CSV and audit manifest."""

    input_path = Path(input_path)
    output_path = Path(output_path)
    if not input_path.is_file():
        raise FileNotFoundError(f"input catalog does not exist: {input_path}")
    if manifest_path is None:
        manifest_path = output_path.with_name(f"{output_path.stem}.manifest.json")
    manifest_path = Path(manifest_path)
    if not overwrite:
        existing = [path for path in (output_path, manifest_path) if path.exists()]
        if existing:
            raise FileExistsError(f"refusing to overwrite existing files: {existing}")

    prepared, manifest = prepare_catalog(
        pd.read_csv(input_path),
        source=source,
        split=split,
        seed=seed,
        skymap_dir=Path(skymap_dir),
        opsim_db=Path(opsim_db),
        mjd_min=mjd_min,
        mjd_max=mjd_max,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    csv_temporary = output_path.with_name(f".{output_path.name}.tmp")
    manifest_temporary = manifest_path.with_name(f".{manifest_path.name}.tmp")
    try:
        prepared.to_csv(csv_temporary, index=False)
        manifest.update(
            input_catalog=str(input_path.resolve()),
            input_catalog_sha256=_sha256(input_path),
            output_catalog=str(output_path.resolve()),
            output_catalog_sha256=_sha256(csv_temporary),
        )
        manifest_temporary.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        csv_temporary.replace(output_path)
        manifest_temporary.replace(manifest_path)
    finally:
        csv_temporary.unlink(missing_ok=True)
        manifest_temporary.unlink(missing_ok=True)
    return manifest


def prepare_run_catalog(
    input_path: Path,
    run_dir: Path,
    *,
    profile_name: str,
    source: str,
    split: str,
    seed: int,
    skymap_dir: Path,
    opsim_db: Path,
    mjd_min: float = DEFAULT_MJD_MIN,
    mjd_max: float = DEFAULT_MJD_MAX,
    profile: dict | None = None,
    overwrite: bool = False,
) -> dict:
    """Validate one GWSamplegen catalog and atomically populate a run directory."""

    input_path = Path(input_path).resolve()
    run_dir = Path(run_dir).resolve()
    if not input_path.is_file():
        raise FileNotFoundError(f"input catalog does not exist: {input_path}")

    copied_input = run_dir / "catalog.csv"
    prepared_path = run_dir / "kn_catalog.csv"
    manifest_path = run_dir / "kn_catalog.manifest.json"
    input_metadata_path = run_dir / "catalog.input.json"
    same_input = input_path == copied_input
    protected = (prepared_path, manifest_path, input_metadata_path)
    if not same_input:
        protected = (copied_input, *protected)
    if not overwrite:
        existing = [path for path in protected if path.exists()]
        if existing:
            raise FileExistsError(f"refusing to overwrite existing files: {existing}")

    input_sha256 = _sha256(input_path)
    prepared, manifest = prepare_catalog(
        pd.read_csv(input_path),
        source=source,
        split=split,
        seed=seed,
        skymap_dir=Path(skymap_dir),
        opsim_db=Path(opsim_db),
        mjd_min=mjd_min,
        mjd_max=mjd_max,
    )

    run_dir.mkdir(parents=True, exist_ok=True)
    temporary_input = run_dir / ".catalog.csv.tmp"
    temporary_prepared = run_dir / ".kn_catalog.csv.tmp"
    temporary_manifest = run_dir / ".kn_catalog.manifest.json.tmp"
    temporary_metadata = run_dir / ".catalog.input.json.tmp"
    temporaries = (
        temporary_input,
        temporary_prepared,
        temporary_manifest,
        temporary_metadata,
    )
    try:
        if not same_input:
            shutil.copyfile(input_path, temporary_input)
        prepared.to_csv(temporary_prepared, index=False)
        output_sha256 = _sha256(temporary_prepared)
        manifest.update(
            profile_name=profile_name,
            source_catalog=str(input_path),
            source_catalog_sha256=input_sha256,
            input_catalog=str(copied_input),
            input_catalog_sha256=input_sha256,
            output_catalog=str(prepared_path),
            output_catalog_sha256=output_sha256,
            profile=profile or {},
        )
        input_metadata = {
            "profile_name": profile_name,
            "source_catalog": str(input_path),
            "source_catalog_sha256": input_sha256,
            "copied_catalog": str(copied_input),
            "rows": int(manifest["input_rows"]),
            "prepared_rows": int(manifest["output_rows"]),
            "model_range_filtered_rows": int(manifest["model_range_filtered_rows"]),
        }
        temporary_manifest.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary_metadata.write_text(
            json.dumps(input_metadata, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

        if not same_input:
            temporary_input.replace(copied_input)
        temporary_prepared.replace(prepared_path)
        temporary_manifest.replace(manifest_path)
        temporary_metadata.replace(input_metadata_path)
    finally:
        for temporary in temporaries:
            temporary.unlink(missing_ok=True)
    return manifest


def validate_type1_negative_catalog(
    catalog: pd.DataFrame,
    *,
    source: str,
    split: str,
    skymap_dir: Path,
) -> tuple[pd.DataFrame, dict]:
    """Validate a complete type-1 GW catalog without preparing SNANA inputs."""
    source = source.lower()
    split = split.lower()
    missing = sorted(set(REQUIRED_COLUMNS) - set(catalog.columns))
    if missing:
        raise ValueError(f"negative catalog is missing required columns: {missing}")
    if catalog.empty:
        raise ValueError("negative catalog contains no complete GW events")

    validated = catalog.copy()
    _require_finite(validated, NUMERIC_COLUMNS)
    _validate_ids(validated)
    _validate_online_ifos(validated)
    _validate_physics(validated, source)
    _normalize_neg_type(validated)
    if not (validated["neg_type"] == 1).all():
        bad = validated.loc[validated["neg_type"] != 1, "simulation_id"].tolist()[:10]
        raise ValueError(f"negative catalog contains non-type1 events: {bad}")

    expected_class = "neg"
    if "sample_class" not in validated:
        raise ValueError("negative catalog is missing sample_class")
    if not (validated["sample_class"].astype(str) == expected_class).all():
        raise ValueError("negative catalog sample_class must be neg")
    if "event_uid" not in validated:
        raise ValueError("negative catalog is missing event_uid")
    expected_uid = (
        source
        + "_"
        + split
        + "_neg_"
        + validated["simulation_id"].astype(np.int64).astype(str)
    )
    if not (validated["event_uid"].astype(str) == expected_uid).all():
        raise ValueError("negative catalog event_uid does not match source/split/id")
    if validated["event_uid"].duplicated().any():
        raise ValueError("negative catalog contains duplicate event_uid values")

    validated = validated.sort_values("simulation_id", kind="stable").reset_index(
        drop=True
    )
    _validate_skymaps(validated, Path(skymap_dir))
    return validated, {
        "rows": len(validated),
        "sample_class": "neg",
        "neg_type": 1,
        "skymap_directory": str(Path(skymap_dir).resolve()),
    }


def prepare_dual_run_catalog(
    positive_input_path: Path,
    negative_input_path: Path,
    run_dir: Path,
    *,
    profile_name: str,
    source: str,
    split: str,
    seed: int,
    positive_skymap_dir: Path,
    negative_skymap_dir: Path,
    opsim_db: Path,
    mjd_min: float = DEFAULT_MJD_MIN,
    mjd_max: float = DEFAULT_MJD_MAX,
    profile: dict | None = None,
    overwrite: bool = False,
) -> dict:
    """Prepare positive events for SNANA and retain type-1 events separately."""
    positive_input_path = Path(positive_input_path).resolve()
    negative_input_path = Path(negative_input_path).resolve()
    run_dir = Path(run_dir).resolve()
    for path in (positive_input_path, negative_input_path):
        if not path.is_file():
            raise FileNotFoundError(f"input catalog does not exist: {path}")

    negative_output = run_dir / "neg_catalog.csv"
    dual_manifest_path = run_dir / "dual_catalog.manifest.json"
    if not overwrite:
        existing = [
            path for path in (negative_output, dual_manifest_path) if path.exists()
        ]
        if existing:
            raise FileExistsError(f"refusing to overwrite existing files: {existing}")

    positive_frame = pd.read_csv(positive_input_path)
    for column in ("sample_class", "event_uid"):
        if column not in positive_frame:
            raise ValueError(f"positive catalog is missing {column}")
    if not (positive_frame["sample_class"].astype(str) == "pos").all():
        raise ValueError("positive catalog sample_class must be pos")
    expected_positive_uid = (
        source.lower()
        + "_"
        + split.lower()
        + "_pos_"
        + positive_frame["simulation_id"].astype(np.int64).astype(str)
    )
    if not (positive_frame["event_uid"].astype(str) == expected_positive_uid).all():
        raise ValueError("positive catalog event_uid does not match source/split/id")
    if positive_frame["event_uid"].duplicated().any():
        raise ValueError("positive catalog contains duplicate event_uid values")

    negative, negative_manifest = validate_type1_negative_catalog(
        pd.read_csv(negative_input_path),
        source=source,
        split=split,
        skymap_dir=negative_skymap_dir,
    )
    positive_manifest = prepare_run_catalog(
        positive_input_path,
        run_dir,
        profile_name=profile_name,
        source=source,
        split=split,
        seed=seed,
        skymap_dir=positive_skymap_dir,
        opsim_db=opsim_db,
        mjd_min=mjd_min,
        mjd_max=mjd_max,
        profile=profile,
        overwrite=overwrite,
    )

    negative_temporary = run_dir / ".neg_catalog.csv.tmp"
    manifest_temporary = run_dir / ".dual_catalog.manifest.json.tmp"
    try:
        negative.to_csv(negative_temporary, index=False)
        result = {
            "schema_version": SCHEMA_VERSION,
            "mode": "dual",
            "profile_name": profile_name,
            "positive": positive_manifest,
            "negative": {
                **negative_manifest,
                "source_catalog": str(negative_input_path),
                "source_catalog_sha256": _sha256(negative_input_path),
                "output_catalog": str(negative_output),
                "output_catalog_sha256": _sha256(negative_temporary),
            },
            "snana_catalog": str(run_dir / "kn_catalog.csv"),
            "snana_contains_sample_class": ["pos"],
        }
        manifest_temporary.write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        negative_temporary.replace(negative_output)
        manifest_temporary.replace(dual_manifest_path)
    finally:
        negative_temporary.unlink(missing_ok=True)
        manifest_temporary.unlink(missing_ok=True)
    return result
