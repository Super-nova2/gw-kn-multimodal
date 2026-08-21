"""HDF5 storage and compaction for KN simulation intermediate products."""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pandas as pd
from config import Profile
from optical import (
    OBSERVATION_DTYPES,
    REALIZATION_DTYPES,
    OpticalPayload,
    normalize_optical_payload,
)

SCHEMA_VERSION = "kn-simulation-intermediates-v2"
LEGACY_SCHEMA_VERSION = "kn-simulation-intermediates-v1"
COORDINATE_COLUMNS = (
    "simulation_id",
    "sample_index",
    "ra",
    "dec",
    "distance_mpc",
    "posterior_probability",
    "is_true_position",
    "redshift",
    "libid",
    "in_baseline_footprint",
    "too_tile_index",
    "too_nobs",
    "too_mode",
)
INTEGER_COLUMNS = {
    "simulation_id",
    "sample_index",
    "libid",
    "too_tile_index",
    "too_nobs",
}
FLOAT_COLUMNS = {"ra", "dec", "distance_mpc", "posterior_probability", "redshift"}
BOOLEAN_COLUMNS = {"is_true_position", "in_baseline_footprint"}
STRING_COLUMNS = {"too_mode"}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_default(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Cannot serialize {type(value).__name__} to JSON")


def _json_text(value: dict[str, Any]) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        default=_json_default,
    )


def _bool_values(series: pd.Series) -> np.ndarray:
    if pd.api.types.is_bool_dtype(series.dtype):
        return series.to_numpy(dtype=np.bool_)
    if pd.api.types.is_numeric_dtype(series.dtype):
        return series.fillna(0).to_numpy(dtype=float).astype(np.bool_)
    normalized = series.fillna("").astype(str).str.strip().str.lower()
    valid = {"true", "false", "1", "0"}
    invalid = sorted(set(normalized) - valid)
    if invalid:
        raise ValueError(f"Invalid boolean coordinate values: {invalid[:5]}")
    return normalized.isin({"true", "1"}).to_numpy(dtype=np.bool_)


def normalize_coordinates(
    frame: pd.DataFrame | None,
    simulation_id: int,
) -> pd.DataFrame:
    """Return one event's coordinates in the stable aggregate schema."""

    if frame is None:
        frame = pd.DataFrame()
    frame = frame.copy()
    count = len(frame)
    defaults: dict[str, Any] = {
        "simulation_id": int(simulation_id),
        "sample_index": np.arange(count, dtype=np.int64),
        "ra": np.nan,
        "dec": np.nan,
        "distance_mpc": np.nan,
        "posterior_probability": np.nan,
        "is_true_position": False,
        "redshift": np.nan,
        "libid": -1,
        "in_baseline_footprint": False,
        "too_tile_index": -1,
        "too_nobs": 0,
        "too_mode": "",
    }
    for column, default in defaults.items():
        if column not in frame:
            if np.isscalar(default):
                frame[column] = np.full(count, default)
            else:
                frame[column] = default

    if count:
        event_ids = pd.to_numeric(frame["simulation_id"], errors="raise").to_numpy(
            dtype=np.int64
        )
        if not np.all(event_ids == int(simulation_id)):
            raise ValueError(
                f"Coordinate rows do not all belong to simulation_id={simulation_id}"
            )
    result: dict[str, Any] = {}
    for column in COORDINATE_COLUMNS:
        series = frame[column]
        if column in INTEGER_COLUMNS:
            result[column] = pd.to_numeric(series, errors="raise").to_numpy(
                dtype=np.int64
            )
        elif column in FLOAT_COLUMNS:
            result[column] = pd.to_numeric(series, errors="raise").to_numpy(
                dtype=np.float64
            )
        elif column in BOOLEAN_COLUMNS:
            result[column] = _bool_values(series)
        else:
            result[column] = series.fillna("").astype(str).to_numpy(dtype=object)
    return pd.DataFrame(result, columns=COORDINATE_COLUMNS)


def _atomic_hdf5_path(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    return path.with_name(f".{path.name}.tmp.{os.getpid()}")


def _create_coordinate_datasets(group: h5py.Group, count: int) -> None:
    options: dict[str, Any] = {"shape": (count,)}
    if count:
        options.update(
            chunks=(min(count, 1_024),),
            compression="gzip",
            compression_opts=4,
            shuffle=True,
        )
    for column in COORDINATE_COLUMNS:
        if column in INTEGER_COLUMNS:
            group.create_dataset(column, dtype=np.int64, **options)
        elif column in FLOAT_COLUMNS:
            group.create_dataset(column, dtype=np.float64, **options)
        elif column in BOOLEAN_COLUMNS:
            group.create_dataset(column, dtype=np.bool_, **options)
        else:
            group.create_dataset(column, dtype=h5py.string_dtype("utf-8"), **options)


def _write_coordinate_slice(
    group: h5py.Group,
    start: int,
    frame: pd.DataFrame,
) -> None:
    stop = start + len(frame)
    for column in COORDINATE_COLUMNS:
        values = frame[column].to_numpy()
        if column in STRING_COLUMNS:
            values = values.astype(object)
        group[column][start:stop] = values


def _create_event_datasets(group: h5py.Group, count: int) -> None:
    text_dtype = h5py.string_dtype("utf-8")
    group.create_dataset("simulation_id", shape=(count,), dtype=np.int64)
    group.create_dataset("status", shape=(count,), dtype=text_dtype)
    group.create_dataset("reason", shape=(count,), dtype=text_dtype)
    group.create_dataset("coordinate_start", shape=(count,), dtype=np.int64)
    group.create_dataset("coordinate_count", shape=(count,), dtype=np.int64)
    group.create_dataset("optical_realization_start", shape=(count,), dtype=np.int64)
    group.create_dataset("optical_realization_count", shape=(count,), dtype=np.int64)
    group.create_dataset("optical_observation_start", shape=(count,), dtype=np.int64)
    group.create_dataset("optical_observation_count", shape=(count,), dtype=np.int64)
    group.create_dataset("mjd_explode", shape=(count,), dtype=np.float64)
    group.create_dataset("optical_payload_sha256", shape=(count,), dtype=text_dtype)
    group.create_dataset("observation_plan_json", shape=(count,), dtype=text_dtype)
    group.create_dataset("submission_id", shape=(count,), dtype=text_dtype)
    group.create_dataset("task_index", shape=(count,), dtype=np.int64)


def _dataset_options(count: int, chunk_rows: int) -> dict[str, Any]:
    options: dict[str, Any] = {"shape": (count,)}
    if count:
        options.update(
            chunks=(min(count, chunk_rows),),
            compression="gzip",
            compression_opts=4,
            shuffle=True,
        )
    return options


def _create_optical_datasets(
    handle: h5py.File,
    realization_count: int,
    observation_count: int,
) -> tuple[h5py.Group, h5py.Group]:
    optical = handle.create_group("optical")
    optical.attrs.update(storage="raw_snana_normalized", band_encoding="ascii_uint8")
    realizations = optical.create_group("realizations")
    observations = optical.create_group("observations")
    for name, dtype in REALIZATION_DTYPES.items():
        realizations.create_dataset(
            name, dtype=dtype, **_dataset_options(realization_count, 1_024)
        )
    for name, dtype in OBSERVATION_DTYPES.items():
        observations.create_dataset(
            name, dtype=dtype, **_dataset_options(observation_count, 4_096)
        )
    return realizations, observations


def _write_optical_slice(
    realization_group: h5py.Group,
    observation_group: h5py.Group,
    realization_start: int,
    observation_start: int,
    payload: OpticalPayload,
) -> None:
    realization_stop = realization_start + payload.realization_count
    observation_stop = observation_start + payload.observation_count
    for name in REALIZATION_DTYPES:
        values = payload.realizations[name]
        if name == "observation_start":
            values = values + observation_start
        realization_group[name][realization_start:realization_stop] = values
    for name in OBSERVATION_DTYPES:
        observation_group[name][observation_start:observation_stop] = (
            payload.observations[name]
        )


def write_task_shard(
    path: Path,
    *,
    profile: Profile,
    submission_id: str,
    task_index: int,
    events: Iterable[dict[str, Any]],
) -> Path:
    """Atomically write all intermediate products from one array task."""

    records = list(events)
    normalized = [
        normalize_coordinates(record.get("coordinates"), int(record["simulation_id"]))
        for record in records
    ]
    optical_payloads = [
        normalize_optical_payload(record.get("optical"), int(record["simulation_id"]))
        for record in records
    ]
    for record, payload in zip(records, optical_payloads):
        if record.get("status") == "success" and payload is None:
            raise ValueError(
                f"Successful simulation_id={record['simulation_id']} has no optical payload"
            )
    coordinate_count = sum(len(frame) for frame in normalized)
    optical_realization_count = sum(
        payload.realization_count for payload in optical_payloads if payload is not None
    )
    optical_observation_count = sum(
        payload.observation_count for payload in optical_payloads if payload is not None
    )
    temporary = _atomic_hdf5_path(path)
    temporary.unlink(missing_ok=True)
    try:
        with h5py.File(temporary, "w") as handle:
            handle.attrs.update(
                schema_version=SCHEMA_VERSION,
                artifact_kind="task_shard",
                profile_name=profile.name,
                profile_source=profile.source,
                profile_split=profile.split,
                prepared_catalog_sha256=file_sha256(profile.prepared_catalog),
                submission_id=str(submission_id),
                task_index=int(task_index),
                created_utc=utc_now(),
                event_count=len(records),
                coordinate_count=coordinate_count,
                optical_realization_count=optical_realization_count,
                optical_observation_count=optical_observation_count,
            )
            event_group = handle.create_group("events")
            coordinate_group = handle.create_group("coordinates")
            _create_event_datasets(event_group, len(records))
            _create_coordinate_datasets(coordinate_group, coordinate_count)
            realization_group, observation_group = _create_optical_datasets(
                handle, optical_realization_count, optical_observation_count
            )
            coordinate_cursor = 0
            realization_cursor = 0
            observation_cursor = 0
            for index, (record, frame, payload) in enumerate(
                zip(records, normalized, optical_payloads)
            ):
                event_group["simulation_id"][index] = int(record["simulation_id"])
                event_group["status"][index] = str(record.get("status", "failed"))
                event_group["reason"][index] = str(record.get("reason") or "")
                event_group["coordinate_start"][index] = coordinate_cursor
                event_group["coordinate_count"][index] = len(frame)
                event_group["optical_realization_start"][index] = realization_cursor
                event_group["optical_realization_count"][index] = (
                    payload.realization_count if payload is not None else 0
                )
                event_group["optical_observation_start"][index] = observation_cursor
                event_group["optical_observation_count"][index] = (
                    payload.observation_count if payload is not None else 0
                )
                event_group["mjd_explode"][index] = (
                    payload.mjd_explode if payload is not None else np.nan
                )
                event_group["optical_payload_sha256"][index] = (
                    payload.checksum if payload is not None else ""
                )
                event_group["observation_plan_json"][index] = _json_text(
                    record.get("observation_plan") or {}
                )
                event_group["submission_id"][index] = str(submission_id)
                event_group["task_index"][index] = int(task_index)
                _write_coordinate_slice(coordinate_group, coordinate_cursor, frame)
                coordinate_cursor += len(frame)
                if payload is not None:
                    _write_optical_slice(
                        realization_group,
                        observation_group,
                        realization_cursor,
                        observation_cursor,
                        payload,
                    )
                    realization_cursor += payload.realization_count
                    observation_cursor += payload.observation_count
            handle.flush()
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
    return path


def _decode(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


@dataclass(frozen=True)
class EventSource:
    simulation_id: int
    kind: str
    path: Path
    event_index: int
    coordinate_start: int
    coordinate_count: int
    optical_realization_start: int
    optical_realization_count: int
    optical_observation_start: int
    optical_observation_count: int
    mjd_explode: float
    optical_payload_sha256: str
    status: str
    reason: str
    submission_id: str
    task_index: int
    plan_path: Path | None = None


def _resolve_shard_path(profile: Profile, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else profile.run_dir / path


def _event_source_from_handle(
    handle: h5py.File,
    *,
    simulation_id: int,
    index: int,
    kind: str,
    path: Path,
) -> EventSource:
    schema_version = str(handle.attrs.get("schema_version", ""))
    is_v2 = schema_version == SCHEMA_VERSION
    return EventSource(
        simulation_id=int(simulation_id),
        kind=kind,
        path=path,
        event_index=int(index),
        coordinate_start=int(handle["events/coordinate_start"][index]),
        coordinate_count=int(handle["events/coordinate_count"][index]),
        optical_realization_start=(
            int(handle["events/optical_realization_start"][index]) if is_v2 else 0
        ),
        optical_realization_count=(
            int(handle["events/optical_realization_count"][index]) if is_v2 else 0
        ),
        optical_observation_start=(
            int(handle["events/optical_observation_start"][index]) if is_v2 else 0
        ),
        optical_observation_count=(
            int(handle["events/optical_observation_count"][index]) if is_v2 else 0
        ),
        mjd_explode=(float(handle["events/mjd_explode"][index]) if is_v2 else np.nan),
        optical_payload_sha256=(
            _decode(handle["events/optical_payload_sha256"][index]) if is_v2 else ""
        ),
        status=_decode(handle["events/status"][index]),
        reason=_decode(handle["events/reason"][index]),
        submission_id=_decode(handle["events/submission_id"][index]),
        task_index=int(handle["events/task_index"][index]),
    )


def _validate_source_header(
    handle: h5py.File,
    *,
    profile: Profile,
    path: Path,
    artifact_kind: str,
    catalog_hash: str,
) -> str:
    schema_version = str(handle.attrs.get("schema_version", ""))
    if schema_version not in {SCHEMA_VERSION, LEGACY_SCHEMA_VERSION}:
        raise ValueError(f"Unsupported artifact schema in {path}")
    if handle.attrs.get("artifact_kind") != artifact_kind:
        raise ValueError(f"Unexpected artifact kind in {path}")
    if handle.attrs.get("profile_name") != profile.name:
        raise ValueError(f"Artifact belongs to another profile: {path}")
    if handle.attrs.get("prepared_catalog_sha256") != catalog_hash:
        raise ValueError(f"Artifact catalog checksum mismatch: {path}")
    return schema_version


def _shard_sources(
    profile: Profile,
    event_statuses: dict[int, dict[str, Any]],
    catalog_hash: str,
) -> dict[int, EventSource]:
    requested: dict[Path, set[int]] = {}
    for simulation_id, status in event_statuses.items():
        value = status.get("artifact_shard")
        if value:
            requested.setdefault(_resolve_shard_path(profile, str(value)), set()).add(
                simulation_id
            )
    result: dict[int, EventSource] = {}
    for path, simulation_ids in requested.items():
        if not path.is_file():
            raise FileNotFoundError(f"Artifact shard is missing: {path}")
        with h5py.File(path, "r") as handle:
            _validate_source_header(
                handle,
                profile=profile,
                path=path,
                artifact_kind="task_shard",
                catalog_hash=catalog_hash,
            )
            ids = handle["events/simulation_id"][:].astype(np.int64)
            if len(ids) != len(set(ids.tolist())):
                raise ValueError(f"Duplicate event IDs in artifact shard: {path}")
            positions = {int(value): index for index, value in enumerate(ids)}
            missing = simulation_ids - positions.keys()
            if missing:
                raise ValueError(
                    f"Artifact shard {path} does not contain events {sorted(missing)}"
                )
            for simulation_id in simulation_ids:
                result[simulation_id] = _event_source_from_handle(
                    handle,
                    simulation_id=simulation_id,
                    index=positions[simulation_id],
                    kind="hdf5",
                    path=path,
                )
    return result


def _aggregate_sources(
    profile: Profile,
    catalog_ids: list[int],
    catalog_hash: str,
    path: Path,
) -> dict[int, EventSource]:
    """Return existing aggregate rows so a resume can rebuild a fresh file."""

    result: dict[int, EventSource] = {}
    with h5py.File(path, "r") as handle:
        _validate_source_header(
            handle,
            profile=profile,
            path=path,
            artifact_kind="aggregate",
            catalog_hash=catalog_hash,
        )
        ids = handle["events/simulation_id"][:].astype(np.int64).tolist()
        positions = {int(value): index for index, value in enumerate(ids)}
        for simulation_id in catalog_ids:
            if simulation_id not in positions:
                continue
            result[simulation_id] = _event_source_from_handle(
                handle,
                simulation_id=simulation_id,
                index=positions[simulation_id],
                kind="aggregate",
                path=path,
            )
    return result


def _legacy_source(
    profile: Profile,
    simulation_id: int,
    status: dict[str, Any],
) -> EventSource:
    coordinate_path = profile.coordinate_manifest_dir / f"{simulation_id}.csv"
    plan_path = profile.observation_plan_dir / f"{simulation_id}.json"
    if not coordinate_path.is_file() or not plan_path.is_file():
        raise FileNotFoundError(
            f"Legacy artifacts are incomplete for simulation_id={simulation_id}: "
            f"{coordinate_path}, {plan_path}"
        )
    with coordinate_path.open("r", encoding="utf-8") as stream:
        coordinate_count = max(0, sum(1 for _ in stream) - 1)
    return EventSource(
        simulation_id=simulation_id,
        kind="legacy",
        path=coordinate_path,
        plan_path=plan_path,
        event_index=0,
        coordinate_start=0,
        coordinate_count=coordinate_count,
        optical_realization_start=0,
        optical_realization_count=0,
        optical_observation_start=0,
        optical_observation_count=0,
        mjd_explode=np.nan,
        optical_payload_sha256="",
        status=str(status.get("status", "failed")),
        reason=str(status.get("reason") or ""),
        submission_id=str(status.get("submission_id") or "legacy"),
        task_index=int(status.get("task_index", -1)),
    )


class EventSourceReader:
    """Read consecutive events while reusing the current task-shard handle."""

    def __init__(self) -> None:
        self._path: Path | None = None
        self._handle: h5py.File | None = None

    def close(self) -> None:
        if self._handle is not None:
            self._handle.close()
        self._handle = None
        self._path = None

    def read(
        self, source: EventSource
    ) -> tuple[dict[str, Any], pd.DataFrame, OpticalPayload | None]:
        if source.kind == "legacy":
            assert source.plan_path is not None
            plan = json.loads(source.plan_path.read_text(encoding="utf-8"))
            coordinates = normalize_coordinates(
                pd.read_csv(source.path), source.simulation_id
            )
            return plan, coordinates, None
        if self._path != source.path:
            self.close()
            self._handle = h5py.File(source.path, "r")
            self._path = source.path
        assert self._handle is not None
        handle = self._handle
        plan = json.loads(
            _decode(handle["events/observation_plan_json"][source.event_index])
        )
        start = source.coordinate_start
        stop = start + source.coordinate_count
        values: dict[str, Any] = {}
        for column in COORDINATE_COLUMNS:
            data = handle[f"coordinates/{column}"][start:stop]
            if column in STRING_COLUMNS:
                data = np.asarray([_decode(value) for value in data], dtype=object)
            values[column] = data
        coordinates = normalize_coordinates(pd.DataFrame(values), source.simulation_id)
        optical = None
        if source.optical_payload_sha256:
            realization_start = source.optical_realization_start
            realization_stop = realization_start + source.optical_realization_count
            observation_start = source.optical_observation_start
            observation_stop = observation_start + source.optical_observation_count
            realizations = {
                name: handle[f"optical/realizations/{name}"][
                    realization_start:realization_stop
                ]
                for name in REALIZATION_DTYPES
            }
            realizations["observation_start"] = (
                realizations["observation_start"].astype(np.int64) - observation_start
            )
            observations = {
                name: handle[f"optical/observations/{name}"][
                    observation_start:observation_stop
                ]
                for name in OBSERVATION_DTYPES
            }
            optical = normalize_optical_payload(
                OpticalPayload(
                    simulation_id=source.simulation_id,
                    mjd_explode=source.mjd_explode,
                    realizations=realizations,
                    observations=observations,
                    checksum=source.optical_payload_sha256,
                ),
                source.simulation_id,
            )
        return plan, coordinates, optical


def _validate_offsets(
    starts: np.ndarray, counts: np.ndarray, total: int, label: str
) -> None:
    if len(starts) and (
        starts[0] != 0
        or not np.array_equal(starts[1:], np.cumsum(counts)[:-1])
        or int(starts[-1] + counts[-1]) != total
    ):
        raise ValueError(f"Aggregate {label} offsets are inconsistent")


def _payload_from_handle(
    handle: h5py.File,
    event_index: int,
    simulation_id: int,
) -> OpticalPayload | None:
    checksum = _decode(handle["events/optical_payload_sha256"][event_index])
    if not checksum:
        return None
    realization_start = int(handle["events/optical_realization_start"][event_index])
    realization_count = int(handle["events/optical_realization_count"][event_index])
    observation_start = int(handle["events/optical_observation_start"][event_index])
    observation_count = int(handle["events/optical_observation_count"][event_index])
    realizations = {
        name: handle[f"optical/realizations/{name}"][
            realization_start : realization_start + realization_count
        ]
        for name in REALIZATION_DTYPES
    }
    realizations["observation_start"] = (
        realizations["observation_start"].astype(np.int64) - observation_start
    )
    observations = {
        name: handle[f"optical/observations/{name}"][
            observation_start : observation_start + observation_count
        ]
        for name in OBSERVATION_DTYPES
    }
    return normalize_optical_payload(
        OpticalPayload(
            simulation_id=simulation_id,
            mjd_explode=float(handle["events/mjd_explode"][event_index]),
            realizations=realizations,
            observations=observations,
            checksum=checksum,
        ),
        simulation_id,
    )


def validate_aggregate(
    path: Path,
    *,
    profile: Profile,
    catalog_ids: list[int],
    catalog_hash: str,
) -> dict[str, Any]:
    """Validate the canonical v2 aggregate before source artifacts are deleted."""

    if not path.is_file():
        raise FileNotFoundError(f"Aggregate artifact does not exist: {path}")
    with h5py.File(path, "r") as handle:
        if handle.attrs.get("schema_version") != SCHEMA_VERSION:
            raise ValueError(f"Unsupported aggregate schema: {path}")
        if handle.attrs.get("artifact_kind") != "aggregate":
            raise ValueError(f"Not an aggregate artifact: {path}")
        if handle.attrs.get("profile_name") != profile.name:
            raise ValueError(f"Aggregate belongs to another profile: {path}")
        if handle.attrs.get("prepared_catalog_sha256") != catalog_hash:
            raise ValueError(f"Aggregate catalog checksum mismatch: {path}")
        ids = handle["events/simulation_id"][:].astype(np.int64).tolist()
        if ids != catalog_ids:
            raise ValueError("Aggregate event IDs/order differ from kn_catalog.csv")
        if int(handle.attrs["event_count"]) != len(catalog_ids):
            raise ValueError("Aggregate event count attribute is inconsistent")
        statuses = [_decode(value) for value in handle["events/status"][:]]
        if any(status not in {"success", "skipped", "failed"} for status in statuses):
            raise ValueError(
                "Aggregate contains an unprocessed or invalid event status"
            )

        coordinate_total = int(handle.attrs["coordinate_count"])
        for column in COORDINATE_COLUMNS:
            if len(handle[f"coordinates/{column}"]) != coordinate_total:
                raise ValueError(
                    f"Aggregate coordinate dataset {column} has an invalid length"
                )
        coordinate_starts = handle["events/coordinate_start"][:].astype(np.int64)
        coordinate_counts = handle["events/coordinate_count"][:].astype(np.int64)
        _validate_offsets(
            coordinate_starts, coordinate_counts, coordinate_total, "coordinate"
        )
        expected_samples = int(profile.samples_per_event)
        for status, count in zip(statuses, coordinate_counts):
            if status in {"success", "skipped"} and count != expected_samples:
                raise ValueError(
                    f"Aggregate {status} event has {count} coordinates; "
                    f"expected {expected_samples}"
                )
            if status == "failed" and not 0 <= count <= expected_samples:
                raise ValueError(f"Failed event has invalid coordinate count {count}")

        realization_total = int(handle.attrs["optical_realization_count"])
        observation_total = int(handle.attrs["optical_observation_count"])
        for name in REALIZATION_DTYPES:
            if len(handle[f"optical/realizations/{name}"]) != realization_total:
                raise ValueError(
                    f"Optical realization dataset {name} has invalid length"
                )
        for name in OBSERVATION_DTYPES:
            if len(handle[f"optical/observations/{name}"]) != observation_total:
                raise ValueError(
                    f"Optical observation dataset {name} has invalid length"
                )
        realization_starts = handle["events/optical_realization_start"][:].astype(
            np.int64
        )
        realization_counts = handle["events/optical_realization_count"][:].astype(
            np.int64
        )
        observation_starts = handle["events/optical_observation_start"][:].astype(
            np.int64
        )
        observation_counts = handle["events/optical_observation_count"][:].astype(
            np.int64
        )
        _validate_offsets(
            realization_starts, realization_counts, realization_total, "realization"
        )
        _validate_offsets(
            observation_starts, observation_counts, observation_total, "observation"
        )
        checksums = [
            _decode(value) for value in handle["events/optical_payload_sha256"][:]
        ]
        for index, (simulation_id, status, checksum) in enumerate(
            zip(catalog_ids, statuses, checksums)
        ):
            if status == "success" and not checksum:
                raise ValueError(
                    f"Successful simulation_id={simulation_id} has no optical payload"
                )
            if checksum:
                payload = _payload_from_handle(handle, index, simulation_id)
                assert payload is not None
                if payload.checksum != checksum:
                    raise ValueError(
                        f"Optical checksum mismatch for simulation_id={simulation_id}"
                    )

        for simulation_id, raw in zip(
            catalog_ids, handle["events/observation_plan_json"]
        ):
            plan = json.loads(_decode(raw))
            if not isinstance(plan, dict):
                raise TypeError("Observation plan is not a JSON object")
            if int(plan.get("simulation_id", -1)) != simulation_id:
                raise ValueError("Aggregate observation plan ID is inconsistent")
    return {
        "path": str(path),
        "schema_version": SCHEMA_VERSION,
        "event_count": len(catalog_ids),
        "coordinate_count": coordinate_total,
        "optical_realization_count": realization_total,
        "optical_observation_count": observation_total,
        "prepared_catalog_sha256": catalog_hash,
    }


def _cleanup_sources(profile: Profile) -> dict[str, int]:
    removed = {"artifact_shards": 0, "coordinate_csv": 0, "observation_plans": 0}
    if profile.artifact_shard_dir.is_dir():
        for path in profile.artifact_shard_dir.glob("*.h5"):
            path.unlink()
            removed["artifact_shards"] += 1
        try:
            profile.artifact_shard_dir.rmdir()
        except OSError:
            pass
    for directory, pattern, key in (
        (profile.coordinate_manifest_dir, "*.csv", "coordinate_csv"),
        (profile.observation_plan_dir, "*.json", "observation_plans"),
    ):
        if directory.is_dir():
            for path in directory.glob(pattern):
                path.unlink()
                removed[key] += 1
            try:
                directory.rmdir()
            except OSError:
                pass
    return removed


def compact_artifacts(
    profile: Profile,
    *,
    catalog_ids: list[int],
    event_statuses: dict[int, dict[str, Any]],
    cleanup: bool = True,
) -> dict[str, Any]:
    """Merge coordinate and raw optical artifacts into one validated v2 HDF5."""

    invalid = [
        simulation_id
        for simulation_id in catalog_ids
        if event_statuses.get(simulation_id, {}).get("status")
        not in {"success", "skipped", "failed"}
    ]
    if invalid:
        raise ValueError(
            f"Cannot compact; {len(invalid)} events are unprocessed or have "
            "an invalid status"
        )
    catalog_hash = file_sha256(profile.prepared_catalog)
    existing_aggregate = profile.aggregate_artifact_file
    has_pending_shards = profile.artifact_shard_dir.is_dir() and any(
        profile.artifact_shard_dir.glob("*.h5")
    )
    if existing_aggregate.is_file() and not has_pending_shards:
        aggregate = validate_aggregate(
            existing_aggregate,
            profile=profile,
            catalog_ids=catalog_ids,
            catalog_hash=catalog_hash,
        )
        aggregate["already_compacted"] = True
        aggregate["cleanup"] = _cleanup_sources(profile) if cleanup else {}
        return aggregate

    catalog_statuses = {
        simulation_id: event_statuses[simulation_id] for simulation_id in catalog_ids
    }
    sources: dict[int, EventSource] = {}
    if existing_aggregate.is_file():
        sources.update(
            _aggregate_sources(profile, catalog_ids, catalog_hash, existing_aggregate)
        )
    shard_statuses = {
        simulation_id: status
        for simulation_id, status in catalog_statuses.items()
        if status.get("artifact_shard")
        and _resolve_shard_path(profile, str(status["artifact_shard"])).is_file()
    }
    sources.update(_shard_sources(profile, shard_statuses, catalog_hash))
    for simulation_id in catalog_ids:
        if simulation_id not in sources:
            sources[simulation_id] = _legacy_source(
                profile, simulation_id, event_statuses[simulation_id]
            )
        source = sources[simulation_id]
        expected_status = str(event_statuses[simulation_id]["status"])
        if source.status != expected_status:
            raise ValueError(
                f"Artifact/status mismatch for simulation_id={simulation_id}: "
                f"{source.status} != {expected_status}"
            )
        expected_samples = int(profile.samples_per_event)
        if source.status in {"success", "skipped"}:
            if source.coordinate_count != expected_samples:
                raise ValueError(
                    f"simulation_id={simulation_id} has {source.coordinate_count} "
                    f"coordinates; expected {expected_samples}"
                )
        elif not 0 <= source.coordinate_count <= expected_samples:
            raise ValueError(
                f"simulation_id={simulation_id} has invalid failed-event "
                f"coordinate count {source.coordinate_count}"
            )
        if source.status == "success" and not source.optical_payload_sha256:
            raise ValueError(
                f"Successful simulation_id={simulation_id} has no archived optical payload; "
                "run kn-sim migrate-optical before compacting"
            )

    coordinate_count = sum(sources[value].coordinate_count for value in catalog_ids)
    optical_realization_count = sum(
        sources[value].optical_realization_count for value in catalog_ids
    )
    optical_observation_count = sum(
        sources[value].optical_observation_count for value in catalog_ids
    )
    output = profile.aggregate_artifact_file
    temporary = _atomic_hdf5_path(output)
    temporary.unlink(missing_ok=True)
    reader = EventSourceReader()
    try:
        with h5py.File(temporary, "w") as handle:
            handle.attrs.update(
                schema_version=SCHEMA_VERSION,
                artifact_kind="aggregate",
                profile_name=profile.name,
                profile_source=profile.source,
                profile_split=profile.split,
                prepared_catalog_sha256=catalog_hash,
                created_utc=utc_now(),
                event_count=len(catalog_ids),
                coordinate_count=coordinate_count,
                optical_realization_count=optical_realization_count,
                optical_observation_count=optical_observation_count,
            )
            event_group = handle.create_group("events")
            coordinate_group = handle.create_group("coordinates")
            _create_event_datasets(event_group, len(catalog_ids))
            _create_coordinate_datasets(coordinate_group, coordinate_count)
            realization_group, observation_group = _create_optical_datasets(
                handle, optical_realization_count, optical_observation_count
            )
            coordinate_cursor = 0
            realization_cursor = 0
            observation_cursor = 0
            for index, simulation_id in enumerate(catalog_ids):
                source = sources[simulation_id]
                plan, coordinates, optical = reader.read(source)
                if len(coordinates) != source.coordinate_count:
                    raise ValueError(
                        f"Coordinate count changed for simulation_id={simulation_id}"
                    )
                if int(plan.get("simulation_id", -1)) != simulation_id:
                    raise ValueError(
                        f"Observation plan ID mismatch for simulation_id={simulation_id}"
                    )
                if source.status == "success" and optical is None:
                    raise ValueError(
                        f"Successful simulation_id={simulation_id} lost optical payload"
                    )
                event_group["simulation_id"][index] = simulation_id
                event_group["status"][index] = source.status
                event_group["reason"][index] = source.reason
                event_group["coordinate_start"][index] = coordinate_cursor
                event_group["coordinate_count"][index] = len(coordinates)
                event_group["optical_realization_start"][index] = realization_cursor
                event_group["optical_realization_count"][index] = (
                    optical.realization_count if optical is not None else 0
                )
                event_group["optical_observation_start"][index] = observation_cursor
                event_group["optical_observation_count"][index] = (
                    optical.observation_count if optical is not None else 0
                )
                event_group["mjd_explode"][index] = (
                    optical.mjd_explode if optical is not None else np.nan
                )
                event_group["optical_payload_sha256"][index] = (
                    optical.checksum if optical is not None else ""
                )
                event_group["observation_plan_json"][index] = _json_text(plan)
                event_group["submission_id"][index] = source.submission_id
                event_group["task_index"][index] = source.task_index
                _write_coordinate_slice(
                    coordinate_group, coordinate_cursor, coordinates
                )
                coordinate_cursor += len(coordinates)
                if optical is not None:
                    _write_optical_slice(
                        realization_group,
                        observation_group,
                        realization_cursor,
                        observation_cursor,
                        optical,
                    )
                    realization_cursor += optical.realization_count
                    observation_cursor += optical.observation_count
            handle.flush()
        validate_aggregate(
            temporary,
            profile=profile,
            catalog_ids=catalog_ids,
            catalog_hash=catalog_hash,
        )
        temporary.replace(output)
        aggregate = validate_aggregate(
            output,
            profile=profile,
            catalog_ids=catalog_ids,
            catalog_hash=catalog_hash,
        )
    finally:
        reader.close()
        temporary.unlink(missing_ok=True)
    aggregate["already_compacted"] = False
    aggregate["cleanup"] = _cleanup_sources(profile) if cleanup else {}
    return aggregate


def aggregate_status(profile: Profile) -> dict[str, Any]:
    """Return lightweight aggregate/shard state without opening all task shards."""

    shard_count = (
        sum(1 for _ in profile.artifact_shard_dir.glob("*.h5"))
        if profile.artifact_shard_dir.is_dir()
        else 0
    )
    result: dict[str, Any] = {
        "path": str(profile.aggregate_artifact_file),
        "exists": profile.aggregate_artifact_file.is_file(),
        "pending_artifact_shards": shard_count,
    }
    if result["exists"]:
        try:
            with h5py.File(profile.aggregate_artifact_file, "r") as handle:
                result.update(
                    valid_schema=(handle.attrs.get("schema_version") == SCHEMA_VERSION),
                    event_count=int(handle.attrs.get("event_count", -1)),
                    coordinate_count=int(handle.attrs.get("coordinate_count", -1)),
                    prepared_catalog_sha256=str(
                        handle.attrs.get("prepared_catalog_sha256", "")
                    ),
                )
        except OSError as error:
            result.update(valid_schema=False, error=str(error))
    return result
