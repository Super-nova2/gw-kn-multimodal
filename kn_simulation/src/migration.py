"""Migrate legacy SNANA directories into verified v2 optical artifacts."""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import h5py
from artifacts import (
    SCHEMA_VERSION,
    EventSource,
    EventSourceReader,
    _aggregate_sources,
    _legacy_source,
    _resolve_shard_path,
    _shard_sources,
    compact_artifacts,
    file_sha256,
    write_task_shard,
)
from config import Profile
from optical import read_snana_optical
from scheduler import latest_event_statuses, read_catalog_ids


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.unlink(missing_ok=True)
    try:
        temporary.write_text(
            json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _source_schema(
    source: EventSource,
    cache: dict[Path, str] | None = None,
) -> str:
    if source.kind == "legacy" or not source.path.is_file():
        return "legacy"
    resolved = source.path.resolve()
    if cache is not None and resolved in cache:
        return cache[resolved]
    with h5py.File(resolved, "r") as handle:
        schema = str(handle.attrs.get("schema_version", ""))
    if cache is not None:
        cache[resolved] = schema
    return schema


def _resolve_sources(
    profile: Profile,
    catalog_ids: list[int],
    statuses: dict[int, dict[str, Any]],
) -> dict[int, EventSource]:
    catalog_hash = file_sha256(profile.prepared_catalog)
    sources: dict[int, EventSource] = {}
    if profile.aggregate_artifact_file.is_file():
        sources.update(
            _aggregate_sources(
                profile, catalog_ids, catalog_hash, profile.aggregate_artifact_file
            )
        )
    shard_statuses = {
        simulation_id: status
        for simulation_id, status in statuses.items()
        if status.get("artifact_shard")
        and _resolve_shard_path(profile, str(status["artifact_shard"])).is_file()
    }
    sources.update(_shard_sources(profile, shard_statuses, catalog_hash))
    for simulation_id, status in statuses.items():
        if status.get("status") not in {"success", "skipped", "failed"}:
            continue
        if simulation_id not in sources:
            sources[simulation_id] = _legacy_source(profile, simulation_id, status)
    return sources


def validate_optical_coverage(profile: Profile) -> dict[str, Any]:
    """Require every latest successful event to have a valid v2 optical payload."""

    catalog_ids = read_catalog_ids(profile.prepared_catalog)
    statuses = latest_event_statuses(profile.status_dir)
    sources = _resolve_sources(profile, catalog_ids, statuses)
    terminal_ids = [
        simulation_id
        for simulation_id in catalog_ids
        if statuses.get(simulation_id, {}).get("status")
        in {"success", "skipped", "failed"}
    ]
    missing_sources: list[int] = []
    missing_optical: list[int] = []
    legacy_sources: list[int] = []
    reader = EventSourceReader()
    schema_cache: dict[Path, str] = {}
    try:
        for simulation_id in terminal_ids:
            source = sources.get(simulation_id)
            if source is None:
                missing_sources.append(simulation_id)
                continue
            if _source_schema(source, schema_cache) != SCHEMA_VERSION:
                legacy_sources.append(simulation_id)
                continue
            if statuses[simulation_id]["status"] == "success":
                _, _, optical = reader.read(source)
                if optical is None:
                    missing_optical.append(simulation_id)
    finally:
        reader.close()
    result = {
        "profile": profile.name,
        "created_utc": _utc_now(),
        "prepared_catalog_sha256": file_sha256(profile.prepared_catalog),
        "catalog_events": len(catalog_ids),
        "terminal_events": len(terminal_ids),
        "success_events": sum(
            statuses.get(value, {}).get("status") == "success" for value in catalog_ids
        ),
        "missing_sources": missing_sources,
        "legacy_sources": legacy_sources,
        "missing_optical": missing_optical,
    }
    result["verified"] = not (missing_sources or legacy_sources or missing_optical)
    return result


def _remove_unreferenced_shards(profile: Profile) -> int:
    statuses = latest_event_statuses(profile.status_dir)
    referenced = {
        _resolve_shard_path(profile, str(status["artifact_shard"])).resolve()
        for status in statuses.values()
        if status.get("artifact_shard")
    }
    removed = 0
    if profile.artifact_shard_dir.is_dir():
        for path in profile.artifact_shard_dir.glob("*.h5"):
            if path.resolve() not in referenced:
                path.unlink()
                removed += 1
    return removed


def migrate_optical(profile: Profile, *, batch_size: int = 200) -> dict[str, Any]:
    """Create v2 shards for all terminal events, preserving current statuses."""

    if batch_size <= 0:
        raise ValueError("Migration batch_size must be positive")
    catalog_ids = read_catalog_ids(profile.prepared_catalog)
    statuses = latest_event_statuses(profile.status_dir)
    terminal_ids = [
        simulation_id
        for simulation_id in catalog_ids
        if statuses.get(simulation_id, {}).get("status")
        in {"success", "skipped", "failed"}
    ]
    sources = _resolve_sources(profile, catalog_ids, statuses)
    schema_cache: dict[Path, str] = {}
    needs_migration = [
        simulation_id
        for simulation_id in terminal_ids
        if simulation_id not in sources
        or _source_schema(sources[simulation_id], schema_cache) != SCHEMA_VERSION
        or (
            statuses[simulation_id]["status"] == "success"
            and not sources[simulation_id].optical_payload_sha256
        )
    ]
    if not needs_migration:
        validation = validate_optical_coverage(profile)
        if not validation["verified"]:
            raise ValueError("Existing v2 optical artifacts failed validation")
        _atomic_json(profile.run_dir / "optical_migration.manifest.json", validation)
        return {**validation, "migrated_events": 0, "created_shards": 0}

    missing = sorted(set(needs_migration) - sources.keys())
    if missing:
        raise FileNotFoundError(
            f"Cannot migrate {len(missing)} terminal events without coordinate artifacts: "
            f"{missing[:20]}"
        )
    migration_id = datetime.now(timezone.utc).strftime("migration_%Y%m%dT%H%M%S%fZ")
    scratch_root = os.environ.get("SLURM_TMPDIR") or os.environ.get("JOBFS")
    if scratch_root is not None:
        scratch_path = Path(scratch_root)
        if not scratch_path.is_dir() or not os.access(scratch_path, os.W_OK):
            raise ValueError(f"Migration scratch is not writable: {scratch_path}")
    created_shards = 0
    print(
        f"[{profile.name}] migrating {len(needs_migration)} terminal events "
        f"in batches of {batch_size}",
        flush=True,
    )
    reader = EventSourceReader()
    try:
        for task_index, offset in enumerate(range(0, len(needs_migration), batch_size)):
            selected = needs_migration[offset : offset + batch_size]
            records = []
            events = []
            for simulation_id in selected:
                source = sources[simulation_id]
                plan, coordinates, optical = reader.read(source)
                status = str(statuses[simulation_id]["status"])
                if status == "success" and optical is None:
                    optical = read_snana_optical(
                        profile.sndata_sim_dir, profile.sim_name, simulation_id
                    )
                records.append(
                    {
                        "simulation_id": simulation_id,
                        "status": status,
                        "reason": statuses[simulation_id].get("reason"),
                        "observation_plan": plan,
                        "coordinates": coordinates,
                        "optical": optical,
                    }
                )
            path = profile.artifact_shard_dir / f"{migration_id}_{task_index}.h5"
            profile.artifact_shard_dir.mkdir(parents=True, exist_ok=True)
            with tempfile.TemporaryDirectory(
                prefix="kn_optical_migration_", dir=scratch_root
            ) as temporary:
                local_path = Path(temporary) / path.name
                write_task_shard(
                    local_path,
                    profile=profile,
                    submission_id=migration_id,
                    task_index=task_index,
                    events=records,
                )
                remote_temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
                remote_temporary.unlink(missing_ok=True)
                try:
                    shutil.copyfile(local_path, remote_temporary)
                    with h5py.File(remote_temporary, "r") as handle:
                        if handle.attrs.get("schema_version") != SCHEMA_VERSION:
                            raise ValueError("Migrated shard has invalid schema")
                    remote_temporary.replace(path)
                finally:
                    remote_temporary.unlink(missing_ok=True)
            artifact_reference = str(path.relative_to(profile.run_dir))
            for record in records:
                events.append(
                    {
                        "simulation_id": int(record["simulation_id"]),
                        "status": str(record["status"]),
                        "reason": record.get("reason"),
                        "artifact_shard": artifact_reference,
                        "submission_id": migration_id,
                        "task_index": task_index,
                    }
                )
            _atomic_json(
                profile.status_dir / "shards" / f"{migration_id}_{task_index}.json",
                {
                    "profile": profile.name,
                    "submission_id": migration_id,
                    "task_index": task_index,
                    "batch_size": batch_size,
                    "started_utc": _utc_now(),
                    "created_utc": _utc_now(),
                    "events": events,
                    "migration": True,
                },
            )
            created_shards += 1
            print(
                f"[{profile.name}] archived "
                f"{min(offset + len(selected), len(needs_migration))}/"
                f"{len(needs_migration)} events",
                flush=True,
            )
    finally:
        reader.close()

    validation = validate_optical_coverage(profile)
    if not validation["verified"]:
        raise ValueError("Migrated optical artifacts failed full validation")
    removed_shards = _remove_unreferenced_shards(profile)
    statuses = latest_event_statuses(profile.status_dir)
    if all(
        statuses.get(value, {}).get("status") in {"success", "skipped", "failed"}
        for value in catalog_ids
    ):
        validation["aggregate"] = compact_artifacts(
            profile,
            catalog_ids=catalog_ids,
            event_statuses=statuses,
        )
    manifest = {
        **validation,
        "migration_id": migration_id,
        "migrated_events": len(needs_migration),
        "created_shards": created_shards,
        "removed_unreferenced_shards": removed_shards,
    }
    _atomic_json(profile.run_dir / "optical_migration.manifest.json", manifest)
    return manifest


def prune_snana(profile: Profile, *, execute: bool = False) -> dict[str, Any]:
    """Report or remove legacy per-event SNANA directories after verified migration."""

    manifest_path = profile.run_dir / "optical_migration.manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"Optical migration manifest is missing: {manifest_path}"
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    validation = validate_optical_coverage(profile)
    if not validation["verified"]:
        raise ValueError("Optical coverage is not verified; refusing to prune SNANA")
    if validation["terminal_events"] != validation["catalog_events"]:
        raise ValueError(
            "The profile still has unprocessed events; refusing to prune SNANA"
        )
    if not profile.aggregate_artifact_file.is_file():
        raise FileNotFoundError(
            f"Validated aggregate is missing: {profile.aggregate_artifact_file}"
        )
    if manifest.get("prepared_catalog_sha256") != file_sha256(profile.prepared_catalog):
        raise ValueError("Migration manifest catalog checksum is stale")

    catalog_ids = set(read_catalog_ids(profile.prepared_catalog))
    candidates: list[Path] = []
    file_count = 0
    byte_count = 0
    root = profile.sndata_sim_dir.resolve()
    if root.is_dir():
        for path in root.iterdir():
            if not path.is_dir() or not path.name.startswith(f"{profile.sim_name}_"):
                continue
            raw_id = path.name[len(profile.sim_name) + 1 :]
            if not raw_id.isdigit() or int(raw_id) not in catalog_ids:
                continue
            if path.resolve().parent != root:
                raise ValueError(f"Unexpected SNANA candidate path: {path}")
            candidates.append(path)
            for child in path.rglob("*"):
                if child.is_file():
                    file_count += 1
                    byte_count += child.stat().st_size
    if execute:
        for path in candidates:
            shutil.rmtree(path)
        try:
            root.rmdir()
        except OSError:
            pass
    return {
        "profile": profile.name,
        "execute": bool(execute),
        "verified": True,
        "event_directories": len(candidates),
        "files": file_count,
        "bytes": byte_count,
        "estimated_inodes": len(candidates) + file_count,
    }
