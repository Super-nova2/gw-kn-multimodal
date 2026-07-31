"""Slurm submission, resume selection, and status reporting."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import subprocess
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from config import PIPELINE_ROOT, REPO_ROOT, Profile, resolve_profile_path


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        temporary.write_text(
            json.dumps(value, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def read_catalog_ids(path: Path) -> list[int]:
    if not path.is_file():
        raise FileNotFoundError(f"Prepared catalog does not exist: {path}")
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if "simulation_id" not in (reader.fieldnames or []):
            raise ValueError(f"Prepared catalog has no simulation_id column: {path}")
        result = [int(row["simulation_id"]) for row in reader if row["simulation_id"]]
    if not result:
        raise ValueError(f"Prepared catalog contains no events: {path}")
    if len(result) != len(set(result)):
        raise ValueError(
            f"Prepared catalog contains duplicate simulation_id values: {path}"
        )
    return result


def validate_prepared_run(profile: Profile) -> dict[str, Any]:
    """Validate the immutable handoff before submitting compute work."""

    required_files = (
        profile.input_catalog,
        profile.input_metadata,
        profile.prepared_catalog,
        profile.prepared_manifest,
        profile.opsim_db,
        profile.template_input,
        profile.too_config,
        profile.snana_bin_dir / "snlc_sim.exe",
    )
    missing = [str(path) for path in required_files if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Required pipeline files are missing: {missing}")
    if not profile.sndata_root.is_dir():
        raise FileNotFoundError(f"SNDATA_ROOT does not exist: {profile.sndata_root}")
    if not profile.skymap_dir.is_dir():
        raise FileNotFoundError(
            f"Skymap directory does not exist: {profile.skymap_dir}"
        )

    manifest = json.loads(profile.prepared_manifest.read_text(encoding="utf-8"))
    if manifest.get("profile_name") != profile.name:
        raise ValueError("Prepared catalog manifest belongs to a different profile")
    if manifest.get("input_catalog_sha256") != _sha256(profile.input_catalog):
        raise ValueError("catalog.csv checksum no longer matches its manifest")
    if manifest.get("output_catalog_sha256") != _sha256(profile.prepared_catalog):
        raise ValueError("kn_catalog.csv checksum no longer matches its manifest")
    if manifest.get("profile") != profile.as_manifest():
        raise ValueError(
            "Prepared catalog profile snapshot differs from the active profile"
        )
    read_catalog_ids(profile.prepared_catalog)
    return manifest


def latest_event_statuses(status_dir: Path) -> dict[int, dict[str, Any]]:
    latest: dict[int, dict[str, Any]] = {}
    shard_dir = status_dir / "shards"
    if not shard_dir.is_dir():
        return latest
    documents = []
    for path in shard_dir.glob("*.json"):
        try:
            document = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        documents.append((str(document.get("created_utc", "")), path.name, document))
    for _, _, document in sorted(documents):
        for event in document.get("events", []):
            latest[int(event["simulation_id"])] = event
    return latest


def _submission_ids(profile: Profile, *, resume: bool) -> list[int]:
    all_ids = read_catalog_ids(profile.prepared_catalog)
    if profile.submission_file.exists() and not resume:
        raise FileExistsError(
            f"A submission already exists for {profile.name}; use --resume to "
            "schedule only failed or unfinished events"
        )
    if not resume:
        return all_ids
    statuses = latest_event_statuses(profile.status_dir)
    terminal = {
        simulation_id
        for simulation_id, event in statuses.items()
        if event.get("status") in {"success", "skipped"}
    }
    return [simulation_id for simulation_id in all_ids if simulation_id not in terminal]


def _command_text(command: list[str]) -> str:
    return " ".join(command)


def submit_profile(
    profile: Profile,
    profile_reference: str | Path,
    *,
    dry_run: bool = False,
    resume: bool = False,
    batch_size: int | None = None,
    max_concurrency: int | None = None,
) -> dict[str, Any]:
    """Submit an event array and an afterany finalizer."""

    validate_prepared_run(profile)
    batch_size = profile.slurm.batch_size if batch_size is None else int(batch_size)
    max_concurrency = (
        profile.slurm.max_concurrency
        if max_concurrency is None
        else int(max_concurrency)
    )
    if batch_size <= 0 or max_concurrency <= 0:
        raise ValueError("batch_size and max_concurrency must be positive")
    event_ids = _submission_ids(profile, resume=resume)
    if not event_ids:
        raise ValueError(f"No unfinished events remain for profile {profile.name}")

    submission_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    submission_id = f"{submission_id}_{uuid.uuid4().hex[:8]}"
    ids_path = profile.status_dir / "submissions" / f"{submission_id}.ids"
    profile_path = resolve_profile_path(profile_reference)
    task_count = math.ceil(len(event_ids) / batch_size)
    worker_script = PIPELINE_ROOT / "slurm" / "worker.sh"
    if not worker_script.is_file():
        raise FileNotFoundError(f"Slurm worker does not exist: {worker_script}")

    export_common = (
        f"KN_PROFILE={profile_path},KN_IDS_FILE={ids_path},"
        f"KN_SUBMISSION_ID={submission_id},KN_BATCH_SIZE={batch_size}"
    )
    array_command = [
        "sbatch",
        "--parsable",
        f"--job-name={profile.sim_name}",
        f"--time={profile.slurm.time_limit}",
        f"--cpus-per-task={profile.slurm.cpus_per_task}",
        f"--mem={profile.slurm.memory}",
        f"--array=0-{task_count - 1}%{max_concurrency}",
        f"--output={profile.log_dir}/%x_%A_%a.out",
        f"--chdir={REPO_ROOT}",
        f"--export=ALL,KN_COMMAND=work,{export_common}",
        str(worker_script),
    ]
    result: dict[str, Any] = {
        "profile": profile.name,
        "submission_id": submission_id,
        "created_utc": _utc_now(),
        "resume": bool(resume),
        "event_count": len(event_ids),
        "batch_size": batch_size,
        "max_concurrency": max_concurrency,
        "task_count": task_count,
        "ids_file": str(ids_path),
        "array_command": array_command,
        "dry_run": bool(dry_run),
    }
    if dry_run:
        result["array_command_text"] = _command_text(array_command)
        result["finalizer_command_text"] = "submitted after array job id is known"
        return result

    profile.log_dir.mkdir(parents=True, exist_ok=True)
    ids_path.parent.mkdir(parents=True, exist_ok=True)
    ids_path.write_text("".join(f"{value}\n" for value in event_ids), encoding="utf-8")
    array = subprocess.run(array_command, check=True, capture_output=True, text=True)
    array_job_id = array.stdout.strip().split(";")[0]
    result["array_job_id"] = array_job_id

    finalize_command = [
        "sbatch",
        "--parsable",
        f"--job-name={profile.sim_name}-finalize",
        "--time=00:20:00",
        "--cpus-per-task=1",
        "--mem=2G",
        f"--dependency=afterany:{array_job_id}",
        f"--output={profile.log_dir}/%x_%j.out",
        f"--chdir={REPO_ROOT}",
        (
            f"--export=ALL,KN_COMMAND=finalize,{export_common},"
            f"KN_EXPECTED_TASKS={task_count}"
        ),
        str(worker_script),
    ]
    result["finalizer_command"] = finalize_command
    try:
        finalizer = subprocess.run(
            finalize_command,
            check=True,
            capture_output=True,
            text=True,
        )
        result["finalizer_job_id"] = finalizer.stdout.strip().split(";")[0]
    finally:
        _atomic_json(profile.submission_file, result)
        _atomic_json(
            profile.status_dir / "submissions" / f"{submission_id}.json",
            result,
        )
    return result


def status_report(profile: Profile) -> dict[str, Any]:
    all_ids = read_catalog_ids(profile.prepared_catalog)
    statuses = latest_event_statuses(profile.status_dir)
    counts = {"success": 0, "failed": 0, "skipped": 0, "unprocessed": 0}
    for simulation_id in all_ids:
        status = statuses.get(simulation_id, {}).get("status", "unprocessed")
        counts[status if status in counts else "failed"] += 1
    submission = None
    if profile.submission_file.is_file():
        submission = json.loads(profile.submission_file.read_text(encoding="utf-8"))
    return {
        "profile": profile.name,
        "catalog_events": len(all_ids),
        "counts": counts,
        "latest_submission": submission,
    }
