"""Slurm array worker and deterministic status finalizer."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import snana
from config import Profile, load_profile
from scheduler import latest_event_statuses, read_catalog_ids


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


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


def _atomic_ids(path: Path, values: list[int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        temporary.write_text(
            "".join(f"{value}\n" for value in sorted(values)),
            encoding="utf-8",
        )
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _read_submission_ids(path: Path) -> list[int]:
    if not path.is_file():
        raise FileNotFoundError(f"Submission ID file does not exist: {path}")
    values = [
        int(line.strip()) for line in path.read_text().splitlines() if line.strip()
    ]
    if len(values) != len(set(values)):
        raise ValueError(f"Submission ID file contains duplicates: {path}")
    return values


def _event_paths(profile: Profile, simulation_id: int) -> tuple[Path, Path, Path]:
    opsim_stem = profile.opsim_db.stem
    simlib = (
        profile.work_dir
        / "SIMLIB"
        / (f"{opsim_stem}_{profile.sim_name}_{simulation_id}.SIMLIB")
    )
    input_file = (
        profile.work_dir
        / "SIM_INPUT"
        / (f"SIMGEN_{profile.sim_name}_{simulation_id}.INPUT")
    )
    plan = profile.observation_plan_dir / f"{simulation_id}.json"
    return simlib, input_file, plan


def _cleanup_transients(simlib: Path, input_file: Path) -> None:
    simlib.unlink(missing_ok=True)
    input_file.unlink(missing_ok=True)


def _snana_head(profile: Profile, simulation_id: int) -> Path:
    version = f"{profile.sim_name}_{simulation_id}"
    return profile.sndata_root / "SIM" / version / f"{version}_HEAD.FITS"


def _generate_documents(profile: Profile, simulation_ids: list[int]) -> None:
    arguments = [
        "--sim_name",
        profile.sim_name,
        "--GW_type",
        profile.source,
        "--skymap_path",
        str(profile.skymap_dir),
        "--sim_ids",
        *[str(value) for value in simulation_ids],
        "--GW_catalog",
        str(profile.prepared_catalog),
        "--Opsim",
        str(profile.opsim_db),
        "--level",
        str(profile.credible_level),
        "--outdir",
        str(profile.work_dir),
        "--template_input",
        str(profile.template_input),
        "--coordinate_mode",
        profile.coordinate_mode,
        "--samples_per_event",
        str(profile.samples_per_event),
        "--sampling_nside",
        str(profile.sampling_nside),
        "--cosmology",
        profile.cosmology,
        "--coordinate_manifest_dir",
        str(profile.coordinate_manifest_dir),
        "--observation_plan_dir",
        str(profile.observation_plan_dir),
        "--too_config",
        str(profile.too_config),
    ]
    snana.main(arguments)


def _run_one(profile: Profile, simulation_id: int) -> dict[str, Any]:
    simlib, input_file, plan_path = _event_paths(profile, simulation_id)
    result: dict[str, Any] = {
        "simulation_id": int(simulation_id),
        "status": "failed",
        "reason": None,
    }
    try:
        if not plan_path.is_file():
            raise FileNotFoundError(f"observation plan was not generated: {plan_path}")
        plan = json.loads(plan_path.read_text(encoding="utf-8"))
        if plan.get("status") != "generated":
            result["reason"] = (
                plan.get("error") or plan.get("reason") or "generation_failed"
            )
            return result
        if not simlib.is_file() or not input_file.is_file():
            raise FileNotFoundError("generated SIMLIB or SNANA input is missing")
        if snana.get_nlibid(simlib) == 0:
            result.update(status="skipped", reason="rubin_footprint_not_covered")
            return result

        environment = os.environ.copy()
        environment["SNDATA_ROOT"] = str(profile.sndata_root)
        environment["PATH"] = f"{profile.snana_bin_dir}:{environment.get('PATH', '')}"
        completed = subprocess.run(
            [str(profile.snana_bin_dir / "snlc_sim.exe"), str(input_file)],
            env=environment,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
        if completed.returncode != 0:
            result["reason"] = (
                completed.stderr.strip()[-2000:]
                or f"snlc_sim_exit_{completed.returncode}"
            )
            return result
        head = _snana_head(profile, simulation_id)
        if not head.is_file() or head.stat().st_size == 0:
            result["reason"] = f"SNANA completed but output is missing: {head}"
            return result
        result.update(status="success", reason=None, snana_head=str(head))
        return result
    except Exception as error:  # noqa: BLE001 - isolate failures by event
        result.update(reason=str(error), error_type=type(error).__name__)
        traceback.print_exc()
        return result
    finally:
        _cleanup_transients(simlib, input_file)


def run_array_task(
    profile: Profile,
    ids_file: Path,
    submission_id: str,
    task_index: int,
    batch_size: int,
) -> dict[str, Any]:
    """Generate documents and run SNANA for one disjoint array slice."""

    all_ids = _read_submission_ids(ids_file)
    start = int(task_index) * int(batch_size)
    selected = all_ids[start : start + int(batch_size)]
    if not selected:
        raise ValueError(f"Array task {task_index} selects no events")
    started = _utc_now()
    profile.work_dir.mkdir(parents=True, exist_ok=True)
    profile.observation_plan_dir.mkdir(parents=True, exist_ok=True)
    profile.coordinate_manifest_dir.mkdir(parents=True, exist_ok=True)
    _generate_documents(profile, selected)
    events = [_run_one(profile, simulation_id) for simulation_id in selected]
    document = {
        "profile": profile.name,
        "submission_id": submission_id,
        "task_index": int(task_index),
        "batch_size": int(batch_size),
        "started_utc": started,
        "created_utc": _utc_now(),
        "events": events,
    }
    shard_path = profile.status_dir / "shards" / f"{submission_id}_{task_index}.json"
    _atomic_json(shard_path, document)
    return document


def finalize_submission(
    profile: Profile,
    submission_id: str,
    expected_tasks: int,
) -> tuple[dict[str, Any], bool]:
    """Consolidate immutable shard sidecars into event-level status lists."""

    shard_dir = profile.status_dir / "shards"
    current_shards = list(shard_dir.glob(f"{submission_id}_*.json"))
    statuses = latest_event_statuses(profile.status_dir)
    all_ids = read_catalog_ids(profile.prepared_catalog)
    grouped = {"success": [], "failed": [], "skipped": [], "unprocessed": []}
    for simulation_id in all_ids:
        status = statuses.get(simulation_id, {}).get("status", "unprocessed")
        if status not in grouped:
            status = "failed"
        grouped[status].append(simulation_id)
    for status in ("success", "failed", "skipped"):
        _atomic_ids(profile.status_dir / f"{status}_sim_ids.txt", grouped[status])

    missing_tasks = max(0, int(expected_tasks) - len(current_shards))
    summary = {
        "profile": profile.name,
        "submission_id": submission_id,
        "created_utc": _utc_now(),
        "catalog_events": len(all_ids),
        "expected_tasks": int(expected_tasks),
        "completed_task_sidecars": len(current_shards),
        "missing_tasks": missing_tasks,
        "counts": {name: len(values) for name, values in grouped.items()},
    }
    _atomic_json(profile.status_dir / "summary.json", summary)
    complete = (
        not missing_tasks and not grouped["failed"] and not grouped["unprocessed"]
    )
    return summary, complete


def main() -> int:
    command = os.environ.get("KN_COMMAND", "work")
    profile = load_profile(os.environ["KN_PROFILE"])
    submission_id = os.environ["KN_SUBMISSION_ID"]
    if command == "work":
        task_index = int(os.environ["SLURM_ARRAY_TASK_ID"])
        run_array_task(
            profile,
            Path(os.environ["KN_IDS_FILE"]),
            submission_id,
            task_index,
            int(os.environ["KN_BATCH_SIZE"]),
        )
        return 0
    if command == "finalize":
        summary, complete = finalize_submission(
            profile,
            submission_id,
            int(os.environ["KN_EXPECTED_TASKS"]),
        )
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0 if complete else 1
    raise ValueError(f"Unknown KN_COMMAND: {command}")


if __name__ == "__main__":
    sys.exit(main())
