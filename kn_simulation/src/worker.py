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
from artifacts import compact_artifacts, write_task_shard
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


def _event_paths(profile: Profile, simulation_id: int) -> tuple[Path, Path]:
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
    return simlib, input_file


def _cleanup_transients(simlib: Path, input_file: Path) -> None:
    simlib.unlink(missing_ok=True)
    input_file.unlink(missing_ok=True)


def _snana_head(profile: Profile, simulation_id: int) -> Path:
    version = f"{profile.sim_name}_{simulation_id}"
    return profile.sndata_root / "SIM" / version / f"{version}_HEAD.FITS"


def _generate_documents(profile: Profile, simulation_ids: list[int]) -> dict[int, Any]:
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
        "--too_config",
        str(profile.too_config),
    ]
    return snana.main(arguments)


def _run_one(
    profile: Profile,
    simulation_id: int,
    plan: dict[str, Any],
) -> dict[str, Any]:
    simlib, input_file = _event_paths(profile, simulation_id)
    result: dict[str, Any] = {
        "simulation_id": int(simulation_id),
        "status": "failed",
        "reason": None,
    }
    try:
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
    profile.artifact_shard_dir.mkdir(parents=True, exist_ok=True)
    generated = _generate_documents(profile, selected)
    events = []
    artifact_events = []
    for simulation_id in selected:
        product = generated.get(simulation_id)
        if product is None:
            product = {
                "plan": {
                    "simulation_id": simulation_id,
                    "status": "failed",
                    "error": "SNANA document generator returned no event artifact",
                },
                "coordinates": None,
            }
        plan = product["plan"]
        result = _run_one(profile, simulation_id, plan)
        events.append(result)
        artifact_events.append(
            {
                **result,
                "observation_plan": plan,
                "coordinates": product.get("coordinates"),
            }
        )
    artifact_path = profile.artifact_shard_dir / f"{submission_id}_{int(task_index)}.h5"
    write_task_shard(
        artifact_path,
        profile=profile,
        submission_id=submission_id,
        task_index=task_index,
        events=artifact_events,
    )
    artifact_reference = str(artifact_path.relative_to(profile.run_dir))
    for event in events:
        event.update(
            artifact_shard=artifact_reference,
            submission_id=submission_id,
            task_index=int(task_index),
        )
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
    """Consolidate statuses and, for complete runs, merge artifact shards."""

    shard_dir = profile.status_dir / "shards"
    expected_indices = set(range(int(expected_tasks)))
    completed_indices: set[int] = set()
    invalid_sidecars: list[str] = []
    for path in shard_dir.glob(f"{submission_id}_*.json"):
        try:
            document = json.loads(path.read_text(encoding="utf-8"))
            task_index = int(document["task_index"])
            if document.get("submission_id") != submission_id:
                raise ValueError("submission_id does not match sidecar filename")
            expected_name = f"{submission_id}_{task_index}.json"
            if path.name != expected_name or task_index not in expected_indices:
                raise ValueError("task_index does not match sidecar filename")
            completed_indices.add(task_index)
        except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError):
            invalid_sidecars.append(path.name)
    missing_task_indices = sorted(expected_indices - completed_indices)
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

    missing_tasks = len(missing_task_indices)
    summary: dict[str, Any] = {
        "profile": profile.name,
        "submission_id": submission_id,
        "created_utc": _utc_now(),
        "catalog_events": len(all_ids),
        "expected_tasks": int(expected_tasks),
        "completed_task_sidecars": len(completed_indices),
        "missing_tasks": missing_tasks,
        "missing_task_indices": missing_task_indices,
        "invalid_task_sidecars": sorted(invalid_sidecars),
        "counts": {name: len(values) for name, values in grouped.items()},
    }
    complete = (
        not missing_tasks
        and not invalid_sidecars
        and not grouped["failed"]
        and not grouped["unprocessed"]
    )
    if complete:
        try:
            summary["aggregate"] = compact_artifacts(
                profile,
                catalog_ids=all_ids,
                event_statuses=statuses,
            )
        except Exception as error:  # noqa: BLE001 - preserve shards on merge failure
            traceback.print_exc()
            summary["aggregate"] = {
                "status": "failed",
                "error_type": type(error).__name__,
                "error": str(error),
            }
            complete = False
    else:
        summary["aggregate"] = {
            "status": "deferred",
            "reason": "run_has_missing_tasks_or_nonterminal_events",
        }
    summary["complete"] = complete
    _atomic_json(profile.status_dir / "summary.json", summary)
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
