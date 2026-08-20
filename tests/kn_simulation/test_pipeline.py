import json
import os
import shutil
import sqlite3
import subprocess
from pathlib import Path

import cli as kn_cli
import h5py
import numpy as np
import pandas as pd
import pytest
from artifacts import SCHEMA_VERSION, write_task_shard
from astropy.time import Time
from catalog import prepare_catalog, prepare_dual_run_catalog, prepare_run_catalog
from config import PIPELINE_ROOT, Profile, SlurmConfig
from scheduler import (
    compact_profile,
    latest_event_statuses,
    status_report,
    submit_profile,
    validate_prepared_run,
)
from worker import (
    _cleanup_snana_unneeded_outputs,
    _generate_documents,
    finalize_submission,
    run_array_task,
)


def make_opsim(path: Path) -> None:
    with sqlite3.connect(path) as connection:
        connection.execute("CREATE TABLE observations (observationStartMJD REAL)")
        connection.executemany(
            "INSERT INTO observations VALUES (?)",
            [(60981.0,), (64632.0,)],
        )


def make_catalog(path: Path, skymap_dir: Path) -> None:
    skymap_dir.mkdir(parents=True)
    for simulation_id in (7, 3):
        (skymap_dir / f"{simulation_id}.fits").touch()
    redshift = np.array([0.05, 0.1])
    mass1 = np.array([1.6, 1.5])
    mass2 = np.array([1.3, 1.2])
    dynamic = np.array([0.005, 0.02])
    wind = np.array([0.04, 0.01])
    frame = pd.DataFrame(
        {
            "simulation_id": [7, 3],
            "mass1_source": mass1,
            "mass2_source": mass2,
            "mass1_detector": mass1 * (1 + redshift),
            "mass2_detector": mass2 * (1 + redshift),
            "spin1z": [0.01, -0.02],
            "spin2z": [0.01, -0.01],
            "redshift": redshift,
            "luminosity_distance": [220.0, 460.0],
            "ra": [1.0, 2.0],
            "dec": [-0.5, 0.25],
            "theta_jn": [0.2, 2.0],
            "psi": [0.3, 0.4],
            "phase": [1.5, 2.5],
            "geocent_time": Time([62000.0, 63000.0], format="mjd").gps,
            "mej_dynamic": dynamic,
            "mej_wind": wind,
            "mej_total": dynamic + wind,
            "recovered_mass1_detector": mass1 * (1 + redshift) + 0.01,
            "recovered_mass2_detector": mass2 * (1 + redshift) - 0.01,
            "recovered_spin1z": [0.02, -0.01],
            "recovered_spin2z": [0.0, 0.0],
            "network_snr": [7.5, 14.0],
            "optimal_network_snr": [8.0, 13.5],
            "online_ifos": ["H1L1", "H1L1V1"],
            "skymap": ["skymaps/7.fits", "skymaps/3.fits"],
        }
    )
    frame.to_csv(path, index=False)


def make_profile(tmp_path: Path) -> tuple[Profile, Path]:
    run_dir = tmp_path / "run"
    skymap_dir = tmp_path / "skymaps"
    opsim_db = tmp_path / "opsim.db"
    make_opsim(opsim_db)
    template = tmp_path / "bns.input"
    template.write_text("template\n", encoding="utf-8")
    too_config = tmp_path / "rubin.yaml"
    too_config.write_text("version: test\n", encoding="utf-8")
    snana_bin = tmp_path / "snana" / "bin"
    snana_bin.mkdir(parents=True)
    executable = snana_bin / "snlc_sim.exe"
    executable.touch()
    (tmp_path / "sndata").mkdir()
    profile_path = tmp_path / "profile.yaml"
    profile_path.touch()
    profile = Profile(
        name="test_profile",
        source="bns",
        split="train",
        run_dir=run_dir,
        skymap_dir=skymap_dir,
        opsim_db=opsim_db,
        sndata_root=tmp_path / "sndata",
        snana_bin_dir=snana_bin,
        log_dir=tmp_path / "logs",
        sim_name="LSST_KN_TEST",
        template_input=template,
        too_config=too_config,
        coordinate_mode="posterior_3d",
        samples_per_event=8,
        sampling_nside=2,
        cosmology="Planck15",
        seed=42,
        mjd_min=61000.0,
        mjd_max=64500.0,
        credible_level=0.9,
        slurm=SlurmConfig(batch_size=20, max_concurrency=20),
    )
    return profile, profile_path


def prepare_profile(tmp_path: Path) -> tuple[Profile, Path]:
    profile, profile_path = make_profile(tmp_path)
    source = tmp_path / "source_catalog.csv"
    make_catalog(source, profile.skymap_dir)
    prepare_run_catalog(
        source,
        profile.run_dir,
        profile_name=profile.name,
        source=profile.source,
        split=profile.split,
        seed=profile.seed,
        skymap_dir=profile.skymap_dir,
        opsim_db=profile.opsim_db,
        profile=profile.as_manifest(),
    )
    return profile, profile_path


def test_worker_removes_only_unused_per_event_snana_outputs(tmp_path):
    profile, _ = make_profile(tmp_path)
    simulation_id = 7
    version = f"{profile.sim_name}_{simulation_id}"
    event_dir = profile.sndata_sim_dir / version
    event_dir.mkdir(parents=True)
    required = [
        event_dir / f"{version}_HEAD.FITS",
        event_dir / f"{version}_PHOT.FITS",
        event_dir / f"{version}.README",
    ]
    unused = [
        event_dir / f"{version}.DUMP",
        event_dir / f"{version}.LIST",
    ]
    global_list = profile.sndata_root / "SIM" / "PATH_SNDATA_SIM.LIST"
    global_list.parent.mkdir(parents=True, exist_ok=True)
    for path in [*required, *unused, global_list]:
        path.touch()

    _cleanup_snana_unneeded_outputs(profile, simulation_id)

    assert all(path.is_file() for path in required)
    assert not any(path.exists() for path in unused)
    assert global_list.is_file()


def test_prepare_keeps_low_snr_and_writes_audited_run(tmp_path):
    profile, _ = prepare_profile(tmp_path)
    prepared = pd.read_csv(profile.prepared_catalog)

    assert prepared["simulation_id"].tolist() == [3, 7]
    assert prepared["network_snr"].tolist() == [14.0, 7.5]
    assert profile.input_catalog.is_file()
    assert profile.input_metadata.is_file()
    manifest = validate_prepared_run(profile)
    assert manifest["input_rows"] == manifest["output_rows"] == 2
    assert manifest["model_range_filtered_rows"] == 0
    assert "snana_mej_dynamic" not in prepared
    assert "mej_dynamic_clipped" not in prepared


def test_dual_prepare_keeps_type1_separate_from_snana_catalog(tmp_path):
    profile, _ = make_profile(tmp_path)
    positive_path = tmp_path / "positive.csv"
    make_catalog(positive_path, profile.skymap_dir)
    positive = pd.read_csv(positive_path)
    positive["sample_class"] = "pos"
    positive["event_uid"] = [
        f"bns_train_pos_{simulation_id}" for simulation_id in positive["simulation_id"]
    ]
    positive.to_csv(positive_path, index=False)

    negative_maps = tmp_path / "negative_skymaps"
    negative_maps.mkdir()
    negative_path = tmp_path / "negative.csv"
    negative = positive.iloc[[0]].copy()
    # Deliberately collide with a positive simulation_id; event_uid is the key.
    negative["mej_dynamic"] = 0.0
    negative["mej_wind"] = 0.0
    negative["mej_total"] = 0.0
    negative["neg_type"] = 1
    negative["sample_class"] = "neg"
    negative["event_uid"] = "bns_train_neg_7"
    negative["skymap"] = "skymaps/7.fits"
    negative.to_csv(negative_path, index=False)
    (negative_maps / "7.fits").touch()

    result = prepare_dual_run_catalog(
        positive_path,
        negative_path,
        profile.run_dir,
        profile_name=profile.name,
        source=profile.source,
        split=profile.split,
        seed=profile.seed,
        positive_skymap_dir=profile.skymap_dir,
        negative_skymap_dir=negative_maps,
        opsim_db=profile.opsim_db,
        profile=profile.as_manifest(),
    )

    prepared = pd.read_csv(profile.prepared_catalog)
    retained_negative = pd.read_csv(profile.run_dir / "neg_catalog.csv")
    assert set(prepared["sample_class"]) == {"pos"}
    assert retained_negative["event_uid"].tolist() == ["bns_train_neg_7"]
    assert result["negative"]["rows"] == 1
    assert result["snana_contains_sample_class"] == ["pos"]


def test_validate_accepts_manifest_before_finalizer_resource_fields(tmp_path):
    profile, _ = prepare_profile(tmp_path)
    manifest = json.loads(profile.prepared_manifest.read_text(encoding="utf-8"))
    manifest["profile"]["slurm"].pop("finalizer_time_limit")
    manifest["profile"]["slurm"].pop("finalizer_memory")
    profile.prepared_manifest.write_text(
        json.dumps(manifest),
        encoding="utf-8",
    )

    assert validate_prepared_run(profile)["profile_name"] == profile.name


def test_prepare_filters_ejecta_outliers_and_keeps_closed_boundaries(tmp_path):
    profile, _ = make_profile(tmp_path)
    source_path = tmp_path / "source_catalog.csv"
    make_catalog(source_path, profile.skymap_dir)
    base = pd.read_csv(source_path).iloc[[0, 0, 0, 0]].copy()
    base["simulation_id"] = [10, 11, 12, 13]
    base["skymap"] = [
        f"skymaps/{simulation_id}.fits" for simulation_id in base["simulation_id"]
    ]
    for simulation_id in base["simulation_id"]:
        (profile.skymap_dir / f"{simulation_id}.fits").touch()
    base["mej_dynamic"] = [0.001, 0.02, 0.0, 0.01]
    base["mej_wind"] = [0.01, 0.13, 0.04, 0.131]
    base["mej_total"] = base["mej_dynamic"] + base["mej_wind"]

    prepared, manifest = prepare_catalog(
        base,
        source="bns",
        split="train",
        seed=42,
        skymap_dir=profile.skymap_dir,
        opsim_db=profile.opsim_db,
    )

    assert prepared["simulation_id"].tolist() == [10, 11]
    assert prepared["mej_dynamic"].tolist() == [0.001, 0.02]
    assert prepared["mej_wind"].tolist() == [0.01, 0.13]
    assert manifest["input_rows"] == 4
    assert manifest["output_rows"] == 2
    assert manifest["model_range_filtered_rows"] == 2
    assert manifest["mej_dynamic_out_of_range_rows"] == 1
    assert manifest["mej_wind_out_of_range_rows"] == 1
    assert manifest["optical_model_parameter_ranges"]["mej_dynamic"] == [
        0.001,
        0.02,
    ]
    assert prepared["viewing_costheta"].between(0.0, 1.0, inclusive="both").all()
    assert prepared["phi_deg"].between(15.0, 75.0, inclusive="both").all()
    assert not {
        "snana_mej_dynamic",
        "snana_mej_wind",
        "mej_dynamic_clipped",
        "mej_wind_clipped",
    } & set(prepared.columns)


def test_prepare_excludes_double_zero_type1_but_not_single_component_from_physical_classification(
    tmp_path,
):
    profile, _ = make_profile(tmp_path)
    source_path = tmp_path / "source_catalog.csv"
    make_catalog(source_path, profile.skymap_dir)
    base = pd.read_csv(source_path).iloc[[0, 0, 0, 0]].copy()
    base["simulation_id"] = [30, 31, 32, 33]
    base["skymap"] = [f"skymaps/{event_id}.fits" for event_id in base["simulation_id"]]
    for event_id in base["simulation_id"]:
        (profile.skymap_dir / f"{event_id}.fits").touch()
    base["mej_dynamic"] = [0.0, 0.0, 0.005, 0.005]
    base["mej_wind"] = [0.0, 0.04, 0.0, 0.04]
    base["mej_total"] = base["mej_dynamic"] + base["mej_wind"]
    base["neg_type"] = [1, 0, 0, 0]

    prepared, manifest = prepare_catalog(
        base,
        source="bns",
        split="train",
        seed=42,
        skymap_dir=profile.skymap_dir,
        opsim_db=profile.opsim_db,
    )

    assert prepared["simulation_id"].tolist() == [33]
    assert manifest["input_type1_rows"] == 1
    assert manifest["excluded_type1_no_ejecta_rows"] == 1
    assert manifest["snana_input_rows"] == 1


def test_prepare_rejects_neg_type_inconsistent_with_double_zero(tmp_path):
    profile, _ = make_profile(tmp_path)
    source_path = tmp_path / "source_catalog.csv"
    make_catalog(source_path, profile.skymap_dir)
    catalog = pd.read_csv(source_path)
    catalog["neg_type"] = [1, 0]

    with pytest.raises(ValueError, match="inconsistent with physical double-zero"):
        prepare_catalog(
            catalog,
            source="bns",
            split="train",
            seed=42,
            skymap_dir=profile.skymap_dir,
            opsim_db=profile.opsim_db,
        )


def test_prepare_nsbh_filter_keeps_closed_boundaries_and_fixed_phi(tmp_path):
    profile, _ = make_profile(tmp_path)
    source_path = tmp_path / "source_catalog.csv"
    make_catalog(source_path, profile.skymap_dir)
    base = pd.read_csv(source_path).iloc[[0, 0, 0, 0]].copy()
    base["simulation_id"] = [20, 21, 22, 23]
    base["skymap"] = [
        f"skymaps/{simulation_id}.fits" for simulation_id in base["simulation_id"]
    ]
    for simulation_id in base["simulation_id"]:
        (profile.skymap_dir / f"{simulation_id}.fits").touch()
    base["mass1_source"] = 5.0
    base["mass2_source"] = 1.4
    base["mass1_detector"] = base["mass1_source"] * (1 + base["redshift"])
    base["mass2_detector"] = base["mass2_source"] * (1 + base["redshift"])
    base["mej_dynamic"] = [0.01, 0.09, 0.009, 0.05]
    base["mej_wind"] = [0.01, 0.09, 0.05, 0.091]
    base["mej_total"] = base["mej_dynamic"] + base["mej_wind"]

    prepared, manifest = prepare_catalog(
        base,
        source="nsbh",
        split="test",
        seed=42,
        skymap_dir=profile.skymap_dir,
        opsim_db=profile.opsim_db,
    )

    assert prepared["simulation_id"].tolist() == [20, 21]
    assert prepared["mej_dynamic"].tolist() == [0.01, 0.09]
    assert prepared["mej_wind"].tolist() == [0.01, 0.09]
    assert prepared["phi_deg"].tolist() == [30.0, 30.0]
    assert prepared["viewing_costheta"].between(0.0, 1.0, inclusive="both").all()
    assert manifest["model_range_filtered_rows"] == 2
    assert manifest["mej_dynamic_out_of_range_rows"] == 1
    assert manifest["mej_wind_out_of_range_rows"] == 1
    assert manifest["optical_model_parameter_ranges"]["phi_deg"] == [30.0, 30.0]


def test_validate_rejects_obsolete_clipping_schema(tmp_path):
    profile, _ = prepare_profile(tmp_path)
    manifest = json.loads(profile.prepared_manifest.read_text(encoding="utf-8"))
    manifest["schema_version"] = "gwsamplegen-kn-v1"
    profile.prepared_manifest.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="obsolete schema"):
        validate_prepared_run(profile)


def test_submit_dry_run_builds_array_without_writing_submission(tmp_path):
    profile, profile_path = prepare_profile(tmp_path)
    result = submit_profile(
        profile,
        profile_path,
        dry_run=True,
        batch_size=1,
        max_concurrency=3,
    )

    assert result["event_count"] == 2
    assert result["task_count"] == 2
    assert "--array=0-1%3" in result["array_command"]
    export_argument = next(
        argument
        for argument in result["array_command"]
        if argument.startswith("--export=")
    )
    assert "KN_PIPELINE_ROOT=" in export_argument
    assert not profile.submission_file.exists()


def test_copied_slurm_launcher_uses_exported_pipeline_root(tmp_path):
    pipeline_root = tmp_path / "pipeline"
    worker = pipeline_root / "src" / "worker.py"
    worker.parent.mkdir(parents=True)
    worker.write_text("# test worker\n", encoding="utf-8")

    spool = tmp_path / "slurm-spool"
    spool.mkdir()
    copied_launcher = spool / "worker.sh"
    shutil.copy2(PIPELINE_ROOT / "slurm" / "worker.sh", copied_launcher)

    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    capture = tmp_path / "python-argument.txt"
    fake_python = fake_bin / "python"
    fake_python.write_text('#!/bin/sh\nprintf "%s\\n" "$1" > "$CAPTURE_PATH"\n')
    fake_python.chmod(0o755)
    environment = os.environ.copy()
    environment["PATH"] = f"{fake_bin}:{environment['PATH']}"
    environment["CAPTURE_PATH"] = str(capture)
    environment["KN_PIPELINE_ROOT"] = str(pipeline_root)

    subprocess.run([copied_launcher], check=True, env=environment)
    assert capture.read_text(encoding="utf-8").strip() == str(worker)


def test_finalize_and_resume_use_event_sidecars(tmp_path):
    profile, profile_path = prepare_profile(tmp_path)
    profile.submission_file.write_text("{}\n", encoding="utf-8")
    shard_dir = profile.status_dir / "shards"
    shard_dir.mkdir(parents=True)
    shard = {
        "created_utc": "2026-07-31T00:00:00+00:00",
        "submission_id": "first",
        "task_index": 0,
        "events": [
            {"simulation_id": 3, "status": "success", "reason": None},
            {"simulation_id": 7, "status": "failed", "reason": "test"},
        ],
    }
    (shard_dir / "first_0.json").write_text(json.dumps(shard), encoding="utf-8")

    summary, complete = finalize_submission(profile, "first", expected_tasks=1)
    assert not complete
    assert summary["counts"] == {
        "success": 1,
        "failed": 1,
        "skipped": 0,
        "unprocessed": 0,
    }
    result = submit_profile(profile, profile_path, dry_run=True, resume=True)
    assert result["event_count"] == 1
    assert status_report(profile)["counts"]["failed"] == 1


def coordinate_frame(simulation_id: int, count: int = 8) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "simulation_id": np.full(count, simulation_id),
            "sample_index": np.arange(count),
            "ra": np.linspace(1.0, 2.0, count),
            "dec": np.linspace(-0.5, 0.5, count),
            "distance_mpc": np.linspace(100.0, 200.0, count),
            "posterior_probability": np.full(count, 1.0 / count),
            "is_true_position": np.arange(count) == 0,
            "redshift": np.r_[np.nan, np.linspace(0.02, 0.08, count - 1)],
            "libid": np.arange(count),
            "in_baseline_footprint": np.arange(count) % 2 == 0,
            "too_tile_index": np.arange(count) - 1,
            "too_nobs": np.arange(count) % 3,
            "too_mode": np.full(count, "baseline_plus_too"),
        }
    )


def generated_products(simulation_ids: list[int]) -> dict[int, dict]:
    return {
        simulation_id: {
            "plan": {
                "simulation_id": simulation_id,
                "status": "generated",
                "nested": {"nights": [0, 1], "enabled": True},
            },
            "coordinates": coordinate_frame(simulation_id),
        }
        for simulation_id in simulation_ids
    }


def test_worker_keeps_coordinates_in_memory_without_per_event_csv(
    tmp_path, monkeypatch
):
    profile, _ = prepare_profile(tmp_path)
    captured = {}

    def fake_snana_main(arguments):
        captured["arguments"] = arguments
        return generated_products([7])

    monkeypatch.setattr("worker.snana.main", fake_snana_main)

    products = _generate_documents(profile, [7])

    assert products[7]["coordinates"].shape == (8, 13)
    assert "--no-coordinate-files" in captured["arguments"]


def test_array_tasks_write_hdf5_shards_then_finalizer_sorts_and_compacts(
    tmp_path, monkeypatch
):
    profile, _ = prepare_profile(tmp_path)
    ids_file = profile.status_dir / "submissions" / "task.ids"
    ids_file.parent.mkdir(parents=True)
    ids_file.write_text("7\n3\n", encoding="utf-8")
    monkeypatch.setattr(
        "worker._generate_documents",
        lambda profile, simulation_ids: generated_products(simulation_ids),
    )
    monkeypatch.setattr(
        "worker._run_one",
        lambda profile, simulation_id, plan: {
            "simulation_id": simulation_id,
            "status": "success",
            "reason": None,
        },
    )

    documents = [
        run_array_task(profile, ids_file, "current", task_index, 1)
        for task_index in (0, 1)
    ]
    artifact_paths = [
        profile.artifact_shard_dir / f"current_{task_index}.h5" for task_index in (0, 1)
    ]

    assert [document["events"][0]["simulation_id"] for document in documents] == [7, 3]
    assert [document["events"][0]["artifact_shard"] for document in documents] == [
        "artifact_shards/current_0.h5",
        "artifact_shards/current_1.h5",
    ]
    assert all(path.is_file() for path in artifact_paths)
    assert not profile.coordinate_manifest_dir.exists()
    assert not profile.observation_plan_dir.exists()
    with h5py.File(artifact_paths[0], "r") as handle:
        assert handle.attrs["schema_version"] == SCHEMA_VERSION
        assert handle["events/simulation_id"][:].tolist() == [7]
        assert len(handle["coordinates/ra"]) == 8
        assert handle["coordinates/ra"].compression == "gzip"
        assert np.isnan(handle["coordinates/redshift"][0])
        plan = json.loads(handle["events/observation_plan_json"][0])
        assert plan["nested"]["nights"] == [0, 1]

    summary, complete = finalize_submission(profile, "current", expected_tasks=2)

    assert complete
    assert summary["complete"]
    assert summary["aggregate"]["event_count"] == 2
    assert profile.aggregate_artifact_file.is_file()
    assert not any(path.exists() for path in artifact_paths)
    assert (profile.status_dir / "shards" / "current_0.json").is_file()
    assert (profile.status_dir / "shards" / "current_1.json").is_file()
    with h5py.File(profile.aggregate_artifact_file, "r") as handle:
        assert handle["events/simulation_id"][:].tolist() == [3, 7]
        assert handle["events/coordinate_start"][:].tolist() == [0, 8]
        assert handle["events/coordinate_count"][:].tolist() == [8, 8]
        assert handle["coordinates/simulation_id"][:].tolist() == [3] * 8 + [7] * 8
    artifact_status = status_report(profile)["intermediate_artifacts"]
    assert artifact_status["exists"]
    assert artifact_status["pending_artifact_shards"] == 0


def test_failed_task_artifact_supports_zero_coordinates(tmp_path):
    profile, _ = prepare_profile(tmp_path)
    path = profile.artifact_shard_dir / "failed_0.h5"

    write_task_shard(
        path,
        profile=profile,
        submission_id="failed",
        task_index=0,
        events=[
            {
                "simulation_id": 3,
                "status": "failed",
                "reason": "generation failed",
                "observation_plan": {
                    "simulation_id": 3,
                    "status": "failed",
                },
                "coordinates": None,
            }
        ],
    )

    with h5py.File(path, "r") as handle:
        assert handle.attrs["coordinate_count"] == 0
        assert handle["coordinates/ra"].shape == (0,)
        assert handle["events/coordinate_count"][:].tolist() == [0]


def test_finalizer_imports_legacy_event_and_cleans_sources(tmp_path):
    profile, _ = prepare_profile(tmp_path)
    profile.coordinate_manifest_dir.mkdir(parents=True)
    profile.observation_plan_dir.mkdir(parents=True)
    coordinate_frame(7).to_csv(
        profile.coordinate_manifest_dir / "7.csv",
        index=False,
    )
    (profile.observation_plan_dir / "7.json").write_text(
        json.dumps(
            {
                "simulation_id": 7,
                "status": "generated",
                "legacy": {"preserved": True},
            }
        ),
        encoding="utf-8",
    )
    status_shards = profile.status_dir / "shards"
    status_shards.mkdir(parents=True)
    (status_shards / "legacy_0.json").write_text(
        json.dumps(
            {
                "created_utc": "2026-08-04T00:00:00+00:00",
                "events": [{"simulation_id": 7, "status": "success", "reason": None}],
            }
        ),
        encoding="utf-8",
    )

    artifact_path = profile.artifact_shard_dir / "current_0.h5"
    write_task_shard(
        artifact_path,
        profile=profile,
        submission_id="current",
        task_index=0,
        events=[
            {
                "simulation_id": 3,
                "status": "success",
                "reason": None,
                "observation_plan": {
                    "simulation_id": 3,
                    "status": "generated",
                },
                "coordinates": coordinate_frame(3),
            }
        ],
    )
    (status_shards / "current_0.json").write_text(
        json.dumps(
            {
                "created_utc": "2026-08-04T01:00:00+00:00",
                "submission_id": "current",
                "task_index": 0,
                "events": [
                    {
                        "simulation_id": 3,
                        "status": "success",
                        "reason": None,
                        "artifact_shard": "artifact_shards/current_0.h5",
                        "submission_id": "current",
                        "task_index": 0,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    summary, complete = finalize_submission(profile, "current", expected_tasks=1)

    assert complete
    assert summary["aggregate"]["cleanup"] == {
        "artifact_shards": 1,
        "coordinate_csv": 1,
        "observation_plans": 1,
    }
    assert not profile.coordinate_manifest_dir.exists()
    assert not profile.observation_plan_dir.exists()
    with h5py.File(profile.aggregate_artifact_file, "r") as handle:
        assert handle["events/simulation_id"][:].tolist() == [3, 7]
        submissions = [
            value.decode() if isinstance(value, bytes) else value
            for value in handle["events/submission_id"][:]
        ]
        assert submissions == ["current", "legacy"]
        legacy_plan = json.loads(handle["events/observation_plan_json"][1])
        assert legacy_plan["legacy"]["preserved"]


def test_corrupt_shard_keeps_all_source_artifacts(tmp_path):
    profile, _ = prepare_profile(tmp_path)
    artifact_path = profile.artifact_shard_dir / "current_0.h5"
    write_task_shard(
        artifact_path,
        profile=profile,
        submission_id="current",
        task_index=0,
        events=[
            {
                "simulation_id": simulation_id,
                "status": "success",
                "reason": None,
                "observation_plan": {
                    "simulation_id": simulation_id,
                    "status": "generated",
                },
                "coordinates": coordinate_frame(simulation_id),
            }
            for simulation_id in (3, 7)
        ],
    )
    with h5py.File(artifact_path, "r+") as handle:
        handle.attrs["prepared_catalog_sha256"] = "corrupt"
    status_shards = profile.status_dir / "shards"
    status_shards.mkdir(parents=True)
    (status_shards / "current_0.json").write_text(
        json.dumps(
            {
                "created_utc": "2026-08-04T01:00:00+00:00",
                "submission_id": "current",
                "task_index": 0,
                "events": [
                    {
                        "simulation_id": simulation_id,
                        "status": "success",
                        "reason": None,
                        "artifact_shard": "artifact_shards/current_0.h5",
                    }
                    for simulation_id in (3, 7)
                ],
            }
        ),
        encoding="utf-8",
    )

    summary, complete = finalize_submission(profile, "current", expected_tasks=1)

    assert not complete
    assert summary["aggregate"]["status"] == "failed"
    assert "checksum mismatch" in summary["aggregate"]["error"]
    assert artifact_path.is_file()
    assert not profile.aggregate_artifact_file.exists()


def test_latest_retry_status_selects_new_artifact_reference(tmp_path):
    profile, _ = prepare_profile(tmp_path)
    shard_dir = profile.status_dir / "shards"
    shard_dir.mkdir(parents=True)
    for name, created, artifact in (
        ("old_0.json", "2026-08-04T00:00:00+00:00", "artifact_shards/old.h5"),
        ("new_0.json", "2026-08-04T01:00:00+00:00", "artifact_shards/new.h5"),
    ):
        (shard_dir / name).write_text(
            json.dumps(
                {
                    "created_utc": created,
                    "events": [
                        {
                            "simulation_id": 3,
                            "status": "success",
                            "artifact_shard": artifact,
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )

    assert latest_event_statuses(profile.status_dir)[3]["artifact_shard"].endswith(
        "new.h5"
    )


def test_compact_cli_routes_to_manual_compactor(monkeypatch, capsys):
    sentinel = object()
    seen = []
    monkeypatch.setattr(kn_cli, "load_profile", lambda reference: sentinel)
    monkeypatch.setattr(
        kn_cli,
        "compact_profile",
        lambda profile: seen.append(profile) or {"event_count": 2},
    )
    result = kn_cli.main(["compact", "bns_test"])

    assert result == 0
    assert seen == [sentinel]
    assert json.loads(capsys.readouterr().out) == {"event_count": 2}


def test_manual_compact_refuses_incomplete_run(tmp_path):
    profile, _ = prepare_profile(tmp_path)

    with pytest.raises(ValueError, match="incomplete run|unprocessed"):
        compact_profile(profile)

    assert not profile.aggregate_artifact_file.exists()
