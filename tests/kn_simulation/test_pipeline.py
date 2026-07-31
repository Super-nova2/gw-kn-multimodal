import json
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
from astropy.time import Time
from catalog import prepare_run_catalog
from config import Profile, SlurmConfig
from scheduler import status_report, submit_profile, validate_prepared_run
from worker import finalize_submission


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
    dynamic = np.array([0.005, 0.03])
    wind = np.array([0.04, 0.005])
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


def test_prepare_keeps_low_snr_and_writes_audited_run(tmp_path):
    profile, _ = prepare_profile(tmp_path)
    prepared = pd.read_csv(profile.prepared_catalog)

    assert prepared["simulation_id"].tolist() == [3, 7]
    assert prepared["network_snr"].tolist() == [14.0, 7.5]
    assert profile.input_catalog.is_file()
    assert profile.input_metadata.is_file()
    manifest = validate_prepared_run(profile)
    assert manifest["input_rows"] == manifest["output_rows"] == 2
    assert manifest["mej_dynamic_clipped"] == 1
    assert manifest["mej_wind_clipped"] == 1


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
    assert not profile.submission_file.exists()


def test_finalize_and_resume_use_event_sidecars(tmp_path):
    profile, profile_path = prepare_profile(tmp_path)
    profile.submission_file.write_text("{}\n", encoding="utf-8")
    shard_dir = profile.status_dir / "shards"
    shard_dir.mkdir(parents=True)
    shard = {
        "created_utc": "2026-07-31T00:00:00+00:00",
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
