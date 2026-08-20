from pathlib import Path

import pytest
from config import DEFAULT_BASE_DIR, REPO_ROOT, load_profile


@pytest.mark.parametrize(
    "name,source,split,mode,samples,sim_name",
    [
        ("bns_train", "bns", "train", "posterior_3d", 1000, "LSST_KN_BNS_TRAIN"),
        ("bns_test", "bns", "test", "posterior_test", 64, "LSST_KN_BNS_TEST"),
        ("nsbh_train", "nsbh", "train", "posterior_3d", 1000, "LSST_KN_NSBH_TRAIN"),
        ("nsbh_test", "nsbh", "test", "posterior_test", 64, "LSST_KN_NSBH_TEST"),
    ],
)
def test_production_profiles(name, source, split, mode, samples, sim_name):
    profile = load_profile(name)
    assert profile.name == name
    assert profile.source == source
    assert profile.split == split
    assert profile.coordinate_mode == mode
    assert profile.samples_per_event == samples
    assert profile.sim_name == sim_name
    assert profile.seed == 1234
    assert profile.mjd_min == 61000.0
    assert profile.mjd_max == 64500.0
    assert profile.slurm.batch_size == 20
    assert profile.slurm.max_concurrency == 50
    assert profile.slurm.finalizer_time_limit == "01:00:00"
    assert profile.slurm.finalizer_memory == "4G"
    assert profile.run_dir == REPO_ROOT / "kn_simulation" / "runs" / name
    assert profile.skymap_dir == (
        DEFAULT_BASE_DIR / "data" / "skymap" / "positive" / f"{source}_skymap_{split}"
    )
    assert profile.negative_skymap_dir == (
        DEFAULT_BASE_DIR / "data" / "skymap" / "negative" / f"{source}_skymap_{split}"
    )
    assert profile.log_dir == DEFAULT_BASE_DIR / "logs" / "kn_simulation" / name
    assert isinstance(profile.template_input, Path)
