from pathlib import Path

import pytest
from config import REPO_ROOT, load_profile


@pytest.mark.parametrize(
    "name,source,split,mode,samples",
    [
        ("bns_train", "bns", "train", "posterior_3d", 1000),
        ("bns_test", "bns", "test", "posterior_test", 64),
        ("nsbh_train", "nsbh", "train", "posterior_3d", 1000),
        ("nsbh_test", "nsbh", "test", "posterior_test", 64),
    ],
)
def test_production_profiles(name, source, split, mode, samples):
    profile = load_profile(name)
    assert profile.name == name
    assert profile.source == source
    assert profile.split == split
    assert profile.coordinate_mode == mode
    assert profile.samples_per_event == samples
    assert profile.seed == 42
    assert profile.mjd_min == 61000.0
    assert profile.mjd_max == 64500.0
    assert profile.slurm.batch_size == 20
    assert profile.slurm.max_concurrency == 50
    assert profile.slurm.finalizer_time_limit == "01:00:00"
    assert profile.slurm.finalizer_memory == "4G"
    assert profile.run_dir == REPO_ROOT / "kn_simulation" / "runs" / name
    assert isinstance(profile.template_input, Path)
