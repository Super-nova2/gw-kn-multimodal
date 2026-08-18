from pathlib import Path

import pytest
from config import DEFAULT_BASE_DIR, REPO_ROOT, load_profile


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
    assert profile.run_dir == REPO_ROOT / "kn_simulation" / "runs_dual" / name
    assert profile.skymap_dir == (
        DEFAULT_BASE_DIR / "data" / "skymap" / "positive" / f"{source}_skymap_{split}"
    )
    assert profile.negative_skymap_dir == (
        DEFAULT_BASE_DIR / "data" / "skymap" / "negative" / f"{source}_skymap_{split}"
    )
    assert isinstance(profile.template_input, Path)


@pytest.mark.parametrize(
    "name,split", [("bns_train_am", "train"), ("bns_test_am", "test")]
)
def test_production_am_profiles_are_positive_only_variants(name, split):
    profile = load_profile(name)
    assert profile.name == name
    assert profile.seed == 1234
    assert profile.negative_skymap_dir is None
    assert profile.skymap_dir == (
        DEFAULT_BASE_DIR
        / "GWSamplegen"
        / "outputs"
        / "production_am_bayestar"
        / f"bns_{split}_seed_1234"
        / "skymaps"
    )
