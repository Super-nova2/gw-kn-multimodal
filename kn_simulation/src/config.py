"""Configuration loading for the production kilonova simulation pipeline."""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
PIPELINE_ROOT = REPO_ROOT / "kn_simulation"
PROFILE_ROOT = PIPELINE_ROOT / "profiles"
DEFAULT_BASE_DIR = Path("/fred/oz016/bgao_kn")


@dataclass(frozen=True)
class SlurmConfig:
    batch_size: int = 20
    max_concurrency: int = 20
    time_limit: str = "04:00:00"
    cpus_per_task: int = 1
    memory: str = "10G"
    finalizer_time_limit: str = "01:00:00"
    finalizer_memory: str = "4G"


@dataclass(frozen=True)
class Profile:
    name: str
    source: str
    split: str
    run_dir: Path
    skymap_dir: Path
    opsim_db: Path
    sndata_root: Path
    snana_bin_dir: Path
    log_dir: Path
    sim_name: str
    template_input: Path
    too_config: Path
    coordinate_mode: str
    samples_per_event: int
    sampling_nside: int
    cosmology: str
    seed: int
    mjd_min: float
    mjd_max: float
    credible_level: float
    slurm: SlurmConfig
    negative_skymap_dir: Path | None = None

    @property
    def input_catalog(self) -> Path:
        return self.run_dir / "catalog.csv"

    @property
    def input_metadata(self) -> Path:
        return self.run_dir / "catalog.input.json"

    @property
    def prepared_catalog(self) -> Path:
        return self.run_dir / "kn_catalog.csv"

    @property
    def prepared_manifest(self) -> Path:
        return self.run_dir / "kn_catalog.manifest.json"

    @property
    def sndata_sim_dir(self) -> Path:
        """Parent directory for this profile's per-event SNANA outputs."""
        return self.sndata_root / "SIM" / self.sim_name

    @property
    def work_dir(self) -> Path:
        return self.run_dir / "work"

    @property
    def observation_plan_dir(self) -> Path:
        return self.run_dir / "observation_plans"

    @property
    def coordinate_manifest_dir(self) -> Path:
        """Legacy per-event coordinate directory used only during migration."""
        return self.run_dir / "coordinate_samples"

    @property
    def artifact_shard_dir(self) -> Path:
        return self.run_dir / "artifact_shards"

    @property
    def aggregate_artifact_file(self) -> Path:
        return self.run_dir / "simulation_intermediates.h5"

    @property
    def status_dir(self) -> Path:
        return self.run_dir / "status"

    @property
    def submission_file(self) -> Path:
        return self.run_dir / "submission.json"

    def as_manifest(self) -> dict[str, Any]:
        result = asdict(self)
        for key in (
            "run_dir",
            "skymap_dir",
            "negative_skymap_dir",
            "opsim_db",
            "sndata_root",
            "snana_bin_dir",
            "log_dir",
            "template_input",
            "too_config",
        ):
            if result[key] is not None:
                result[key] = str(result[key])
        return result


def _expand_path(value: str, variables: dict[str, str]) -> Path:
    expanded = str(value)
    for name, replacement in variables.items():
        expanded = expanded.replace(f"${{{name}}}", replacement)
    if "${" in expanded:
        raise ValueError(f"Unresolved variable in configured path: {value}")
    return Path(expanded).expanduser().resolve()


def _require(mapping: dict[str, Any], name: str) -> Any:
    if name not in mapping:
        raise ValueError(f"Profile is missing required key: {name}")
    return mapping[name]


def resolve_profile_path(profile: str | Path) -> Path:
    candidate = Path(profile)
    if candidate.is_file():
        return candidate.resolve()
    name = candidate.name
    if not name.endswith((".yaml", ".yml")):
        name = f"{name}.yaml"
    resolved = PROFILE_ROOT / name
    if not resolved.is_file():
        raise FileNotFoundError(f"Unknown profile: {profile} ({resolved})")
    return resolved.resolve()


def load_profile(profile: str | Path) -> Profile:
    """Load and validate one source/split production profile."""

    path = resolve_profile_path(profile)
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise TypeError(f"Profile must contain a YAML mapping: {path}")

    base_dir = Path(os.environ.get("BASE_DIR", DEFAULT_BASE_DIR)).resolve()
    variables = {
        "BASE_DIR": str(base_dir),
        "REPO_ROOT": str(REPO_ROOT),
    }
    paths = _require(raw, "paths")
    snana = _require(raw, "snana")
    catalog = _require(raw, "catalog")
    slurm_raw = raw.get("slurm", {})
    if not all(isinstance(item, dict) for item in (paths, snana, catalog, slurm_raw)):
        raise ValueError("paths, catalog, snana, and slurm must be YAML mappings")

    source = str(_require(raw, "source")).lower()
    split = str(_require(raw, "split")).lower()
    if source not in {"bns", "nsbh"}:
        raise ValueError("source must be bns or nsbh")
    if split not in {"train", "test"}:
        raise ValueError("split must be train or test")

    slurm = SlurmConfig(
        batch_size=int(slurm_raw.get("batch_size", 20)),
        max_concurrency=int(slurm_raw.get("max_concurrency", 20)),
        time_limit=str(slurm_raw.get("time_limit", "04:00:00")),
        cpus_per_task=int(slurm_raw.get("cpus_per_task", 1)),
        memory=str(slurm_raw.get("memory", "10G")),
        finalizer_time_limit=str(slurm_raw.get("finalizer_time_limit", "01:00:00")),
        finalizer_memory=str(slurm_raw.get("finalizer_memory", "4G")),
    )
    if slurm.batch_size <= 0 or slurm.max_concurrency <= 0:
        raise ValueError("Slurm batch_size and max_concurrency must be positive")
    if slurm.cpus_per_task <= 0:
        raise ValueError("Slurm cpus_per_task must be positive")

    result = Profile(
        name=str(_require(raw, "name")),
        source=source,
        split=split,
        run_dir=_expand_path(_require(paths, "run_dir"), variables),
        skymap_dir=_expand_path(_require(paths, "skymap_dir"), variables),
        negative_skymap_dir=(
            _expand_path(paths["negative_skymap_dir"], variables)
            if paths.get("negative_skymap_dir")
            else None
        ),
        opsim_db=_expand_path(_require(paths, "opsim_db"), variables),
        sndata_root=_expand_path(_require(paths, "sndata_root"), variables),
        snana_bin_dir=_expand_path(_require(paths, "snana_bin_dir"), variables),
        log_dir=_expand_path(_require(paths, "log_dir"), variables),
        sim_name=str(_require(snana, "sim_name")),
        template_input=_expand_path(_require(snana, "template_input"), variables),
        too_config=_expand_path(_require(snana, "too_config"), variables),
        coordinate_mode=str(_require(snana, "coordinate_mode")),
        samples_per_event=int(_require(snana, "samples_per_event")),
        sampling_nside=int(snana.get("sampling_nside", 256)),
        cosmology=str(snana.get("cosmology", "Planck15")),
        seed=int(catalog.get("seed", 42)),
        mjd_min=float(catalog.get("mjd_min", 61_000.0)),
        mjd_max=float(catalog.get("mjd_max", 64_500.0)),
        credible_level=float(snana.get("credible_level", 0.9)),
        slurm=slurm,
    )
    if result.coordinate_mode not in {"posterior_3d", "posterior_test"}:
        raise ValueError(
            "Production coordinate_mode must be posterior_3d or posterior_test"
        )
    if result.samples_per_event <= 0 or result.sampling_nside <= 0:
        raise ValueError("SNANA samples_per_event and sampling_nside must be positive")
    if not 0 < result.credible_level <= 1:
        raise ValueError("credible_level must be in (0, 1]")
    if result.mjd_min >= result.mjd_max:
        raise ValueError("catalog MJD bounds must be increasing")
    if result.name != path.stem:
        raise ValueError(
            f"Profile name {result.name!r} must match filename {path.stem!r}"
        )
    expected_name = f"{source}_{split}"
    if result.name != expected_name:
        raise ValueError(
            f"Profile name {result.name!r} must be {expected_name!r} for its source/split"
        )
    expected_mode = "posterior_3d" if split == "train" else "posterior_test"
    if result.coordinate_mode != expected_mode:
        raise ValueError(f"{split} profiles must use coordinate_mode={expected_mode}")
    return result
