# gw-kn-multimodal

[English](README.md) | [中文](README_zh.md)

A multimodal research repository for kilonova (KN) identification: starting from GWSamplegen gravitational-wave catalogs, it runs Rubin/SNANA optical simulations, builds HDF5 datasets, and trains and evaluates the joint GW + optical model MAGIKS alongside optical-only, Skymap-only, and Fink Random Forest baselines.

## Research Pipeline

```text
External GWSamplegen positive/negative catalogs and separate skymaps
    -> kn_simulation: positive Rubin/SNANA simulations, retained physical negatives
    -> simulation_intermediates.h5
    -> joint HDF5 / optical-only HDF5
    -> MAGIKS / optical-only training
    -> classification, retrieval ablations, physical sensitivity, GW170817A scenarios
```

The research inputs include BNS/NSBH events, GW scalars and skymaps, first-detection-aligned luptitude light curves, and GW-to-optical associations. Datasets, checkpoints, and runtime results must be prepared separately and are not distributed with the code.

## Layout and Documentation

```text
gw-kn-multimodal/
├── Model/
│   ├── model.py, data_loader.py       # MAGIKS, data loading and sampling
│   ├── args/                         # Tracked templates and shared defaults
│   └── scripts/                      # data / train / eval / hpo
├── optical_only/
│   ├── args/                         # Baseline and pretrained optical templates
│   └── scripts/                      # data / train / eval / analysis / plot
├── kn_simulation/
│   ├── bin/kn-sim                    # Optical production entry point
│   ├── profiles/                     # BNS/NSBH train/test YAML
│   ├── gw170817a/                    # GW170817A LSST scenario experiment
│   └── runs/                         # Git-ignored runtime products
├── plots_scripts/                    # Paper plots and result regeneration
├── figures/                          # Saved figures
├── tests/                            # Config, data, model and simulation tests
├── pyproject.toml, uv.lock           # Full research environment and locked versions
└── requirements*.txt                 # Minimal dependencies / historical snapshot
```

| Guide | Contents |
| --- | --- |
| [Environment](ENVIRONMENT.md#english-version) | Python, CUDA, development tools and external simulation dependencies |
| [MAGIKS configuration](Model/args/README.md) | Current training, ablation, default and evaluation templates |
| [Optical-only configuration](optical_only/args/README.md) | Randomly initialized baseline and v15/v16 recipes |
| [Optical simulation](kn_simulation/README.md) | Dual catalogs, Slurm, aggregation, resume and migration |
| [Runtime products](kn_simulation/runs/README.md) | Directories, manifests, status and HDF5 |
| [GW170817A experiment](kn_simulation/gw170817a/README.md) | Fixed-physics retrieval across observing scenarios |

The repository was previously named `ML+GW+KN`. The historical `dataset/` directory has been removed; the GW170817A workflow lives in `kn_simulation/gw170817a/`. Local historical backups and experiment notes are not required components of a GitHub checkout.

## Environment and Paths

The full environment uses Python 3.10. From the repository root:

```bash
uv sync --locked
source .venv/bin/activate
```

`requirements.txt` covers core model dependencies only, not the complete simulation environment. See [ENVIRONMENT.md](ENVIRONMENT.md#english-version) for development dependencies, the historical pip snapshot, SNANA, OpSim and `opsimsummaryv2` prerequisites.

Run the commands below from the repository root. Distinguish the data workspace from the code directory:

```bash
export BASE_DIR=/your/data/workspace
cd /path/to/gw-kn-multimodal
export REPO_ROOT="$(pwd)"
```

Entry points that support `BASE_DIR` default to `/fred/oz016/bgao_kn`; `REPO_ROOT` denotes the code checkout. JSON placeholders `<BASE_DIR>` / `<REPO_ROOT>` must be replaced before use. The simulation configuration loader expands YAML `${BASE_DIR}` / `${REPO_ROOT}` variables. Some evaluation wrappers use `WORKSPACE_ROOT`, and some Slurm log paths, partitions and notebook paths remain OzSTAR-specific. Setting `BASE_DIR` alone does not rewrite them.

### External Data Locations

| Purpose | Default location or configuration source |
| --- | --- |
| Dual GW catalogs | `$BASE_DIR/GWSamplegen/outputs/production_am_bayestar/dual/<source>_<split>_seed_1234/` |
| Positive/negative skymaps | `$BASE_DIR/data/skymap/{positive,negative}/<source>_skymap_<split>/` |
| Simulation aggregate | `$REPO_ROOT/kn_simulation/runs/<profile>/simulation_intermediates.h5` |
| Joint HDF5 | `$BASE_DIR/data/ALBEF_dataset/` |
| Optical HDF5 | `$BASE_DIR/data/Optical_Only_dataset/` |
| External optical negatives for MAGIKS | `$BASE_DIR/data/Optical_Negative_dataset/` |
| Model weights | `$BASE_DIR/data/model/` |
| Rubin / SNANA | OpSim database, SNANA executable and SNDATA_ROOT in the YAML profile |

`ALBEF_dataset`, checkpoint artifact `ALBEF/albef_best.pth`, and some legacy configuration keys retain their names for compatibility. Training and evaluation can use different negative files and HDF5 groups; check each input before running. Fink RF comparisons additionally require a model artifact from the external `fink-rf-reproduction` project.

<a id="configuration"></a>

## Generate Runtime Configuration

Git tracks `*.json.example`; runtime `*.json` files are ignored. After setting the environment above, generate missing configurations from tracked templates, replacing both placeholders and the legacy OzSTAR path prefixes in the GW170817A template. Existing configurations are preserved:

```bash
python - <<'PY'
import json
import os
import subprocess
from pathlib import Path

repo = Path.cwd()
base = str(Path(os.environ["BASE_DIR"]).expanduser().resolve())

def expand(value):
    if isinstance(value, str):
        for old, new in (
            ("/fred/oz016/bgao_kn/gw-kn-multimodal", str(repo)),
            ("/fred/oz016/bgao_kn", base),
        ):
            if value == old or value.startswith(old + "/"):
                return new + value[len(old):]
        return value.replace("<BASE_DIR>", base).replace("<REPO_ROOT>", str(repo))
    if isinstance(value, list):
        return [expand(item) for item in value]
    if isinstance(value, dict):
        return {key: expand(item) for key, item in value.items()}
    return value

names = subprocess.check_output([
    "git", "ls-files", "-z", "Model/args/*.json.example",
    "optical_only/args/*.json.example",
]).decode().split("\0")
for name in filter(None, names):
    template = repo / name
    target = template.with_suffix("")
    if target.exists():
        continue
    config = expand(json.loads(template.read_text(encoding="utf-8")))
    with target.open("x", encoding="utf-8") as stream:
        json.dump(config, stream, indent=2, ensure_ascii=False)
        stream.write("\n")
PY
```

After pulling updates, compare existing JSON files with their templates; generation does not migrate local configurations. Template checkpoint paths, directories and HDF5 groups are not guaranteed to exist on your machine.

## Workflows

### 1. Generate Rubin/SNANA Optical Simulations

Prepare matching positive/negative GW catalogs and their separate skymaps. The four production profiles are `bns_train`, `nsbh_train`, `bns_test` and `nsbh_test`:

```bash
kn_simulation/bin/kn-sim prepare bns_train \
  --pos-catalog "$BASE_DIR/GWSamplegen/outputs/production_am_bayestar/dual/bns_train_seed_1234/pos_catalog.csv" \
  --neg-catalog "$BASE_DIR/GWSamplegen/outputs/production_am_bayestar/dual/bns_train_seed_1234/neg_catalog.csv"
kn_simulation/bin/kn-sim submit bns_train --dry-run
kn_simulation/bin/kn-sim submit bns_train
kn_simulation/bin/kn-sim status bns_train
```

`prepare` writes a local run directory; `submit` schedules compute work. Only filtered positives enter SNANA; physical double-zero ejecta negatives are retained separately. Completed tasks are automatically consolidated into `simulation_intermediates.h5`. See the [simulation guide](kn_simulation/README.md) for resume, manual compaction and legacy migration.

### 2. Build Joint HDF5 Datasets

After completing both BNS and NSBH simulation aggregates for the corresponding split:

```bash
PROFILE=final_train DATASET_MODE=train \
  bash Model/scripts/data/submit_create_dataset_bns_nsbh.sh
PROFILE=astro_test DATASET_MODE=test \
  bash Model/scripts/data/submit_create_dataset_bns_nsbh.sh
```

The default outputs are `combined_dataset_train.h5` and `combined_dataset_astro_test.h5` under `data/ALBEF_dataset/`. The launcher reads the four profile aggregates and separate negative catalogs. Override paths or resources with `OUTPUT_H5_PATH`, `NUM_WORKERS`, `BNS_SIM_ARTIFACT` and `NSBH_SIM_ARTIFACT`.

The dataset distinguishes positives with usable KN light curves, physical double-zero ejecta type-1 negatives, and type-2 negatives with ejecta but no usable light curve. Optical rows reference positive GW parents through `parent_gw_idx`. These GW negative classes differ from external non-KN optical distractors.

### 3. Train MAGIKS and Component Ablations

```bash
mkdir -p logs/train
bash Model/scripts/train/train.sh Model/args/MAGIKS_BNS_NSBH_full.json

# Submit Full and five component ablations as six Slurm jobs
bash Model/scripts/train/submit_ablation_train.sh
```

The launcher merges shared defaults with experiment JSON, with experiment values taking precedence. Current Full uses staged training, mixed KN/non-KN galleries, training-aligned validation, and checkpoint selection by a mixed-gallery retrieval metric. Shared defaults specify a different gallery and selection metric until overridden; see the [MAGIKS configuration guide](Model/args/README.md).

### 4. Classification and Retrieval Comparisons

```bash
bash Model/scripts/eval/submit_test_evaluate.sh \
  Model/args/eval/cls_test/MAGIKS_BNS_NSBH_eval_full.json
bash Model/scripts/eval/submit_retrieval_comparison.sh \
  Model/args/eval/retrieval_comparison.json
```

The current retrieval template compares nine methods: Full, five ablations, Optical-only baseline, Skymap-only and Fink Random Forest. It uses `synthetic_time_sky_hard` non-KN distractors, a different test protocol from the `mixed_kn_nonkn` training galleries. Verify every model path. Redshift-analysis catalog paths still reference earlier `production_rubin_dual` data; check that they match the test HDF5 before use.

Standalone mixed-retrieval and fixed-checkpoint attribution scripts have been removed. Historical configuration or result names do not imply a current runnable entry point. See the [evaluation template index](Model/args/README.md#evaluation).

### 5. GW-KN Single-Parameter Physical Sensitivity

```bash
DRY_RUN=true bash Model/scripts/eval/submit_gw_kn_pairing_sensitivity.sh \
  Model/args/eval/gw_kn_single_parameter_sensitivity.json
bash Model/scripts/eval/submit_gw_kn_pairing_sensitivity.sh \
  Model/args/eval/gw_kn_single_parameter_sensitivity.json
```

This configuration constructs crossed pairs within BNS or NSBH while controlling non-target parameter differences. Shared candidate coordinates and GW-to-first-detection delays support interaction and directional-win comparisons. Brightness, colour, errors and internal light-curve evolution are preserved. Outputs include pair manifests, per-pair scores, bootstrap/permutation statistics and parameter trends. Validation does not submit jobs, though the shell wrapper creates a log directory.

Training-set fitting for the Physics Ejecta Bridge remains available:

```bash
bash Model/scripts/eval/submit_physics_ejecta_bridge_fit.sh \
  Model/args/eval/physics_ejecta_bridge_fit.json
```

The Bridge fits nearest-neighbour posteriors from training GW/optical features and scores their overlap in a shared physical space. It is an empirical physical reference. The fit configuration uses test data only for overlap checks. The directional-win v2 implementation remains, but no corresponding `.json.example` is currently distributed: provide a configuration with compatible models, a frozen Bridge artifact, pairing settings and the expected manifest digest. The v1 template above is not a substitute. See the [configuration guide](Model/args/README.md#evaluation).

### 6. Optical-Only Data and Training

```bash
DATASET_MODE=train BUILD_POSITIVE=true BUILD_NEGATIVE=true \
  bash optical_only/scripts/data/submit_create_datasets.sh
bash optical_only/scripts/train/train.sh optical_only/args/optical_only_kn_baseline.json

# Requires the MAGIKS Full checkpoint, evaluation inputs and time-offset distribution
bash optical_only/scripts/train/train.sh optical_only/args/optical_only_kn_v16.json
```

Optical preprocessing includes first-detection alignment, 2-hour same-band merging and a luptitude transform. The randomly initialized baseline provides an independent comparison. v16 initializes from the MAGIKS Full optical encoder and uses single-view prefix fine-tuning without adversarial heads or GRL. The training wrapper subsequently runs evaluation, so evaluation inputs must also be ready. See the [optical-only configuration guide](optical_only/args/README.md).

### 7. GW170817A LSST Scenario Retrieval

```bash
WORKSPACE_ROOT="$(dirname "$REPO_ROOT")" \
  bash kn_simulation/gw170817a/submit_gw170817a_lsst_pipeline.sh
bash Model/scripts/eval/submit_gw170817a_retrieval.sh \
  Model/args/eval/retrieval_gw170817a_lsst.json
```

Submit retrieval after the dataset build completes. The build wrapper requires the checkout name `gw-kn-multimodal` and uses `WORKSPACE_ROOT` for its parent directory. This experiment fixes GW170817A physics and host distance, comparing 10 Rubin observing scenarios over the same 50 coordinates, for 500 light curves. It is not a redshift scan. Inputs, stage selection and statistical methods are documented in the [GW170817A guide](kn_simulation/gw170817a/README.md).

## Checks and Reproduction

With development dependencies installed, run lightweight configuration and entry-point checks from the repository root:

```bash
python -m pytest tests/config/test_config_templates.py tests/config/test_optical_only_layout.py
```

The current `retrieval_gw170817a_lsst.json.example` still contains hardcoded paths, causing the template portability subtest to fail. The generation example above translates only the generated local JSON, leaving the source template unchanged. Some checks also depend on local historical configurations or scientific dependencies; configuration checks do not validate the complete scientific pipeline.

The full suite lives under `tests/`. Preserve actual runtime JSON, data versions and checkpoint provenance, and check Slurm partitions, resources, log directories and node-local storage. Plot scripts may write into a sibling paper directory; check output locations before regenerating figures.

## Data, Citation and License

Generate data through external GWSamplegen and the simulation/dataset workflows in this repository; contact the authors for pre-built datasets. The paper is in preparation; see [CITATION.cff](CITATION.cff). This project uses the [MIT License](LICENSE).
