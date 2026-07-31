# gw-kn-multimodal

`gw-kn-multimodal` is a multimodal research repository for kilonova (KN) identification, covering the full pipeline from simulated data generation and HDF5 dataset construction to joint GW + optical training, optical-only baseline training, and evaluation.

> The repository was previously named `ML+GW+KN`. It has been renamed to avoid path and import compatibility issues with the `+` character.

## Goals

This repository addresses three problems:

1. Organise gravitational-wave event parameters and optical photometric time series into a unified training dataset.
2. Train a joint GW + optical model for KN matching, retrieval, and classification.
3. Train an optical-only baseline that does not depend on GW inputs, and analyse its generalisation on real and simulated negative samples.

The core research objects include:

- BNS / NSBH simulated events and their corresponding kilonova light curves.
- GW scalar parameters and skymap representations.
- Optical sequence representations based on luptitude, first-detection alignment, and time-offset modelling.
- Multimodal MAGIKS / contrastive learning training, and an optical-only classification baseline.

## Repository Structure

```text
gw-kn-multimodal/
├── Model/
│   ├── data_loader.py                # HDF5 reading and samplers
│   ├── model.py                      # MAGIKS model and encoders
│   ├── args/                         # Current configs and portable templates
│   └── scripts/
│       ├── train/                    # Training entry points
│       ├── eval/                     # Evaluation, retrieval, and benchmarks
│       ├── data/                     # Dataset construction
│       └── hpo/                      # Hyperparameter optimisation
├── optical_only/
│   ├── args/                             # current templates and archived configs
│   ├── scripts/                          # train / eval / data / analysis / plot
│   └── notebooks/                        # optical-only analysis notebooks
├── kn_simulation/                    # Maintained GWSamplegen → Rubin/SNANA pipeline
│   ├── bin/kn-sim                    # Unified user entry point
│   ├── src/                          # Python implementation
│   ├── profiles/                     # BNS/NSBH train/test profiles
│   └── runs/                         # Git-ignored runtime directories
├── dataset/                          # Frozen historical data and workflows
└── docs/                             # Design notes and experiment records
```

## Dependencies

Python dependencies are listed in `requirements.txt`. Install with:

```bash
pip install -r requirements.txt
```

Main dependencies: Python 3.10+, PyTorch, h5py, numpy, pandas, astropy, healpy, ligo.skymap, tqdm, matplotlib, optuna, tensorboard, graphviz.

Additional system dependencies: jq, Slurm (HPC environment), SNANA + opsimsummaryv2 (data generation only).

Running on an HPC / Slurm environment is recommended. All `.sh` entry scripts are written for Slurm job submission and resource allocation.

## Data & Path Conventions

### Environment Variable `BASE_DIR`

All shell scripts and Python source code resolve data paths through the `BASE_DIR` environment variable. If unset, it defaults to `/fred/oz016/bgao_kn`.

```bash
export BASE_DIR=/your/data/root
```

### JSON Configuration (`.json.example` Templates)

Configuration files use a template pattern:

- `*.json.example` files are tracked in Git and use `<BASE_DIR>` as a path placeholder
- `*.json` files are the actual runtime configs (gitignored)

On first use, copy templates and substitute your path:

```bash
cd /path/to/gw-kn-multimodal
REPO_ROOT="$(pwd)"
find Model/args -name '*.json.example' -print0 | while IFS= read -r -d '' f; do
    sed -e "s|<BASE_DIR>|$BASE_DIR|g" -e "s|<REPO_ROOT>|$REPO_ROOT|g" "$f" > "${f%.example}"
done

# Repeat for optical_only/args/; kn_simulation uses tracked YAML profiles
```

### Data Directory Layout

| Purpose | Path |
|---------|------|
| Multimodal HDF5 | `$BASE_DIR/data/ALBEF_dataset/` |
| optical-only HDF5 | `$BASE_DIR/data/Optical_Only_dataset/` |
| Checkpoints | `$BASE_DIR/data/model/` |
| Skymap / SNANA data | `$BASE_DIR/data/` and `$BASE_DIR/SNANA/` |

`ALBEF_dataset` is the existing external data location and remains unchanged for HDF5 compatibility.

## Workflows

### 1. Build GW + Optical Dataset

Organises BNS / NSBH GW parameters, skymaps, and optical light curves into a single HDF5 file.

```bash
cd /path/to/gw-kn-multimodal

PROFILE=final_train DATASET_MODE=train \
bash Model/scripts/data/submit_create_dataset_bns_nsbh.sh
```

Key environment variables:

| Variable | Options |
|----------|---------|
| `PROFILE` | `test_aug`, `final_train` |
| `DATASET_MODE` | `train`, `test` |
| `OUTPUT_H5_PATH` | `/path/to/output.h5` |
| `NUM_WORKERS` | e.g. `6` |

Relevant files:

- `Model/scripts/data/submit_create_dataset_bns_nsbh.sh`
- `Model/scripts/data/create_dataset_bns_nsbh.py`

### 2. Train the Multimodal GW + Optical Model

```bash
cd /path/to/gw-kn-multimodal

bash Model/scripts/train/train.sh Model/args/MAGIKS_BNS_NSBH_full.json
```

Relevant files:

- `Model/scripts/train/train.py`
- `Model/model.py`
- `Model/data_loader.py`
- `Model/args/MAGIKS_BNS_NSBH_full.json`

The training script reads data paths, negative sample paths, time-offset settings, model hyperparameters, and checkpoint directory from the JSON config.

### 3. Evaluate the Multimodal Model

```bash
cd /path/to/gw-kn-multimodal

bash Model/scripts/eval/submit_test_evaluate.sh /path/to/eval_args.json
```

Relevant files:

- `Model/scripts/eval/evaluate.py`
- `Model/scripts/eval/submit_test_evaluate.sh`

Supports retrieval, classification, OOD monitoring, and negative-sample time-offset evaluation.

### 4. Build the Optical-Only Dataset

Applies first-detection alignment, 2-hour same-band merging, luptitude transformation, and outputs an optical-only HDF5.

```bash
cd /path/to/gw-kn-multimodal

DATASET_MODE=train \
BUILD_POSITIVE=true \
BUILD_NEGATIVE=true \
bash optical_only/scripts/data/submit_create_datasets.sh
```

Relevant files:

- `optical_only/scripts/data/submit_create_datasets.sh`
- `optical_only/scripts/data/create_datasets.py`

### 5. Train the Optical-Only Baseline

```bash
cd /path/to/gw-kn-multimodal

bash optical_only/scripts/train/train.sh optical_only/args/optical_only_kn_v16.json
```

Relevant files:

- `optical_only/scripts/train/train.py`
- `optical_only/scripts/eval/evaluate.py`
- `optical_only/args/optical_only_kn_v16.json`

The script automatically invokes the optical-only evaluation after training completes.

## Configuration Guide

Configurations are stored in two directories:

- `Model/args/`
- `optical_only/args/`

Copy an existing JSON and edit the following fields:

| Field | Description |
|-------|-------------|
| `data_path` / `pos_data_path` / `neg_data_path` | Paths to HDF5 datasets |
| `ckpt_path` / `output_dir` | Output and checkpoint directories |
| `batch_size` / `num_epochs` / `num_workers` | Training settings |
| Time-offset modelling parameters | `delta_time_*` fields |
| `prefix` / `universal` / `OOD` / `shortcut_audit` | Experimental switches |

## Notes

- Most scripts assume a Slurm environment; on local machines they will attempt `sbatch` self-submission.
- When transferring to a new environment, regenerate config files from `.json.example` templates.
- New optical simulations use `kn_simulation/bin/kn-sim` and depend on external
  SNANA, OpSim data, and database files.
- `dataset/` is retained for historical experiments, GW170817A, and existing
  model consumers; new production jobs must not invoke scripts from it.
- Some notebooks and experiment documents retain cluster-specific paths from earlier development; verify paths before running.

## Recommended Starting Order

1. Install dependencies: `pip install -r requirements.txt`
2. Set the environment variable: `export BASE_DIR=/your/data/root`
3. Generate local config files from `.json.example` templates (see above)
4. Confirm that HDF5, skymap, and SNANA data files under `$BASE_DIR/data/` are present
5. Run dataset construction -> training -> evaluation in order

## Data

This repository does not include training data or large simulation files. Datasets can be obtained by:

- **GW-associated optical simulations**: use `kn_simulation/bin/kn-sim` to
  convert a GWSamplegen `catalog.csv` to `kn_catalog.csv` and submit Rubin/SNANA.
- **Training HDF5**: Build from simulation data using `Model/scripts/data/create_dataset_bns_nsbh.py` and `optical_only/scripts/data/create_datasets.py`.
- For pre-built datasets, please contact the authors.

## Citation

If you use this code, please cite our paper:

```
[Paper in preparation]
```

See `CITATION.cff` for details.

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
