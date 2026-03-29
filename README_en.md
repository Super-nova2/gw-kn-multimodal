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
- Multimodal ALBEF / contrastive learning training, and an optical-only classification baseline.

## Repository Structure

```text
gw-kn-multimodal/
├── Model/
│   ├── ALBEF_train.py / .sh          # Multimodal training entry point
│   ├── data_loader.py                # HDF5 reading and sampler
│   ├── model.py                      # GW / optical encoders and classification head
│   ├── args/                         # Training and HPO configurations
│   └── script/
│       ├── create_dataset_bns_nsbh.py
│       ├── submit_create_dataset_bns_nsbh.sh
│       └── submit_test_evaluate.sh
├── optical_only/
│   ├── create_optical_only_datasets.py   # optical-only dataset construction
│   ├── train_optical_only.py / .sh       # optical-only training and auto-evaluation
│   ├── test_evaluate_optical_only.py     # optical-only evaluation
│   └── args/                             # optical-only configurations
├── dataset/
│   ├── O5_sim_*                      # Simulated events and analysis materials
│   └── KN_sim/                       # SNANA / OpSim workflow scripts
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

Configuration files use `<BASE_DIR>` as a placeholder. Replace all occurrences of `<BASE_DIR>` with your actual root directory before running. Key directories:

| Purpose | Example path |
|---------|-------------|
| Multimodal HDF5 | `<BASE_DIR>/data/ALBEF_dataset/` |
| optical-only HDF5 | `<BASE_DIR>/data/Optical_Only_dataset/` |
| Checkpoints | `<BASE_DIR>/data/model/` |
| Skymap / SNANA data | `<BASE_DIR>/data/` and `<BASE_DIR>/SNANA/` |

To run on a different machine, update:

- `Model/args/*.json`
- `optical_only/args/*.json`
- Environment variables passed to submission scripts

## Workflows

### 1. Build GW + Optical Dataset

Organises BNS / NSBH GW parameters, skymaps, and optical light curves into a single HDF5 file.

```bash
cd /path/to/gw-kn-multimodal

PROFILE=final_train DATASET_MODE=train \
bash Model/script/submit_create_dataset_bns_nsbh.sh
```

Key environment variables:

| Variable | Options |
|----------|---------|
| `PROFILE` | `test_aug`, `final_train` |
| `DATASET_MODE` | `train`, `test` |
| `OUTPUT_H5_PATH` | `/path/to/output.h5` |
| `NUM_WORKERS` | e.g. `6` |

Relevant files:

- `Model/script/submit_create_dataset_bns_nsbh.sh`
- `Model/script/create_dataset_bns_nsbh.py`

### 2. Train the Multimodal GW + Optical Model

```bash
cd /path/to/gw-kn-multimodal

bash Model/ALBEF_train.sh Model/args/ALBEF_BNS_NSBH.json
```

Relevant files:

- `Model/ALBEF_train.py`
- `Model/model.py`
- `Model/data_loader.py`
- `Model/args/ALBEF_BNS_NSBH.json`

The training script reads data paths, negative sample paths, time-offset settings, model hyperparameters, and checkpoint directory from the JSON config.

### 3. Evaluate the Multimodal Model

```bash
cd /path/to/gw-kn-multimodal

bash Model/script/submit_test_evaluate.sh /path/to/eval_args.json
```

Relevant files:

- `Model/test_evaluate.py`
- `Model/script/submit_test_evaluate.sh`

Supports retrieval, classification, OOD monitoring, and negative-sample time-offset evaluation.

### 4. Build the Optical-Only Dataset

Applies first-detection alignment, 2-hour same-band merging, luptitude transformation, and outputs an optical-only HDF5.

```bash
cd /path/to/gw-kn-multimodal

DATASET_MODE=train \
BUILD_POSITIVE=true \
BUILD_NEGATIVE=true \
bash optical_only/submit_create_optical_only_datasets.sh
```

Relevant files:

- `optical_only/submit_create_optical_only_datasets.sh`
- `optical_only/create_optical_only_datasets.py`

### 5. Train the Optical-Only Baseline

```bash
cd /path/to/gw-kn-multimodal

bash optical_only/train_optical_only.sh optical_only/args/optical_only_kn_v14.json
```

Relevant files:

- `optical_only/train_optical_only.py`
- `optical_only/test_evaluate_optical_only.py`
- `optical_only/args/optical_only_kn_v14.json`

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
- Example configurations use absolute paths — always check JSON files before transferring to a new environment.
- Scripts under `dataset/KN_sim/` depend on external SNANA, OpSim data and database files.
- Some notebooks and experiment documents retain cluster-specific paths from earlier development; verify paths before running.

## Recommended Starting Order

1. Install dependencies: `pip install -r requirements.txt`
2. Replace `<BASE_DIR>` in configuration files with your actual root directory.
3. Confirm that HDF5, skymap, and SNANA data files under your data directory are present.
4. Check `Model/args/ALBEF_BNS_NSBH.json` or `optical_only/args/optical_only_kn_v14.json`.
5. Run dataset construction -> training -> evaluation in order.

## Data

This repository does not include training data or large simulation files. Datasets can be obtained by:

- **GW simulated events**: Generate using scripts under `dataset/KN_sim/` with SNANA/OpSim.
- **Training HDF5**: Build from simulation data using `Model/script/create_dataset_bns_nsbh.py` and `optical_only/create_optical_only_datasets.py`.
- For pre-built datasets, please contact the authors.

## Citation

If you use this code, please cite our paper:

```
[Paper in preparation]
```

See `CITATION.cff` for details.

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
