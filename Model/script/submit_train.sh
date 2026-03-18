#!/bin/bash
#SBATCH --job-name=GW_Opt_Contrastive_learning   # Job name
#SBATCH --output=logs/train/%x_%j.out         # Standard output log (%x=Job Name, %j=Job ID)
#SBATCH --nodes=1                       # Number of nodes requested
#SBATCH --ntasks=1                      # Number of tasks
#SBATCH --cpus-per-task=8               # CPU cores per task (Recommended >= num_workers)
#SBATCH --mem=16G                       # Memory request (Adjust based on HDF5 size)
#SBATCH --gres=gpu:1                    # Request 1 GPU (Can specify model, e.g., gpu:a100:1)
#SBATCH --time=12:00:00                 # Max runtime (HH:MM:SS)
#SBATCH --partition=gpu                 # Partition name (Modify based on your server, e.g., gpu, defq)

# ================= Configuration Area =================
set -euo pipefail
SCRIPT_SUBDIR="Model/script"
SCRIPT_REL_PATH="Model/script/submit_train.sh"
REPO_NAME="gw-kn-multimodal"
if [[ -n "${SLURM_JOB_ID:-}" && -n "${SLURM_SUBMIT_DIR:-}" ]]; then
    if [[ "$(basename "${SLURM_SUBMIT_DIR}")" == "${REPO_NAME}" ]]; then
        REPO_ROOT="${SLURM_SUBMIT_DIR}"
    else
        REPO_ROOT="${SLURM_SUBMIT_DIR}/${REPO_NAME}"
    fi
else
    LOCAL_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    REPO_ROOT="${LOCAL_SCRIPT_DIR%/${SCRIPT_SUBDIR}}"
fi
SCRIPT_DIR="${REPO_ROOT}/${SCRIPT_SUBDIR}"
SCRIPT_PATH="${REPO_ROOT}/${SCRIPT_REL_PATH}"
if [[ ! -f "${SCRIPT_PATH}" ]]; then
    echo "Resolved script path not found: ${SCRIPT_PATH}" >&2
    exit 1
fi
which python

args_file=${1:-}
if [[ -z "${args_file}" || ! -f "${args_file}" ]]; then
    echo "Usage: $0 <args.json>"
    exit 1
fi

# ================= Training Parameters =================
DATA_PATH=$(jq -r '.data_path' "$args_file")     # Path to HDF5 data
CKPT_PATH=$(jq -r '.ckpt_path' "$args_file")          # Path to save checkpoints
EPOCHS=$(jq -r '.epochs' "$args_file")                        # Number of training epochs
BATCH_SIZE=$(jq -r '.batch_size' "$args_file")                    # Batch size
LR=$(jq -r '.lr' "$args_file")                          # Learning rate
NUM_WORKERS=$(jq -r '.num_workers' "$args_file")                    # Number of data loading workers (Should be <= cpus-per-task)
STEPS_PER_EPOCH=$(jq -r '.steps_per_epoch' "$args_file")                    # Steps per epoch
Float32=$(jq -r '.float32' "$args_file")                    # Use Float32 precision if true

mkdir -p "$CKPT_PATH"

echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Start time: $(date)"
echo "------------------------------------------------"

# ================= Run Command =================
# 'python -u' disables stdout buffering, allowing real-time logging in the output file
if [ "$Float32" = true ]; then
    python -u "${SCRIPT_DIR}/Contrastive_train.py" \
        --data_path "$DATA_PATH" \
        --ckpt_path "$CKPT_PATH" \
        --epochs $EPOCHS \
        --batch_size $BATCH_SIZE \
        --lr $LR \
        --num_workers $NUM_WORKERS \
        --float32 \
        # --steps_per_epoch 1000  # Optional: Uncomment to manually set steps per epoch
    exit_code=$?
    if [ $exit_code -ne 0 ]; then
        echo "Training script failed with exit code $exit_code"
        exit $exit_code
    fi
else
    python -u "${SCRIPT_DIR}/Contrastive_train.py" \
        --data_path "$DATA_PATH" \
        --ckpt_path "$CKPT_PATH" \
        --epochs $EPOCHS \
        --batch_size $BATCH_SIZE \
        --lr $LR \
        --num_workers $NUM_WORKERS
        # --steps_per_epoch 1000  # Optional: Uncomment to manually set steps per epoch
    exit_code=$?
    if [ $exit_code -ne 0 ]; then
        echo "Training script failed with exit code $exit_code"
        exit $exit_code
    fi
fi

echo "------------------------------------------------"
echo "End time: $(date)"
