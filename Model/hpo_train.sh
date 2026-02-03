#!/bin/bash

#SBATCH --job-name=ALBEF_HPO
#SBATCH --output=logs/hpo/%x_%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=150G
#SBATCH --gres=gpu:1
#SBATCH --time=72:00:00
#SBATCH --partition=gpu
#SBATCH --tmp=110G

set -euo pipefail

args_file=${1:-}
if [[ -z "${args_file}" || ! -f "${args_file}" ]]; then
    echo "Usage: $0 <hpo_config.json>"
    exit 1
fi

args_dir="$(cd "$(dirname "${args_file}")" && pwd)"
args_file="${args_dir}/$(basename "${args_file}")"

# ─── Auto-submit via sbatch if not inside a SLURM allocation ──────────
if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    if ! command -v sbatch >/dev/null 2>&1; then
        echo "sbatch not found; run inside a Slurm allocation or install Slurm tools."
        exit 1
    fi

    script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    script_path="${script_dir}/$(basename "${BASH_SOURCE[0]}")"

    sbatch_opts=()
    if [[ -n "${JOB_NAME:-}" ]]; then
        sbatch_opts+=(--job-name="${JOB_NAME}")
    fi
    if [[ -n "${TIME_LIMIT:-}" ]]; then
        sbatch_opts+=(--time="${TIME_LIMIT}")
    fi
    if [[ -n "${PARTITION:-}" ]]; then
        sbatch_opts+=(--partition="${PARTITION}")
    fi

    echo "Submitting job with: sbatch ${sbatch_opts[*]} ${script_path} ${args_file}"
    sbatch "${sbatch_opts[@]}" "${script_path}" "${args_file}"
    exit 0
fi

which python

# ─── Read configuration from JSON ─────────────────────────────────────
N_TRIALS=$(jq -r '.n_trials' "$args_file")
STUDY_NAME=$(jq -r '.study_name' "$args_file")
EPOCHS_PER_TRIAL=$(jq -r '.epochs_per_trial' "$args_file")
OUTPUT_DIR=$(jq -r '.output_dir' "$args_file")
N_STARTUP_TRIALS=$(jq -r '.n_startup_trials // 10' "$args_file")

DATA_PATH=$(jq -r '.data_path' "$args_file")
NEG_DATA_PATH=$(jq -r '.neg_data_path // empty' "$args_file")
NEG_GROUP=$(jq -r '.neg_group // empty' "$args_file")
CACHE_IN_MEMORY=$(jq -r '.cache_in_memory // false' "$args_file")

# ─── SLURM Info ────────────────────────────────────────────────────────
echo "========================================"
echo "SLURM Job Information"
echo "========================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node: $SLURMD_NODENAME"
echo "Partition: $SLURM_JOB_PARTITION"
echo "CPUs: $SLURM_CPUS_PER_TASK"
echo "Memory: ${SLURM_MEM_PER_NODE}MB"
echo "Start time: $(date)"
echo "========================================"
echo ""
echo "Configuration File: $args_file"
echo "HPO Configuration:"
echo "  Trials: $N_TRIALS"
echo "  Study: $STUDY_NAME"
echo "  Epochs/trial: $EPOCHS_PER_TRIAL"
echo "  Output: $OUTPUT_DIR"
echo "  Startup trials: $N_STARTUP_TRIALS"
echo ""

# ─── Stage data to local disk ─────────────────────────────────────────
JOBFS_DIR="${SLURM_TMPDIR:-${TMPDIR:-${JOBFS:-}}}"
if [ -n "$JOBFS_DIR" ]; then
    echo "Staging datasets to local disk: $JOBFS_DIR"
    cp -f "$DATA_PATH" "$JOBFS_DIR"/
    DATA_PATH="$JOBFS_DIR/$(basename "$DATA_PATH")"
    if [[ -n "$NEG_DATA_PATH" && "$NEG_DATA_PATH" != "null" ]]; then
        cp -f "$NEG_DATA_PATH" "$JOBFS_DIR"/
        NEG_DATA_PATH="$JOBFS_DIR/$(basename "$NEG_DATA_PATH")"
    fi
    echo "Staging complete."
else
    echo "No local tmp dir found; skip staging."
fi

# ─── Create directories ───────────────────────────────────────────────
mkdir -p logs/hpo
mkdir -p "$OUTPUT_DIR"

# ─── Build command ─────────────────────────────────────────────────────
cmd=(
    python -u /fred/oz016/bgao_kn/ML+GW+KN/Model/hpo_optuna.py
    --n_trials "$N_TRIALS"
    --study_name "$STUDY_NAME"
    --output_dir "$OUTPUT_DIR"
    --epochs_per_trial "$EPOCHS_PER_TRIAL"
    --n_startup_trials "$N_STARTUP_TRIALS"
    --data_path "$DATA_PATH"
    --num_workers "$SLURM_CPUS_PER_TASK"
)

if [[ -n "$NEG_DATA_PATH" && "$NEG_DATA_PATH" != "null" ]]; then
    cmd+=(--neg_data_path "$NEG_DATA_PATH")
fi
if [[ -n "$NEG_GROUP" && "$NEG_GROUP" != "null" ]]; then
    cmd+=(--neg_group "$NEG_GROUP")
fi
if [[ "$CACHE_IN_MEMORY" == "true" ]]; then
    cmd+=(--cache_in_memory)
fi

# ─── Run HPO ───────────────────────────────────────────────────────────
echo "Command: ${cmd[*]}"
"${cmd[@]}"
exit_code=$?

echo "------------------------------------------------"
echo "End time: $(date)"
echo "Exit code: $exit_code"

# ─── Run analysis if HPO succeeded ────────────────────────────────────
if [ $exit_code -eq 0 ]; then
    echo "Running post-HPO analysis..."
    python -u /fred/oz016/bgao_kn/ML+GW+KN/Model/hpo_analyze.py \
        --study_name "$STUDY_NAME" \
        --storage "sqlite:///$OUTPUT_DIR/optuna_study.db" \
        --output_dir "$OUTPUT_DIR/analysis" \
        --top_k 5
fi

exit $exit_code
