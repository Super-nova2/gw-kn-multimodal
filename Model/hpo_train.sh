#!/bin/bash

#SBATCH --job-name=ALBEF_HPO
#SBATCH --output=logs/hpo/%x_%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=150G
#SBATCH --gres=gpu:1
#SBATCH --time=168:00:00
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

DATA_PATH=$(jq -r '.data_path' "$args_file")
NEG_DATA_PATH=$(jq -r '.neg_data_path // empty' "$args_file")
NEG_GROUP=$(jq -r '.neg_group // empty' "$args_file")

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

# ─── Build runtime config (inject staged paths + worker count) ───────
RUNTIME_CONFIG="$OUTPUT_DIR/runtime_hpo_config_${SLURM_JOB_ID}.json"
jq \
  --arg data_path "$DATA_PATH" \
  --arg neg_data_path "$NEG_DATA_PATH" \
  --arg neg_group "$NEG_GROUP" \
  --argjson num_workers "${SLURM_CPUS_PER_TASK}" \
  '
  .data_path = $data_path
  | .num_workers = $num_workers
  | (if ($neg_data_path | length) > 0 and $neg_data_path != "null"
     then .neg_data_path = $neg_data_path
     else .
     end)
  | (if ($neg_group | length) > 0 and $neg_group != "null"
     then .neg_group = $neg_group
     else .
     end)
  ' \
  "$args_file" > "$RUNTIME_CONFIG"

# ─── Build command ─────────────────────────────────────────────────────
cmd=(
    python -u /fred/oz016/bgao_kn/ML+GW+KN/Model/hpo_optuna.py
    --config "$RUNTIME_CONFIG"
)

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
    STORAGE=$(jq -r '.storage // empty' "$RUNTIME_CONFIG")
    if [[ -z "$STORAGE" || "$STORAGE" == "null" ]]; then
        STORAGE="sqlite:///$OUTPUT_DIR/optuna_study.db"
    fi
    python -u /fred/oz016/bgao_kn/ML+GW+KN/Model/hpo_analyze.py \
        --study_name "$STUDY_NAME" \
        --storage "$STORAGE" \
        --output_dir "$OUTPUT_DIR/analysis" \
        --top_k 5
fi

exit $exit_code
