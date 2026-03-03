#!/bin/bash

#SBATCH --job-name=ALBEF_HPO
#SBATCH --output=logs/hpo/%x_%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=200G
#SBATCH --gres=gpu:1
#SBATCH --time=168:00:00
#SBATCH --partition=gpu
#SBATCH --tmp=180G

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
OBJECTIVE_METRIC=$(jq -r '.objective_metric // "combined_auroc_g2o_r5"' "$args_file")
BEST_CKPT_METRIC=$(jq -r '.best_ckpt_metric // "auprc"' "$args_file")
OOD_MONITORING=$(jq -r '.enable_ood_monitoring // false' "$args_file")

BASE_TRAIN_CONFIG=$(jq -r '.base_train_config // empty' "$args_file")
if [[ -n "$BASE_TRAIN_CONFIG" && "$BASE_TRAIN_CONFIG" != "null" ]]; then
    if [[ "$BASE_TRAIN_CONFIG" != /* ]]; then
        BASE_TRAIN_CONFIG="$args_dir/$BASE_TRAIN_CONFIG"
    fi
    if [[ ! -f "$BASE_TRAIN_CONFIG" ]]; then
        echo "base_train_config not found: $BASE_TRAIN_CONFIG"
        exit 1
    fi
fi

DATA_PATH=$(jq -r '.data_path // empty' "$args_file")
NEG_DATA_PATH=$(jq -r '.neg_data_path // empty' "$args_file")
NEG_GROUP=$(jq -r '.neg_group // empty' "$args_file")
TEST_DATA_PATH=$(jq -r '.test_data_path // empty' "$args_file")
NEG_OFFSET_DIST_NPZ=$(jq -r '.neg_offset_dist_npz // empty' "$args_file")

if [[ ( -z "$DATA_PATH" || "$DATA_PATH" == "null" ) && -n "$BASE_TRAIN_CONFIG" && "$BASE_TRAIN_CONFIG" != "null" ]]; then
    DATA_PATH=$(jq -r '.data_path // empty' "$BASE_TRAIN_CONFIG")
fi
if [[ ( -z "$NEG_DATA_PATH" || "$NEG_DATA_PATH" == "null" ) && -n "$BASE_TRAIN_CONFIG" && "$BASE_TRAIN_CONFIG" != "null" ]]; then
    NEG_DATA_PATH=$(jq -r '.neg_data_path // empty' "$BASE_TRAIN_CONFIG")
fi
if [[ ( -z "$NEG_GROUP" || "$NEG_GROUP" == "null" ) && -n "$BASE_TRAIN_CONFIG" && "$BASE_TRAIN_CONFIG" != "null" ]]; then
    NEG_GROUP=$(jq -r '.neg_group // empty' "$BASE_TRAIN_CONFIG")
fi
if [[ ( -z "$TEST_DATA_PATH" || "$TEST_DATA_PATH" == "null" ) && -n "$BASE_TRAIN_CONFIG" && "$BASE_TRAIN_CONFIG" != "null" ]]; then
    TEST_DATA_PATH=$(jq -r '.test_data_path // empty' "$BASE_TRAIN_CONFIG")
fi
if [[ ( -z "$NEG_OFFSET_DIST_NPZ" || "$NEG_OFFSET_DIST_NPZ" == "null" ) && -n "$BASE_TRAIN_CONFIG" && "$BASE_TRAIN_CONFIG" != "null" ]]; then
    NEG_OFFSET_DIST_NPZ=$(jq -r '.neg_offset_dist_npz // empty' "$BASE_TRAIN_CONFIG")
fi

if [[ -z "$DATA_PATH" || "$DATA_PATH" == "null" ]]; then
    echo "data_path missing in HPO config and base_train_config."
    exit 1
fi
if [[ ! -f "$DATA_PATH" ]]; then
    echo "Training data file not found: $DATA_PATH"
    exit 1
fi
if [[ -n "$NEG_DATA_PATH" && "$NEG_DATA_PATH" != "null" && ! -f "$NEG_DATA_PATH" ]]; then
    echo "Negative data file not found: $NEG_DATA_PATH"
    exit 1
fi
if [[ -n "$TEST_DATA_PATH" && "$TEST_DATA_PATH" != "null" && ! -f "$TEST_DATA_PATH" ]]; then
    echo "Test/OOD data file not found: $TEST_DATA_PATH"
    exit 1
fi
if [[ -n "$NEG_OFFSET_DIST_NPZ" && "$NEG_OFFSET_DIST_NPZ" != "null" && ! -f "$NEG_OFFSET_DIST_NPZ" ]]; then
    echo "Negative offset distribution file not found: $NEG_OFFSET_DIST_NPZ"
    exit 1
fi

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
echo "  Objective metric: $OBJECTIVE_METRIC"
echo "  Best ckpt metric: $BEST_CKPT_METRIC"
echo "  enable_ood_monitoring (config): $OOD_MONITORING"
echo "  enable_ood_monitoring (runtime override): false"
echo "  data_path (resolved): $DATA_PATH"
if [[ -n "$NEG_DATA_PATH" && "$NEG_DATA_PATH" != "null" ]]; then
    echo "  neg_data_path (resolved): $NEG_DATA_PATH"
fi
if [[ -n "$NEG_GROUP" && "$NEG_GROUP" != "null" ]]; then
    echo "  neg_group (resolved): $NEG_GROUP"
fi
if [[ -n "$TEST_DATA_PATH" && "$TEST_DATA_PATH" != "null" ]]; then
    echo "  test_data_path (resolved): $TEST_DATA_PATH"
fi
if [[ -n "$NEG_OFFSET_DIST_NPZ" && "$NEG_OFFSET_DIST_NPZ" != "null" ]]; then
    echo "  neg_offset_dist_npz (resolved): $NEG_OFFSET_DIST_NPZ"
fi
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
    if [[ -n "$TEST_DATA_PATH" && "$TEST_DATA_PATH" != "null" ]]; then
        cp -f "$TEST_DATA_PATH" "$JOBFS_DIR"/
        TEST_DATA_PATH="$JOBFS_DIR/$(basename "$TEST_DATA_PATH")"
    fi
    if [[ -n "$NEG_OFFSET_DIST_NPZ" && "$NEG_OFFSET_DIST_NPZ" != "null" ]]; then
        cp -f "$NEG_OFFSET_DIST_NPZ" "$JOBFS_DIR"/
        NEG_OFFSET_DIST_NPZ="$JOBFS_DIR/$(basename "$NEG_OFFSET_DIST_NPZ")"
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
  --arg test_data_path "$TEST_DATA_PATH" \
  --arg neg_offset_dist_npz "$NEG_OFFSET_DIST_NPZ" \
  --arg base_train_config "$BASE_TRAIN_CONFIG" \
  --arg best_ckpt_metric "$BEST_CKPT_METRIC" \
  --argjson num_workers "${SLURM_CPUS_PER_TASK}" \
  '
  .data_path = $data_path
  | .num_workers = $num_workers
  | .best_ckpt_metric = $best_ckpt_metric
  | .enable_ood_monitoring = false
  | (if ($base_train_config | length) > 0 and $base_train_config != "null"
     then .base_train_config = $base_train_config
     else .
     end)
  | (if ($neg_data_path | length) > 0 and $neg_data_path != "null"
     then .neg_data_path = $neg_data_path
     else .
     end)
  | (if ($neg_group | length) > 0 and $neg_group != "null"
     then .neg_group = $neg_group
     else .
     end)
  | (if ($test_data_path | length) > 0 and $test_data_path != "null"
     then .test_data_path = $test_data_path
     else .
     end)
  | (if ($neg_offset_dist_npz | length) > 0 and $neg_offset_dist_npz != "null"
     then .neg_offset_dist_npz = $neg_offset_dist_npz
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
