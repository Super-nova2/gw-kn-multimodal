#!/bin/bash

#SBATCH --job-name=MAGIKS_HPO
#SBATCH --output=logs/hpo/%x_%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=200G
#SBATCH --gres=gpu:1
#SBATCH --time=168:00:00
#SBATCH --partition=milan-gpu
#SBATCH --tmp=180G

set -euo pipefail

SCRIPT_SUBDIR="Model/scripts/hpo"
SCRIPT_REL_PATH="Model/scripts/hpo/hpo_train.sh"
DEFAULT_HPO_CONFIG_REL="Model/args/hpo/hpo_v6_main_guardrail85.json"
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
WORKSPACE_ROOT="$(dirname "${REPO_ROOT}")"
DEFAULT_HPO_CONFIG="${REPO_ROOT}/${DEFAULT_HPO_CONFIG_REL}"
if [[ ! -f "${SCRIPT_PATH}" ]]; then
    echo "Resolved script path not found: ${SCRIPT_PATH}" >&2
    exit 1
fi

resolve_config_path() {
    local raw_path="$1"
    local field_name="$2"
    local raw_dir=""

    if [[ -z "${raw_path}" || "${raw_path}" == "null" ]]; then
        echo ""
        return 0
    fi

    raw_path="${raw_path//<REPO_ROOT>/${REPO_ROOT}}"
    raw_path="${raw_path//<BASE_DIR>/${WORKSPACE_ROOT}}"
    if [[ "${raw_path}" != /* ]]; then
        raw_path="${args_dir}/${raw_path}"
    fi

    raw_dir="$(dirname "${raw_path}")"
    if [[ ! -d "${raw_dir}" ]]; then
        echo "${field_name} not found: ${raw_path}" >&2
        exit 1
    fi
    raw_path="$(cd "${raw_dir}" && pwd)/$(basename "${raw_path}")"
    if [[ ! -f "${raw_path}" ]]; then
        echo "${field_name} not found: ${raw_path}" >&2
        exit 1
    fi
    echo "${raw_path}"
}

resolve_data_path() {
    local raw_path="$1"

    if [[ -z "${raw_path}" || "${raw_path}" == "null" ]]; then
        echo ""
        return 0
    fi

    raw_path="${raw_path//<REPO_ROOT>/${REPO_ROOT}}"
    raw_path="${raw_path//<BASE_DIR>/${WORKSPACE_ROOT}}"
    if [[ "${raw_path}" != /* ]]; then
        raw_path="${REPO_ROOT}/${raw_path}"
    fi
    echo "${raw_path}"
}

read_train_config_value() {
    local key="$1"
    local value=""
    local override=""

    if [[ -n "${DEFAULT_TRAIN_CONFIG}" && "${DEFAULT_TRAIN_CONFIG}" != "null" ]]; then
        value=$(jq -r --arg key "${key}" '.[$key] // empty' "${DEFAULT_TRAIN_CONFIG}")
    fi
    if [[ -n "${BASE_TRAIN_CONFIG}" && "${BASE_TRAIN_CONFIG}" != "null" ]]; then
        override=$(jq -r --arg key "${key}" '.[$key] // empty' "${BASE_TRAIN_CONFIG}")
        if [[ -n "${override}" && "${override}" != "null" ]]; then
            value="${override}"
        fi
    fi
    echo "${value}"
}

args_file=${1:-${HPO_CONFIG:-${DEFAULT_HPO_CONFIG}}}
if [[ "${args_file}" != /* && ! -f "${args_file}" && -f "${REPO_ROOT}/${args_file}" ]]; then
    args_file="${REPO_ROOT}/${args_file}"
fi
if [[ -z "${args_file}" || ! -f "${args_file}" ]]; then
    echo "Usage: $0 [hpo_config.json]"
    echo "Default config: ${DEFAULT_HPO_CONFIG}"
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

    script_path="${SCRIPT_PATH}"

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

    mkdir -p "${REPO_ROOT}/logs/hpo"
    echo "Submitting job with: sbatch ${sbatch_opts[*]} ${script_path} ${args_file}"
    (
        cd "${REPO_ROOT}"
        sbatch "${sbatch_opts[@]}" "${script_path}" "${args_file}"
    )
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
TUNABLE_PARAMS=$(jq -r '.tunable_params // [] | join(", ")' "$args_file")
MAX_GALLERY_QUERIES=$(jq -r '.fixed_overrides.max_gallery_queries // .max_gallery_queries // "unset"' "$args_file")
GALLERY_TOPK_SPACE=$(jq -r '.search_space.gallery_hard_neg_topk.choices // [] | map(tostring) | join(", ")' "$args_file")

DEFAULT_TRAIN_CONFIG=$(jq -r '.default_train_config // empty' "$args_file")
BASE_TRAIN_CONFIG=$(jq -r '.base_train_config // empty' "$args_file")
DEFAULT_TRAIN_CONFIG=$(resolve_config_path "$DEFAULT_TRAIN_CONFIG" "default_train_config")
BASE_TRAIN_CONFIG=$(resolve_config_path "$BASE_TRAIN_CONFIG" "base_train_config")

DATA_PATH=$(jq -r '.data_path // empty' "$args_file")
NEG_DATA_PATH=$(jq -r '.neg_data_path // empty' "$args_file")
NEG_GROUP=$(jq -r '.neg_group // empty' "$args_file")
TEST_DATA_PATH=$(jq -r '.test_data_path // empty' "$args_file")

if [[ -z "$DATA_PATH" || "$DATA_PATH" == "null" ]]; then
    DATA_PATH=$(read_train_config_value "data_path")
fi
if [[ -z "$NEG_DATA_PATH" || "$NEG_DATA_PATH" == "null" ]]; then
    NEG_DATA_PATH=$(read_train_config_value "neg_data_path")
fi
if [[ -z "$NEG_GROUP" || "$NEG_GROUP" == "null" ]]; then
    NEG_GROUP=$(read_train_config_value "neg_group")
fi
if [[ -z "$TEST_DATA_PATH" || "$TEST_DATA_PATH" == "null" ]]; then
    TEST_DATA_PATH=$(read_train_config_value "test_data_path")
fi

DATA_PATH=$(resolve_data_path "$DATA_PATH")
NEG_DATA_PATH=$(resolve_data_path "$NEG_DATA_PATH")
TEST_DATA_PATH=$(resolve_data_path "$TEST_DATA_PATH")

if [[ -z "$DATA_PATH" || "$DATA_PATH" == "null" ]]; then
    echo "data_path missing in HPO config and train configs."
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
if [[ -n "$TUNABLE_PARAMS" ]]; then
    echo "  Tunable params: $TUNABLE_PARAMS"
fi
echo "  max_gallery_queries: $MAX_GALLERY_QUERIES"
if [[ -n "$GALLERY_TOPK_SPACE" ]]; then
    echo "  gallery_hard_neg_topk choices: $GALLERY_TOPK_SPACE"
fi
echo "  enable_ood_monitoring (config): $OOD_MONITORING"
echo "  enable_ood_monitoring (runtime override): false"
if [[ -n "$DEFAULT_TRAIN_CONFIG" && "$DEFAULT_TRAIN_CONFIG" != "null" ]]; then
    echo "  default_train_config (resolved): $DEFAULT_TRAIN_CONFIG"
fi
if [[ -n "$BASE_TRAIN_CONFIG" && "$BASE_TRAIN_CONFIG" != "null" ]]; then
    echo "  base_train_config (resolved): $BASE_TRAIN_CONFIG"
fi
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
  --arg default_train_config "$DEFAULT_TRAIN_CONFIG" \
  --arg base_train_config "$BASE_TRAIN_CONFIG" \
  --arg best_ckpt_metric "$BEST_CKPT_METRIC" \
  --argjson num_workers "${SLURM_CPUS_PER_TASK}" \
  '
  .data_path = $data_path
  | .num_workers = $num_workers
  | .best_ckpt_metric = $best_ckpt_metric
  | .enable_ood_monitoring = false
  | (if ($default_train_config | length) > 0 and $default_train_config != "null"
     then .default_train_config = $default_train_config
     else .
     end)
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
  ' \
  "$args_file" > "$RUNTIME_CONFIG"

# ─── Build command ─────────────────────────────────────────────────────
cmd=(
    python -u "${SCRIPT_DIR}/hpo_optuna.py"
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
    python -u "${SCRIPT_DIR}/hpo_analyze.py" \
        --study_name "$STUDY_NAME" \
        --storage "$STORAGE" \
        --output_dir "$OUTPUT_DIR/analysis" \
        --top_k 5
fi

exit $exit_code
