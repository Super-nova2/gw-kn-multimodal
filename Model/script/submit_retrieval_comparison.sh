#!/bin/bash

#SBATCH --job-name=ALBEF_retrieval_cmp
#SBATCH --output=/fred/oz016/bgao_kn/logs/eval/%x_%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=160G
#SBATCH --gres=gpu:1
#SBATCH --time=16:00:00
#SBATCH --partition=gpu
#SBATCH --tmp=100G

set -euo pipefail

SCRIPT_SUBDIR="Model/script"
SCRIPT_REL_PATH="Model/script/submit_retrieval_comparison.sh"
REPO_NAME="gw-kn-multimodal"
WORKSPACE_ROOT_DEFAULT="/fred/oz016/bgao_kn"
DEFAULT_CONFIG_REL="Model/args/eval/retrieval_comparison_v11.json"

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
MODEL_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
SCRIPT_PATH="${REPO_ROOT}/${SCRIPT_REL_PATH}"
EVAL_SCRIPT="${MODEL_DIR}/script/eval_retrieval_comparison.py"
DEFAULT_CONFIG_PATH="${REPO_ROOT}/${DEFAULT_CONFIG_REL}"
DEFAULT_OUTPUT_DIR="${MODEL_DIR}/eval_results/ablation_comparison"
WORKSPACE_ROOT="${WORKSPACE_ROOT:-${WORKSPACE_ROOT_DEFAULT}}"
LOG_DIR="${WORKSPACE_ROOT}/logs/eval"
DEFAULT_OUTPUT_LOG="${LOG_DIR}/%x_%j.out"

if [[ ! -f "${SCRIPT_PATH}" ]]; then
    echo "Resolved script path not found: ${SCRIPT_PATH}" >&2
    exit 1
fi
if [[ ! -f "${EVAL_SCRIPT}" ]]; then
    echo "Evaluation script not found: ${EVAL_SCRIPT}" >&2
    exit 1
fi

config_file=${1:-${DEFAULT_CONFIG_PATH}}
if [[ ! -f "${config_file}" ]]; then
    echo "Config not found: ${config_file}" >&2
    echo "Usage: $0 [retrieval_comparison.json]" >&2
    exit 1
fi
config_dir="$(cd "$(dirname "${config_file}")" && pwd)"
config_file="${config_dir}/$(basename "${config_file}")"

mkdir -p "${LOG_DIR}"

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    if ! command -v sbatch >/dev/null 2>&1; then
        echo "sbatch not found; run inside a Slurm allocation or install Slurm tools." >&2
        exit 1
    fi

    sbatch_opts=()
    if [[ -n "${JOB_NAME:-}" ]]; then
        sbatch_opts+=(--job-name="${JOB_NAME}")
    fi
    if [[ -n "${OUTPUT_LOG:-}" ]]; then
        sbatch_opts+=(--output="${OUTPUT_LOG}")
    else
        sbatch_opts+=(--output="${DEFAULT_OUTPUT_LOG}")
    fi
    if [[ -n "${TIME_LIMIT:-}" ]]; then
        sbatch_opts+=(--time="${TIME_LIMIT}")
    fi
    if [[ -n "${PARTITION:-}" ]]; then
        sbatch_opts+=(--partition="${PARTITION}")
    fi
    if [[ -n "${CPUS_PER_TASK:-}" ]]; then
        sbatch_opts+=(--cpus-per-task="${CPUS_PER_TASK}")
    fi
    if [[ -n "${MEM_PER_TASK:-}" ]]; then
        sbatch_opts+=(--mem="${MEM_PER_TASK}")
    fi
    if [[ -n "${GPUS:-}" ]]; then
        sbatch_opts+=(--gres="gpu:${GPUS}")
    fi
    if [[ -n "${NODES:-}" ]]; then
        sbatch_opts+=(--nodes="${NODES}")
    fi
    if [[ -n "${NTASKS:-}" ]]; then
        sbatch_opts+=(--ntasks="${NTASKS}")
    fi
    if [[ -n "${CHDIR:-}" ]]; then
        sbatch_opts+=(--chdir="${CHDIR}")
    fi

    echo "Submitting job with: sbatch ${sbatch_opts[*]} ${SCRIPT_PATH} ${config_file}"
    (
        cd "${WORKSPACE_ROOT}"
        sbatch "${sbatch_opts[@]}" "${SCRIPT_PATH}" "${config_file}"
    )
    exit 0
fi

which python

if ! command -v jq >/dev/null 2>&1; then
    echo "jq not found; required to parse retrieval comparison config." >&2
    exit 1
fi

TEST_DATA_PATH=$(jq -r '.test_data_path // empty' "$config_file")
NEG_DATA_PATH=$(jq -r '.neg_data_path // empty' "$config_file")
NEG_GROUP=$(jq -r '.neg_group // empty' "$config_file")
OUTPUT_DIR=$(jq -r '.output_dir // empty' "$config_file")
DEVICE=$(jq -r '.device // "cuda"' "$config_file")
AMP_DTYPE=$(jq -r '.amp_dtype // empty' "$config_file")
GALLERY_SIZES=$(jq -r '.gallery_sizes // empty' "$config_file")
GALLERY_TRIALS=$(jq -r '.gallery_trials // empty' "$config_file")
GALLERY_CANDIDATE_MODE=$(jq -r '.gallery_candidate_mode // "time_sky_hard"' "$config_file")
GALLERY_CANDIDATE_TIME_WINDOW_DAYS=$(jq -r '.gallery_candidate_time_window_days // empty' "$config_file")
GALLERY_CANDIDATE_CREDIBLE_LEVEL_MAX=$(jq -r '.gallery_candidate_credible_level_max // empty' "$config_file")
GALLERY_INCLUDE_UNDERSIZED=$(jq -r '.gallery_include_undersized // "true"' "$config_file")
STAGE_TO_JOBFS=$(jq -r '.stage_to_jobfs // false' "$config_file")
TUTORIAL_NEG_DATA_PATH=$(jq -r '.tutorial_neg_data_path // empty' "$config_file")
MODEL_NAMES=$(jq -r '[.models[].name] | join(", ")' "$config_file")
MODEL_COUNT=$(jq -r '.models | length' "$config_file")

if [[ -z "$TEST_DATA_PATH" || "$TEST_DATA_PATH" == "null" ]]; then
    echo "Required field missing in config: test_data_path" >&2
    exit 1
fi
if [[ ! -f "$TEST_DATA_PATH" ]]; then
    echo "Test dataset not found: $TEST_DATA_PATH" >&2
    exit 1
fi
if [[ -n "$NEG_DATA_PATH" && "$NEG_DATA_PATH" != "null" && ! -f "$NEG_DATA_PATH" ]]; then
    echo "Negative dataset not found: $NEG_DATA_PATH" >&2
    exit 1
fi
if [[ -z "$OUTPUT_DIR" || "$OUTPUT_DIR" == "null" ]]; then
    OUTPUT_DIR="${DEFAULT_OUTPUT_DIR}"
fi
mkdir -p "${OUTPUT_DIR}"

echo "========================================"
echo "SLURM Job Information"
echo "========================================"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Job Name: ${SLURM_JOB_NAME:-unknown}"
echo "Node: ${SLURMD_NODENAME:-unknown}"
echo "Partition: ${SLURM_JOB_PARTITION:-unknown}"
echo "CPUs: ${SLURM_CPUS_PER_TASK:-unknown}"
echo "Memory: ${SLURM_MEM_PER_NODE:-unknown}MB"
echo "GPUs: ${CUDA_VISIBLE_DEVICES:-unknown}"
echo "Start time: $(date)"
echo "========================================"
echo

echo "Workspace root: ${WORKSPACE_ROOT}"
echo "Log dir: ${LOG_DIR}"
echo "Config: ${config_file}"
echo "Test data: ${TEST_DATA_PATH}"
echo "Negative data: ${NEG_DATA_PATH:-none}"
echo "Negative group: ${NEG_GROUP:-none}"
echo "Output dir: ${OUTPUT_DIR}"
echo "Device: ${DEVICE}"
echo "AMP dtype: ${AMP_DTYPE:-auto}"
echo "Models (${MODEL_COUNT}): ${MODEL_NAMES}"
echo "Gallery sizes: ${GALLERY_SIZES:-default}"
echo "Gallery trials: ${GALLERY_TRIALS:-default}"
echo "Gallery candidate mode: ${GALLERY_CANDIDATE_MODE}"
if [[ -n "${GALLERY_CANDIDATE_TIME_WINDOW_DAYS}" && "${GALLERY_CANDIDATE_TIME_WINDOW_DAYS}" != "null" ]]; then
    echo "Candidate time window: +/-${GALLERY_CANDIDATE_TIME_WINDOW_DAYS} days"
fi
if [[ -n "${GALLERY_CANDIDATE_CREDIBLE_LEVEL_MAX}" && "${GALLERY_CANDIDATE_CREDIBLE_LEVEL_MAX}" != "null" ]]; then
    echo "Candidate credible max: ${GALLERY_CANDIDATE_CREDIBLE_LEVEL_MAX}"
fi
echo "Include undersized galleries: ${GALLERY_INCLUDE_UNDERSIZED}"
echo "Expected outputs:"
echo "  ${OUTPUT_DIR}/ablation_comparison.json"
echo "  ${OUTPUT_DIR}/retrieval_curves.png"
echo "  ${OUTPUT_DIR}/retrieval_coverage.png"
if [[ -n "${redshift_analysis_enable:-}" && "${redshift_analysis_enable}" != "false" ]]; then
    echo "  ${OUTPUT_DIR}/redshift_metrics.csv"
fi

# --- checkpoint pre-check ---
DRY_RUN="${DRY_RUN:-false}"
ALLOW_MISSING="${ALLOW_MISSING:-false}"

check_checkpoints() {
    local missing=0
    while IFS= read -r entry; do
        local name=$(echo "$entry" | jq -r '.name')
        local ckpt=$(echo "$entry" | jq -r '.checkpoint // empty')
        local type=$(echo "$entry" | jq -r '.type')
        if [[ "$type" == "skymap" ]]; then continue; fi
        if [[ -z "$ckpt" || "$ckpt" == "null" ]]; then
            echo "MISSING checkpoint path: $name (type=$type)"
            missing=1
            continue
        fi
        if [[ ! -d "$ckpt" && ! -f "$ckpt" ]]; then
            echo "MISSING: $name → $ckpt"
            missing=1
        fi
    done < <(jq -c '.models[]' "$config_file")
    if [[ "$missing" -eq 1 ]]; then
        if [[ "$ALLOW_MISSING" == "true" ]]; then
            echo "WARNING: Some checkpoints missing, continuing (ALLOW_MISSING=true)"
        else
            echo "ERROR: Missing checkpoints. Set ALLOW_MISSING=true to bypass, or DRY_RUN=true for config-only check."
            exit 1
        fi
    else
        echo "All checkpoints found."
    fi
}

check_checkpoints

if [[ "$DRY_RUN" == "true" ]]; then
    echo "DRY_RUN=true: Config and checkpoint check complete. Exiting without running eval."
    exit 0
fi
# --- end checkpoint pre-check ---

if [[ "$STAGE_TO_JOBFS" == "true" ]]; then
    JOBFS_DIR="${SLURM_TMPDIR:-${TMPDIR:-${JOBFS:-}}}"
    if [[ -n "$JOBFS_DIR" ]]; then
        echo "Staging data files to local disk: $JOBFS_DIR"
        cp -f "$TEST_DATA_PATH" "$JOBFS_DIR"/
        TEST_DATA_PATH="$JOBFS_DIR/$(basename "$TEST_DATA_PATH")"

        if [[ -n "$NEG_DATA_PATH" && "$NEG_DATA_PATH" != "null" ]]; then
            cp -f "$NEG_DATA_PATH" "$JOBFS_DIR"/
            NEG_DATA_PATH="$JOBFS_DIR/$(basename "$NEG_DATA_PATH")"
        fi
        if [[ -n "$TUTORIAL_NEG_DATA_PATH" && "$TUTORIAL_NEG_DATA_PATH" != "null" ]]; then
            cp -f "$TUTORIAL_NEG_DATA_PATH" "$JOBFS_DIR"/
        fi
        if [[ -n "${NEG_OFFSET_DIST_NPZ:-}" && "${NEG_OFFSET_DIST_NPZ}" != "null" ]]; then
            cp -f "$NEG_OFFSET_DIST_NPZ" "$JOBFS_DIR"/
        fi
        export JOBFS_DIR
        echo "Staging complete. Using JOBFS_DIR=$JOBFS_DIR"
    else
        echo "No local tmp dir found; skip staging."
    fi
fi

echo "Command: python -u ${EVAL_SCRIPT} --config ${config_file}"

cd "${REPO_ROOT}"
python -u "${EVAL_SCRIPT}" --config "${config_file}"

exit_code=$?
if [[ ${exit_code} -ne 0 ]]; then
    echo "Retrieval comparison failed with exit code ${exit_code}" >&2
    exit ${exit_code}
fi

echo "----------------------------------------"
echo "End time: $(date)"
