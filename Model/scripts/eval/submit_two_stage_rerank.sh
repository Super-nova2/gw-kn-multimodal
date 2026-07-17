#!/bin/bash

#SBATCH --job-name=MAGIKS_two_stage
#SBATCH --output=/fred/oz016/bgao_kn/logs/eval/%x_%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=80G
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --partition=gpu
#SBATCH --tmp=200G

set -euo pipefail

SCRIPT_SUBDIR="Model/scripts/eval"
SCRIPT_REL_PATH="Model/scripts/eval/submit_two_stage_rerank.sh"
REPO_NAME="gw-kn-multimodal"
WORKSPACE_ROOT_DEFAULT="/fred/oz016/bgao_kn"
DEFAULT_CONFIG_REL="Model/args/eval/two_stage_rerank_g1000_top64.json"

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
MODEL_DIR="${REPO_ROOT}/Model"
SCRIPT_PATH="${REPO_ROOT}/${SCRIPT_REL_PATH}"
EVAL_SCRIPT="${SCRIPT_DIR}/eval_two_stage_rerank.py"
DEFAULT_CONFIG_PATH="${REPO_ROOT}/${DEFAULT_CONFIG_REL}"
WORKSPACE_ROOT="${WORKSPACE_ROOT:-${WORKSPACE_ROOT_DEFAULT}}"
LOG_DIR="${WORKSPACE_ROOT}/logs/eval"
DEFAULT_OUTPUT_LOG="${LOG_DIR}/%x_%j.out"

config_file=${1:-${DEFAULT_CONFIG_PATH}}
if [[ ! -f "${config_file}" ]]; then
    echo "Config not found: ${config_file}" >&2
    echo "Usage: $0 [two_stage_rerank.json]" >&2
    exit 1
fi
config_dir="$(cd "$(dirname "${config_file}")" && pwd)"
config_file="${config_dir}/$(basename "${config_file}")"

if [[ ! -f "${EVAL_SCRIPT}" ]]; then
    echo "Evaluation script not found: ${EVAL_SCRIPT}" >&2
    exit 1
fi

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

    echo "Submitting job with: sbatch ${sbatch_opts[*]} ${SCRIPT_PATH} ${config_file}"
    (
        cd "${WORKSPACE_ROOT}"
        sbatch "${sbatch_opts[@]}" "${SCRIPT_PATH}" "${config_file}"
    )
    exit 0
fi

if ! command -v jq >/dev/null 2>&1; then
    echo "jq not found; required to parse two-stage rerank config." >&2
    exit 1
fi

TEST_DATA_PATH=$(jq -r '.test_data_path // empty' "$config_file")
NEG_DATA_PATH=$(jq -r '.neg_data_path // empty' "$config_file")
OUTPUT_DIR=$(jq -r '.output_dir // empty' "$config_file")
STAGE_TO_JOBFS=$(jq -r '.stage_to_jobfs // false' "$config_file")

if [[ -z "${TEST_DATA_PATH}" || "${TEST_DATA_PATH}" == "null" || ! -f "${TEST_DATA_PATH}" ]]; then
    echo "Test dataset not found: ${TEST_DATA_PATH}" >&2
    exit 1
fi
if [[ -n "${NEG_DATA_PATH}" && "${NEG_DATA_PATH}" != "null" && ! -f "${NEG_DATA_PATH}" ]]; then
    echo "Negative dataset not found: ${NEG_DATA_PATH}" >&2
    exit 1
fi
if [[ -n "${OUTPUT_DIR}" && "${OUTPUT_DIR}" != "null" ]]; then
    mkdir -p "${OUTPUT_DIR}"
fi

echo "========================================"
echo "Two-Stage Rerank Evaluation"
echo "========================================"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: ${SLURMD_NODENAME:-unknown}"
echo "Config: ${config_file}"
echo "Test data: ${TEST_DATA_PATH}"
echo "Negative data: ${NEG_DATA_PATH:-none}"
echo "Output dir: ${OUTPUT_DIR:-default}"
echo "Start time: $(date)"
echo "========================================"

if [[ "${STAGE_TO_JOBFS}" == "true" ]]; then
    JOBFS_DIR="${SLURM_TMPDIR:-${TMPDIR:-${JOBFS:-}}}"
    if [[ -n "${JOBFS_DIR}" ]]; then
        echo "Staging data files to local disk: ${JOBFS_DIR}"
        cp -f "${TEST_DATA_PATH}" "${JOBFS_DIR}/"
        if [[ -n "${NEG_DATA_PATH}" && "${NEG_DATA_PATH}" != "null" ]]; then
            cp -f "${NEG_DATA_PATH}" "${JOBFS_DIR}/"
        fi
        export JOBFS_DIR
    else
        echo "No local tmp dir found; skip staging."
    fi
fi

python -u "${EVAL_SCRIPT}" --config "${config_file}"

echo "End time: $(date)"
