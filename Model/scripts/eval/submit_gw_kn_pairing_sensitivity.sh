#!/bin/bash

#SBATCH --job-name=MAGIKS_pair_sens
#SBATCH --output=/fred/oz016/bgao_kn/logs/eval/%x_%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --gres=gpu:a100:1
#SBATCH --time=04:00:00
#SBATCH --partition=gpu
#SBATCH --tmp=80G

set -euo pipefail

SCRIPT_SUBDIR="Model/scripts/eval"
REPO_NAME="gw-kn-multimodal"
WORKSPACE_ROOT_DEFAULT="/fred/oz016/bgao_kn"
DEFAULT_CONFIG_REL="Model/args/eval/gw_kn_pairing_sensitivity.json"

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

SCRIPT_PATH="${REPO_ROOT}/${SCRIPT_SUBDIR}/submit_gw_kn_pairing_sensitivity.sh"
EVAL_SCRIPT="${REPO_ROOT}/${SCRIPT_SUBDIR}/eval_gw_kn_pairing_sensitivity.py"
WORKSPACE_ROOT="${WORKSPACE_ROOT:-${WORKSPACE_ROOT_DEFAULT}}"
LOG_DIR="${WORKSPACE_ROOT}/logs/eval"
CONFIG_FILE="${1:-${REPO_ROOT}/${DEFAULT_CONFIG_REL}}"

if [[ ! -f "${CONFIG_FILE}" ]]; then
    echo "Config not found: ${CONFIG_FILE}" >&2
    echo "Generate it from Model/args/eval/gw_kn_pairing_sensitivity.json.example." >&2
    exit 1
fi
CONFIG_FILE="$(cd "$(dirname "${CONFIG_FILE}")" && pwd)/$(basename "${CONFIG_FILE}")"
mkdir -p "${LOG_DIR}"

if [[ "${DRY_RUN:-false}" == "true" && -z "${SLURM_JOB_ID:-}" ]]; then
    python -u "${EVAL_SCRIPT}" --config "${CONFIG_FILE}" --validate-only
    exit 0
fi

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    opts=(--output="${OUTPUT_LOG:-${LOG_DIR}/%x_%j.out}")
    [[ -n "${JOB_NAME:-}" ]] && opts+=(--job-name="${JOB_NAME}")
    [[ -n "${TIME_LIMIT:-}" ]] && opts+=(--time="${TIME_LIMIT}")
    [[ -n "${PARTITION:-}" ]] && opts+=(--partition="${PARTITION}")
    [[ -n "${MEM_PER_TASK:-}" ]] && opts+=(--mem="${MEM_PER_TASK}")
    echo "Submitting: sbatch ${opts[*]} ${SCRIPT_PATH} ${CONFIG_FILE}"
    cd "${WORKSPACE_ROOT}"
    sbatch "${opts[@]}" "${SCRIPT_PATH}" "${CONFIG_FILE}"
    exit 0
fi

command -v jq >/dev/null
TEST_DATA_PATH="$(jq -r '.test_data_path // empty' "${CONFIG_FILE}")"
OUTPUT_DIR="$(jq -r '.output_dir // empty' "${CONFIG_FILE}")"
STAGE_TO_JOBFS="$(jq -r '.stage_to_jobfs // false' "${CONFIG_FILE}")"
[[ -f "${TEST_DATA_PATH}" ]] || { echo "Dataset not found: ${TEST_DATA_PATH}" >&2; exit 1; }
[[ -n "${OUTPUT_DIR}" ]] || { echo "output_dir is required" >&2; exit 1; }

echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: ${SLURMD_NODENAME:-unknown}"
echo "Config: ${CONFIG_FILE}"
echo "Output: ${OUTPUT_DIR}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

if [[ "${STAGE_TO_JOBFS}" == "true" ]]; then
    JOBFS_BASE="${SLURM_TMPDIR:-${TMPDIR:-${JOBFS:-}}}"
    [[ -n "${JOBFS_BASE}" && -d "${JOBFS_BASE}" && -w "${JOBFS_BASE}" ]] || {
        echo "No writable job-local temporary directory" >&2
        exit 1
    }
    JOBFS_DIR="$(mktemp -d "${JOBFS_BASE%/}/magiks_pair_sens_${SLURM_JOB_ID}.XXXXXX")"
    cleanup_jobfs() {
        local prefix="${JOBFS_BASE%/}/magiks_pair_sens_${SLURM_JOB_ID}."
        if [[ -n "${JOBFS_DIR:-}" && "${JOBFS_DIR}" == "${prefix}"* && -d "${JOBFS_DIR}" ]]; then
            rm -rf -- "${JOBFS_DIR}"
        fi
    }
    trap cleanup_jobfs EXIT
    cp -f "${TEST_DATA_PATH}" "${JOBFS_DIR}/"
    export JOBFS_DIR
    echo "Staged test dataset to ${JOBFS_DIR}"
fi

cd "${REPO_ROOT}"
python -u "${EVAL_SCRIPT}" --config "${CONFIG_FILE}"
echo "Completed at $(date)"
