#!/bin/bash
#SBATCH --job-name=MAGIKS_mixed_ret
#SBATCH --output=/fred/oz016/bgao_kn/logs/eval/%x_%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=100G
#SBATCH --gres=gpu:a100:1
#SBATCH --time=24:00:00
#SBATCH --partition=gpu
#SBATCH --tmp=250G

set -euo pipefail

REPO_ROOT="/fred/oz016/bgao_kn/gw-kn-multimodal"
EVAL_SCRIPT="${REPO_ROOT}/Model/scripts/eval/eval_mixed_retrieval_comparison.py"
SUBMIT_SCRIPT="${REPO_ROOT}/Model/scripts/eval/submit_mixed_retrieval_comparison.sh"
DEFAULT_CONFIG="${REPO_ROOT}/Model/args/eval/retrieval_comparison_mixed_kn_nonkn.json"
LOG_DIR="/fred/oz016/bgao_kn/logs/eval"
CONFIG_FILE="${1:-${DEFAULT_CONFIG}}"

if [[ ! -f "${CONFIG_FILE}" ]]; then
    echo "Config not found: ${CONFIG_FILE}" >&2
    exit 1
fi
CONFIG_FILE="$(cd "$(dirname "${CONFIG_FILE}")" && pwd)/$(basename "${CONFIG_FILE}")"
mkdir -p "${LOG_DIR}"

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    opts=(--output="${OUTPUT_LOG:-${LOG_DIR}/%x_%j.out}")
    [[ -n "${JOB_NAME:-}" ]] && opts+=(--job-name="${JOB_NAME}")
    [[ -n "${TIME_LIMIT:-}" ]] && opts+=(--time="${TIME_LIMIT}")
    [[ -n "${DEPENDENCY:-}" ]] && opts+=(--dependency="${DEPENDENCY}")
    echo "Submitting: sbatch ${opts[*]} ${SUBMIT_SCRIPT} ${CONFIG_FILE}"
    cd "/fred/oz016/bgao_kn"
    sbatch "${opts[@]}" "${SUBMIT_SCRIPT}" "${CONFIG_FILE}"
    exit 0
fi

command -v jq >/dev/null
TEST_DATA_PATH="$(jq -r '.test_data_path' "${CONFIG_FILE}")"
NEG_DATA_PATH="$(jq -r '.neg_data_path' "${CONFIG_FILE}")"
OUTPUT_DIR="$(jq -r '.output_dir' "${CONFIG_FILE}")"
STAGE_TO_JOBFS="$(jq -r '.stage_to_jobfs // false' "${CONFIG_FILE}")"
for path in "${TEST_DATA_PATH}" "${NEG_DATA_PATH}"; do
    [[ -f "${path}" ]] || { echo "Dataset not found: ${path}" >&2; exit 1; }
done

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
    JOBFS_DIR="$(mktemp -d "${JOBFS_BASE%/}/magiks_mixed_${SLURM_JOB_ID}.XXXXXX")"
    cleanup_jobfs() {
        local prefix="${JOBFS_BASE%/}/magiks_mixed_${SLURM_JOB_ID}."
        if [[ -n "${JOBFS_DIR:-}" && "${JOBFS_DIR}" == "${prefix}"* && -d "${JOBFS_DIR}" ]]; then
            rm -rf -- "${JOBFS_DIR}"
        fi
    }
    trap cleanup_jobfs EXIT
    cp -f "${TEST_DATA_PATH}" "${JOBFS_DIR}/"
    cp -f "${NEG_DATA_PATH}" "${JOBFS_DIR}/"
    export JOBFS_DIR
    echo "Staged datasets to ${JOBFS_DIR}"
fi

cd "${REPO_ROOT}"
python -u "${EVAL_SCRIPT}" --config "${CONFIG_FILE}"
echo "Completed at $(date)"
