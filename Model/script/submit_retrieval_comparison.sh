#!/bin/bash

#SBATCH --job-name=ALBEF_retrieval_cmp
#SBATCH --output=/fred/oz016/bgao_kn/logs/eval/%x_%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=4:00:00
#SBATCH --partition=gpu

set -euo pipefail

SCRIPT_SUBDIR="Model/script"
SCRIPT_REL_PATH="Model/script/submit_retrieval_comparison.sh"
REPO_NAME="gw-kn-multimodal"
WORKSPACE_ROOT_DEFAULT="/fred/oz016/bgao_kn"
DEFAULT_CONFIG_REL="Model/args/eval/retrieval_comparison.json"

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

echo "========================================"
echo "SLURM Job Information"
echo "========================================"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Job Name: ${SLURM_JOB_NAME}"
echo "Node: ${SLURMD_NODENAME:-unknown}"
echo "Partition: ${SLURM_JOB_PARTITION:-unknown}"
echo "CPUs: ${SLURM_CPUS_PER_TASK:-unknown}"
echo "GPUs: ${CUDA_VISIBLE_DEVICES:-unknown}"
echo "Start time: $(date)"
echo "========================================"
echo

echo "Workspace root: ${WORKSPACE_ROOT}"
echo "Log dir: ${LOG_DIR}"
echo "Config: ${config_file}"
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
