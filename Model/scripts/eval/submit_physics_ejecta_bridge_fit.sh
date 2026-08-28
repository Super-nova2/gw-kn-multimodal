#!/bin/bash
#SBATCH --job-name=PhysicsBridge_fit
#SBATCH --output=/fred/oz016/bgao_kn/logs/eval/%x_%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=08:00:00
#SBATCH --partition=skylake

set -euo pipefail

REPO_ROOT="/fred/oz016/bgao_kn/gw-kn-multimodal"
WORKSPACE_ROOT="/fred/oz016/bgao_kn"
SCRIPT_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/$(basename "${BASH_SOURCE[0]}")"
FIT_SCRIPT="${REPO_ROOT}/Model/scripts/eval/fit_physics_ejecta_bridge.py"
CONFIG_FILE="${1:-${REPO_ROOT}/Model/args/eval/physics_ejecta_bridge_fit.json}"
LOG_DIR="${WORKSPACE_ROOT}/logs/eval"

[[ -f "${CONFIG_FILE}" ]] || { echo "Config not found: ${CONFIG_FILE}" >&2; exit 1; }
CONFIG_FILE="$(cd "$(dirname "${CONFIG_FILE}")" && pwd)/$(basename "${CONFIG_FILE}")"
mkdir -p "${LOG_DIR}"

if [[ "${DRY_RUN:-false}" == "true" && -z "${SLURM_JOB_ID:-}" ]]; then
    python -u "${FIT_SCRIPT}" --config "${CONFIG_FILE}" --validate-only
    exit 0
fi

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    opts=(--output="${OUTPUT_LOG:-${LOG_DIR}/%x_%j.out}")
    [[ -n "${JOB_NAME:-}" ]] && opts+=(--job-name="${JOB_NAME}")
    [[ -n "${TIME_LIMIT:-}" ]] && opts+=(--time="${TIME_LIMIT}")
    [[ -n "${PARTITION:-}" ]] && opts+=(--partition="${PARTITION}")
    [[ -n "${MEM_PER_TASK:-}" ]] && opts+=(--mem="${MEM_PER_TASK}")
    echo "Submitting: sbatch ${opts[*]} $0 ${CONFIG_FILE}"
    cd "${WORKSPACE_ROOT}"
    sbatch "${opts[@]}" "${SCRIPT_PATH}" "${CONFIG_FILE}"
    exit 0
fi

echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: ${SLURMD_NODENAME:-unknown}"
echo "Config: ${CONFIG_FILE}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
echo "CPU threads: ${SLURM_CPUS_PER_TASK:-1}"
cd "${REPO_ROOT}"
python -u "${FIT_SCRIPT}" --config "${CONFIG_FILE}"
echo "Completed at $(date)"
