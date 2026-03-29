#!/bin/bash

#SBATCH --job-name=FD_DELAY_SNR
#SBATCH --output=<BASE_DIR>/logs/optical_only/%x_%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=8G
#SBATCH --time=04:00:00
#SBATCH --partition=cpu

set -euo pipefail

SCRIPT_SUBDIR="optical_only"
SCRIPT_REL_PATH="optical_only/submit_snana_first_detection_delay_bns_nsbh.sh"
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
PYTHON_SCRIPT="${SCRIPT_DIR}/snana_first_detection_delay_bns_nsbh.py"
if [[ ! -f "${PYTHON_SCRIPT}" ]]; then
    echo "Resolved python script not found: ${PYTHON_SCRIPT}" >&2
    exit 1
fi

mkdir -p <BASE_DIR>/logs/optical_only
PYTHON_BIN="${PYTHON_BIN:-python}"
export MPLBACKEND=Agg

cd "${BASE_DIR:-.}"
which "${PYTHON_BIN}"
"${PYTHON_BIN}" --version

echo "Running: ${PYTHON_BIN} -u ${PYTHON_SCRIPT} --num-workers ${SLURM_CPUS_PER_TASK:-8} $*"
"${PYTHON_BIN}" -u "${PYTHON_SCRIPT}" --num-workers "${SLURM_CPUS_PER_TASK:-8}" "$@"
