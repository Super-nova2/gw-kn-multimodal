#!/bin/bash

#SBATCH --job-name=BUILD_NEG_OPTICAL_H5
#SBATCH --output=/fred/oz016/bgao_kn/logs/data/%x_%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --partition=cpu

set -euo pipefail

BASE_DIR="${BASE_DIR:-/fred/oz016/bgao_kn}"
SCRIPT_SUBDIR="optical_only"
SCRIPT_REL_PATH="optical_only/submit_creat_optical_negative_dataset.sh"
REPO_NAME="gw-kn-multimodal"

# Required manual parameters. Define these before running, either by editing this
# block or exporting the same variable names in your shell.
NEG_SIM_ROOT="${NEG_SIM_ROOT:-/fred/oz016/bgao_kn/data/ELASTICC2_TRAIN_02}"
OUTPUT_NEG_DIR="${OUTPUT_NEG_DIR:-/fred/oz016/bgao_kn/data/Optical_Negative_dataset}"
OUTPUT_NEG_FILENAME="${OUTPUT_NEG_FILENAME:-ELASTICC2_negative_dataset.h5}"
NEG_GROUP="${NEG_GROUP:-ELASTICC2/optical_data}"
NEG_MATCH_POS_DENSITY="${NEG_MATCH_POS_DENSITY:-false}"
DENSITY_MATCH_POS_H5="${DENSITY_MATCH_POS_H5:-}"
BUFFER_LIMIT="${BUFFER_LIMIT:-3000}"
NUM_WORKERS="${NUM_WORKERS:-4}"
PYTHON_BIN="${PYTHON_BIN:-python}"

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
BUILD_SCRIPT="${SCRIPT_DIR}/create_optical_only_datasets.py"

usage() {
    cat <<'USAGE'
Usage:
  NEG_SIM_ROOT=/path/to/negative/root \
  OUTPUT_NEG_DIR=/path/to/output/dir \
  OUTPUT_NEG_FILENAME=negative_dataset.h5 \
  NEG_GROUP=SomeName/optical_data \
  NEG_MATCH_POS_DENSITY=false \
  bash gw-kn-multimodal/optical_only/submit_creat_optical_negative_dataset.sh

Required manual parameters:
  NEG_SIM_ROOT            Negative optical sample root directory.
  OUTPUT_NEG_DIR          Directory where the negative H5 will be written.
  OUTPUT_NEG_FILENAME     Output H5 file name, for example ELASTICC_negative.h5.
  NEG_GROUP               H5 group to create, for example ELASTICC/optical_data.
  NEG_MATCH_POS_DENSITY   true or false. Set false for standalone negative builds.

Conditionally required:
  DENSITY_MATCH_POS_H5    Positive H5 reference when NEG_MATCH_POS_DENSITY=true.

Optional:
  BUFFER_LIMIT            HDF5 write buffer size. Defaults to 3000.
  NUM_WORKERS             FITS parsing workers. Defaults to 4, capped by SLURM_CPUS_PER_TASK.
  PYTHON_BIN              Python executable. Defaults to python.
USAGE
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    usage
    exit 0
fi

if [[ "$#" -ne 0 ]]; then
    echo "This script does not accept positional arguments. Define variables before running." >&2
    usage >&2
    exit 2
fi

require_var() {
    local name value
    name="$1"
    value="${!name:-}"
    if [[ -z "${value}" ]]; then
        echo "Required parameter ${name} is not set." >&2
        usage >&2
        exit 1
    fi
}

normalize_bool() {
    local v
    v="$(echo "${1:-}" | tr '[:upper:]' '[:lower:]')"
    case "${v}" in
        1|true|yes|y|on) echo "true" ;;
        0|false|no|n|off) echo "false" ;;
        *)
            echo "Invalid boolean value '${1}'. Use true/false (or 1/0)." >&2
            return 1
            ;;
    esac
}

if [[ ! -f "${SCRIPT_PATH}" ]]; then
    echo "Resolved script path not found: ${SCRIPT_PATH}" >&2
    exit 1
fi
if [[ ! -f "${BUILD_SCRIPT}" ]]; then
    echo "Build script not found: ${BUILD_SCRIPT}" >&2
    exit 1
fi

require_var NEG_SIM_ROOT
require_var OUTPUT_NEG_DIR
require_var OUTPUT_NEG_FILENAME
require_var NEG_GROUP
require_var NEG_MATCH_POS_DENSITY

NEG_MATCH_POS_DENSITY="$(normalize_bool "${NEG_MATCH_POS_DENSITY}")"
OUTPUT_NEG_DIR="${OUTPUT_NEG_DIR%/}"
OUTPUT_NEG_H5="${OUTPUT_NEG_DIR}/${OUTPUT_NEG_FILENAME}"

if [[ "${OUTPUT_NEG_FILENAME}" == */* ]]; then
    echo "OUTPUT_NEG_FILENAME must be a file name only, not a path: ${OUTPUT_NEG_FILENAME}" >&2
    exit 1
fi
if [[ ! -d "${NEG_SIM_ROOT}" ]]; then
    echo "NEG_SIM_ROOT not found: ${NEG_SIM_ROOT}" >&2
    exit 1
fi
if [[ -n "${SLURM_CPUS_PER_TASK:-}" && "${NUM_WORKERS}" -gt "${SLURM_CPUS_PER_TASK}" ]]; then
    NUM_WORKERS="${SLURM_CPUS_PER_TASK}"
fi

if [[ "${NEG_MATCH_POS_DENSITY}" == "true" ]]; then
    require_var DENSITY_MATCH_POS_H5
    if [[ ! -f "${DENSITY_MATCH_POS_H5}" ]]; then
        echo "Density-match positive H5 not found: ${DENSITY_MATCH_POS_H5}" >&2
        exit 1
    fi
fi

# Self-submit: run this script directly on a Slurm login node to submit the job.
if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    if ! command -v sbatch >/dev/null 2>&1; then
        echo "sbatch not found. Please run on a Slurm login node." >&2
        exit 1
    fi
    mkdir -p "${BASE_DIR}/logs/data" "${OUTPUT_NEG_DIR}"
    echo "Submitting: sbatch --export=ALL ${SCRIPT_PATH}"
    (
        cd "${REPO_ROOT}"
        sbatch --export=ALL "${SCRIPT_PATH}"
    )
    exit 0
fi

mkdir -p "${OUTPUT_NEG_DIR}" "${BASE_DIR}/logs/data"

LOCK_FILE="${OUTPUT_NEG_H5}.lock"
exec 200>"${LOCK_FILE}"
if ! flock -n 200; then
    echo "Another negative optical build is already running for output: ${OUTPUT_NEG_H5}" >&2
    echo "Lock file: ${LOCK_FILE}" >&2
    exit 1
fi

cmd=(
    "${PYTHON_BIN}" -u "${BUILD_SCRIPT}"
    --build_negative
    --output_neg_h5 "${OUTPUT_NEG_H5}"
    --neg_group "${NEG_GROUP}"
    --neg_sim_root "${NEG_SIM_ROOT}"
    --buffer_limit "${BUFFER_LIMIT}"
    --num_workers "${NUM_WORKERS}"
    --neg_match_pos_density "${NEG_MATCH_POS_DENSITY}"
)

if [[ "${NEG_MATCH_POS_DENSITY}" == "true" ]]; then
    cmd+=(--density_match_pos_h5 "${DENSITY_MATCH_POS_H5}")
fi

echo "========================================"
echo "Slurm Job: ${SLURM_JOB_ID}"
echo "Node: ${SLURMD_NODENAME:-unknown}"
echo "Start: $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
echo "Build script: ${BUILD_SCRIPT}"
echo "Negative sim root: ${NEG_SIM_ROOT}"
echo "Output NEG: ${OUTPUT_NEG_H5}"
echo "Negative group: ${NEG_GROUP}"
echo "Buffer limit: ${BUFFER_LIMIT}"
echo "Workers: ${NUM_WORKERS}"
echo "Input scan: recursive *_HEAD.FITS(.gz) under NEG_SIM_ROOT; KN/BNS/NSBH token folders are skipped by the builder"
echo "Density matching: ${NEG_MATCH_POS_DENSITY}"
if [[ "${NEG_MATCH_POS_DENSITY}" == "true" ]]; then
    echo "Density ref POS H5: ${DENSITY_MATCH_POS_H5}"
fi
echo "Command: ${cmd[*]}"
echo "========================================"

cd "${REPO_ROOT}"
"${cmd[@]}"

echo "========================================"
echo "Finished: $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
echo "Saved NEG: ${OUTPUT_NEG_H5}"
echo "========================================"
