#!/bin/bash

#SBATCH --job-name=GW170817A_LSST
#SBATCH --output=/fred/oz016/bgao_kn/logs/data/%x_%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=8:00:00

set -euo pipefail

SCRIPT_SUBDIR="dataset/GW170817A_lsst"
SCRIPT_REL_PATH="dataset/GW170817A_lsst/submit_gw170817a_lsst_pipeline.sh"
REPO_NAME="gw-kn-multimodal"
BASE_DIR="${BASE_DIR:-/fred/oz016/bgao_kn}"
WORKSPACE_ROOT="${WORKSPACE_ROOT:-${BASE_DIR}}"

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
GEN_SCRIPT="${SCRIPT_DIR}/generate_gw170817a_lsst_docs.py"
BUILD_SCRIPT="${SCRIPT_DIR}/build_gw170817a_retrieval_h5.py"
LOG_DIR="${WORKSPACE_ROOT}/logs/data"
DEFAULT_OUTPUT_LOG="${LOG_DIR}/%x_%j.out"

GW170817A_DIR="${GW170817A_DIR:-${BASE_DIR}/data/GW_real_events/GW_data/GW170817A}"
SKYMAP="${SKYMAP:-${GW170817A_DIR}/bayestar_no_virgo.fits}"
OUTPUT_DIR="${OUTPUT_DIR:-${GW170817A_DIR}/lsst_redshift_experiment_balanced}"
OPSIM_DB="${OPSIM_DB:-${BASE_DIR}/data/rubin_sim/baseline/baseline_v5.0.1_10yrs.db}"
GENVERSION="${GENVERSION:-LSST_KN_GW170817A_REDSHIFT_GRID_BALANCED}"
REDSHIFTS="${REDSHIFTS:-0.01,0.02,0.03,0.05,0.08,0.12,0.16}"
N_PER_REDSHIFT="${N_PER_REDSHIFT:-800,800,1200,1300,2000,3000,6000}"
CREDIBLE_LEVEL_MAX="${CREDIBLE_LEVEL_MAX:-0.9}"
SEED="${SEED:-170817}"
MIN_NOBS="${MIN_NOBS:-5}"
FLUXCAL_ZP="${FLUXCAL_ZP:-27.5}"
PSFFLUX_ZP="${PSFFLUX_ZP:-31.4}"
LUPT_K="${LUPT_K:-1.0}"
LUPT_M5_MAG="${LUPT_M5_MAG:-23.9,25.0,24.7,24.0,23.3,22.1}"
OUTPUT_H5="${OUTPUT_H5:-${BASE_DIR}/data/ALBEF_dataset/gw170817a_lsst_redshift_balanced_test.h5}"
MANIFEST="${MANIFEST:-${OUTPUT_DIR}/gw170817a_lsst_manifest.csv}"
INPUT_PATH="${INPUT_PATH:-${OUTPUT_DIR}/SIM_INPUT/SIMGEN_${GENVERSION}.INPUT}"
SNDATA_ROOT="${SNDATA_ROOT:-${BASE_DIR}/SNANA/SNDATA_ROOT}"
SIM_DIR="${SIM_DIR:-${SNDATA_ROOT}/SIM/${GENVERSION}}"
SNANA_BIN_DIR="${SNANA_BIN_DIR:-${BASE_DIR}/SNANA/SNANA/bin}"
STAGE="${STAGE:-all}"

if [[ ! -f "${SCRIPT_PATH}" ]]; then
    echo "Resolved script path not found: ${SCRIPT_PATH}" >&2
    exit 1
fi
if [[ ! -f "${GEN_SCRIPT}" ]]; then
    echo "Generator script not found: ${GEN_SCRIPT}" >&2
    exit 1
fi
if [[ ! -f "${BUILD_SCRIPT}" ]]; then
    echo "HDF5 builder script not found: ${BUILD_SCRIPT}" >&2
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
    if [[ -n "${NODES:-}" ]]; then
        sbatch_opts+=(--nodes="${NODES}")
    fi
    if [[ -n "${NTASKS:-}" ]]; then
        sbatch_opts+=(--ntasks="${NTASKS}")
    fi
    if [[ -n "${CHDIR:-}" ]]; then
        sbatch_opts+=(--chdir="${CHDIR}")
    fi

    echo "Submitting job with: sbatch ${sbatch_opts[*]} ${SCRIPT_PATH}"
    (
        cd "${WORKSPACE_ROOT}"
        sbatch "${sbatch_opts[@]}" "${SCRIPT_PATH}"
    )
    exit 0
fi

case "${STAGE}" in
    all|docs|sim|h5)
        ;;
    *)
        echo "Unsupported STAGE='${STAGE}'. Use all, docs, sim, or h5." >&2
        exit 1
        ;;
esac

if [[ ! -f "${SKYMAP}" ]]; then
    echo "GW170817A bayestar skymap not found: ${SKYMAP}" >&2
    exit 1
fi
if [[ "${STAGE}" == "all" || "${STAGE}" == "docs" ]]; then
    if [[ ! -f "${OPSIM_DB}" ]]; then
        echo "OpSim database not found: ${OPSIM_DB}" >&2
        exit 1
    fi
fi

export SNDATA_ROOT
export PATH="${SNANA_BIN_DIR}:${PATH}"

echo "========================================"
echo "GW170817A LSST Redshift Pipeline"
echo "========================================"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Job Name: ${SLURM_JOB_NAME:-unknown}"
echo "Node: ${SLURMD_NODENAME:-unknown}"
echo "Start time: $(date)"
echo "Stage: ${STAGE}"
echo "Repo root: ${REPO_ROOT}"
echo "Skymap: ${SKYMAP}"
echo "OpSim DB: ${OPSIM_DB}"
echo "Output dir: ${OUTPUT_DIR}"
echo "Manifest: ${MANIFEST}"
echo "GENVERSION: ${GENVERSION}"
echo "Redshifts: ${REDSHIFTS}"
echo "N per redshift: ${N_PER_REDSHIFT}"
echo "SNDATA_ROOT: ${SNDATA_ROOT}"
echo "SNANA SIM dir: ${SIM_DIR}"
echo "Output HDF5: ${OUTPUT_H5}"
echo "========================================"

cd "${REPO_ROOT}"

if [[ "${STAGE}" == "all" || "${STAGE}" == "docs" ]]; then
    echo "Generating manifest, SIMLIB, and SNANA INPUT..."
    python -u "${GEN_SCRIPT}" \
        --skymap "${SKYMAP}" \
        --opsim-db "${OPSIM_DB}" \
        --output-dir "${OUTPUT_DIR}" \
        --genversion "${GENVERSION}" \
        --redshifts "${REDSHIFTS}" \
        --n-per-redshift "${N_PER_REDSHIFT}" \
        --credible-level-max "${CREDIBLE_LEVEL_MAX}" \
        --seed "${SEED}"
fi

if [[ "${STAGE}" == "all" || "${STAGE}" == "sim" ]]; then
    if [[ ! -f "${INPUT_PATH}" ]]; then
        echo "SNANA INPUT not found: ${INPUT_PATH}" >&2
        exit 1
    fi
    if ! command -v snlc_sim.exe >/dev/null 2>&1; then
        echo "snlc_sim.exe not found in PATH. Checked SNANA_BIN_DIR=${SNANA_BIN_DIR}" >&2
        exit 1
    fi
    echo "Running SNANA: snlc_sim.exe ${INPUT_PATH}"
    snlc_sim.exe "${INPUT_PATH}"
fi

if [[ "${STAGE}" == "all" || "${STAGE}" == "h5" ]]; then
    if [[ ! -f "${MANIFEST}" ]]; then
        echo "Manifest not found: ${MANIFEST}" >&2
        exit 1
    fi
    echo "Building retrieval HDF5..."
    python -u "${BUILD_SCRIPT}" \
        --sim-dir "${SIM_DIR}" \
        --genversion "${GENVERSION}" \
        --manifest "${MANIFEST}" \
        --skymap "${SKYMAP}" \
        --output-h5 "${OUTPUT_H5}" \
        --min-nobs "${MIN_NOBS}" \
        --fluxcal-zp "${FLUXCAL_ZP}" \
        --psfflux-zp "${PSFFLUX_ZP}" \
        --lupt-k "${LUPT_K}" \
        --lupt-m5-mag "${LUPT_M5_MAG}"
fi

echo "----------------------------------------"
echo "End time: $(date)"
