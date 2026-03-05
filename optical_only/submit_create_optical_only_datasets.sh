#!/bin/bash

#SBATCH --job-name=BUILD_OPTICAL_ONLT_DATASET
#SBATCH --output=logs/data/%x_%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --time=8:00:00
#SBATCH --partition=cpu

set -euo pipefail

DATASET_MODE="${DATASET_MODE:-train}"   # train | test
BUILD_POSITIVE="${BUILD_POSITIVE:-true}" # true|false
BUILD_NEGATIVE="${BUILD_NEGATIVE:-true}" # true|false

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

BUILD_POSITIVE="$(normalize_bool "${BUILD_POSITIVE}")"
BUILD_NEGATIVE="$(normalize_bool "${BUILD_NEGATIVE}")"
if [[ "${BUILD_POSITIVE}" != "true" && "${BUILD_NEGATIVE}" != "true" ]]; then
    echo "Nothing to build: BUILD_POSITIVE=${BUILD_POSITIVE}, BUILD_NEGATIVE=${BUILD_NEGATIVE}"
    echo "Set at least one of BUILD_POSITIVE/BUILD_NEGATIVE to true."
    exit 1
fi

# Self-submit: run this script directly to submit job to Slurm.
if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    if ! command -v sbatch >/dev/null 2>&1; then
        echo "sbatch not found. Please run on a Slurm login node."
        exit 1
    fi

    script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    script_path="${script_dir}/$(basename "${BASH_SOURCE[0]}")"

    mkdir -p /fred/oz016/bgao_kn/logs/optical_only
    echo "Submitting: sbatch ${script_path}"
    sbatch "${script_path}"
    exit 0
fi

# ------------------------------
# Hard-coded parameters
# ------------------------------
PYTHON_BIN="python"
BUILD_SCRIPT="/fred/oz016/bgao_kn/ML+GW+KN/optical_only/create_optical_only_datasets.py"

case "${DATASET_MODE}" in
    train)
        BNS_SIM_ROOT="/fred/oz016/bgao_kn/SNANA/SNDATA_ROOT/SIM/LSST_KN_BNS_AUG"
        BNS_SIM_NAME="LSST_KN_BNS_AUG"

        NSBH_SIM_ROOT="/fred/oz016/bgao_kn/SNANA/SNDATA_ROOT/SIM/LSST_KN_NSBH_TRAIN"
        NSBH_SIM_NAME="LSST_KN_NSBH_TRAIN"

        NEG_SIM_ROOT="/fred/oz016/bgao_kn/data/ELASTICC2_TRAIN_02"
        NEG_GROUP="ELASTICC2/optical_data"

        OUTPUT_POS_H5="/fred/oz016/bgao_kn/data/Optical_Only_dataset/combined_dataset_train.h5"
        OUTPUT_NEG_H5="/fred/oz016/bgao_kn/data/Optical_Only_dataset/ELASTICC2_negative_dataset.h5"
        CLS_TIME_ANCHOR_GW_H5_DEFAULT="/fred/oz016/bgao_kn/data/ALBEF_dataset/combined_dataset_train.h5"
        ;;
    test)
        BNS_SIM_ROOT="/fred/oz016/bgao_kn/SNANA/SNDATA_ROOT/SIM/LSST_KN_BNS"
        BNS_SIM_NAME="LSST_KN_BNS"

        NSBH_SIM_ROOT="/fred/oz016/bgao_kn/SNANA/SNDATA_ROOT/SIM/LSST_KN_NSBH_AUG"
        NSBH_SIM_NAME="LSST_KN_NSBH_AUG"

        NEG_SIM_ROOT="/fred/oz016/bgao_kn/data/Tutorial_LSST_sims_2025"
        NEG_GROUP="Tutorial/optical_data"

        OUTPUT_POS_H5="/fred/oz016/bgao_kn/data/Optical_Only_dataset/combined_dataset_test.h5"
        OUTPUT_NEG_H5="/fred/oz016/bgao_kn/data/Optical_Only_dataset/Tutorial_negative_dataset.h5"
        # C2: use a global GW-time prior from training set for cls base anchoring.
        CLS_TIME_ANCHOR_GW_H5_DEFAULT="/fred/oz016/bgao_kn/data/ALBEF_dataset/combined_dataset_train.h5"
        ;;
    *)
        echo "Unsupported DATASET_MODE='${DATASET_MODE}'. Use train or test."
        exit 1
        ;;
esac

CLS_TIME_ANCHOR_GW_H5="${CLS_TIME_ANCHOR_GW_H5:-$CLS_TIME_ANCHOR_GW_H5_DEFAULT}"
CLS_TIME_ANCHOR_SEED="${CLS_TIME_ANCHOR_SEED:-42}"

MIN_NOBS=5
SNR_THRESHOLD=5.0
BUFFER_LIMIT=3000
FIXED_OFFSET_DAYS=0.0
MAX_LCS_PER_EVENT=1000
FLUXCAL_ZP="${FLUXCAL_ZP:-27.5}"
PSFFLUX_ZP="${PSFFLUX_ZP:-31.4}"
LUPT_K="${LUPT_K:-1.0}"
LUPT_M5_MAG="${LUPT_M5_MAG:-23.9,25.0,24.7,24.0,23.3,22.1}"
# Conservative default to reduce worker crashes on large FITS parsing.
NUM_WORKERS="${NUM_WORKERS:-4}"
if [[ -n "${SLURM_CPUS_PER_TASK:-}" && "${NUM_WORKERS}" -gt "${SLURM_CPUS_PER_TASK}" ]]; then
    NUM_WORKERS="${SLURM_CPUS_PER_TASK}"
fi

if [[ -z "${LUPT_M5_MAG}" ]]; then
    echo "LUPT_M5_MAG is required (6 comma-separated m5 values in order u,g,r,i,z,Y)."
    echo "Example: LUPT_M5_MAG='23.9,25.0,24.7,24.0,23.3,22.1'"
    exit 1
fi

if [[ "${BUILD_POSITIVE}" == "true" ]]; then
    mkdir -p "$(dirname "${OUTPUT_POS_H5}")"
fi
if [[ "${BUILD_NEGATIVE}" == "true" ]]; then
    mkdir -p "$(dirname "${OUTPUT_NEG_H5}")"
fi

# Prevent concurrent jobs from writing the same dataset outputs.
LOCK_FILE="/fred/oz016/bgao_kn/data/Optical_Only_dataset/.build_optical_only_${DATASET_MODE}.lock"
exec 200>"${LOCK_FILE}"
if ! flock -n 200; then
    echo "Another build job is already running for DATASET_MODE=${DATASET_MODE}."
    echo "Lock file: ${LOCK_FILE}"
    echo "If this is unexpected, check active jobs and remove stale lock after confirming no writer is running."
    exit 1
fi

echo "========================================"
echo "Slurm Job: ${SLURM_JOB_ID}"
echo "Node: ${SLURMD_NODENAME:-unknown}"
echo "Start: $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
echo "========================================"
echo "Build script: ${BUILD_SCRIPT}"
echo "Dataset mode: ${DATASET_MODE}"
echo "Build positive: ${BUILD_POSITIVE}"
echo "Build negative: ${BUILD_NEGATIVE}"
echo "Output POS: ${OUTPUT_POS_H5}"
echo "Output NEG: ${OUTPUT_NEG_H5}"
echo "Lock file: ${LOCK_FILE}"
echo "Detection rule: PHOTFLAG!=0, fallback SNR>${SNR_THRESHOLD}"
echo "Workers: ${NUM_WORKERS}"
echo "CLS anchor GW prior: ${CLS_TIME_ANCHOR_GW_H5}"
echo "Flux zeropoints: FLUXCAL_ZP=${FLUXCAL_ZP}, PSFFLUX_ZP=${PSFFLUX_ZP}"
echo "Luptitude params: LUPT_K=${LUPT_K}, LUPT_M5_MAG=${LUPT_M5_MAG}"
echo "========================================"

if [[ "${BUILD_NEGATIVE}" == "true" && ! -f "${CLS_TIME_ANCHOR_GW_H5}" ]]; then
    echo "GW anchor H5 not found: ${CLS_TIME_ANCHOR_GW_H5}"
    exit 1
fi

cmd=(
    "${PYTHON_BIN}" -u "${BUILD_SCRIPT}"
    --min_nobs "${MIN_NOBS}"
    --snr_threshold "${SNR_THRESHOLD}"
    --buffer_limit "${BUFFER_LIMIT}"
    --fixed_offset_days "${FIXED_OFFSET_DAYS}"
    --max_lcs_per_event "${MAX_LCS_PER_EVENT}"
    --num_workers "${NUM_WORKERS}"
    --fluxcal_zp "${FLUXCAL_ZP}"
    --psfflux_zp "${PSFFLUX_ZP}"
    --lupt_k "${LUPT_K}"
    --lupt_m5_mag "${LUPT_M5_MAG}"
)

if [[ "${BUILD_POSITIVE}" == "true" ]]; then
    cmd+=(
        --build_positive
        --output_pos_h5 "${OUTPUT_POS_H5}"
        --bns_sim_root "${BNS_SIM_ROOT}"
        --bns_sim_name "${BNS_SIM_NAME}"
        --nsbh_sim_root "${NSBH_SIM_ROOT}"
        --nsbh_sim_name "${NSBH_SIM_NAME}"
    )
fi

if [[ "${BUILD_NEGATIVE}" == "true" ]]; then
    cmd+=(
        --build_negative
        --output_neg_h5 "${OUTPUT_NEG_H5}"
        --neg_group "${NEG_GROUP}"
        --neg_sim_root "${NEG_SIM_ROOT}"
        --cls_time_anchor_gw_h5 "${CLS_TIME_ANCHOR_GW_H5}"
        --cls_time_anchor_seed "${CLS_TIME_ANCHOR_SEED}"
    )
fi

echo "Command: ${cmd[*]}"
"${cmd[@]}"

echo "========================================"
echo "Finished: $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
if [[ "${BUILD_POSITIVE}" == "true" ]]; then
    echo "Saved POS: ${OUTPUT_POS_H5}"
fi
if [[ "${BUILD_NEGATIVE}" == "true" ]]; then
    echo "Saved NEG: ${OUTPUT_NEG_H5}"
fi
echo "========================================"
