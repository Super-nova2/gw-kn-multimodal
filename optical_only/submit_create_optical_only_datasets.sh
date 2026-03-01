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

DATASET_MODE="${DATASET_MODE:-test}"   # train | test

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
        ;;
    *)
        echo "Unsupported DATASET_MODE='${DATASET_MODE}'. Use train or test."
        exit 1
        ;;
esac

MIN_NOBS=5
SNR_THRESHOLD=5.0
BUFFER_LIMIT=3000
FIXED_OFFSET_DAYS=0.0
MAX_LCS_PER_EVENT=1000
# Conservative default to reduce worker crashes on large FITS parsing.
NUM_WORKERS="${NUM_WORKERS:-4}"
if [[ -n "${SLURM_CPUS_PER_TASK:-}" && "${NUM_WORKERS}" -gt "${SLURM_CPUS_PER_TASK}" ]]; then
    NUM_WORKERS="${SLURM_CPUS_PER_TASK}"
fi

mkdir -p "$(dirname "${OUTPUT_POS_H5}")"
mkdir -p "$(dirname "${OUTPUT_NEG_H5}")"

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
echo "Output POS: ${OUTPUT_POS_H5}"
echo "Output NEG: ${OUTPUT_NEG_H5}"
echo "Lock file: ${LOCK_FILE}"
echo "Detection rule: PHOTFLAG!=0, fallback SNR>${SNR_THRESHOLD}"
echo "Workers: ${NUM_WORKERS}"
echo "========================================"

cmd=(
    "${PYTHON_BIN}" -u "${BUILD_SCRIPT}"
    --build_positive
    --build_negative
    --output_pos_h5 "${OUTPUT_POS_H5}"
    --output_neg_h5 "${OUTPUT_NEG_H5}"
    --neg_group "${NEG_GROUP}"
    --bns_sim_root "${BNS_SIM_ROOT}"
    --bns_sim_name "${BNS_SIM_NAME}"
    --nsbh_sim_root "${NSBH_SIM_ROOT}"
    --nsbh_sim_name "${NSBH_SIM_NAME}"
    --neg_sim_root "${NEG_SIM_ROOT}"
    --min_nobs "${MIN_NOBS}"
    --snr_threshold "${SNR_THRESHOLD}"
    --buffer_limit "${BUFFER_LIMIT}"
    --fixed_offset_days "${FIXED_OFFSET_DAYS}"
    --max_lcs_per_event "${MAX_LCS_PER_EVENT}"
    --num_workers "${NUM_WORKERS}"
)

echo "Command: ${cmd[*]}"
"${cmd[@]}"

echo "========================================"
echo "Finished: $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
echo "Saved POS: ${OUTPUT_POS_H5}"
echo "Saved NEG: ${OUTPUT_NEG_H5}"
echo "========================================"
