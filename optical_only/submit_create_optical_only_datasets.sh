#!/bin/bash

#SBATCH --job-name=BUILD_OPTICAL_ONLY_DATASET
#SBATCH --output=logs/data/%x_%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --time=8:00:00
#SBATCH --partition=cpu

set -euo pipefail

SCRIPT_SUBDIR="optical_only"
SCRIPT_REL_PATH="optical_only/submit_create_optical_only_datasets.sh"
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
SCRIPT_PATH="${REPO_ROOT}/${SCRIPT_REL_PATH}"
if [[ ! -f "${SCRIPT_PATH}" ]]; then
    echo "Resolved script path not found: ${SCRIPT_PATH}" >&2
    exit 1
fi

DATASET_MODE="${DATASET_MODE:-train}"   # train | test
BUILD_POSITIVE="${BUILD_POSITIVE:-true}" # true|false
BUILD_NEGATIVE="${BUILD_NEGATIVE:-true}" # true|false
ENFORCE_TIME_WINDOW="${ENFORCE_TIME_WINDOW:-true}"  # true|false
WRITE_META_FEATURES="${WRITE_META_FEATURES:-true}"  # true|false
NEG_MATCH_POS_DENSITY="${NEG_MATCH_POS_DENSITY:-true}" # true|false
PREFIX_TASK_ENABLE="${PREFIX_TASK_ENABLE:-false}"   # true|false
DATASET_TAG="${DATASET_TAG:-}"

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
ENFORCE_TIME_WINDOW="$(normalize_bool "${ENFORCE_TIME_WINDOW}")"
WRITE_META_FEATURES="$(normalize_bool "${WRITE_META_FEATURES}")"
NEG_MATCH_POS_DENSITY="$(normalize_bool "${NEG_MATCH_POS_DENSITY}")"
PREFIX_TASK_ENABLE="$(normalize_bool "${PREFIX_TASK_ENABLE}")"
if [[ "${BUILD_POSITIVE}" != "true" && "${BUILD_NEGATIVE}" != "true" ]]; then
    echo "Nothing to build: BUILD_POSITIVE=${BUILD_POSITIVE}, BUILD_NEGATIVE=${BUILD_NEGATIVE}"
    echo "Set at least one of BUILD_POSITIVE/BUILD_NEGATIVE to true."
    exit 1
fi
if [[ "${PREFIX_TASK_ENABLE}" == "true" && "${WRITE_META_FEATURES}" != "true" ]]; then
    echo "PREFIX_TASK_ENABLE=true requires WRITE_META_FEATURES=true so meta_n_obs/meta_n_det_snr5 are written."
    exit 1
fi
if [[ -z "${DATASET_TAG}" && "${PREFIX_TASK_ENABLE}" == "true" ]]; then
    DATASET_TAG="v7_prefix_full"
fi
DATASET_TAG_SLUG="${DATASET_TAG//[^A-Za-z0-9._-]/_}"
if [[ -n "${DATASET_TAG_SLUG}" ]]; then
    LOCK_TAG="_${DATASET_TAG_SLUG}"
else
    LOCK_TAG=""
fi

# Self-submit: run this script directly to submit job to Slurm.
if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    if ! command -v sbatch >/dev/null 2>&1; then
        echo "sbatch not found. Please run on a Slurm login node."
        exit 1
    fi

    script_path="${SCRIPT_PATH}"

    mkdir -p <BASE_DIR>/logs/data
    echo "Submitting: sbatch ${script_path}"
    (
        cd "${REPO_ROOT}"
        sbatch "${script_path}"
    )
    exit 0
fi

# ------------------------------
# Hard-coded parameters
# ------------------------------
PYTHON_BIN="python"
BUILD_SCRIPT="${SCRIPT_DIR}/create_optical_only_datasets.py"

case "${DATASET_MODE}" in
    train)
        BNS_SIM_ROOT="<BASE_DIR>/SNANA/SNDATA_ROOT/SIM/LSST_KN_BNS_AUG"
        BNS_SIM_NAME="LSST_KN_BNS_AUG"

        NSBH_SIM_ROOT="<BASE_DIR>/SNANA/SNDATA_ROOT/SIM/LSST_KN_NSBH_TRAIN"
        NSBH_SIM_NAME="LSST_KN_NSBH_TRAIN"

        NEG_SIM_ROOT="<BASE_DIR>/data/ELASTICC2_TRAIN_02"
        NEG_GROUP="ELASTICC2/optical_data"

        if [[ "${PREFIX_TASK_ENABLE}" == "true" ]]; then
            OUTPUT_POS_H5_DEFAULT="<BASE_DIR>/data/Optical_Only_dataset/combined_dataset_train_${DATASET_TAG_SLUG}.h5"
            OUTPUT_NEG_H5_DEFAULT="<BASE_DIR>/data/Optical_Only_dataset/ELASTICC2_negative_dataset_${DATASET_TAG_SLUG}.h5"
        else
            OUTPUT_POS_H5_DEFAULT="<BASE_DIR>/data/Optical_Only_dataset/combined_dataset_train.h5"
            OUTPUT_NEG_H5_DEFAULT="<BASE_DIR>/data/Optical_Only_dataset/ELASTICC2_negative_dataset.h5"
        fi
        ;;
    test)
        BNS_SIM_ROOT="<BASE_DIR>/SNANA/SNDATA_ROOT/SIM/LSST_KN_BNS"
        BNS_SIM_NAME="LSST_KN_BNS"

        NSBH_SIM_ROOT="<BASE_DIR>/SNANA/SNDATA_ROOT/SIM/LSST_KN_NSBH_AUG"
        NSBH_SIM_NAME="LSST_KN_NSBH_AUG"

        NEG_SIM_ROOT="<BASE_DIR>/data/Tutorial_LSST_sims_2025"
        NEG_GROUP="Tutorial/optical_data"

        if [[ "${PREFIX_TASK_ENABLE}" == "true" ]]; then
            OUTPUT_POS_H5_DEFAULT="<BASE_DIR>/data/Optical_Only_dataset/combined_dataset_test_${DATASET_TAG_SLUG}.h5"
            OUTPUT_NEG_H5_DEFAULT="<BASE_DIR>/data/Optical_Only_dataset/Tutorial_negative_dataset_${DATASET_TAG_SLUG}.h5"
        else
            OUTPUT_POS_H5_DEFAULT="<BASE_DIR>/data/Optical_Only_dataset/combined_dataset_test.h5"
            OUTPUT_NEG_H5_DEFAULT="<BASE_DIR>/data/Optical_Only_dataset/Tutorial_negative_dataset.h5"
        fi
        ;;
    *)
        echo "Unsupported DATASET_MODE='${DATASET_MODE}'. Use train or test."
        exit 1
        ;;
esac

OUTPUT_POS_H5="${OUTPUT_POS_H5:-$OUTPUT_POS_H5_DEFAULT}"
OUTPUT_NEG_H5="${OUTPUT_NEG_H5:-$OUTPUT_NEG_H5_DEFAULT}"

if [[ "${PREFIX_TASK_ENABLE}" == "true" ]]; then
    MIN_NOBS_DEFAULT=2
    DENSITY_BINS_N_DET_DEFAULT="2,3,4,5,6,8,10,12,20,40,80,200"
else
    MIN_NOBS_DEFAULT=5
    DENSITY_BINS_N_DET_DEFAULT="3,5,8,12,20,40,80,200"
fi

MIN_NOBS="${MIN_NOBS:-$MIN_NOBS_DEFAULT}"
SNR_THRESHOLD="${SNR_THRESHOLD:-5.0}"
BUFFER_LIMIT="${BUFFER_LIMIT:-3000}"
FIXED_OFFSET_DAYS="${FIXED_OFFSET_DAYS:-0.0}"
MAX_LCS_PER_EVENT="${MAX_LCS_PER_EVENT:-1000}"
MAX_NEGATIVE_HEADS="${MAX_NEGATIVE_HEADS:-}"
TIME_WINDOW_START="${TIME_WINDOW_START:--0.3}"
TIME_WINDOW_END="${TIME_WINDOW_END:-0.6}"
DENSITY_BINS_N_DET="${DENSITY_BINS_N_DET:-$DENSITY_BINS_N_DET_DEFAULT}"
DENSITY_BINS_N_BANDS="${DENSITY_BINS_N_BANDS:-1,2,3,4,5,6}"
DENSITY_BINS_T_SPAN="${DENSITY_BINS_T_SPAN:-0,0.01,0.05,0.1,0.2,0.5,1.0}"
FLUXCAL_ZP="${FLUXCAL_ZP:-27.5}"
PSFFLUX_ZP="${PSFFLUX_ZP:-31.4}"
LUPT_K="${LUPT_K:-1.0}"
LUPT_M5_MAG="${LUPT_M5_MAG:-23.9,25.0,24.7,24.0,23.3,22.1}"
DENSITY_MATCH_POS_H5="${DENSITY_MATCH_POS_H5:-$OUTPUT_POS_H5}"
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
LOCK_FILE="<BASE_DIR>/data/Optical_Only_dataset/.build_optical_only_${DATASET_MODE}${LOCK_TAG}.lock"
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
echo "Prefix task enable: ${PREFIX_TASK_ENABLE}"
echo "Dataset tag: ${DATASET_TAG_SLUG:-<legacy-default>}"
echo "Build positive: ${BUILD_POSITIVE}"
echo "Build negative: ${BUILD_NEGATIVE}"
echo "Output POS: ${OUTPUT_POS_H5}"
echo "Output NEG: ${OUTPUT_NEG_H5}"
echo "Lock file: ${LOCK_FILE}"
echo "Merge policy: 2h same-band inverse-variance merge in psfFlux domain"
echo "Detection rule: merged psfFlux SNR>${SNR_THRESHOLD}"
echo "min_nobs stage: post-merge"
echo "Workers: ${NUM_WORKERS}"
echo "Flux zeropoints: FLUXCAL_ZP=${FLUXCAL_ZP}, PSFFLUX_ZP=${PSFFLUX_ZP}"
echo "Luptitude params: LUPT_K=${LUPT_K}, LUPT_M5_MAG=${LUPT_M5_MAG}"
echo "POS zero_time policy: first detection + fixed_offset_days"
echo "NEG zero_time policy: synthetic first detection sampled in [61000,64500]"
echo "Time window: enforce=${ENFORCE_TIME_WINDOW}, range=[${TIME_WINDOW_START}, ${TIME_WINDOW_END}]"
echo "Meta features: write_meta_features=${WRITE_META_FEATURES}"
echo "Density matching: neg_match_pos_density=${NEG_MATCH_POS_DENSITY}"
echo "Density ref POS H5: ${DENSITY_MATCH_POS_H5}"
echo "Density bins: n_det=${DENSITY_BINS_N_DET}"
echo "Density bins: n_bands=${DENSITY_BINS_N_BANDS}"
echo "Density bins: t_span=${DENSITY_BINS_T_SPAN}"
echo "========================================"

if [[ "${BUILD_NEGATIVE}" == "true" && "${NEG_MATCH_POS_DENSITY}" == "true" ]]; then
    if [[ "${BUILD_POSITIVE}" == "true" && "${DENSITY_MATCH_POS_H5}" == "${OUTPUT_POS_H5}" ]]; then
        :
    elif [[ ! -f "${DENSITY_MATCH_POS_H5}" ]]; then
        echo "Density-match positive H5 not found: ${DENSITY_MATCH_POS_H5}"
        echo "Set DENSITY_MATCH_POS_H5 to an existing positive H5, or build positives in the same run."
        exit 1
    fi
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
    --enforce_time_window "${ENFORCE_TIME_WINDOW}"
    --time_window_start "${TIME_WINDOW_START}"
    --time_window_end "${TIME_WINDOW_END}"
    --write_meta_features "${WRITE_META_FEATURES}"
    --neg_match_pos_density "${NEG_MATCH_POS_DENSITY}"
    --density_bins_n_det "${DENSITY_BINS_N_DET}"
    --density_bins_n_bands "${DENSITY_BINS_N_BANDS}"
    --density_bins_t_span "${DENSITY_BINS_T_SPAN}"
)

if [[ -n "${MAX_NEGATIVE_HEADS}" ]]; then
    cmd+=(--max_negative_heads "${MAX_NEGATIVE_HEADS}")
fi

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
        --density_match_pos_h5 "${DENSITY_MATCH_POS_H5}"
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
