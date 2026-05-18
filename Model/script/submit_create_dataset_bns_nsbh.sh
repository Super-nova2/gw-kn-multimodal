#!/bin/bash

#SBATCH --job-name=GW_BNS_NSBH_Dataset
#SBATCH --output=logs/data/%x_%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=8:00:00

set -euo pipefail

BASE_DIR="${BASE_DIR:-/fred/oz016/bgao_kn}"

SCRIPT_SUBDIR="Model/script"
SCRIPT_REL_PATH="Model/script/submit_create_dataset_bns_nsbh.sh"
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
MODEL_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
SCRIPT_PATH="${REPO_ROOT}/${SCRIPT_REL_PATH}"
if [[ ! -f "${SCRIPT_PATH}" ]]; then
    echo "Resolved script path not found: ${SCRIPT_PATH}" >&2
    exit 1
fi

PROFILE="${PROFILE:-astro_test}"       # test_aug | final_train | astro_test
DATASET_MODE="${DATASET_MODE:-}"        # train | test

BUFFER_LIMIT="${BUFFER_LIMIT:-2000}"
NUM_WORKERS="${NUM_WORKERS:-8}"
SEED="${SEED:-42}"

BNS_MAX_LC_PER_GW="${BNS_MAX_LC_PER_GW:-1000}"
NSBH_MAX_LC_PER_GW="${NSBH_MAX_LC_PER_GW:-1000}"

BNS_MAX_NEG_GW="${BNS_MAX_NEG_GW:-}"
NSBH_MAX_NEG_GW="${NSBH_MAX_NEG_GW:-}"
BNS_MAX_POS_GW="${BNS_MAX_POS_GW:-}"
NSBH_MAX_POS_GW="${NSBH_MAX_POS_GW:-}"
NSBH_MAX_NEG_TYPE1_GW="${NSBH_MAX_NEG_TYPE1_GW:-}"
NSBH_MAX_NEG_TYPE2_GW="${NSBH_MAX_NEG_TYPE2_GW:-}"
NSBH_MEJ_COL="${NSBH_MEJ_COL:-mej_tot}"
NSBH_TYPE1_THRESHOLD="${NSBH_TYPE1_THRESHOLD:-0.0}"
NSBH_REQUIRE_SUCCESS_FOR_MEJ_POS="${NSBH_REQUIRE_SUCCESS_FOR_MEJ_POS:-1}"
FLUXCAL_ZP="${FLUXCAL_ZP:-27.5}"
PSFFLUX_ZP="${PSFFLUX_ZP:-31.4}"
LUPT_K="${LUPT_K:-1.0}"
LUPT_M5_MAG="${LUPT_M5_MAG:-23.9,25.0,24.7,24.0,23.3,22.1}"

set_profile_defaults() {
    case "$PROFILE" in
        test_aug)
            DATASET_MODE="${DATASET_MODE:-test}"
            BNS_FULL_CATALOG_PATH="${BNS_FULL_CATALOG_PATH:-${REPO_ROOT}/dataset/O5_sim_bns/injections_final.csv}"
            BNS_SKYMAP_DIR="${BNS_SKYMAP_DIR:-${BASE_DIR}/data/skymap/bns_skymap_v0}"
            BNS_SIM_ROOT="${BNS_SIM_ROOT:-${BASE_DIR}/SNANA/SNDATA_ROOT/SIM/LSST_KN_BNS}"
            BNS_SIM_NAME="${BNS_SIM_NAME:-LSST_KN_BNS}"
            BNS_SUCCESS_IDS_PATH="${BNS_SUCCESS_IDS_PATH:-${BASE_DIR}/data/LSST_KN_BNS/success_sim_ids.txt}"

            NSBH_FULL_CATALOG_PATH="${NSBH_FULL_CATALOG_PATH:-${REPO_ROOT}/dataset/O5_sim_nsbh_aug/injections_full.csv}"
            NSBH_SKYMAP_DIR="${NSBH_SKYMAP_DIR:-${BASE_DIR}/data/skymap/nsbh_skymap}"
            NSBH_SIM_ROOT="${NSBH_SIM_ROOT:-${BASE_DIR}/SNANA/SNDATA_ROOT/SIM/LSST_KN_NSBH_AUG}"
            NSBH_SIM_NAME="${NSBH_SIM_NAME:-LSST_KN_NSBH_AUG}"
            NSBH_SUCCESS_IDS_PATH="${NSBH_SUCCESS_IDS_PATH:-${BASE_DIR}/data/LSST_KN_NSBH_AUG/success_sim_ids.txt}"

            OUTPUT_H5_PATH="${OUTPUT_H5_PATH:-${BASE_DIR}/data/ALBEF_dataset/combined_dataset_${DATASET_MODE}.h5}"
            ;;
        final_train)
            DATASET_MODE="${DATASET_MODE:-train}"
            BNS_FULL_CATALOG_PATH="${BNS_FULL_CATALOG_PATH:-${REPO_ROOT}/dataset/O5_sim_bns_aug/injections_final.csv}"
            BNS_SKYMAP_DIR="${BNS_SKYMAP_DIR:-${BASE_DIR}/data/skymap/bns_skymap}"
            # Keep overridable because BNS_AUG raw SNANA outputs may be compressed/offline.
            BNS_SIM_ROOT="${BNS_SIM_ROOT:-${BASE_DIR}/SNANA/SNDATA_ROOT/SIM/LSST_KN_BNS_AUG}"
            BNS_SIM_NAME="${BNS_SIM_NAME:-LSST_KN_BNS_AUG}"
            BNS_SUCCESS_IDS_PATH="${BNS_SUCCESS_IDS_PATH:-${BASE_DIR}/data/LSST_KN_BNS_AUG/success_sim_ids.txt}"

            NSBH_FULL_CATALOG_PATH="${NSBH_FULL_CATALOG_PATH:-${REPO_ROOT}/dataset/O5_sim_nsbh_train/injections_full.csv}"
            NSBH_SKYMAP_DIR="${NSBH_SKYMAP_DIR:-${BASE_DIR}/data/skymap/nsbh_skymap_train}"
            NSBH_SIM_ROOT="${NSBH_SIM_ROOT:-${BASE_DIR}/SNANA/SNDATA_ROOT/SIM/LSST_KN_NSBH_TRAIN}"
            NSBH_SIM_NAME="${NSBH_SIM_NAME:-LSST_KN_NSBH_TRAIN}"
            NSBH_SUCCESS_IDS_PATH="${NSBH_SUCCESS_IDS_PATH:-${BASE_DIR}/data/LSST_KN_NSBH_TRAIN/success_sim_ids.txt}"

            OUTPUT_H5_PATH="${OUTPUT_H5_PATH:-${BASE_DIR}/data/ALBEF_dataset/combined_dataset_${DATASET_MODE}.h5}"
            ;;
        astro_test)
            DATASET_MODE="${DATASET_MODE:-test}"
            BNS_FULL_CATALOG_PATH="${BNS_FULL_CATALOG_PATH:-${REPO_ROOT}/dataset/test/bns_test/injections_final.csv}"
            BNS_SKYMAP_DIR="${BNS_SKYMAP_DIR:-${BASE_DIR}/data_zjq/skymap/bns_test}"
            BNS_SIM_ROOT="${BNS_SIM_ROOT:-${BASE_DIR}/SNANA/SNDATA_ROOT/SIM/LSST_KN_BNS_Test}"
            BNS_SIM_NAME="${BNS_SIM_NAME:-LSST_KN_BNS_Test}"
            BNS_SUCCESS_IDS_PATH="${BNS_SUCCESS_IDS_PATH:-${BASE_DIR}/data_zjq/LSST_KN_BNS_Test/success_sim_ids.txt}"

            NSBH_FULL_CATALOG_PATH="${NSBH_FULL_CATALOG_PATH:-${REPO_ROOT}/dataset/test/nsbh_test/injections_full.csv}"
            NSBH_SKYMAP_DIR="${NSBH_SKYMAP_DIR:-${BASE_DIR}/data_zjq/skymap/nsbh_test}"
            NSBH_SIM_ROOT="${NSBH_SIM_ROOT:-${BASE_DIR}/SNANA/SNDATA_ROOT/SIM/LSST_KN_NSBH_TEST}"
            NSBH_SIM_NAME="${NSBH_SIM_NAME:-LSST_KN_NSBH_TEST}"
            NSBH_SUCCESS_IDS_PATH="${NSBH_SUCCESS_IDS_PATH:-${BASE_DIR}/data_zjq/LSST_KN_NSBH_TEST/success_sim_ids.txt}"

            OUTPUT_H5_PATH="${OUTPUT_H5_PATH:-${BASE_DIR}/data_zjq/ALBEF_dataset/combined_dataset_astro_${DATASET_MODE}.h5}"
            ;;
        *)
            echo "Unsupported PROFILE=$PROFILE. Use PROFILE=test_aug, PROFILE=final_train, or PROFILE=astro_test."
            exit 1
            ;;
    esac
}

require_file() {
    local p="$1"
    if [[ ! -f "$p" ]]; then
        echo "Required file not found: $p"
        exit 1
    fi
}

require_dir() {
    local p="$1"
    if [[ ! -d "$p" ]]; then
        echo "Required directory not found: $p"
        exit 1
    fi
}

append_optional_arg() {
    local flag="$1"
    local val="$2"
    if [[ -n "$val" ]]; then
        cmd+=("$flag" "$val")
    fi
}

validate_output_h5_schema() {
    local h5_path="$1"
    python - "$h5_path" <<'PY'
import sys
import numpy as np
import h5py

h5_path = sys.argv[1]
with h5py.File(h5_path, "r") as f:
    scalars_path = "events/gw_data/scalars"
    event_time_path = "events/gw_data/event_time_mjd"
    if scalars_path not in f:
        raise SystemExit(f"[SchemaError] Missing dataset: {scalars_path}")
    if event_time_path not in f:
        raise SystemExit(f"[SchemaError] Missing dataset: {event_time_path}")

    scalars = f[scalars_path]
    event_time = f[event_time_path]
    if scalars.ndim != 2 or scalars.shape[1] != 7:
        raise SystemExit(
            f"[SchemaError] {scalars_path} shape must be (n_gw, 7), got {tuple(scalars.shape)}"
        )
    if event_time.ndim != 1:
        raise SystemExit(
            f"[SchemaError] {event_time_path} shape must be (n_gw,), got {tuple(event_time.shape)}"
        )
    if event_time.shape[0] != scalars.shape[0]:
        raise SystemExit(
            f"[SchemaError] Length mismatch: {event_time_path}={event_time.shape[0]} "
            f"vs {scalars_path} n_gw={scalars.shape[0]}"
        )

    invalid = int((~np.isfinite(event_time[:])).sum())
    print(
        f"[SchemaOK] n_gw={scalars.shape[0]} scalars_dim={scalars.shape[1]} "
        f"event_time_len={event_time.shape[0]} invalid_event_time={invalid}"
    )
PY
}

set_profile_defaults

if [[ "$DATASET_MODE" != "train" && "$DATASET_MODE" != "test" ]]; then
    echo "DATASET_MODE must be train or test, got: $DATASET_MODE"
    exit 1
fi

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    if ! command -v sbatch >/dev/null 2>&1; then
        echo "sbatch not found; run inside a Slurm allocation or install Slurm tools."
        exit 1
    fi

    script_path="${SCRIPT_PATH}"

    mkdir -p logs/data

    sbatch_opts=()
    if [[ -n "${JOB_NAME:-}" ]]; then
        sbatch_opts+=(--job-name="${JOB_NAME}")
    fi
    if [[ -n "${OUTPUT_LOG:-}" ]]; then
        sbatch_opts+=(--output="${OUTPUT_LOG}")
    fi
    if [[ -n "${TIME_LIMIT:-}" ]]; then
        sbatch_opts+=(--time="${TIME_LIMIT}")
    fi
    if [[ -n "${PARTITION:-}" ]]; then
        sbatch_opts+=(--partition="${PARTITION}")
    fi
    REQUESTED_CPUS_PER_TASK="${CPUS_PER_TASK:-${NUM_WORKERS:-}}"
    if [[ -n "${REQUESTED_CPUS_PER_TASK:-}" ]]; then
        sbatch_opts+=(--cpus-per-task="${REQUESTED_CPUS_PER_TASK}")
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

    echo "Submitting job with: sbatch ${sbatch_opts[*]} ${script_path}"
    (
        cd "${REPO_ROOT}"
        sbatch "${sbatch_opts[@]}" "${script_path}"
    )
    exit 0
fi

require_file "$BNS_FULL_CATALOG_PATH"
require_file "$NSBH_FULL_CATALOG_PATH"
require_dir "$BNS_SKYMAP_DIR"
require_dir "$NSBH_SKYMAP_DIR"
require_dir "$BNS_SIM_ROOT"
require_dir "$NSBH_SIM_ROOT"

if [[ -n "${BNS_SUCCESS_IDS_PATH:-}" && ! -f "$BNS_SUCCESS_IDS_PATH" ]]; then
    echo "WARNING: BNS success ids file not found, skip filtering: $BNS_SUCCESS_IDS_PATH"
    BNS_SUCCESS_IDS_PATH=""
fi

if [[ "${NSBH_REQUIRE_SUCCESS_FOR_MEJ_POS}" == "1" ]]; then
    if [[ -z "${NSBH_SUCCESS_IDS_PATH:-}" ]]; then
        echo "NSBH_SUCCESS_IDS_PATH is required when NSBH_REQUIRE_SUCCESS_FOR_MEJ_POS=1."
        exit 1
    fi
    require_file "$NSBH_SUCCESS_IDS_PATH"
elif [[ -n "${NSBH_SUCCESS_IDS_PATH:-}" && ! -f "$NSBH_SUCCESS_IDS_PATH" ]]; then
    echo "WARNING: NSBH success ids file not found, skip filtering: $NSBH_SUCCESS_IDS_PATH"
    NSBH_SUCCESS_IDS_PATH=""
fi

mkdir -p "$(dirname "$OUTPUT_H5_PATH")"

if [[ -z "${NUM_WORKERS:-}" || "${NUM_WORKERS}" == "null" ]]; then
    NUM_WORKERS="${SLURM_CPUS_PER_TASK:-1}"
fi
if ! [[ "$NUM_WORKERS" =~ ^[0-9]+$ ]] || [[ "$NUM_WORKERS" -lt 1 ]]; then
    echo "NUM_WORKERS must be a positive integer, got: $NUM_WORKERS"
    exit 1
fi
if [[ -n "${SLURM_CPUS_PER_TASK:-}" && "$NUM_WORKERS" -gt "$SLURM_CPUS_PER_TASK" ]]; then
    echo "NUM_WORKERS=$NUM_WORKERS exceeds SLURM_CPUS_PER_TASK=$SLURM_CPUS_PER_TASK; capping to allocated CPUs."
    NUM_WORKERS="$SLURM_CPUS_PER_TASK"
fi

py_script="${SCRIPT_DIR}/create_dataset_bns_nsbh.py"

cmd=(
    python -u "$py_script"
    --output_h5_path "$OUTPUT_H5_PATH"
    --dataset_mode "$DATASET_MODE"
    --buffer_limit "$BUFFER_LIMIT"
    --num_workers "$NUM_WORKERS"
    --seed "$SEED"
    --fluxcal_zp "$FLUXCAL_ZP"
    --psfflux_zp "$PSFFLUX_ZP"
    --lupt_k "$LUPT_K"
    --lupt_m5_mag "$LUPT_M5_MAG"
    --bns_full_catalog_path "$BNS_FULL_CATALOG_PATH"
    --bns_skymap_dir "$BNS_SKYMAP_DIR"
    --bns_sim_root "$BNS_SIM_ROOT"
    --bns_sim_name "$BNS_SIM_NAME"
    --bns_max_lc_per_gw "$BNS_MAX_LC_PER_GW"
    --nsbh_full_catalog_path "$NSBH_FULL_CATALOG_PATH"
    --nsbh_skymap_dir "$NSBH_SKYMAP_DIR"
    --nsbh_sim_root "$NSBH_SIM_ROOT"
    --nsbh_sim_name "$NSBH_SIM_NAME"
    --nsbh_max_lc_per_gw "$NSBH_MAX_LC_PER_GW"
    --nsbh_mej_col "$NSBH_MEJ_COL"
    --nsbh_type1_threshold "$NSBH_TYPE1_THRESHOLD"
    --nsbh_require_success_for_mej_pos "$NSBH_REQUIRE_SUCCESS_FOR_MEJ_POS"
)

append_optional_arg --bns_success_ids_path "${BNS_SUCCESS_IDS_PATH:-}"
append_optional_arg --nsbh_success_ids_path "${NSBH_SUCCESS_IDS_PATH:-}"
append_optional_arg --bns_max_neg_gw "${BNS_MAX_NEG_GW:-}"
append_optional_arg --nsbh_max_neg_gw "${NSBH_MAX_NEG_GW:-}"
append_optional_arg --bns_max_pos_gw "${BNS_MAX_POS_GW:-}"
append_optional_arg --nsbh_max_pos_gw "${NSBH_MAX_POS_GW:-}"
append_optional_arg --nsbh_max_neg_type1_gw "${NSBH_MAX_NEG_TYPE1_GW:-}"
append_optional_arg --nsbh_max_neg_type2_gw "${NSBH_MAX_NEG_TYPE2_GW:-}"

echo "PROFILE=$PROFILE DATASET_MODE=$DATASET_MODE"
echo "Output H5: $OUTPUT_H5_PATH"
echo "Luptitude params: FLUXCAL_ZP=$FLUXCAL_ZP PSFFLUX_ZP=$PSFFLUX_ZP LUPT_K=$LUPT_K LUPT_M5_MAG=$LUPT_M5_MAG"
echo "Parallel preprocessing workers: $NUM_WORKERS"
echo "Light-curve preprocessing: 2h same-band inverse-variance merge in psfFlux domain before luptitude conversion"
"${cmd[@]}"
validate_output_h5_schema "$OUTPUT_H5_PATH"
