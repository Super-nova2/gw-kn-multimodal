#!/bin/bash

#SBATCH --job-name=GW_Dataset_Gen
#SBATCH --output=logs/data/%x_%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=4:00:00

set -euo pipefail

BASE_DIR="${BASE_DIR:-/fred/oz016/bgao_kn}"

PROFILE="${PROFILE:-nsbh_train}"           # bns_aug | nsbh_aug | nsbh_train
SOURCE_TYPE="${SOURCE_TYPE:-nsbh}"

MAX_LC_PER_GW="${MAX_LC_PER_GW:-1000}"
BUFFER_LIMIT="${BUFFER_LIMIT:-10000}"
MAX_NEG_GW="${MAX_NEG_GW:-}"
SEED="${SEED:-42}"

NSBH_MEJ_COL="${NSBH_MEJ_COL:-mej_tot}"
NSBH_TYPE1_THRESHOLD="${NSBH_TYPE1_THRESHOLD:-0.0}"
NSBH_REQUIRE_SUCCESS_FOR_MEJ_POS="${NSBH_REQUIRE_SUCCESS_FOR_MEJ_POS:-1}"
NSBH_MAX_NEG_TYPE1_GW="${NSBH_MAX_NEG_TYPE1_GW:-}"
NSBH_MAX_NEG_TYPE2_GW="${NSBH_MAX_NEG_TYPE2_GW:-}"

set_profile_defaults() {
    case "$PROFILE" in
        bns_aug)
            SOURCE_TYPE="${SOURCE_TYPE:-bns}"
            FULL_CATALOG_PATH="${FULL_CATALOG_PATH:-${BASE_DIR}/gw-kn-multimodal/dataset/O5_sim_bns_aug/injections_final.csv}"
            SUCCESS_IDS_PATH="${SUCCESS_IDS_PATH:-${BASE_DIR}/data/LSST_KN_BNS_AUG/success_sim_ids.txt}"
            SKYMAP_DIR="${SKYMAP_DIR:-${BASE_DIR}/data/skymap/bns_skymap}"
            SIM_ROOT="${SIM_ROOT:-${BASE_DIR}/SNANA/SNDATA_ROOT/SIM/LSST_KN_BNS_AUG}"
            SIM_NAME="${SIM_NAME:-LSST_KN_BNS_AUG}"
            OUTPUT_H5_PATH="${OUTPUT_H5_PATH:-${BASE_DIR}/data/LSST_KN_BNS_AUG/combined_dataset_with_neg_gw.h5}"
            ;;
        nsbh_aug)
            SOURCE_TYPE="${SOURCE_TYPE:-nsbh}"
            FULL_CATALOG_PATH="${FULL_CATALOG_PATH:-${BASE_DIR}/gw-kn-multimodal/dataset/O5_sim_nsbh_aug/injections_full.csv}"
            SUCCESS_IDS_PATH="${SUCCESS_IDS_PATH:-${BASE_DIR}/data/LSST_KN_NSBH_AUG/success_sim_ids.txt}"
            SKYMAP_DIR="${SKYMAP_DIR:-${BASE_DIR}/data/skymap/nsbh_skymap}"
            SIM_ROOT="${SIM_ROOT:-${BASE_DIR}/SNANA/SNDATA_ROOT/SIM/LSST_KN_NSBH_AUG}"
            SIM_NAME="${SIM_NAME:-LSST_KN_NSBH_AUG}"
            OUTPUT_H5_PATH="${OUTPUT_H5_PATH:-${BASE_DIR}/data/LSST_KN_NSBH_AUG/combined_dataset_with_neg_gw.h5}"
            ;;
        nsbh_train)
            SOURCE_TYPE="${SOURCE_TYPE:-nsbh}"
            FULL_CATALOG_PATH="${FULL_CATALOG_PATH:-${BASE_DIR}/gw-kn-multimodal/dataset/O5_sim_nsbh_train/injections_full.csv}"
            SUCCESS_IDS_PATH="${SUCCESS_IDS_PATH:-${BASE_DIR}/data/LSST_KN_NSBH_TRAIN/success_sim_ids.txt}"
            SKYMAP_DIR="${SKYMAP_DIR:-${BASE_DIR}/data/skymap/nsbh_skymap_train}"
            SIM_ROOT="${SIM_ROOT:-${BASE_DIR}/SNANA/SNDATA_ROOT/SIM/LSST_KN_NSBH_TRAIN}"
            SIM_NAME="${SIM_NAME:-LSST_KN_NSBH_TRAIN}"
            OUTPUT_H5_PATH="${OUTPUT_H5_PATH:-${BASE_DIR}/data/LSST_KN_NSBH_TRAIN/combined_dataset_with_neg_gw.h5}"
            ;;
        *)
            echo "Unsupported PROFILE='$PROFILE'. Use PROFILE=bns_aug, PROFILE=nsbh_aug, or PROFILE=nsbh_train."
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

if [[ "$SOURCE_TYPE" != "bns" && "$SOURCE_TYPE" != "nsbh" ]]; then
    echo "SOURCE_TYPE must be bns or nsbh, got: $SOURCE_TYPE"
    exit 1
fi

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    if ! command -v sbatch >/dev/null 2>&1; then
        echo "sbatch not found; run inside a Slurm allocation or install Slurm tools."
        exit 1
    fi

    script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    script_path="${script_dir}/$(basename "${BASH_SOURCE[0]}")"

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

    echo "Submitting job with: sbatch ${sbatch_opts[*]} ${script_path}"
    sbatch "${sbatch_opts[@]}" "${script_path}"
    exit 0
fi

require_file "$FULL_CATALOG_PATH"
require_dir "$SKYMAP_DIR"
require_dir "$SIM_ROOT"

if [[ "$SOURCE_TYPE" == "nsbh" && "$NSBH_REQUIRE_SUCCESS_FOR_MEJ_POS" == "1" ]]; then
    if [[ -z "${SUCCESS_IDS_PATH:-}" ]]; then
        echo "SUCCESS_IDS_PATH is required for NSBH when NSBH_REQUIRE_SUCCESS_FOR_MEJ_POS=1."
        exit 1
    fi
    require_file "$SUCCESS_IDS_PATH"
elif [[ -n "${SUCCESS_IDS_PATH:-}" && ! -f "$SUCCESS_IDS_PATH" ]]; then
    echo "WARNING: SUCCESS_IDS_PATH not found, skip success filtering: $SUCCESS_IDS_PATH"
    SUCCESS_IDS_PATH=""
fi

mkdir -p "$(dirname "$OUTPUT_H5_PATH")"

py_script="${BASE_DIR}/gw-kn-multimodal/Model/notebook/create_dataset.py"

cmd=(
    python -u "$py_script"
    --full_catalog_path "$FULL_CATALOG_PATH"
    --skymap_dir "$SKYMAP_DIR"
    --sim_root "$SIM_ROOT"
    --output_h5_path "$OUTPUT_H5_PATH"
    --sim_name "$SIM_NAME"
    --buffer_limit "$BUFFER_LIMIT"
    --max_lc_per_gw "$MAX_LC_PER_GW"
    --source_type "$SOURCE_TYPE"
    --nsbh_mej_col "$NSBH_MEJ_COL"
    --nsbh_type1_threshold "$NSBH_TYPE1_THRESHOLD"
    --nsbh_require_success_for_mej_pos "$NSBH_REQUIRE_SUCCESS_FOR_MEJ_POS"
)

append_optional_arg --success_ids_path "${SUCCESS_IDS_PATH:-}"
append_optional_arg --max_neg_gw "${MAX_NEG_GW:-}"
append_optional_arg --seed "${SEED:-}"
append_optional_arg --nsbh_max_neg_type1_gw "${NSBH_MAX_NEG_TYPE1_GW:-}"
append_optional_arg --nsbh_max_neg_type2_gw "${NSBH_MAX_NEG_TYPE2_GW:-}"

echo "PROFILE=$PROFILE SOURCE_TYPE=$SOURCE_TYPE"
echo "Output H5: $OUTPUT_H5_PATH"
"${cmd[@]}"
validate_output_h5_schema "$OUTPUT_H5_PATH"
