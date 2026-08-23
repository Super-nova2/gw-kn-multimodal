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

SCRIPT_SUBDIR="Model/scripts/data"
SCRIPT_REL_PATH="Model/scripts/data/submit_create_dataset_bns_nsbh.sh"
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
MODEL_DIR="${REPO_ROOT}/Model"
SCRIPT_PATH="${REPO_ROOT}/${SCRIPT_REL_PATH}"
if [[ ! -f "${SCRIPT_PATH}" ]]; then
    echo "Resolved script path not found: ${SCRIPT_PATH}" >&2
    exit 1
fi

PROFILE="${PROFILE:-astro_test}"       # final_train | astro_test
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
BNS_MAX_NEG_TYPE1_GW="${BNS_MAX_NEG_TYPE1_GW:-}"
BNS_MAX_NEG_TYPE2_GW="${BNS_MAX_NEG_TYPE2_GW:-}"
NSBH_MAX_NEG_TYPE1_GW="${NSBH_MAX_NEG_TYPE1_GW:-}"
NSBH_MAX_NEG_TYPE2_GW="${NSBH_MAX_NEG_TYPE2_GW:-}"
FLUXCAL_ZP="${FLUXCAL_ZP:-27.5}"
PSFFLUX_ZP="${PSFFLUX_ZP:-31.4}"
LUPT_K="${LUPT_K:-1.0}"
LUPT_M5_MAG="${LUPT_M5_MAG:-23.9,25.0,24.7,24.0,23.3,22.1}"

set_profile_defaults() {
    case "$PROFILE" in
        final_train)
            DATASET_MODE="${DATASET_MODE:-train}"
            BNS_MAX_NEG_GW="${BNS_MAX_NEG_GW:-10000}"
            BNS_MAX_NEG_TYPE1_GW="${BNS_MAX_NEG_TYPE1_GW:-5000}"
            BNS_MAX_NEG_TYPE2_GW="${BNS_MAX_NEG_TYPE2_GW:-5000}"
            NSBH_MAX_NEG_GW="${NSBH_MAX_NEG_GW:-10000}"
            NSBH_MAX_NEG_TYPE1_GW="${NSBH_MAX_NEG_TYPE1_GW:-5000}"
            NSBH_MAX_NEG_TYPE2_GW="${NSBH_MAX_NEG_TYPE2_GW:-5000}"
            BNS_FULL_CATALOG_PATH="${BNS_FULL_CATALOG_PATH:-${BASE_DIR}/GWSamplegen/outputs/production_am_bayestar/dual/bns_train_seed_1234/pos_catalog.csv}"
            BNS_SKYMAP_DIR="${BNS_SKYMAP_DIR:-${BASE_DIR}/data/skymap/positive/bns_skymap_train}"
            BNS_SIM_ARTIFACT="${BNS_SIM_ARTIFACT-${REPO_ROOT}/kn_simulation/runs/bns_train/simulation_intermediates.h5}"
            BNS_SIM_ROOT="${BNS_SIM_ROOT:-${BASE_DIR}/SNANA/SNDATA_ROOT/SIM/LSST_KN_BNS_TRAIN}"
            BNS_SIM_NAME="${BNS_SIM_NAME:-LSST_KN_BNS_TRAIN}"
            BNS_SUCCESS_IDS_PATH="${BNS_SUCCESS_IDS_PATH:-${REPO_ROOT}/kn_simulation/runs/bns_train/success_sim_ids.txt}"

            NSBH_FULL_CATALOG_PATH="${NSBH_FULL_CATALOG_PATH:-${BASE_DIR}/GWSamplegen/outputs/production_am_bayestar/dual/nsbh_train_seed_1234/pos_catalog.csv}"
            NSBH_SKYMAP_DIR="${NSBH_SKYMAP_DIR:-${BASE_DIR}/data/skymap/positive/nsbh_skymap_train}"
            NSBH_SIM_ARTIFACT="${NSBH_SIM_ARTIFACT-${REPO_ROOT}/kn_simulation/runs/nsbh_train/simulation_intermediates.h5}"
            NSBH_SIM_ROOT="${NSBH_SIM_ROOT:-${BASE_DIR}/SNANA/SNDATA_ROOT/SIM/LSST_KN_NSBH_TRAIN}"
            NSBH_SIM_NAME="${NSBH_SIM_NAME:-LSST_KN_NSBH_TRAIN}"
            NSBH_SUCCESS_IDS_PATH="${NSBH_SUCCESS_IDS_PATH:-${REPO_ROOT}/kn_simulation/runs/nsbh_train/success_sim_ids.txt}"

            OUTPUT_H5_PATH="${OUTPUT_H5_PATH:-${BASE_DIR}/data/ALBEF_dataset/combined_dataset_${DATASET_MODE}.h5}"
            ;;
        astro_test)
            DATASET_MODE="${DATASET_MODE:-test}"
            BNS_MAX_NEG_GW="${BNS_MAX_NEG_GW:-2000}"
            BNS_MAX_NEG_TYPE1_GW="${BNS_MAX_NEG_TYPE1_GW:-500}"
            BNS_MAX_NEG_TYPE2_GW="${BNS_MAX_NEG_TYPE2_GW:-1000}"
            NSBH_MAX_NEG_GW="${NSBH_MAX_NEG_GW:-1500}"
            NSBH_MAX_NEG_TYPE1_GW="${NSBH_MAX_NEG_TYPE1_GW:-500}"
            NSBH_MAX_NEG_TYPE2_GW="${NSBH_MAX_NEG_TYPE2_GW:-500}"
            BNS_FULL_CATALOG_PATH="${BNS_FULL_CATALOG_PATH:-${BASE_DIR}/GWSamplegen/outputs/production_am_bayestar/dual/bns_test_seed_1234/pos_catalog.csv}"
            BNS_SKYMAP_DIR="${BNS_SKYMAP_DIR:-${BASE_DIR}/data/skymap/positive/bns_skymap_test}"
            BNS_SIM_ARTIFACT="${BNS_SIM_ARTIFACT-${REPO_ROOT}/kn_simulation/runs/bns_test/simulation_intermediates.h5}"
            BNS_SIM_ROOT="${BNS_SIM_ROOT:-${BASE_DIR}/SNANA/SNDATA_ROOT/SIM/LSST_KN_BNS_TEST}"
            BNS_SIM_NAME="${BNS_SIM_NAME:-LSST_KN_BNS_TEST}"
            BNS_SUCCESS_IDS_PATH="${BNS_SUCCESS_IDS_PATH:-${REPO_ROOT}/kn_simulation/runs/bns_test/success_sim_ids.txt}"

            NSBH_FULL_CATALOG_PATH="${NSBH_FULL_CATALOG_PATH:-${BASE_DIR}/GWSamplegen/outputs/production_am_bayestar/dual/nsbh_test_seed_1234/pos_catalog.csv}"
            NSBH_SKYMAP_DIR="${NSBH_SKYMAP_DIR:-${BASE_DIR}/data/skymap/positive/nsbh_skymap_test}"
            NSBH_SIM_ARTIFACT="${NSBH_SIM_ARTIFACT-${REPO_ROOT}/kn_simulation/runs/nsbh_test/simulation_intermediates.h5}"
            NSBH_SIM_ROOT="${NSBH_SIM_ROOT:-${BASE_DIR}/SNANA/SNDATA_ROOT/SIM/LSST_KN_NSBH_TEST}"
            NSBH_SIM_NAME="${NSBH_SIM_NAME:-LSST_KN_NSBH_TEST}"
            NSBH_SUCCESS_IDS_PATH="${NSBH_SUCCESS_IDS_PATH:-${REPO_ROOT}/kn_simulation/runs/nsbh_test/success_sim_ids.txt}"

            OUTPUT_H5_PATH="${OUTPUT_H5_PATH:-${BASE_DIR}/data/ALBEF_dataset/combined_dataset_astro_${DATASET_MODE}.h5}"
            ;;
        *)
            echo "Unsupported PROFILE=$PROFILE. Use PROFILE=final_train or PROFILE=astro_test."
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
    python - "$h5_path" <<'PY_VALIDATE_H5'
import sys
import h5py
import numpy as np

h5_path = sys.argv[1]
base = "events/gw_data"
required = (
    "scalars", "skymaps", "ids", "event_uid", "simulation_id", "sample_class",
    "has_kn", "neg_type",
    "mej_dynamic", "mej_wind", "mej_tot", "event_time_mjd", "source_type",
)
with h5py.File(h5_path, "r") as f:
    missing = [f"{base}/{name}" for name in required if f"{base}/{name}" not in f]
    if missing:
        raise SystemExit(f"[SchemaError] Missing datasets: {missing}")
    n_gw = int(f[f"{base}/scalars"].shape[0])
    if f[f"{base}/scalars"].shape != (n_gw, 7):
        raise SystemExit(f"[SchemaError] scalars must have shape (n_gw, 7)")
    if f[f"{base}/skymaps"].shape != (n_gw, 7, 19200):
        raise SystemExit(f"[SchemaError] skymaps must have shape (n_gw, 7, 19200)")
    for name in required[2:]:
        if f[f"{base}/{name}"].shape != (n_gw,):
            raise SystemExit(
                f"[SchemaError] {base}/{name} must have shape ({n_gw},), "
                f"got {f[f'{base}/{name}'].shape}"
            )

    has_kn = np.asarray(f[f"{base}/has_kn"][:], dtype=np.int8)
    neg_type = np.asarray(f[f"{base}/neg_type"][:], dtype=np.int8)
    dynamic = np.asarray(f[f"{base}/mej_dynamic"][:], dtype=np.float64)
    wind = np.asarray(f[f"{base}/mej_wind"][:], dtype=np.float64)
    total = np.asarray(f[f"{base}/mej_tot"][:], dtype=np.float64)
    event_time = np.asarray(f[f"{base}/event_time_mjd"][:], dtype=np.float64)
    decode = lambda values: np.asarray(
        [value.decode() if isinstance(value, bytes) else str(value) for value in values]
    )
    ids = decode(f[f"{base}/ids"][:])
    event_uid = decode(f[f"{base}/event_uid"][:])
    simulation_id = np.asarray(f[f"{base}/simulation_id"][:], dtype=np.int64)
    sample_class = decode(f[f"{base}/sample_class"][:])
    source_type = decode(f[f"{base}/source_type"][:])
    if not np.array_equal(ids, event_uid) or len(set(event_uid.tolist())) != n_gw:
        raise SystemExit("[SchemaError] event_uid must be unique and match legacy ids")
    if (simulation_id < 0).any() or not np.isin(sample_class, ("pos", "neg")).all():
        raise SystemExit("[SchemaError] invalid simulation_id or sample_class")
    if not np.array_equal(sample_class == "neg", neg_type == 1):
        raise SystemExit("[SchemaError] only type-1 events may come from the neg stream")
    split = str(f.attrs["dataset_mode"])
    expected_uid = np.asarray(
        [f"{source}_{split}_{klass}_{sim_id}" for source, klass, sim_id in zip(
            source_type, sample_class, simulation_id
        )]
    )
    if not np.array_equal(event_uid, expected_uid):
        raise SystemExit("[SchemaError] non-canonical event_uid")
    if not np.isin(has_kn, (0, 1)).all() or not np.isin(neg_type, (0, 1, 2)).all():
        raise SystemExit("[SchemaError] has_kn or neg_type contains invalid labels")
    if not np.isfinite(dynamic).all() or not np.isfinite(wind).all():
        raise SystemExit("[SchemaError] ejecta components must be finite")
    if (dynamic < 0).any() or (wind < 0).any():
        raise SystemExit("[SchemaError] physical ejecta components must be non-negative")
    if not np.allclose(total, dynamic + wind, rtol=1e-5, atol=1e-8):
        raise SystemExit("[SchemaError] mej_tot != mej_dynamic + mej_wind")
    double_zero = (dynamic == 0.0) & (wind == 0.0)
    if not np.array_equal(neg_type == 1, double_zero):
        raise SystemExit("[SchemaError] type-1 must be exactly physical double-zero ejecta")
    if not np.all(total[neg_type != 1] > 0.0):
        raise SystemExit("[SchemaError] positive and type-2 events require total ejecta > 0")
    if not np.array_equal(has_kn == 1, neg_type == 0):
        raise SystemExit("[SchemaError] has_kn=1 must be equivalent to neg_type=0")

    parent_path = "events/optical_data/parent_gw_idx"
    if parent_path not in f:
        raise SystemExit(f"[SchemaError] Missing dataset: {parent_path}")
    parents = np.asarray(f[parent_path][:], dtype=np.int64)
    if ((parents < 0) | (parents >= n_gw)).any():
        raise SystemExit("[SchemaError] optical parent_gw_idx out of range")
    if parents.size and not np.all(has_kn[parents] == 1):
        raise SystemExit("[SchemaError] optical samples may only reference has_kn=1 GW events")

    invalid_time = int((~np.isfinite(event_time)).sum())
    counts = {label: int((neg_type == label).sum()) for label in (0, 1, 2)}
    print(
        f"[SchemaOK] n_gw={n_gw} n_optical={len(parents)} "
        f"positive={counts[0]} type1={counts[1]} type2={counts[2]} "
        f"invalid_event_time={invalid_time}"
    )
PY_VALIDATE_H5
}

set_profile_defaults

BNS_BUNDLE_ROOT="${BASE_DIR}/GWSamplegen/outputs/production_am_bayestar/dual/bns_${DATASET_MODE}_seed_1234"
NSBH_BUNDLE_ROOT="${BASE_DIR}/GWSamplegen/outputs/production_am_bayestar/dual/nsbh_${DATASET_MODE}_seed_1234"
BNS_NEGATIVE_CATALOG_PATH="${BNS_NEGATIVE_CATALOG_PATH:-${BNS_BUNDLE_ROOT}/neg_catalog.csv}"
BNS_NEGATIVE_SKYMAP_DIR="${BNS_NEGATIVE_SKYMAP_DIR:-${BASE_DIR}/data/skymap/negative/bns_skymap_${DATASET_MODE}}"
NSBH_NEGATIVE_CATALOG_PATH="${NSBH_NEGATIVE_CATALOG_PATH:-${NSBH_BUNDLE_ROOT}/neg_catalog.csv}"
NSBH_NEGATIVE_SKYMAP_DIR="${NSBH_NEGATIVE_SKYMAP_DIR:-${BASE_DIR}/data/skymap/negative/nsbh_skymap_${DATASET_MODE}}"

if [[ "$DATASET_MODE" != "train" && "$DATASET_MODE" != "test" ]]; then
    echo "DATASET_MODE must be train or test, got: $DATASET_MODE"
    exit 1
fi

echo "PROFILE=$PROFILE DATASET_MODE=$DATASET_MODE"

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

    export BASE_DIR PROFILE DATASET_MODE
    export BUFFER_LIMIT NUM_WORKERS SEED
    export BNS_MAX_LC_PER_GW NSBH_MAX_LC_PER_GW
    export BNS_MAX_NEG_GW NSBH_MAX_NEG_GW BNS_MAX_POS_GW NSBH_MAX_POS_GW
    export BNS_MAX_NEG_TYPE1_GW BNS_MAX_NEG_TYPE2_GW
    export NSBH_MAX_NEG_TYPE1_GW NSBH_MAX_NEG_TYPE2_GW
    export FLUXCAL_ZP PSFFLUX_ZP LUPT_K LUPT_M5_MAG
    export BNS_FULL_CATALOG_PATH BNS_SKYMAP_DIR BNS_SIM_ARTIFACT BNS_SIM_ROOT BNS_SIM_NAME BNS_SUCCESS_IDS_PATH
    export BNS_NEGATIVE_CATALOG_PATH BNS_NEGATIVE_SKYMAP_DIR
    export NSBH_FULL_CATALOG_PATH NSBH_SKYMAP_DIR NSBH_SIM_ARTIFACT NSBH_SIM_ROOT NSBH_SIM_NAME NSBH_SUCCESS_IDS_PATH
    export NSBH_NEGATIVE_CATALOG_PATH NSBH_NEGATIVE_SKYMAP_DIR
    export OUTPUT_H5_PATH
    sbatch_opts+=(--export=ALL)

    echo "Submitting job with: sbatch ${sbatch_opts[*]} ${script_path}"
    (
        cd "${REPO_ROOT}"
        sbatch "${sbatch_opts[@]}" "${script_path}"
    )
    exit 0
fi

require_file "$BNS_FULL_CATALOG_PATH"
require_file "$NSBH_FULL_CATALOG_PATH"
require_file "$BNS_NEGATIVE_CATALOG_PATH"
require_file "$NSBH_NEGATIVE_CATALOG_PATH"
require_dir "$BNS_SKYMAP_DIR"
require_dir "$NSBH_SKYMAP_DIR"
require_dir "$BNS_NEGATIVE_SKYMAP_DIR"
require_dir "$NSBH_NEGATIVE_SKYMAP_DIR"
if [[ -n "${BNS_SIM_ARTIFACT:-}" ]]; then
    require_file "$BNS_SIM_ARTIFACT"
else
    require_dir "$BNS_SIM_ROOT"
fi
if [[ -n "${NSBH_SIM_ARTIFACT:-}" ]]; then
    require_file "$NSBH_SIM_ARTIFACT"
else
    require_dir "$NSBH_SIM_ROOT"
fi

if [[ -z "${BNS_SUCCESS_IDS_PATH:-}" || -z "${NSBH_SUCCESS_IDS_PATH:-}" ]]; then
    echo "BNS_SUCCESS_IDS_PATH and NSBH_SUCCESS_IDS_PATH are required to distinguish type-2 negatives from missing observation coverage."
    exit 1
fi
require_file "$BNS_SUCCESS_IDS_PATH"
require_file "$NSBH_SUCCESS_IDS_PATH"

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
    --bns_negative_catalog_path "$BNS_NEGATIVE_CATALOG_PATH"
    --bns_negative_skymap_dir "$BNS_NEGATIVE_SKYMAP_DIR"
    --bns_sim_name "$BNS_SIM_NAME"
    --bns_max_lc_per_gw "$BNS_MAX_LC_PER_GW"
    --nsbh_full_catalog_path "$NSBH_FULL_CATALOG_PATH"
    --nsbh_skymap_dir "$NSBH_SKYMAP_DIR"
    --nsbh_negative_catalog_path "$NSBH_NEGATIVE_CATALOG_PATH"
    --nsbh_negative_skymap_dir "$NSBH_NEGATIVE_SKYMAP_DIR"
    --nsbh_sim_name "$NSBH_SIM_NAME"
    --nsbh_max_lc_per_gw "$NSBH_MAX_LC_PER_GW"
)

append_optional_arg --bns_sim_artifact "${BNS_SIM_ARTIFACT:-}"
append_optional_arg --bns_sim_root "${BNS_SIM_ROOT:-}"
append_optional_arg --bns_success_ids_path "${BNS_SUCCESS_IDS_PATH:-}"
append_optional_arg --nsbh_sim_artifact "${NSBH_SIM_ARTIFACT:-}"
append_optional_arg --nsbh_sim_root "${NSBH_SIM_ROOT:-}"
append_optional_arg --nsbh_success_ids_path "${NSBH_SUCCESS_IDS_PATH:-}"
append_optional_arg --bns_max_neg_gw "${BNS_MAX_NEG_GW:-}"
append_optional_arg --nsbh_max_neg_gw "${NSBH_MAX_NEG_GW:-}"
append_optional_arg --bns_max_pos_gw "${BNS_MAX_POS_GW:-}"
append_optional_arg --nsbh_max_pos_gw "${NSBH_MAX_POS_GW:-}"
append_optional_arg --bns_max_neg_type1_gw "${BNS_MAX_NEG_TYPE1_GW:-}"
append_optional_arg --bns_max_neg_type2_gw "${BNS_MAX_NEG_TYPE2_GW:-}"
append_optional_arg --nsbh_max_neg_type1_gw "${NSBH_MAX_NEG_TYPE1_GW:-}"
append_optional_arg --nsbh_max_neg_type2_gw "${NSBH_MAX_NEG_TYPE2_GW:-}"

echo "Output H5: $OUTPUT_H5_PATH"
echo "Luptitude params: FLUXCAL_ZP=$FLUXCAL_ZP PSFFLUX_ZP=$PSFFLUX_ZP LUPT_K=$LUPT_K LUPT_M5_MAG=$LUPT_M5_MAG"
echo "Parallel preprocessing workers: $NUM_WORKERS"
echo "Light-curve preprocessing: 2h same-band inverse-variance merge in psfFlux domain before luptitude conversion"
"${cmd[@]}"
validate_output_h5_schema "$OUTPUT_H5_PATH"
