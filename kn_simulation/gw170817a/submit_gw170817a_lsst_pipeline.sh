#!/bin/bash
# Fixed-physics GW170817A counterfactual LSST scenario pipeline.
# Stages: catalog -> snana_prepare -> snana_sim -> h5

#SBATCH --job-name=GW170817A_LSST
#SBATCH --output=/fred/oz016/bgao_kn/logs/data/%x_%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=8:00:00
#SBATCH --partition=milan

set -euo pipefail

SCRIPT_SUBDIR="kn_simulation/gw170817a"
REPO_NAME="gw-kn-multimodal"
BASE_DIR="${BASE_DIR:-/fred/oz016/bgao_kn}"
WORKSPACE_ROOT="${WORKSPACE_ROOT:-${BASE_DIR}}"
REPO_ROOT="${WORKSPACE_ROOT}/${REPO_NAME}"
SCRIPT_DIR="${REPO_ROOT}/${SCRIPT_SUBDIR}"
SCRIPT_PATH="${SCRIPT_DIR}/submit_gw170817a_lsst_pipeline.sh"
KN_SRC="${REPO_ROOT}/kn_simulation/src"

GEN_SCRIPT="${SCRIPT_DIR}/generate_gw170817a_lsst_docs.py"
BUILD_SCRIPT="${SCRIPT_DIR}/build_gw170817a_retrieval_h5.py"
SNANA_PREPARE="${KN_SRC}/snana.py"

GW170817A_DIR="${GW170817A_DIR:-${BASE_DIR}/data/GW_real_events/GW_data/GW170817A}"
OUTPUT_DIR="${OUTPUT_DIR:-${GW170817A_DIR}/lsst_scenario_experiment}"
SKYMAP_DIR="${SKYMAP_DIR:-${GW170817A_DIR}}"
SKYMAP="${SKYMAP:-${GW170817A_DIR}/bayestar_no_virgo.fits}"
POSTERIOR_H5="${POSTERIOR_H5:-${GW170817A_DIR}/GW170817_GWTC-1.hdf5}"
POSTERIOR_DATASET="${POSTERIOR_DATASET:-IMRPhenomPv2NRT_lowSpin_posterior}"
OPSIM_DB="${OPSIM_DB:-${BASE_DIR}/data/rubin_sim/baseline_v5.1/baseline_v5.1.1_10yrs.db}"
TEMPLATE_INPUT="${TEMPLATE_INPUT:-${REPO_ROOT}/kn_simulation/templates/bns.input}"
TOO_CONFIG="${TOO_CONFIG:-${REPO_ROOT}/kn_simulation/config/rubin_too_2024.yaml}"
SNDATA_ROOT="${SNDATA_ROOT:-${BASE_DIR}/SNANA/SNDATA_ROOT}"
SNANA_BIN_DIR="${SNANA_BIN_DIR:-${BASE_DIR}/SNANA/SNANA/bin}"
SNDATA_SIM_DIR="${SNDATA_SIM_DIR:-${SNDATA_ROOT}/SIM/gw170817a_scenarios}"
COORDINATE_DIR="${COORDINATE_DIR:-${OUTPUT_DIR}/COORDINATES}"
SIM_NAME="${SIM_NAME:-LSST_KN_GW170817A}"
GENVERSION="${GENVERSION:-LSST_KN_GW170817A_SCENARIOS}"
N_SCENARIOS="${N_SCENARIOS:-10}"
CANDIDATE_COORDINATES="${CANDIDATE_COORDINATES:-80}"
TARGET_PER_SCENARIO="${TARGET_PER_SCENARIO:-50}"
SEED="${SEED:-170817}"
MIN_NOBS="${MIN_NOBS:-5}"
SAMPLING_NSIDE="${SAMPLING_NSIDE:-256}"
BATCH_SIZE="${BATCH_SIZE:-500}"
NETWORK_SNR="${NETWORK_SNR:-32.4}"
REAL_TRIGGER_MJD="${REAL_TRIGGER_MJD:-57982.528523}"
HOST_DISTANCE_MPC="${HOST_DISTANCE_MPC:-40.7}"
HOST_REDSHIFT_OBSERVED="${HOST_REDSHIFT_OBSERVED:-0.009783}"
OUTPUT_H5="${OUTPUT_H5:-${BASE_DIR}/data/ALBEF_dataset/gw170817a_lsst_scenarios.h5}"
STAGE="${STAGE:-all}"

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    if ! command -v sbatch >/dev/null 2>&1; then
        echo "sbatch not found; run inside a Slurm allocation." >&2
        exit 1
    fi
    sbatch_opts=(--export=ALL)
    [[ -z "${JOB_NAME:-}" ]] || sbatch_opts+=(--job-name="${JOB_NAME}")
    [[ -z "${OUTPUT_LOG:-}" ]] || sbatch_opts+=(--output="${OUTPUT_LOG}")
    [[ -z "${TIME_LIMIT:-}" ]] || sbatch_opts+=(--time="${TIME_LIMIT}")
    [[ -z "${PARTITION:-}" ]] || sbatch_opts+=(--partition="${PARTITION}")
    [[ -z "${CPUS_PER_TASK:-}" ]] || sbatch_opts+=(--cpus-per-task="${CPUS_PER_TASK}")
    [[ -z "${MEM_PER_TASK:-}" ]] || sbatch_opts+=(--mem="${MEM_PER_TASK}")
    echo "Submitting: sbatch ${sbatch_opts[*]} ${SCRIPT_PATH}"
    (
        cd "${WORKSPACE_ROOT}"
        sbatch "${sbatch_opts[@]}" "${SCRIPT_PATH}"
    )
    exit 0
fi

case "${STAGE}" in
  all|catalog|snana|sim|h5) ;;
  *) echo "Unsupported STAGE=${STAGE}" >&2; exit 1 ;;
esac

export SNDATA_ROOT
export PATH="${SNANA_BIN_DIR}:${PATH}"
NPROC="${NPROC:-${SLURM_CPUS_PER_TASK:-4}}"
mkdir -p "${OUTPUT_DIR}" "${SNDATA_SIM_DIR}"

if [[ "${STAGE}" == "all" || "${STAGE}" == "catalog" ]]; then
  echo "== Stage: scenario catalog =="
  python -u "${GEN_SCRIPT}" \
    --skymap "${SKYMAP}" \
    --posterior-h5 "${POSTERIOR_H5}" \
    --posterior-dataset "${POSTERIOR_DATASET}" \
    --opsim-db "${OPSIM_DB}" \
    --output-dir "${OUTPUT_DIR}" \
    --n-scenarios "${N_SCENARIOS}" \
    --candidate-coordinates "${CANDIDATE_COORDINATES}" \
    --real-trigger-mjd "${REAL_TRIGGER_MJD}" \
    --host-distance-mpc "${HOST_DISTANCE_MPC}" \
    --host-redshift-observed "${HOST_REDSHIFT_OBSERVED}" \
    --network-snr "${NETWORK_SNR}" \
    --seed "${SEED}"
fi

CATALOG="${OUTPUT_DIR}/gw170817a_prepared_catalog.csv"
IDS_FILE="${OUTPUT_DIR}/simulation_ids.txt"

if [[ "${STAGE}" == "all" || "${STAGE}" == "snana" ]]; then
  echo "== Stage: SNANA preparation =="
  rm -rf "${OUTPUT_DIR}/SIM_INPUT" "${OUTPUT_DIR}/SIMLIB" "${COORDINATE_DIR}"
  rm -f "${OUTPUT_DIR}"/sim_ids_chunk_*
  mkdir -p "${OUTPUT_DIR}/SIM_INPUT" "${OUTPUT_DIR}/SIMLIB" "${COORDINATE_DIR}"
  split -l "${BATCH_SIZE}" -d -a 4 "${IDS_FILE}" "${OUTPUT_DIR}/sim_ids_chunk_"
  export SNANA_PREPARE CATALOG SKYMAP_DIR SIM_NAME OPSIM_DB TEMPLATE_INPUT TOO_CONFIG
  export SAMPLING_NSIDE OUTPUT_DIR SNDATA_SIM_DIR
  find "${OUTPUT_DIR}" -maxdepth 1 -name 'sim_ids_chunk_*' -print0 | sort -z | \
    xargs -0 -P "${NPROC}" -I{} bash -c '
      python -u "$SNANA_PREPARE" \
        --GW_type bns --sim-id-file "$1" --GW_catalog "$CATALOG" \
        --skymap_path "$SKYMAP_DIR" --sim_name "$SIM_NAME" --Opsim "$OPSIM_DB" \
        --template_input "$TEMPLATE_INPUT" --too_config "$TOO_CONFIG" \
        --coordinate_mode posterior_fixed_distance_with_truth \
        --sampling_nside "$SAMPLING_NSIDE" --cosmology Planck15 \
        --outdir "$OUTPUT_DIR" --sndata-sim-dir "$SNDATA_SIM_DIR"
    ' _ {}
fi

if [[ "${STAGE}" == "all" || "${STAGE}" == "sim" ]]; then
  echo "== Stage: SNANA simulation =="
  find "${OUTPUT_DIR}/SIM_INPUT" -maxdepth 1 -name 'SIMGEN_*' -print0 | sort -z | \
    xargs -0 -P "${NPROC}" -I{} bash -c 'snlc_sim.exe "$1"' _ {}
fi

if [[ "${STAGE}" == "all" || "${STAGE}" == "h5" ]]; then
  echo "== Stage: HDF5 =="
  python -u "${BUILD_SCRIPT}" \
    --sim-dir "${SNDATA_SIM_DIR}" \
    --genversion "${GENVERSION}" \
    --manifest "${OUTPUT_DIR}/gw170817a_lsst_manifest.csv" \
    --coordinate-dir "${COORDINATE_DIR}" \
    --skymap "${SKYMAP}" \
    --posterior-h5 "${POSTERIOR_H5}" \
    --posterior-dataset "${POSTERIOR_DATASET}" \
    --output-h5 "${OUTPUT_H5}" \
    --min-nobs "${MIN_NOBS}" \
    --target-per-scenario "${TARGET_PER_SCENARIO}" \
    --selection-seed "${SEED}"
fi

echo "GW170817A scenario pipeline finished."
echo "Output H5: ${OUTPUT_H5}"
