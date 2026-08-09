#!/bin/bash
# GW170817A bright-source ToO pipeline.
# Stages: catalog -> snana_prepare -> snana_sim -> h5
# The output H5 overwrites gw170817a_lsst_redshift_test.h5 (no version suffix).

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
SCRIPT_PATH="${REPO_ROOT}/${SCRIPT_SUBDIR}/submit_gw170817a_lsst_pipeline.sh"
KN_SRC="${REPO_ROOT}/kn_simulation/src"

GEN_SCRIPT="${SCRIPT_DIR}/generate_gw170817a_lsst_docs.py"
BUILD_SCRIPT="${SCRIPT_DIR}/build_gw170817a_retrieval_h5.py"
SNANA_PREPARE="${KN_SRC}/snana.py"

GW170817A_DIR="${GW170817A_DIR:-${BASE_DIR}/data/GW_real_events/GW_data/GW170817A}"
OUTPUT_DIR="${OUTPUT_DIR:-${GW170817A_DIR}/lsst_redshift_experiment}"
SKYMAP_DIR="${SKYMAP_DIR:-${BASE_DIR}/data/skymap/gw170817a}"
SKYMAP="${SKYMAP:-${GW170817A_DIR}/bayestar_no_virgo.fits}"
POSTERIOR_H5="${POSTERIOR_H5:-${GW170817A_DIR}/GW170817_GWTC-1.hdf5}"
POSTERIOR_DATASET="${POSTERIOR_DATASET:-IMRPhenomPv2NRT_lowSpin_posterior}"
OPSIM_DB="${OPSIM_DB:-${BASE_DIR}/data/rubin_sim/baseline_v5.1/baseline_v5.1.1_10yrs.db}"
TEMPLATE_INPUT="${TEMPLATE_INPUT:-${REPO_ROOT}/kn_simulation/templates/bns.input}"
TOO_CONFIG="${TOO_CONFIG:-${REPO_ROOT}/kn_simulation/config/rubin_too_2024.yaml}"
SNDATA_ROOT="${SNDATA_ROOT:-${BASE_DIR}/SNANA/SNDATA_ROOT}"
SNANA_BIN_DIR="${SNANA_BIN_DIR:-${BASE_DIR}/SNANA/SNANA/bin}"
SNDATA_SIM_DIR="${SNDATA_SIM_DIR:-${SNDATA_ROOT}/SIM/gw170817a}"
COORDINATE_DIR="${COORDINATE_DIR:-${OUTPUT_DIR}/COORDINATES}"
SIM_NAME="${SIM_NAME:-LSST_KN_GW170817A}"
GENVERSION="${GENVERSION:-LSST_KN_GW170817A_REDSHIFT_GRID}"
REDSHIFTS="${REDSHIFTS:-0.01,0.02,0.03,0.05,0.08,0.12,0.16}"
N_PER_REDSHIFT="${N_PER_REDSHIFT:-800,800,1200,1300,2000,3000,6000}"
SEED="${SEED:-170817}"
MIN_NOBS="${MIN_NOBS:-5}"
SAMPLES_PER_EVENT="${SAMPLES_PER_EVENT:-64}"
SAMPLING_NSIDE="${SAMPLING_NSIDE:-256}"
BATCH_SIZE="${BATCH_SIZE:-500}"
NETWORK_SNR="${NETWORK_SNR:-32.4}"
TRIGGER_MJD="${TRIGGER_MJD:-62500.0}"
OUTPUT_H5="${OUTPUT_H5:-${BASE_DIR}/data/ALBEF_dataset/gw170817a_lsst_redshift_test.h5}"
STAGE="${STAGE:-all}"

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
    sbatch_opts+=(--export=ALL)
    echo "Submitting job with: sbatch ${sbatch_opts[*]} ${SCRIPT_PATH}"
    (
        cd "${WORKSPACE_ROOT}"
        sbatch "${sbatch_opts[@]}" "${SCRIPT_PATH}"
    )
    exit 0
fi

export SNDATA_ROOT
export PATH="${SNANA_BIN_DIR}:${PATH}"
NPROC="${NPROC:-${SLURM_CPUS_PER_TASK:-4}}"
export NPROC SNANA_PREPARE CATALOG SKYMAP_DIR SIM_NAME OPSIM_DB TEMPLATE_INPUT TOO_CONFIG
export SAMPLES_PER_EVENT SAMPLING_NSIDE OUTPUT_DIR SNDATA_SIM_DIR

case "${STAGE}" in
  all|catalog|snana|sim|h5) ;;
  *) echo "Unsupported STAGE=${STAGE}"; exit 1 ;;
esac

mkdir -p "${OUTPUT_DIR}" "${SKYMAP_DIR}" "${SNDATA_SIM_DIR}"

if [[ "${STAGE}" == "all" || "${STAGE}" == "catalog" ]]; then
  echo "== Stage: catalog =="
  python -u "${GEN_SCRIPT}" \
    --skymap "${SKYMAP}" \
    --output-dir "${OUTPUT_DIR}" \
    --skymap-dir "${SKYMAP_DIR}" \
    --redshifts "${REDSHIFTS}" \
    --n-per-redshift "${N_PER_REDSHIFT}" \
    --seed "${SEED}" \
    --posterior-h5 "${POSTERIOR_H5}" \
    --posterior-dataset "${POSTERIOR_DATASET}" \
    --network-snr "${NETWORK_SNR}" \
    --trigger-mjd "${TRIGGER_MJD}" \
    --prepared-catalog
fi

CATALOG="${OUTPUT_DIR}/gw170817a_prepared_catalog.csv"
IDS_FILE="${OUTPUT_DIR}/simulation_ids.txt"

if [[ "${STAGE}" == "all" || "${STAGE}" == "snana" ]]; then
  echo "== Stage: snana prepare =="
  rm -rf "${OUTPUT_DIR}/SIM_INPUT" "${OUTPUT_DIR}/SIMLIB" "${COORDINATE_DIR}"
  rm -f "${OUTPUT_DIR}"/sim_ids_chunk_*
  mkdir -p "${OUTPUT_DIR}/SIM_INPUT" "${OUTPUT_DIR}/SIMLIB"
  mkdir -p "${COORDINATE_DIR}"
  split -l "${BATCH_SIZE}" -d -a 4 "${IDS_FILE}" "${OUTPUT_DIR}/sim_ids_chunk_"
  find "${OUTPUT_DIR}" -maxdepth 1 -name 'sim_ids_chunk_*' -print0 | sort -z | \
    xargs -0 -P "${NPROC}" -I{} bash -c '
      echo "Preparing chunk: $1"
      python -u "$SNANA_PREPARE"         --GW_type bns         --sim-id-file "$1"         --GW_catalog "$CATALOG"         --skymap_path "$SKYMAP_DIR"         --sim_name "$SIM_NAME"         --Opsim "$OPSIM_DB"         --template_input "$TEMPLATE_INPUT"         --too_config "$TOO_CONFIG"         --coordinate_mode posterior_test         --samples_per_event "$SAMPLES_PER_EVENT"         --sampling_nside "$SAMPLING_NSIDE"         --cosmology Planck15         --outdir "$OUTPUT_DIR"         --sndata-sim-dir "$SNDATA_SIM_DIR"
    ' _ {}
fi

if [[ "${STAGE}" == "all" || "${STAGE}" == "sim" ]]; then
  echo "== Stage: snana simulation =="
  find "${OUTPUT_DIR}/SIM_INPUT" -maxdepth 1 -name 'SIMGEN_*' -print0 | sort -z | \
    xargs -0 -P "${NPROC}" -I{} bash -c '
      echo "Running SNANA: $1"
      snlc_sim.exe "$1"
    ' _ {}
fi

if [[ "${STAGE}" == "all" || "${STAGE}" == "h5" ]]; then
  echo "== Stage: h5 =="
  python -u "${BUILD_SCRIPT}" \
    --sim-dir "${SNDATA_SIM_DIR}" \
    --genversion "${GENVERSION}" \
    --manifest "${OUTPUT_DIR}/gw170817a_lsst_manifest.csv" \
    --coordinate-dir "${COORDINATE_DIR}" \
    --skymap "${SKYMAP}" \
    --posterior-h5 "${POSTERIOR_H5}" \
    --posterior-dataset "${POSTERIOR_DATASET}" \
    --output-h5 "${OUTPUT_H5}" \
    --min-nobs "${MIN_NOBS}"
fi

echo "GW170817A pipeline finished."
echo "Output H5: ${OUTPUT_H5}"
