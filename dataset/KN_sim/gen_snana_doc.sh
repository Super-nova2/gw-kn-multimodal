#!/bin/bash
#SBATCH --job-name=gen_snana_doc
#SBATCH --time=00:30:00             # 运行时间上限
#SBATCH --cpus-per-task=1
#SBATCH --array=0-1
#SBATCH --mem=8G
#SBATCH --output=${BASE_DIR}/logs/%x_%j.out

# ml gcc/11.3/0 python/3.10.4
# ml gsl/2.7 cfitsio/4.2.0

BASE_DIR="${BASE_DIR:-/fred/oz016/bgao_kn}"

echo "Starting SNANA doc generation..."

SCRIPT_SUBDIR="dataset/KN_sim"
SCRIPT_REL_PATH="dataset/KN_sim/gen_snana_doc.sh"
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

args_file=${1:-}
if [[ -z "${args_file}" || ! -f "${args_file}" ]]; then
    echo "Usage: $0 <args.json>" >&2
    exit 1
fi

sim_name=$(jq -r '.SIM_NAME' "${args_file}")
opsim_db=$(jq -r '.OpsimDB' "${args_file}")
nside=$(jq -r '.Nside' "${args_file}")
data_dir=$(jq -r '.DATA_DIR' "${args_file}")
input_dir=$(jq -r '.INPUT_DIR' "${args_file}")
simlib_dir=$(jq -r '.SIMLIB_DIR' "${args_file}")
inj_file=$(jq -r '.injections_file' "${args_file}")
out_dir=$(jq -r '.OUTPUT_DIR' "${args_file}")

# get all simulation IDs from the injections file
mapfile -t SIM_IDS < <(awk -F',' 'NR>1 {print $1}' "${inj_file}")
# SIM_IDS=$2

# select a subset of SIM_IDS based on SLURM_ARRAY_TASK_ID
BATCH_SIZE=10
task_id=${SLURM_ARRAY_TASK_ID}
start=$((task_id * BATCH_SIZE))
end=$((start + BATCH_SIZE - 1))
chunk=("${SIM_IDS[@]:start:BATCH_SIZE}")
echo "Processing simulation ids: ${chunk[@]}"

echo "Task $task_id: handling index [$start, $end]"

python "${SCRIPT_DIR}/gen_SNANA_doc.py" --sim_ids "${chunk[@]}" \
    --GW_params ${inj_file} \
    --Opsim ${opsim_db} \
    --within \
    --outdir ${data_dir} \
    --template_input ${input_dir}/SIMGEN_KN_LSST_TEMPLATE.INPUT
