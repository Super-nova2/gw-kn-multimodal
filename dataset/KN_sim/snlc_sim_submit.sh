#!/bin/bash

# Submit SNANA simulations to Slurm, automatically computing sbatch array size.
# Usage: ./snlc_sim_submit.sh args.json

set -euo pipefail

args_file=${1:-}
if [[ -z "${args_file}" || ! -f "${args_file}" ]]; then
    echo "Usage: $0 <args.json>"
    exit 1
fi


# Configurable resource knobs (override via env if needed)
TIME_LIMIT=${TIME_LIMIT:-4:00:00}
CPUS_PER_TASK=${CPUS_PER_TASK:-1}
MEM_PER_TASK=${MEM_PER_TASK:-10G}
BATCH_SIZE=${BATCH_SIZE:-20}          # must match batch size in snlc_sim_workflow.sh
MAX_ARRAY_CONCURRENCY=${MAX_ARRAY_CONCURRENCY:-20}

sim_name=$(jq -r '.SIM_NAME' "${args_file}")
inj_file=$(jq -r '.injections_file' "${args_file}")
log_dir=$(jq -r '.LOG_DIR' "${args_file}")
data_dir=$(jq -r '.DATA_DIR' "${args_file}")

# Count events (skip header row)
N=$(tail -n +2 "${inj_file}" | wc -l)
NTASK=$(((N + BATCH_SIZE - 1) / BATCH_SIZE))

if (( NTASK <= 0 )); then
    echo "No events found in ${inj_file}"
    exit 1
fi

mkdir -p "${log_dir}"

echo "SIM_NAME           : ${sim_name}"
echo "Injections file    : ${inj_file}"
echo "Total events (N)   : ${N}"
echo "Batch size         : ${BATCH_SIZE}"
echo "Array tasks needed : ${NTASK} (0-$((${NTASK} - 1)))"
echo "Max concurrency    : ${MAX_ARRAY_CONCURRENCY}"
echo "Log directory      : ${log_dir}"

# Resolve workflow path relative to this script
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
workflow="${script_dir}/snlc_sim_workflow.sh"

# remove all old failed ids files
rm -f "${data_dir}failed_sim_ids.txt"
echo "Have removed old failed_sim_ids.txt"

sbatch \
    -J "${sim_name}" \
    --time="${TIME_LIMIT}" \
    --cpus-per-task="${CPUS_PER_TASK}" \
    --mem="${MEM_PER_TASK}" \
    --array=0-$((${NTASK} - 1))%${MAX_ARRAY_CONCURRENCY} \
    --output="${log_dir%/}/${sim_name}/%x_%a.out" \
    "${workflow}" "${args_file}"
