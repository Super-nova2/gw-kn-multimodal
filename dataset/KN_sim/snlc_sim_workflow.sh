#!/bin/bash
#SBATCH --job-name=LSST_KN_BNS
#SBATCH --time=5:00:00            
#SBATCH --cpus-per-task=1
#SBATCH --mem=10G
#SBATCH --array=0-182%10   # 1822/268 simulations, each job handles 10 sims, max 10 jobs running simultaneously
#SBATCH --output=logs/LSST_KN_BNS/%x_%a.out

# ml gcc/11.3/0 python/3.10.4
# ml gsl/2.7 cfitsio/4.2.0
set -euo pipefail

args_file=$1

sim_name=$(jq -r '.SIM_NAME' $args_file)
gw_type=$(jq -r '.GW_type' $args_file)
opsim_db=$(jq -r '.OpsimDB' $args_file)
nside=$(jq -r '.Nside' $args_file)
data_dir=$(jq -r '.DATA_DIR' $args_file)
input_dir=$(jq -r '.INPUT_DIR' $args_file)
simlib_dir=$(jq -r '.SIMLIB_DIR' $args_file)
inj_file=$(jq -r '.injections_file' $args_file)
out_dir=$(jq -r '.OUTPUT_DIR' $args_file)
log_dir=$(jq -r '.LOG_DIR' $args_file)
tem_input=$(jq -r '.TEMPLATE_INPUT' $args_file)

echo "Starting SNANA doc generation for simulation: $sim_name"
echo "Using Opsim DB: $opsim_db"
echo "Data directory: $data_dir"
echo "Input directory: $input_dir"
echo "Simlib directory: $simlib_dir"
echo "Injections file: $inj_file"
echo "SNANA simulation results output directory: ${out_dir}/${sim_name}"

# get all simulation IDs from the injections file
mapfile -t SIM_IDS < <(awk -F',' 'NR>1 {print $1}' ${inj_file})

N=$(tail -n +2 ${inj_file} | wc -l)     # total number of GW events
BATCH_SIZE=10
NTASK=$(((N + BATCH_SIZE - 1) / BATCH_SIZE))
echo "Number of GW events = $N, need array 0-$(($NTASK - 1)), each task handling up to $BATCH_SIZE events."

# select a subset of SIM_IDS based on SLURM_ARRAY_TASK_ID

task_id=${SLURM_ARRAY_TASK_ID}
start=$((task_id * BATCH_SIZE))
end=$((start + BATCH_SIZE - 1))
group_ids=("${SIM_IDS[@]:start:BATCH_SIZE}")

if (( end >= N )); then
    end=$((N - 1))
fi

echo "Task $task_id handles indices [$start, $end] / N=$N"
echo "Processing simulation ids: ${group_ids[@]}"

# gen_snana_doc=$(sbatch -J ${sim_name}_gen_snana_doc_%a --array=0-$(($NTASK - 1)) --cpus-per-task=1 --mem=8G \
#     --output=${log_dir}%x_%A_%a.out --parsable /fred/oz016/bgao_kn/ML+GW+KN/dataset/KN_sim/gen_snana_doc.sh ${args_file} ${SIM_IDS})

###################################################################
#-------Generating SIMLIB and INPUT files for snlc_sim.exe--------#
###################################################################

python /fred/oz016/bgao_kn/ML+GW+KN/dataset/KN_sim/gen_SNANA_doc.py --sim_name ${sim_name} \
    --GW_type ${gw_type} \
    --sim_ids "${group_ids[@]}" \
    --GW_params ${inj_file} \
    --Opsim ${opsim_db} \
    --within \
    --outdir ${data_dir} \
    --template_input ${tem_input}

###################################################################
#------------------Running snlc_sim.exe---------------------------#
###################################################################

failed_ids=()
for sim_id in "${group_ids[@]}"; do
    simlib="${simlib_dir}baseline_v5.0.1_10yrs_${sim_name}_${sim_id}.SIMLIB"
    input="${input_dir}SIMGEN_${sim_name}_${sim_id}.INPUT"

    echo "sim_id=$sim_id : checking files"
    #  check NLIBID in SIMLIB file, skip if NLIBID=0
    nlibid=$(grep -m1 'NLIBID' "$simlib" | awk '{print $2}')
    echo "    simid=$sim_id : NLIBID=$nlibid"
    if [[ "$nlibid" -eq 0 ]]; then
        echo "    SIMLIB has NLIBID=0, skip $sim_id"
        echo "    LSST do not cover the skymap of this event."
        echo "  [$sim_id] Skipped."
        rm -f "${simlib}"
        continue
    fi

    if [[ ! -f "$simlib" ]]; then
        echo "    ERROR: SIMLIB not found: $simlib"
        failed_ids+=("$sim_id")
        continue
    fi
    if [[ ! -f "$input" ]]; then
        echo "    ERROR: INPUT not found: $input"
        rm -f "$simlib"   # remove SIMLIB if INPUT is missing
        echo "    removed SIMLIB $simlib due to missing INPUT"
        failed_ids+=("$sim_id")
        continue
    fi

    echo "    running snlc_sim.exe $input"
    if ! snlc_sim.exe "$input" > /dev/null ; then
        echo "    snlc_sim failed for $sim_id"
        echo "  [$sim_id] Failed."
        # rm -f "$simlib"  # do not remove SIMLIB if snlc_sim fails
        failed_ids+=("$sim_id")
        continue
    fi

    # clean up SIMLIB file to save space
    echo "    snlc_sim success for $sim_id"
    echo "    removing SIMLIB $simlib"
    rm -f "$simlib"

    echo "  [$sim_id] done."
done

echo "Failed simulation IDs: ${failed_ids[@]}"
printf "%s " "${failed_ids[@]}" | tr -s ' ' >> "${data_dir}failed_sim_ids.txt"
echo "Failed simulation ids saved to ${data_dir}failed_sim_ids.txt"

echo "Task $task_id finished."
