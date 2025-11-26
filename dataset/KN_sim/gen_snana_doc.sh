#!/bin/bash
#SBATCH --job-name=gen_snana_doc
#SBATCH --time=00:30:00             # 运行时间上限
#SBATCH --cpus-per-task=1
#SBATCH --array=0-1
#SBATCH --mem=8G
#SBATCH --output=/fred/oz016/bgao_kn/logs/%x_%j.out

# ml gcc/11.3/0 python/3.10.4
# ml gsl/2.7 cfitsio/4.2.0

echo "Starting SNANA doc generation..."

args_file=$1
sim_name=$(jq -r '.SIM_NAME' $args_file)
opsim_db=$(jq -r '.OpsimDB' $args_file)
nside=$(jq -r '.Nside' $args_file)
data_dir=$(jq -r '.DATA_DIR' $args_file)
input_dir=$(jq -r '.INPUT_DIR' $args_file)
simlib_dir=$(jq -r '.SIMLIB_DIR' $args_file)
inj_file=$(jq -r '.injections_file' $args_file)
out_dir=$(jq -r '.OUTPUT_DIR' $args_file)

# get all simulation IDs from the injections file
mapfile -t SIM_IDS < <(awk -F',' 'NR>1 {print $1}' ${inj_file})
# SIM_IDS=$2

# select a subset of SIM_IDS based on SLURM_ARRAY_TASK_ID
BATCH_SIZE=10
task_id=${SLURM_ARRAY_TASK_ID}
start=$((task_id * BATCH_SIZE))
end=$((start + BATCH_SIZE - 1))
chunk=("${SIM_IDS[@]:start:BATCH_SIZE}")
echo "Processing simulation ids: ${chunk[@]}"

echo "Task $task_id: handling index [$start, $end]"

python /fred/oz016/bgao_kn/ML+GW+KN/dataset/KN_sim/gen_SNANA_doc.py --sim_ids "${chunk[@]}" \
    --GW_params ${inj_file} \
    --Opsim ${opsim_db} \
    --within \
    --outdir ${data_dir} \
    --template_input ${input_dir}/SIMGEN_KN_LSST_TEMPLATE.INPUT

