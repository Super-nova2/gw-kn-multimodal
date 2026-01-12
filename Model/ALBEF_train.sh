#!/bin/bash
#SBATCH --job-name=GW_Opt_ALBEF_train
#SBATCH --output=logs/train/%x_%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --partition=gpu

set -euo pipefail
which python

args_file=${1:-}
if [[ -z "${args_file}" || ! -f "${args_file}" ]]; then
    echo "Usage: $0 <args.json>"
    exit 1
fi

DATA_PATH=$(jq -r '.data_path' "$args_file")
NEG_DATA_PATH=$(jq -r '.neg_data_path // empty' "$args_file")
NEG_GROUP=$(jq -r '.neg_group // empty' "$args_file")
CKPT_PATH=$(jq -r '.ckpt_path' "$args_file")
RESUME=$(jq -r '.resume // empty' "$args_file")
EPOCHS=$(jq -r '.epochs' "$args_file")
BATCH_SIZE=$(jq -r '.batch_size' "$args_file")
LR=$(jq -r '.lr' "$args_file")
NUM_WORKERS=$(jq -r '.num_workers' "$args_file")
STEPS_PER_EPOCH=$(jq -r '.steps_per_epoch // empty' "$args_file")
N_REF=$(jq -r '.n_ref // empty' "$args_file")
REF_START=$(jq -r '.ref_start // empty' "$args_file")
REF_END=$(jq -r '.ref_end // empty' "$args_file")
REF_DIM=$(jq -r '.ref_dim // empty' "$args_file")
ENC_DIM=$(jq -r '.enc_dim // empty' "$args_file")
PROJ_DIM=$(jq -r '.proj_dim // empty' "$args_file")
FUSION_ATTN_DIM=$(jq -r '.fusion_attn_dim // empty' "$args_file")
FUSION_HIDDEN_DIM=$(jq -r '.fusion_hidden_dim // empty' "$args_file")
FUSION_DROPOUT=$(jq -r '.fusion_dropout // empty' "$args_file")
TEMP_INIT=$(jq -r '.temp_init // empty' "$args_file")
ITC_WEIGHT=$(jq -r '.itc_weight // empty' "$args_file")
CLS_WEIGHT=$(jq -r '.cls_weight // empty' "$args_file")
HARD_NEG_START_EPOCH=$(jq -r '.hard_neg_start_epoch // empty' "$args_file")
MASK_ITC=$(jq -r '.mask_itc // false' "$args_file")

mkdir -p "$CKPT_PATH"

echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Start time: $(date)"
echo "------------------------------------------------"

cmd=(
    python -u /fred/oz016/bgao_kn/ML+GW+KN/Model/ALBEF_train.py
    --data_path "$DATA_PATH"
    --ckpt_path "$CKPT_PATH"
    --epochs "$EPOCHS"
    --batch_size "$BATCH_SIZE"
    --lr "$LR"
    --num_workers "$NUM_WORKERS"
)

if [[ -n "$NEG_DATA_PATH" && "$NEG_DATA_PATH" != "null" ]]; then
    cmd+=(--neg_data_path "$NEG_DATA_PATH")
fi
if [[ -n "$NEG_GROUP" && "$NEG_GROUP" != "null" ]]; then
    cmd+=(--neg_group "$NEG_GROUP")
fi
if [[ -n "$STEPS_PER_EPOCH" && "$STEPS_PER_EPOCH" != "null" ]]; then
    cmd+=(--steps_per_epoch "$STEPS_PER_EPOCH")
fi
if [[ -n "$RESUME" && "$RESUME" != "null" ]]; then
    cmd+=(--resume "$RESUME")
fi
if [[ -n "$N_REF" && "$N_REF" != "null" ]]; then
    cmd+=(--n_ref "$N_REF")
fi
if [[ -n "$REF_START" && "$REF_START" != "null" ]]; then
    cmd+=(--ref_start "$REF_START")
fi
if [[ -n "$REF_END" && "$REF_END" != "null" ]]; then
    cmd+=(--ref_end "$REF_END")
fi
if [[ -n "$REF_DIM" && "$REF_DIM" != "null" ]]; then
    cmd+=(--ref_dim "$REF_DIM")
fi
if [[ -n "$ENC_DIM" && "$ENC_DIM" != "null" ]]; then
    cmd+=(--enc_dim "$ENC_DIM")
fi
if [[ -n "$PROJ_DIM" && "$PROJ_DIM" != "null" ]]; then
    cmd+=(--proj_dim "$PROJ_DIM")
fi
if [[ -n "$FUSION_ATTN_DIM" && "$FUSION_ATTN_DIM" != "null" ]]; then
    cmd+=(--fusion_attn_dim "$FUSION_ATTN_DIM")
fi
if [[ -n "$FUSION_HIDDEN_DIM" && "$FUSION_HIDDEN_DIM" != "null" ]]; then
    cmd+=(--fusion_hidden_dim "$FUSION_HIDDEN_DIM")
fi
if [[ -n "$FUSION_DROPOUT" && "$FUSION_DROPOUT" != "null" ]]; then
    cmd+=(--fusion_dropout "$FUSION_DROPOUT")
fi
if [[ -n "$TEMP_INIT" && "$TEMP_INIT" != "null" ]]; then
    cmd+=(--temp_init "$TEMP_INIT")
fi
if [[ -n "$ITC_WEIGHT" && "$ITC_WEIGHT" != "null" ]]; then
    cmd+=(--itc_weight "$ITC_WEIGHT")
fi
if [[ -n "$CLS_WEIGHT" && "$CLS_WEIGHT" != "null" ]]; then
    cmd+=(--cls_weight "$CLS_WEIGHT")
fi
if [[ -n "$HARD_NEG_START_EPOCH" && "$HARD_NEG_START_EPOCH" != "null" ]]; then
    cmd+=(--hard_neg_start_epoch "$HARD_NEG_START_EPOCH")
fi
if [[ "$MASK_ITC" == "true" ]]; then
    cmd+=(--mask_itc)
fi

echo "Command: ${cmd[*]}"
"${cmd[@]}"
exit_code=$?
if [ $exit_code -ne 0 ]; then
    echo "Training script failed with exit code $exit_code"
    exit $exit_code
fi

echo "------------------------------------------------"
echo "End time: $(date)"
