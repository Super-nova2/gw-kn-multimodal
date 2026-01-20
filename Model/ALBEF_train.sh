#!/bin/bash

#SBATCH --job-name=ALBEF_anti_overfit
#SBATCH --output=logs/train/%x_%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
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
STAGE_TO_JOBFS=$(jq -r '.stage_to_jobfs // false' "$args_file")
RESUME=$(jq -r '.resume // empty' "$args_file")
PRETRAINED=$(jq -r '.pretrained // empty' "$args_file")
EPOCHS=$(jq -r '.epochs' "$args_file")
BATCH_SIZE=$(jq -r '.batch_size' "$args_file")
LR=$(jq -r '.lr' "$args_file")
WEIGHT_DECAY=$(jq -r '.weight_decay // empty' "$args_file")
GRAD_CLIP_NORM=$(jq -r '.grad_clip_norm // empty' "$args_file")
LR_SCHEDULER=$(jq -r '.lr_scheduler // empty' "$args_file")
WARMUP_EPOCHS=$(jq -r '.warmup_epochs // empty' "$args_file")
MIN_LR=$(jq -r '.min_lr // empty' "$args_file")
NUM_WORKERS=$(jq -r '.num_workers' "$args_file")
STEPS_PER_EPOCH=$(jq -r '.steps_per_epoch // empty' "$args_file")
PIN_MEMORY=$(jq -r '.pin_memory // empty' "$args_file")
PERSISTENT_WORKERS=$(jq -r '.persistent_workers // empty' "$args_file")
PREFETCH_FACTOR=$(jq -r '.prefetch_factor // empty' "$args_file")
CACHE_IN_MEMORY=$(jq -r '.cache_in_memory // false' "$args_file")
VAL_SPLIT=$(jq -r '.val_split // empty' "$args_file")
VAL_BATCH_SIZE=$(jq -r '.val_batch_size // empty' "$args_file")
VAL_STEPS_PER_EPOCH=$(jq -r '.val_steps_per_epoch // empty' "$args_file")
SPLIT_SEED=$(jq -r '.split_seed // empty' "$args_file")
EARLY_STOP_PATIENCE=$(jq -r '.early_stop_patience // empty' "$args_file")
EARLY_STOP_MIN_DELTA=$(jq -r '.early_stop_min_delta // empty' "$args_file")
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
TEMP_FINAL=$(jq -r '.temp_final // empty' "$args_file")
TEMP_MIN=$(jq -r '.temp_min // empty' "$args_file")
TEMP_MAX=$(jq -r '.temp_max // empty' "$args_file")
TEMP_SCHEDULE=$(jq -r '.temp_schedule // empty' "$args_file")
GW_DROPOUT=$(jq -r '.gw_dropout // empty' "$args_file")
OPT_DROPOUT=$(jq -r '.opt_dropout // empty' "$args_file")
PROJ_DROPOUT=$(jq -r '.proj_dropout // empty' "$args_file")
LABEL_SMOOTHING=$(jq -r '.label_smoothing // empty' "$args_file")
FEATURE_DROPOUT=$(jq -r '.feature_dropout // empty' "$args_file")
FREEZE_ENCODER_EPOCHS=$(jq -r '.freeze_encoder_epochs // empty' "$args_file")
FREEZE_ITC_EPOCHS=$(jq -r '.freeze_itc_epochs // empty' "$args_file")
ITC_WEIGHT=$(jq -r '.itc_weight // empty' "$args_file")
CLS_WEIGHT=$(jq -r '.cls_weight // empty' "$args_file")
CLS_POS_WEIGHT=$(jq -r '.cls_pos_weight // empty' "$args_file")
CLS_NEG_WEIGHT=$(jq -r '.cls_neg_weight // empty' "$args_file")
CLS_EXTRA_NEG_WEIGHT=$(jq -r '.cls_extra_neg_weight // empty' "$args_file")
CLS_RAMP_EPOCHS=$(jq -r '.cls_ramp_epochs // empty' "$args_file")
ITC_DECAY_START_EPOCH=$(jq -r '.itc_decay_start_epoch // empty' "$args_file")
ITC_DECAY_EPOCHS=$(jq -r '.itc_decay_epochs // empty' "$args_file")
ITC_DECAY_RATIO=$(jq -r '.itc_decay_ratio // empty' "$args_file")
ITC_LABEL_SMOOTHING=$(jq -r '.itc_label_smoothing // empty' "$args_file")
HARD_NEG_START_EPOCH=$(jq -r '.hard_neg_start_epoch // empty' "$args_file")
HARD_NEG_RAMP_EPOCHS=$(jq -r '.hard_neg_ramp_epochs // empty' "$args_file")
CLS_START_EPOCH=$(jq -r '.cls_start_epoch // empty' "$args_file")
MASK_ITC=$(jq -r '.mask_itc // false' "$args_file")
USE_LIGHTWEIGHT_GW=$(jq -r '.use_lightweight_gw // false' "$args_file")
GW_AUG_NOISE=$(jq -r '.gw_aug_noise // empty' "$args_file")
GW_AUG_JITTER=$(jq -r '.gw_aug_jitter // empty' "$args_file")
GW_AUG_DROPOUT=$(jq -r '.gw_aug_dropout // empty' "$args_file")
OPT_AUG_NOISE=$(jq -r '.opt_aug_noise // empty' "$args_file")
OPT_AUG_TIME_JITTER=$(jq -r '.opt_aug_time_jitter // empty' "$args_file")
OPT_AUG_DROPOUT=$(jq -r '.opt_aug_dropout // empty' "$args_file")
OPT_AUG_BAND_DROPOUT=$(jq -r '.opt_aug_band_dropout // empty' "$args_file")

mkdir -p "$CKPT_PATH"

echo "========================================"
echo "SLURM Job Information"
echo "========================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node: $SLURMD_NODENAME"
echo "Partition: $SLURM_JOB_PARTITION"
echo "CPUs: $SLURM_CPUS_PER_TASK"
echo "Memory: ${SLURM_MEM_PER_NODE}MB"
echo "Start time: $(date)"
echo "========================================"
echo ""
echo "Configuration File: $args_file"
echo "Checkpoint Path: $CKPT_PATH"
echo ""

# Optional: stage large HDF5 to local disk to reduce Lustre I/O
if [ "$STAGE_TO_JOBFS" = "true" ]; then
    JOBFS_DIR="${SLURM_TMPDIR:-${TMPDIR:-${JOBFS:-}}}"
    if [ -n "$JOBFS_DIR" ]; then
        echo "Staging datasets to local disk: $JOBFS_DIR"
        cp -f "$DATA_PATH" "$JOBFS_DIR"/
        DATA_PATH="$JOBFS_DIR/$(basename "$DATA_PATH")"
        if [[ -n "$NEG_DATA_PATH" && "$NEG_DATA_PATH" != "null" ]]; then
            cp -f "$NEG_DATA_PATH" "$JOBFS_DIR"/
            NEG_DATA_PATH="$JOBFS_DIR/$(basename "$NEG_DATA_PATH")"
        fi
    else
        echo "No local tmp dir found; skip staging."
    fi
fi

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
if [[ -n "$PRETRAINED" && "$PRETRAINED" != "null" ]]; then
    cmd+=(--pretrained "$PRETRAINED")
fi
if [[ -n "$LR_SCHEDULER" && "$LR_SCHEDULER" != "null" ]]; then
    cmd+=(--lr_scheduler "$LR_SCHEDULER")
fi
if [[ -n "$WARMUP_EPOCHS" && "$WARMUP_EPOCHS" != "null" ]]; then
    cmd+=(--warmup_epochs "$WARMUP_EPOCHS")
fi
if [[ -n "$MIN_LR" && "$MIN_LR" != "null" ]]; then
    cmd+=(--min_lr "$MIN_LR")
fi
if [[ -n "$WEIGHT_DECAY" && "$WEIGHT_DECAY" != "null" ]]; then
    cmd+=(--weight_decay "$WEIGHT_DECAY")
fi
if [[ -n "$GRAD_CLIP_NORM" && "$GRAD_CLIP_NORM" != "null" ]]; then
    cmd+=(--grad_clip_norm "$GRAD_CLIP_NORM")
fi
if [[ -n "$PIN_MEMORY" && "$PIN_MEMORY" != "null" ]]; then
    cmd+=(--pin_memory "$PIN_MEMORY")
fi
if [[ -n "$PERSISTENT_WORKERS" && "$PERSISTENT_WORKERS" != "null" ]]; then
    cmd+=(--persistent_workers "$PERSISTENT_WORKERS")
fi
if [[ -n "$PREFETCH_FACTOR" && "$PREFETCH_FACTOR" != "null" ]]; then
    cmd+=(--prefetch_factor "$PREFETCH_FACTOR")
fi
if [[ "$CACHE_IN_MEMORY" == "true" ]]; then
    cmd+=(--cache_in_memory)
fi
if [[ -n "$VAL_SPLIT" && "$VAL_SPLIT" != "null" ]]; then
    cmd+=(--val_split "$VAL_SPLIT")
fi
if [[ -n "$VAL_BATCH_SIZE" && "$VAL_BATCH_SIZE" != "null" ]]; then
    cmd+=(--val_batch_size "$VAL_BATCH_SIZE")
fi
if [[ -n "$VAL_STEPS_PER_EPOCH" && "$VAL_STEPS_PER_EPOCH" != "null" ]]; then
    cmd+=(--val_steps_per_epoch "$VAL_STEPS_PER_EPOCH")
fi
if [[ -n "$SPLIT_SEED" && "$SPLIT_SEED" != "null" ]]; then
    cmd+=(--split_seed "$SPLIT_SEED")
fi
if [[ -n "$EARLY_STOP_PATIENCE" && "$EARLY_STOP_PATIENCE" != "null" ]]; then
    cmd+=(--early_stop_patience "$EARLY_STOP_PATIENCE")
fi
if [[ -n "$EARLY_STOP_MIN_DELTA" && "$EARLY_STOP_MIN_DELTA" != "null" ]]; then
    cmd+=(--early_stop_min_delta "$EARLY_STOP_MIN_DELTA")
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
if [[ -n "$TEMP_FINAL" && "$TEMP_FINAL" != "null" ]]; then
    cmd+=(--temp_final "$TEMP_FINAL")
fi
if [[ -n "$TEMP_MIN" && "$TEMP_MIN" != "null" ]]; then
    cmd+=(--temp_min "$TEMP_MIN")
fi
if [[ -n "$TEMP_MAX" && "$TEMP_MAX" != "null" ]]; then
    cmd+=(--temp_max "$TEMP_MAX")
fi
if [[ -n "$TEMP_SCHEDULE" && "$TEMP_SCHEDULE" != "null" ]]; then
    cmd+=(--temp_schedule "$TEMP_SCHEDULE")
fi
if [[ -n "$GW_DROPOUT" && "$GW_DROPOUT" != "null" ]]; then
    cmd+=(--gw_dropout "$GW_DROPOUT")
fi
if [[ -n "$OPT_DROPOUT" && "$OPT_DROPOUT" != "null" ]]; then
    cmd+=(--opt_dropout "$OPT_DROPOUT")
fi
if [[ -n "$PROJ_DROPOUT" && "$PROJ_DROPOUT" != "null" ]]; then
    cmd+=(--proj_dropout "$PROJ_DROPOUT")
fi
if [[ -n "$LABEL_SMOOTHING" && "$LABEL_SMOOTHING" != "null" ]]; then
    cmd+=(--label_smoothing "$LABEL_SMOOTHING")
fi
if [[ -n "$FEATURE_DROPOUT" && "$FEATURE_DROPOUT" != "null" ]]; then
    cmd+=(--feature_dropout "$FEATURE_DROPOUT")
fi
if [[ -n "$FREEZE_ENCODER_EPOCHS" && "$FREEZE_ENCODER_EPOCHS" != "null" ]]; then
    cmd+=(--freeze_encoder_epochs "$FREEZE_ENCODER_EPOCHS")
fi
if [[ -n "$FREEZE_ITC_EPOCHS" && "$FREEZE_ITC_EPOCHS" != "null" ]]; then
    cmd+=(--freeze_itc_epochs "$FREEZE_ITC_EPOCHS")
fi
if [[ -n "$ITC_WEIGHT" && "$ITC_WEIGHT" != "null" ]]; then
    cmd+=(--itc_weight "$ITC_WEIGHT")
fi
if [[ -n "$CLS_WEIGHT" && "$CLS_WEIGHT" != "null" ]]; then
    cmd+=(--cls_weight "$CLS_WEIGHT")
fi
if [[ -n "$CLS_POS_WEIGHT" && "$CLS_POS_WEIGHT" != "null" ]]; then
    cmd+=(--cls_pos_weight "$CLS_POS_WEIGHT")
fi
if [[ -n "$CLS_NEG_WEIGHT" && "$CLS_NEG_WEIGHT" != "null" ]]; then
    cmd+=(--cls_neg_weight "$CLS_NEG_WEIGHT")
fi
if [[ -n "$CLS_EXTRA_NEG_WEIGHT" && "$CLS_EXTRA_NEG_WEIGHT" != "null" ]]; then
    cmd+=(--cls_extra_neg_weight "$CLS_EXTRA_NEG_WEIGHT")
fi
if [[ -n "$CLS_RAMP_EPOCHS" && "$CLS_RAMP_EPOCHS" != "null" ]]; then
    cmd+=(--cls_ramp_epochs "$CLS_RAMP_EPOCHS")
fi
if [[ -n "$ITC_DECAY_START_EPOCH" && "$ITC_DECAY_START_EPOCH" != "null" ]]; then
    cmd+=(--itc_decay_start_epoch "$ITC_DECAY_START_EPOCH")
fi
if [[ -n "$ITC_DECAY_EPOCHS" && "$ITC_DECAY_EPOCHS" != "null" ]]; then
    cmd+=(--itc_decay_epochs "$ITC_DECAY_EPOCHS")
fi
if [[ -n "$ITC_DECAY_RATIO" && "$ITC_DECAY_RATIO" != "null" ]]; then
    cmd+=(--itc_decay_ratio "$ITC_DECAY_RATIO")
fi
if [[ -n "$ITC_LABEL_SMOOTHING" && "$ITC_LABEL_SMOOTHING" != "null" ]]; then
    cmd+=(--itc_label_smoothing "$ITC_LABEL_SMOOTHING")
fi
if [[ -n "$HARD_NEG_START_EPOCH" && "$HARD_NEG_START_EPOCH" != "null" ]]; then
    cmd+=(--hard_neg_start_epoch "$HARD_NEG_START_EPOCH")
fi
if [[ -n "$HARD_NEG_RAMP_EPOCHS" && "$HARD_NEG_RAMP_EPOCHS" != "null" ]]; then
    cmd+=(--hard_neg_ramp_epochs "$HARD_NEG_RAMP_EPOCHS")
fi
if [[ -n "$CLS_START_EPOCH" && "$CLS_START_EPOCH" != "null" ]]; then
    cmd+=(--cls_start_epoch "$CLS_START_EPOCH")
fi
if [[ "$MASK_ITC" == "true" ]]; then
    cmd+=(--mask_itc)
fi
if [[ "$USE_LIGHTWEIGHT_GW" == "true" ]]; then
    cmd+=(--use_lightweight_gw)
fi
if [[ -n "$GW_AUG_NOISE" && "$GW_AUG_NOISE" != "null" ]]; then
    cmd+=(--gw_aug_noise "$GW_AUG_NOISE")
fi
if [[ -n "$GW_AUG_JITTER" && "$GW_AUG_JITTER" != "null" ]]; then
    cmd+=(--gw_aug_jitter "$GW_AUG_JITTER")
fi
if [[ -n "$GW_AUG_DROPOUT" && "$GW_AUG_DROPOUT" != "null" ]]; then
    cmd+=(--gw_aug_dropout "$GW_AUG_DROPOUT")
fi
if [[ -n "$OPT_AUG_NOISE" && "$OPT_AUG_NOISE" != "null" ]]; then
    cmd+=(--opt_aug_noise "$OPT_AUG_NOISE")
fi
if [[ -n "$OPT_AUG_TIME_JITTER" && "$OPT_AUG_TIME_JITTER" != "null" ]]; then
    cmd+=(--opt_aug_time_jitter "$OPT_AUG_TIME_JITTER")
fi
if [[ -n "$OPT_AUG_DROPOUT" && "$OPT_AUG_DROPOUT" != "null" ]]; then
    cmd+=(--opt_aug_dropout "$OPT_AUG_DROPOUT")
fi
if [[ -n "$OPT_AUG_BAND_DROPOUT" && "$OPT_AUG_BAND_DROPOUT" != "null" ]]; then
    cmd+=(--opt_aug_band_dropout "$OPT_AUG_BAND_DROPOUT")
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
