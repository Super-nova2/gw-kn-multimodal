#!/bin/bash

#SBATCH --job-name=GWOptical_Fusion_v1
#SBATCH --output=logs/train/%x_%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=136G
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --partition=gpu
#SBATCH --tmp=110G

set -euo pipefail

args_file=${1:-}
if [[ -z "${args_file}" || ! -f "${args_file}" ]]; then
    echo "Usage: $0 <args.json>"
    exit 1
fi

args_dir="$(cd "$(dirname "${args_file}")" && pwd)"
args_file="${args_dir}/$(basename "${args_file}")"

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    if ! command -v sbatch >/dev/null 2>&1; then
        echo "sbatch not found; run inside a Slurm allocation or install Slurm tools."
        exit 1
    fi

    script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    script_path="${script_dir}/$(basename "${BASH_SOURCE[0]}")"

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
    if [[ -n "${GPUS:-}" ]]; then
        sbatch_opts+=(--gres="gpu:${GPUS}")
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

    echo "Submitting job with: sbatch ${sbatch_opts[*]} ${script_path} ${args_file}"
    sbatch "${sbatch_opts[@]}" "${script_path}" "${args_file}"
    exit 0
fi

which python

DATA_PATH=$(jq -r '.data_path' "$args_file")
NEG_DATA_PATH=$(jq -r '.neg_data_path // empty' "$args_file")
NEG_GROUP=$(jq -r '.neg_group // empty' "$args_file")
USE_NEG_GW=$(jq -r '.use_neg_gw // false' "$args_file")
NEG_GW_RATIO=$(jq -r '.neg_gw_ratio // empty' "$args_file")
SAMPLES_PER_GW=$(jq -r '.samples_per_gw // empty' "$args_file")
CKPT_PATH=$(jq -r '.ckpt_path' "$args_file")
STAGE_TO_JOBFS=$(jq -r '.stage_to_jobfs // false' "$args_file")
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
OPTICAL_DIM=$(jq -r '.optical_dim // empty' "$args_file")
FUSION_ATTN_DIM=$(jq -r '.fusion_attn_dim // empty' "$args_file")
FUSION_HIDDEN_DIM=$(jq -r '.fusion_hidden_dim // empty' "$args_file")
FUSION_DROPOUT=$(jq -r '.fusion_dropout // empty' "$args_file")
GW_DROPOUT=$(jq -r '.gw_dropout // empty' "$args_file")
OPT_DROPOUT=$(jq -r '.opt_dropout // empty' "$args_file")
LABEL_SMOOTHING=$(jq -r '.label_smoothing // empty' "$args_file")
POS_WEIGHT=$(jq -r '.pos_weight // empty' "$args_file")
NEG_WEIGHT=$(jq -r '.neg_weight // empty' "$args_file")
EXTRA_NEG_WEIGHT=$(jq -r '.extra_neg_weight // empty' "$args_file")
LOG_EVERY=$(jq -r '.log_every // empty' "$args_file")

mkdir -p "$CKPT_PATH"

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
    python -u /fred/oz016/bgao_kn/ML+GW+KN/Model_v1/train_v1.py
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
if [[ "$USE_NEG_GW" = "true" ]]; then
    cmd+=(--use_neg_gw)
fi
if [[ -n "$NEG_GW_RATIO" && "$NEG_GW_RATIO" != "null" ]]; then
    cmd+=(--neg_gw_ratio "$NEG_GW_RATIO")
fi
if [[ -n "$SAMPLES_PER_GW" && "$SAMPLES_PER_GW" != "null" ]]; then
    cmd+=(--samples_per_gw "$SAMPLES_PER_GW")
fi
if [[ -n "$WEIGHT_DECAY" && "$WEIGHT_DECAY" != "null" ]]; then
    cmd+=(--weight_decay "$WEIGHT_DECAY")
fi
if [[ -n "$GRAD_CLIP_NORM" && "$GRAD_CLIP_NORM" != "null" ]]; then
    cmd+=(--grad_clip_norm "$GRAD_CLIP_NORM")
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
if [[ -n "$STEPS_PER_EPOCH" && "$STEPS_PER_EPOCH" != "null" ]]; then
    cmd+=(--steps_per_epoch "$STEPS_PER_EPOCH")
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
if [[ -n "$CACHE_IN_MEMORY" && "$CACHE_IN_MEMORY" != "null" ]]; then
    if [[ "$CACHE_IN_MEMORY" = "true" ]]; then
        cmd+=(--cache_in_memory)
    fi
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
if [[ -n "$OPTICAL_DIM" && "$OPTICAL_DIM" != "null" ]]; then
    cmd+=(--optical_dim "$OPTICAL_DIM")
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
if [[ -n "$GW_DROPOUT" && "$GW_DROPOUT" != "null" ]]; then
    cmd+=(--gw_dropout "$GW_DROPOUT")
fi
if [[ -n "$OPT_DROPOUT" && "$OPT_DROPOUT" != "null" ]]; then
    cmd+=(--opt_dropout "$OPT_DROPOUT")
fi
if [[ -n "$LABEL_SMOOTHING" && "$LABEL_SMOOTHING" != "null" ]]; then
    cmd+=(--label_smoothing "$LABEL_SMOOTHING")
fi
if [[ -n "$POS_WEIGHT" && "$POS_WEIGHT" != "null" ]]; then
    cmd+=(--pos_weight "$POS_WEIGHT")
fi
if [[ -n "$NEG_WEIGHT" && "$NEG_WEIGHT" != "null" ]]; then
    cmd+=(--neg_weight "$NEG_WEIGHT")
fi
if [[ -n "$EXTRA_NEG_WEIGHT" && "$EXTRA_NEG_WEIGHT" != "null" ]]; then
    cmd+=(--extra_neg_weight "$EXTRA_NEG_WEIGHT")
fi
if [[ -n "$LOG_EVERY" && "$LOG_EVERY" != "null" ]]; then
    cmd+=(--log_every "$LOG_EVERY")
fi

printf 'Running: %q ' "${cmd[@]}"
echo

"${cmd[@]}"
