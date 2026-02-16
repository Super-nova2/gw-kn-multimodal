#!/bin/bash

#SBATCH --job-name=OPTICAL_ONLY_KN
#SBATCH --output=logs/optical_only/%x_%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --partition=gpu
#SBATCH --tmp=110G

set -euo pipefail

args_file=${1:-}
if [[ -z "${args_file}" || ! -f "${args_file}" ]]; then
    echo "Usage: $0 <optical_only_args.json>"
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

POS_DATA_PATH=$(jq -r '.pos_data_path' "$args_file")
NEG_DATA_PATH=$(jq -r '.neg_data_path' "$args_file")
NEG_GROUP=$(jq -r '.neg_group // empty' "$args_file")
CKPT_PATH=$(jq -r '.ckpt_path' "$args_file")
PRETRAINED_ALBEF_CKPT=$(jq -r '.pretrained_albef_ckpt // empty' "$args_file")
STAGE_TO_JOBFS=$(jq -r '.stage_to_jobfs // false' "$args_file")

EPOCHS_STAGE1=$(jq -r '.epochs_stage1 // empty' "$args_file")
EPOCHS_STAGE2=$(jq -r '.epochs_stage2 // empty' "$args_file")
BATCH_SIZE=$(jq -r '.batch_size // empty' "$args_file")
VAL_BATCH_SIZE=$(jq -r '.val_batch_size // empty' "$args_file")
STEPS_PER_EPOCH=$(jq -r '.steps_per_epoch // empty' "$args_file")
VAL_STEPS_PER_EPOCH=$(jq -r '.val_steps_per_epoch // empty' "$args_file")
VAL_SPLIT=$(jq -r '.val_split // empty' "$args_file")
SPLIT_SEED=$(jq -r '.split_seed // empty' "$args_file")

LR_HEAD_STAGE1=$(jq -r '.lr_head_stage1 // empty' "$args_file")
LR_HEAD_STAGE2=$(jq -r '.lr_head_stage2 // empty' "$args_file")
LR_ENCODER_STAGE2=$(jq -r '.lr_encoder_stage2 // empty' "$args_file")
WEIGHT_DECAY=$(jq -r '.weight_decay // empty' "$args_file")
GRAD_CLIP_NORM=$(jq -r '.grad_clip_norm // empty' "$args_file")

N_REF=$(jq -r '.n_ref // empty' "$args_file")
REF_START=$(jq -r '.ref_start // empty' "$args_file")
REF_END=$(jq -r '.ref_end // empty' "$args_file")
REF_DIM=$(jq -r '.ref_dim // empty' "$args_file")
ENC_DIM=$(jq -r '.enc_dim // empty' "$args_file")

OPT_DROPOUT=$(jq -r '.opt_dropout // empty' "$args_file")
FEATURE_DROPOUT=$(jq -r '.feature_dropout // empty' "$args_file")
HEAD_HIDDEN_DIM=$(jq -r '.head_hidden_dim // empty' "$args_file")
HEAD_DROPOUT=$(jq -r '.head_dropout // empty' "$args_file")
INCLUDE_COORDS=$(jq -r '.include_coords // false' "$args_file")

OPT_AUG_NOISE=$(jq -r '.opt_aug_noise // empty' "$args_file")
OPT_AUG_TIME_JITTER=$(jq -r '.opt_aug_time_jitter // empty' "$args_file")
OPT_AUG_DROPOUT=$(jq -r '.opt_aug_dropout // empty' "$args_file")
OPT_AUG_BAND_DROPOUT=$(jq -r '.opt_aug_band_dropout // empty' "$args_file")

TARGET_RECALL=$(jq -r '.target_recall // empty' "$args_file")
EARLY_STOP_PATIENCE=$(jq -r '.early_stop_patience // empty' "$args_file")
EARLY_STOP_MIN_DELTA=$(jq -r '.early_stop_min_delta // empty' "$args_file")

NUM_WORKERS=$(jq -r '.num_workers // empty' "$args_file")
PIN_MEMORY=$(jq -r '.pin_memory // empty' "$args_file")
PERSISTENT_WORKERS=$(jq -r '.persistent_workers // empty' "$args_file")
PREFETCH_FACTOR=$(jq -r '.prefetch_factor // empty' "$args_file")
CACHE_IN_MEMORY=$(jq -r '.cache_in_memory // false' "$args_file")
SEED=$(jq -r '.seed // empty' "$args_file")
TB_LOG_DIR=$(jq -r '.tb_log_dir // empty' "$args_file")
TB_FLUSH_SECS=$(jq -r '.tb_flush_secs // empty' "$args_file")
DISABLE_TENSORBOARD=$(jq -r '.disable_tensorboard // false' "$args_file")

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

# Optional: stage large HDF5 to local disk to reduce shared filesystem I/O
if [[ "$STAGE_TO_JOBFS" == "true" ]]; then
    JOBFS_DIR="${SLURM_TMPDIR:-${TMPDIR:-${JOBFS:-}}}"
    if [[ -n "$JOBFS_DIR" ]]; then
        echo "Staging datasets to local disk: $JOBFS_DIR"
        cp -f "$POS_DATA_PATH" "$JOBFS_DIR"/
        POS_DATA_PATH="$JOBFS_DIR/$(basename "$POS_DATA_PATH")"
        cp -f "$NEG_DATA_PATH" "$JOBFS_DIR"/
        NEG_DATA_PATH="$JOBFS_DIR/$(basename "$NEG_DATA_PATH")"
        echo "Staging complete."
    else
        echo "No local tmp dir found; skip staging."
    fi
fi

cmd=(
    python -u /fred/oz016/bgao_kn/ML+GW+KN/Model/optical_only/train_optical_only.py
    --pos_data_path "$POS_DATA_PATH"
    --neg_data_path "$NEG_DATA_PATH"
    --ckpt_path "$CKPT_PATH"
)

if [[ -n "$NEG_GROUP" && "$NEG_GROUP" != "null" ]]; then
    cmd+=(--neg_group "$NEG_GROUP")
fi
if [[ -n "$PRETRAINED_ALBEF_CKPT" && "$PRETRAINED_ALBEF_CKPT" != "null" ]]; then
    cmd+=(--pretrained_albef_ckpt "$PRETRAINED_ALBEF_CKPT")
fi

if [[ -n "$EPOCHS_STAGE1" && "$EPOCHS_STAGE1" != "null" ]]; then
    cmd+=(--epochs_stage1 "$EPOCHS_STAGE1")
fi
if [[ -n "$EPOCHS_STAGE2" && "$EPOCHS_STAGE2" != "null" ]]; then
    cmd+=(--epochs_stage2 "$EPOCHS_STAGE2")
fi
if [[ -n "$BATCH_SIZE" && "$BATCH_SIZE" != "null" ]]; then
    cmd+=(--batch_size "$BATCH_SIZE")
fi
if [[ -n "$VAL_BATCH_SIZE" && "$VAL_BATCH_SIZE" != "null" ]]; then
    cmd+=(--val_batch_size "$VAL_BATCH_SIZE")
fi
if [[ -n "$STEPS_PER_EPOCH" && "$STEPS_PER_EPOCH" != "null" ]]; then
    cmd+=(--steps_per_epoch "$STEPS_PER_EPOCH")
fi
if [[ -n "$VAL_STEPS_PER_EPOCH" && "$VAL_STEPS_PER_EPOCH" != "null" ]]; then
    cmd+=(--val_steps_per_epoch "$VAL_STEPS_PER_EPOCH")
fi
if [[ -n "$VAL_SPLIT" && "$VAL_SPLIT" != "null" ]]; then
    cmd+=(--val_split "$VAL_SPLIT")
fi
if [[ -n "$SPLIT_SEED" && "$SPLIT_SEED" != "null" ]]; then
    cmd+=(--split_seed "$SPLIT_SEED")
fi

if [[ -n "$LR_HEAD_STAGE1" && "$LR_HEAD_STAGE1" != "null" ]]; then
    cmd+=(--lr_head_stage1 "$LR_HEAD_STAGE1")
fi
if [[ -n "$LR_HEAD_STAGE2" && "$LR_HEAD_STAGE2" != "null" ]]; then
    cmd+=(--lr_head_stage2 "$LR_HEAD_STAGE2")
fi
if [[ -n "$LR_ENCODER_STAGE2" && "$LR_ENCODER_STAGE2" != "null" ]]; then
    cmd+=(--lr_encoder_stage2 "$LR_ENCODER_STAGE2")
fi
if [[ -n "$WEIGHT_DECAY" && "$WEIGHT_DECAY" != "null" ]]; then
    cmd+=(--weight_decay "$WEIGHT_DECAY")
fi
if [[ -n "$GRAD_CLIP_NORM" && "$GRAD_CLIP_NORM" != "null" ]]; then
    cmd+=(--grad_clip_norm "$GRAD_CLIP_NORM")
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

if [[ -n "$OPT_DROPOUT" && "$OPT_DROPOUT" != "null" ]]; then
    cmd+=(--opt_dropout "$OPT_DROPOUT")
fi
if [[ -n "$FEATURE_DROPOUT" && "$FEATURE_DROPOUT" != "null" ]]; then
    cmd+=(--feature_dropout "$FEATURE_DROPOUT")
fi
if [[ -n "$HEAD_HIDDEN_DIM" && "$HEAD_HIDDEN_DIM" != "null" ]]; then
    cmd+=(--head_hidden_dim "$HEAD_HIDDEN_DIM")
fi
if [[ -n "$HEAD_DROPOUT" && "$HEAD_DROPOUT" != "null" ]]; then
    cmd+=(--head_dropout "$HEAD_DROPOUT")
fi
if [[ "$INCLUDE_COORDS" == "true" ]]; then
    cmd+=(--include_coords)
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

if [[ -n "$TARGET_RECALL" && "$TARGET_RECALL" != "null" ]]; then
    cmd+=(--target_recall "$TARGET_RECALL")
fi
if [[ -n "$EARLY_STOP_PATIENCE" && "$EARLY_STOP_PATIENCE" != "null" ]]; then
    cmd+=(--early_stop_patience "$EARLY_STOP_PATIENCE")
fi
if [[ -n "$EARLY_STOP_MIN_DELTA" && "$EARLY_STOP_MIN_DELTA" != "null" ]]; then
    cmd+=(--early_stop_min_delta "$EARLY_STOP_MIN_DELTA")
fi

if [[ -n "$NUM_WORKERS" && "$NUM_WORKERS" != "null" ]]; then
    cmd+=(--num_workers "$NUM_WORKERS")
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
if [[ -n "$SEED" && "$SEED" != "null" ]]; then
    cmd+=(--seed "$SEED")
fi
if [[ -n "$TB_LOG_DIR" && "$TB_LOG_DIR" != "null" ]]; then
    cmd+=(--tb_log_dir "$TB_LOG_DIR")
fi
if [[ -n "$TB_FLUSH_SECS" && "$TB_FLUSH_SECS" != "null" ]]; then
    cmd+=(--tb_flush_secs "$TB_FLUSH_SECS")
fi
if [[ "$DISABLE_TENSORBOARD" == "true" ]]; then
    cmd+=(--disable_tensorboard)
fi

echo "Command: ${cmd[*]}"
"${cmd[@]}"
exit_code=$?

echo "------------------------------------------------"
echo "End time: $(date)"
echo "Exit code: $exit_code"

exit $exit_code
