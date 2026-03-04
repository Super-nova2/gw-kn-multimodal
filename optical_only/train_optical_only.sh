#!/bin/bash

#SBATCH --job-name=OPTICAL_ONLY_KN
#SBATCH --output=logs/optical_only/%x_%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=6:00:00
#SBATCH --partition=gpu
#SBATCH --tmp=160G

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
ARCH_VERSION=$(jq -r '.arch_version // empty' "$args_file")

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
RUN_NAME=$(jq -r '.run_name // empty' "$args_file")
EVAL_BATCH_SIZE=$(jq -r '.eval_batch_size // empty' "$args_file")
EVAL_NUM_WORKERS=$(jq -r '.eval_num_workers // empty' "$args_file")
EVAL_MAX_POS_SAMPLES=$(jq -r '.eval_max_pos_samples // empty' "$args_file")
EVAL_MAX_NEG_SAMPLES=$(jq -r '.eval_max_neg_samples // empty' "$args_file")
EVAL_SAMPLE_SEED=$(jq -r '.eval_sample_seed // empty' "$args_file")
EVAL_TARGET_RECALL=$(jq -r '.eval_target_recall // empty' "$args_file")
EVAL_NO_PLOTS=$(jq -r '.eval_no_plots // false' "$args_file")
EVAL_POS_DATA_PATH=$(jq -r '.eval_pos_data_path // empty' "$args_file")
EVAL_NEG_DATA_PATH=$(jq -r '.eval_neg_data_path // empty' "$args_file")
EVAL_NEG_GROUP=$(jq -r '.eval_neg_group // empty' "$args_file")

TIME_OFFSET_ENABLE=$(jq -r '.time_offset_enable // false' "$args_file")
OFFSET_DIST_NPZ=$(jq -r '.offset_dist_npz // empty' "$args_file")
OFFSET_DIST_KEY=$(jq -r '.offset_dist_key // empty' "$args_file")
OFFSET_TRAIN_SAMPLING=$(jq -r '.offset_train_sampling // empty' "$args_file")
OFFSET_EVAL_MODE=$(jq -r '.offset_eval_mode // empty' "$args_file")
OFFSET_EVAL_QUANTILES=$(jq -r '.offset_eval_quantiles // empty' "$args_file")
OFFSET_SCALE_DAYS_DIVISOR=$(jq -r '.offset_scale_days_divisor // empty' "$args_file")
OFFSET_SEED=$(jq -r '.offset_seed // empty' "$args_file")
OFFSET_BANK_SIZE=$(jq -r '.offset_bank_size // empty' "$args_file")

OPTICAL_V2_EVAL_POS_DEFAULT="/fred/oz016/bgao_kn/data/Optical_Only_dataset/combined_dataset_test.h5"
OPTICAL_V2_EVAL_NEG_DEFAULT="/fred/oz016/bgao_kn/data/Optical_Only_dataset/Tutorial_negative_dataset.h5"
OPTICAL_V2_EVAL_GROUP_DEFAULT="Tutorial/optical_data"

if [[ -z "$EVAL_POS_DATA_PATH" || "$EVAL_POS_DATA_PATH" == "null" ]]; then
    if [[ -n "$POS_DATA_PATH" && "$POS_DATA_PATH" != "null" ]]; then
        EVAL_POS_DATA_PATH="$POS_DATA_PATH"
    else
        EVAL_POS_DATA_PATH="$OPTICAL_V2_EVAL_POS_DEFAULT"
    fi
fi
if [[ -z "$EVAL_NEG_DATA_PATH" || "$EVAL_NEG_DATA_PATH" == "null" ]]; then
    if [[ -n "$NEG_DATA_PATH" && "$NEG_DATA_PATH" != "null" ]]; then
        EVAL_NEG_DATA_PATH="$NEG_DATA_PATH"
    else
        EVAL_NEG_DATA_PATH="$OPTICAL_V2_EVAL_NEG_DEFAULT"
    fi
fi
if [[ -z "$EVAL_NEG_GROUP" || "$EVAL_NEG_GROUP" == "null" ]]; then
    if [[ -n "$NEG_GROUP" && "$NEG_GROUP" != "null" ]]; then
        EVAL_NEG_GROUP="$NEG_GROUP"
    else
        EVAL_NEG_GROUP="$OPTICAL_V2_EVAL_GROUP_DEFAULT"
    fi
fi

if [[ -z "$RUN_NAME" || "$RUN_NAME" == "null" ]]; then
    if [[ -n "${SLURM_JOB_ID:-}" ]]; then
        RUN_NAME="job${SLURM_JOB_ID}"
    else
        RUN_NAME="run_$(date +%Y%m%d_%H%M%S)"
    fi
fi
RUN_NAME="${RUN_NAME//\//_}"
RUN_NAME="${RUN_NAME//\\/_}"
RUN_NAME="${RUN_NAME// /_}"
if [[ -z "$RUN_NAME" || "$RUN_NAME" == "." || "$RUN_NAME" == ".." ]]; then
    RUN_NAME="run_$(date +%Y%m%d_%H%M%S)"
fi

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
echo "Run Name: $RUN_NAME"
echo "Train POS data: $POS_DATA_PATH"
echo "Train NEG data: $NEG_DATA_PATH"
if [[ -n "$NEG_GROUP" && "$NEG_GROUP" != "null" ]]; then
    echo "Train NEG group: $NEG_GROUP"
fi
echo "Eval POS data (resolved): $EVAL_POS_DATA_PATH"
echo "Eval NEG data (resolved): $EVAL_NEG_DATA_PATH"
echo "Eval NEG group (resolved): $EVAL_NEG_GROUP"
echo ""

# Optional: stage large HDF5 to local disk to reduce shared filesystem I/O
if [[ "$STAGE_TO_JOBFS" == "true" ]]; then
    JOBFS_DIR="${SLURM_TMPDIR:-${TMPDIR:-${JOBFS:-}}}"
    if [[ -n "$JOBFS_DIR" ]]; then
        echo "Staging datasets to local disk: $JOBFS_DIR"
        TRAIN_POS_LOCAL="$JOBFS_DIR/train_pos_$(basename "$POS_DATA_PATH")"
        TRAIN_NEG_LOCAL="$JOBFS_DIR/train_neg_$(basename "$NEG_DATA_PATH")"
        EVAL_POS_LOCAL="$JOBFS_DIR/eval_pos_$(basename "$EVAL_POS_DATA_PATH")"
        EVAL_NEG_LOCAL="$JOBFS_DIR/eval_neg_$(basename "$EVAL_NEG_DATA_PATH")"
        OFFSET_DIST_LOCAL=""

        cp -f "$POS_DATA_PATH" "$TRAIN_POS_LOCAL"
        POS_DATA_PATH="$TRAIN_POS_LOCAL"
        cp -f "$NEG_DATA_PATH" "$TRAIN_NEG_LOCAL"
        NEG_DATA_PATH="$TRAIN_NEG_LOCAL"
        cp -f "$EVAL_POS_DATA_PATH" "$EVAL_POS_LOCAL"
        EVAL_POS_DATA_PATH="$EVAL_POS_LOCAL"
        cp -f "$EVAL_NEG_DATA_PATH" "$EVAL_NEG_LOCAL"
        EVAL_NEG_DATA_PATH="$EVAL_NEG_LOCAL"
        if [[ "$TIME_OFFSET_ENABLE" == "true" && -n "$OFFSET_DIST_NPZ" && "$OFFSET_DIST_NPZ" != "null" ]]; then
            OFFSET_DIST_LOCAL="$JOBFS_DIR/offset_dist_$(basename "$OFFSET_DIST_NPZ")"
            cp -f "$OFFSET_DIST_NPZ" "$OFFSET_DIST_LOCAL"
            OFFSET_DIST_NPZ="$OFFSET_DIST_LOCAL"
        fi
        echo "Staging complete."
    else
        echo "No local tmp dir found; skip staging."
    fi
fi

cmd=(
    python -u /fred/oz016/bgao_kn/ML+GW+KN/optical_only/train_optical_only.py
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
if [[ -n "$ARCH_VERSION" && "$ARCH_VERSION" != "null" ]]; then
    cmd+=(--arch_version "$ARCH_VERSION")
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
if [[ -n "$RUN_NAME" && "$RUN_NAME" != "null" ]]; then
    cmd+=(--run_name "$RUN_NAME")
fi
if [[ "$TIME_OFFSET_ENABLE" == "true" ]]; then
    cmd+=(--time_offset_enable)
fi
if [[ -n "$OFFSET_DIST_NPZ" && "$OFFSET_DIST_NPZ" != "null" ]]; then
    cmd+=(--offset_dist_npz "$OFFSET_DIST_NPZ")
fi
if [[ -n "$OFFSET_DIST_KEY" && "$OFFSET_DIST_KEY" != "null" ]]; then
    cmd+=(--offset_dist_key "$OFFSET_DIST_KEY")
fi
if [[ -n "$OFFSET_TRAIN_SAMPLING" && "$OFFSET_TRAIN_SAMPLING" != "null" ]]; then
    cmd+=(--offset_train_sampling "$OFFSET_TRAIN_SAMPLING")
fi
if [[ -n "$OFFSET_EVAL_MODE" && "$OFFSET_EVAL_MODE" != "null" ]]; then
    cmd+=(--offset_eval_mode "$OFFSET_EVAL_MODE")
fi
if [[ -n "$OFFSET_EVAL_QUANTILES" && "$OFFSET_EVAL_QUANTILES" != "null" ]]; then
    cmd+=(--offset_eval_quantiles "$OFFSET_EVAL_QUANTILES")
fi
if [[ -n "$OFFSET_SCALE_DAYS_DIVISOR" && "$OFFSET_SCALE_DAYS_DIVISOR" != "null" ]]; then
    cmd+=(--offset_scale_days_divisor "$OFFSET_SCALE_DAYS_DIVISOR")
fi
if [[ -n "$OFFSET_SEED" && "$OFFSET_SEED" != "null" ]]; then
    cmd+=(--offset_seed "$OFFSET_SEED")
fi
if [[ -n "$OFFSET_BANK_SIZE" && "$OFFSET_BANK_SIZE" != "null" ]]; then
    cmd+=(--offset_bank_size "$OFFSET_BANK_SIZE")
fi

echo "Training Command: ${cmd[*]}"
set +e
"${cmd[@]}"
train_exit_code=$?
set -e
if [ $train_exit_code -ne 0 ]; then
    echo "Training script failed with exit code $train_exit_code"
    echo "------------------------------------------------"
    echo "End time: $(date)"
    echo "Exit code: $train_exit_code"
    exit $train_exit_code
fi

BEST_CKPT="${CKPT_PATH}/optical_only/${RUN_NAME}/optical_only_best.pth"
EVAL_OUTPUT_DIR="${CKPT_PATH}/optical_only/eval_results/${RUN_NAME}"
EVAL_PY="/fred/oz016/bgao_kn/ML+GW+KN/optical_only/test_evaluate_optical_only.py"

if [[ ! -f "$BEST_CKPT" ]]; then
    echo "Best checkpoint not found after training: $BEST_CKPT"
    exit 1
fi
if [[ ! -f "$EVAL_PY" ]]; then
    echo "Evaluation script not found: $EVAL_PY"
    exit 1
fi
if [[ ! -f "$EVAL_POS_DATA_PATH" ]]; then
    echo "Evaluation positive dataset not found: $EVAL_POS_DATA_PATH"
    exit 1
fi
if [[ ! -f "$EVAL_NEG_DATA_PATH" ]]; then
    echo "Evaluation negative dataset not found: $EVAL_NEG_DATA_PATH"
    exit 1
fi
mkdir -p "$EVAL_OUTPUT_DIR"

echo "Evaluation POS data: $EVAL_POS_DATA_PATH"
echo "Evaluation NEG data: $EVAL_NEG_DATA_PATH"
if [[ -n "$EVAL_NEG_GROUP" && "$EVAL_NEG_GROUP" != "null" ]]; then
    echo "Evaluation NEG group: $EVAL_NEG_GROUP"
elif [[ -n "$NEG_GROUP" && "$NEG_GROUP" != "null" ]]; then
    echo "Evaluation NEG group: $NEG_GROUP (fallback from training args)"
fi

eval_cmd=(
    python -u "$EVAL_PY"
    --checkpoint "$BEST_CKPT"
    --config "$args_file"
    --pos_data_path "$EVAL_POS_DATA_PATH"
    --neg_data_path "$EVAL_NEG_DATA_PATH"
    --output_dir "$EVAL_OUTPUT_DIR"
    --device cuda
)

if [[ -n "$EVAL_NEG_GROUP" && "$EVAL_NEG_GROUP" != "null" ]]; then
    eval_cmd+=(--neg_group "$EVAL_NEG_GROUP")
elif [[ -n "$NEG_GROUP" && "$NEG_GROUP" != "null" ]]; then
    eval_cmd+=(--neg_group "$NEG_GROUP")
fi
if [[ -n "$EVAL_BATCH_SIZE" && "$EVAL_BATCH_SIZE" != "null" ]]; then
    eval_cmd+=(--batch_size "$EVAL_BATCH_SIZE")
elif [[ -n "$BATCH_SIZE" && "$BATCH_SIZE" != "null" ]]; then
    eval_cmd+=(--batch_size "$BATCH_SIZE")
fi
if [[ -n "$EVAL_NUM_WORKERS" && "$EVAL_NUM_WORKERS" != "null" ]]; then
    eval_cmd+=(--num_workers "$EVAL_NUM_WORKERS")
elif [[ -n "$NUM_WORKERS" && "$NUM_WORKERS" != "null" ]]; then
    eval_cmd+=(--num_workers "$NUM_WORKERS")
fi
if [[ -n "$PIN_MEMORY" && "$PIN_MEMORY" != "null" ]]; then
    eval_cmd+=(--pin_memory "$PIN_MEMORY")
fi
if [[ -n "$PERSISTENT_WORKERS" && "$PERSISTENT_WORKERS" != "null" ]]; then
    eval_cmd+=(--persistent_workers "$PERSISTENT_WORKERS")
fi
if [[ -n "$PREFETCH_FACTOR" && "$PREFETCH_FACTOR" != "null" ]]; then
    eval_cmd+=(--prefetch_factor "$PREFETCH_FACTOR")
fi
if [[ -n "$EVAL_MAX_POS_SAMPLES" && "$EVAL_MAX_POS_SAMPLES" != "null" ]]; then
    eval_cmd+=(--max_pos_samples "$EVAL_MAX_POS_SAMPLES")
fi
if [[ -n "$EVAL_MAX_NEG_SAMPLES" && "$EVAL_MAX_NEG_SAMPLES" != "null" ]]; then
    eval_cmd+=(--max_neg_samples "$EVAL_MAX_NEG_SAMPLES")
fi
if [[ -n "$EVAL_SAMPLE_SEED" && "$EVAL_SAMPLE_SEED" != "null" ]]; then
    eval_cmd+=(--sample_seed "$EVAL_SAMPLE_SEED")
fi
if [[ -n "$EVAL_TARGET_RECALL" && "$EVAL_TARGET_RECALL" != "null" ]]; then
    eval_cmd+=(--target_recall "$EVAL_TARGET_RECALL")
fi
if [[ "$EVAL_NO_PLOTS" == "true" ]]; then
    eval_cmd+=(--no_plots)
fi
if [[ "$TIME_OFFSET_ENABLE" == "true" ]]; then
    eval_cmd+=(--time_offset_enable)
fi
if [[ -n "$OFFSET_DIST_NPZ" && "$OFFSET_DIST_NPZ" != "null" ]]; then
    eval_cmd+=(--offset_dist_npz "$OFFSET_DIST_NPZ")
fi
if [[ -n "$OFFSET_DIST_KEY" && "$OFFSET_DIST_KEY" != "null" ]]; then
    eval_cmd+=(--offset_dist_key "$OFFSET_DIST_KEY")
fi
if [[ -n "$OFFSET_EVAL_MODE" && "$OFFSET_EVAL_MODE" != "null" ]]; then
    eval_cmd+=(--offset_eval_mode "$OFFSET_EVAL_MODE")
fi
if [[ -n "$OFFSET_EVAL_QUANTILES" && "$OFFSET_EVAL_QUANTILES" != "null" ]]; then
    eval_cmd+=(--offset_eval_quantiles "$OFFSET_EVAL_QUANTILES")
fi
if [[ -n "$OFFSET_SCALE_DAYS_DIVISOR" && "$OFFSET_SCALE_DAYS_DIVISOR" != "null" ]]; then
    eval_cmd+=(--offset_scale_days_divisor "$OFFSET_SCALE_DAYS_DIVISOR")
fi

echo "Evaluation Command: ${eval_cmd[*]}"
set +e
"${eval_cmd[@]}"
eval_exit_code=$?
set -e
if [ $eval_exit_code -ne 0 ]; then
    echo "Evaluation script failed with exit code $eval_exit_code"
    echo "------------------------------------------------"
    echo "End time: $(date)"
    echo "Exit code: $eval_exit_code"
    exit $eval_exit_code
fi

echo "Evaluation complete. Results saved to: $EVAL_OUTPUT_DIR"
echo "------------------------------------------------"
echo "End time: $(date)"
echo "Exit code: 0"

exit 0
