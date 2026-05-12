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
#SBATCH --tmp=200G

set -euo pipefail

BASE_DIR="${BASE_DIR:-/fred/oz016/bgao_kn}"

SCRIPT_SUBDIR="optical_only"
SCRIPT_REL_PATH="optical_only/train_optical_only.sh"
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
DEFAULT_ARGS_FILE="${SCRIPT_DIR}/args/optical_only_kn_baseline.json"
args_file=${1:-${OPTICAL_ONLY_ARGS_FILE:-${DEFAULT_ARGS_FILE}}}
if [[ ! -f "${args_file}" ]]; then
    echo "Args file not found: ${args_file}"
    echo "Usage: $0 [optical_only_args.json]"
    echo "Default args file: ${DEFAULT_ARGS_FILE}"
    exit 1
fi

args_dir="$(cd "$(dirname "${args_file}")" && pwd)"
args_file="${args_dir}/$(basename "${args_file}")"

is_truthy() {
    local v
    v="$(echo "${1:-}" | tr '[:upper:]' '[:lower:]')"
    case "${v}" in
        1|true|yes|y|on) return 0 ;;
    esac
    return 1
}

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    if ! command -v sbatch >/dev/null 2>&1; then
        echo "sbatch not found; run inside a Slurm allocation or install Slurm tools."
        exit 1
    fi

    script_path="${SCRIPT_PATH}"

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
    (
        cd "${REPO_ROOT}"
        sbatch "${sbatch_opts[@]}" "${script_path}" "${args_file}"
    )
    exit 0
fi

which python

POS_DATA_PATH=$(jq -r '.pos_data_path' "$args_file")
NEG_DATA_PATH=$(jq -r '.neg_data_path' "$args_file")
NEG_GROUP=$(jq -r '.neg_group // empty' "$args_file")
CKPT_PATH=$(jq -r '.ckpt_path' "$args_file")
PRETRAINED_ALBEF_CKPT=$(jq -r '.pretrained_albef_ckpt // empty' "$args_file")
STAGE_TO_JOBFS=$(jq -r '.stage_to_jobfs // false' "$args_file")

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
OPTICAL_CURVE_DIM=$(jq -r '.optical_curve_dim // empty' "$args_file")
OPTICAL_CURVE_HIDDEN_DIM=$(jq -r '.optical_curve_hidden_dim // empty' "$args_file")
NUM_HEADS=$(jq -r '.num_heads // empty' "$args_file")
K_DIM=$(jq -r '.k_dim // empty' "$args_file")
MTAN_SNR_S0=$(jq -r '.mtan_snr_s0 // empty' "$args_file")
MTAN_SNR_BETA=$(jq -r '.mtan_snr_beta // empty' "$args_file")
MTAN_SNR_CLIP_MIN=$(jq -r '.mtan_snr_clip_min // empty' "$args_file")
MTAN_SNR_CLIP_MAX=$(jq -r '.mtan_snr_clip_max // empty' "$args_file")
MTAN_SNR_EPS=$(jq -r '.mtan_snr_eps // empty' "$args_file")
MTAN_LUPT_PSFFLUX_ZP=$(jq -r '.mtan_lupt_psfflux_zp // empty' "$args_file")
MTAN_LUPT_K=$(jq -r '.mtan_lupt_k // empty' "$args_file")
MTAN_LUPT_M5_MAG=$(jq -r 'if .mtan_lupt_m5_mag == null then empty elif (.mtan_lupt_m5_mag | type) == "array" then .mtan_lupt_m5_mag | join(",") else .mtan_lupt_m5_mag end' "$args_file")
MTAN_PERIOD_RANGE_DAYS=$(jq -r 'if .mtan_period_range_days == null then empty elif (.mtan_period_range_days | type) == "array" then .mtan_period_range_days | join(",") else .mtan_period_range_days end' "$args_file")
MTAN_TIME_SCALE_DIVISOR=$(jq -r '.mtan_time_scale_divisor // empty' "$args_file")

OPT_DROPOUT=$(jq -r '.opt_dropout // empty' "$args_file")
FEATURE_DROPOUT=$(jq -r '.feature_dropout // empty' "$args_file")
HEAD_HIDDEN_DIM=$(jq -r '.head_hidden_dim // empty' "$args_file")
HEAD_DROPOUT=$(jq -r '.head_dropout // empty' "$args_file")
ARCH_VERSION=$(jq -r '.arch_version // empty' "$args_file")

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

META_MATCHED_SAMPLING=$(jq -r '.meta_matched_sampling // empty' "$args_file")
META_MATCH_FALLBACK=$(jq -r '.meta_match_fallback // empty' "$args_file")
META_FILTER_N_DET_MIN=$(jq -r '.meta_filter_n_det_min // empty' "$args_file")
META_FILTER_N_DET_MAX=$(jq -r '.meta_filter_n_det_max // empty' "$args_file")
META_FILTER_N_BANDS_MAX=$(jq -r '.meta_filter_n_bands_max // empty' "$args_file")
META_FILTER_T_SPAN_MAX=$(jq -r '.meta_filter_t_span_max // empty' "$args_file")
META_FILTER_RELAX_T_SPAN_IF_BELOW_ROWS=$(jq -r '.meta_filter_relax_t_span_if_below_rows // empty' "$args_file")
SHORTCUT_AUDIT_ENABLE=$(jq -r '.shortcut_audit_enable // empty' "$args_file")
SHORTCUT_AUDIT_VAL_SAMPLES=$(jq -r '.shortcut_audit_val_samples // empty' "$args_file")
PREFIX_MIN_DET=$(jq -r '.prefix_min_det // empty' "$args_file")
PREFIX_TRAIN_SAMPLING=$(jq -r '.prefix_train_sampling // empty' "$args_file")
PREFIX_TERMINAL_MIX_PROB=$(jq -r '.prefix_terminal_mix_prob // empty' "$args_file")
PREFIX_BUCKET_UNIFORM_MIX_WEIGHT=$(jq -r '.prefix_bucket_uniform_mix_weight // empty' "$args_file")
PREFIX_TERMINAL_MIX_WEIGHT=$(jq -r '.prefix_terminal_mix_weight // empty' "$args_file")
PREFIX_EVAL_DET_SUPPORT=$(jq -r '.prefix_eval_det_support // empty' "$args_file")
PREFIX_EVAL_ENABLE=$(jq -r '.prefix_eval_enable // empty' "$args_file")
PREFIX_MANIFEST_OUT=$(jq -r '(.prefix_manifest_out // .eval_prefix_manifest_out) // empty' "$args_file")
SECONDARY_PREFIX_EVAL_DET_SUPPORT=$(jq -r '.secondary_prefix_eval_det_support // empty' "$args_file")
UNIVERSAL_TRAIN_ENABLE=$(jq -r '.universal_train_enable // false' "$args_file")
UNIVERSAL_STAGE1_EPOCHS=$(jq -r '.universal_stage1_epochs // empty' "$args_file")
UNIVERSAL_STAGE2_EPOCHS=$(jq -r '.universal_stage2_epochs // empty' "$args_file")
UNIVERSAL_STAGE3_EPOCHS=$(jq -r '.universal_stage3_epochs // empty' "$args_file")
UNIVERSAL_STAGE2_MODE=$(jq -r '.universal_stage2_mode // empty' "$args_file")
UNIVERSAL_STAGE3_MODE=$(jq -r '.universal_stage3_mode // empty' "$args_file")
VIEW_KEEP_PROB_MIN=$(jq -r '.view_keep_prob_min // empty' "$args_file")
VIEW_KEEP_PROB_MAX=$(jq -r '.view_keep_prob_max // empty' "$args_file")
VIEW_BAND_DROPOUT_MAX=$(jq -r '.view_band_dropout_max // empty' "$args_file")
CONSISTENCY_EMBED_WEIGHT=$(jq -r '.consistency_embed_weight // empty' "$args_file")
CONSISTENCY_PROB_WEIGHT=$(jq -r '.consistency_prob_weight // empty' "$args_file")
ADV_DET_WEIGHT=$(jq -r '.adv_det_weight // empty' "$args_file")
ADV_BAND_WEIGHT=$(jq -r '.adv_band_weight // empty' "$args_file")
ADV_SPAN_WEIGHT=$(jq -r '.adv_span_weight // empty' "$args_file")
GRL_LAMBDA=$(jq -r '.grl_lambda // empty' "$args_file")
REGIME_EVAL_ENABLE=$(jq -r '.regime_eval_enable // false' "$args_file")

OPTICAL_V2_EVAL_POS_DEFAULT="${BASE_DIR}/data/Optical_Only_dataset/combined_dataset_test.h5"
OPTICAL_V2_EVAL_NEG_DEFAULT="${BASE_DIR}/data/Optical_Only_dataset/Tutorial_negative_dataset.h5"
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
if is_truthy "$STAGE_TO_JOBFS"; then
    JOBFS_DIR="${SLURM_TMPDIR:-${TMPDIR:-${JOBFS:-}}}"
    if [[ -n "$JOBFS_DIR" ]]; then
        echo "Staging datasets to local disk: $JOBFS_DIR"
        TRAIN_POS_LOCAL="$JOBFS_DIR/train_pos_$(basename "$POS_DATA_PATH")"
        TRAIN_NEG_LOCAL="$JOBFS_DIR/train_neg_$(basename "$NEG_DATA_PATH")"
        EVAL_POS_LOCAL="$JOBFS_DIR/eval_pos_$(basename "$EVAL_POS_DATA_PATH")"
        EVAL_NEG_LOCAL="$JOBFS_DIR/eval_neg_$(basename "$EVAL_NEG_DATA_PATH")"
        cp -f "$POS_DATA_PATH" "$TRAIN_POS_LOCAL"
        POS_DATA_PATH="$TRAIN_POS_LOCAL"
        cp -f "$NEG_DATA_PATH" "$TRAIN_NEG_LOCAL"
        NEG_DATA_PATH="$TRAIN_NEG_LOCAL"
        cp -f "$EVAL_POS_DATA_PATH" "$EVAL_POS_LOCAL"
        EVAL_POS_DATA_PATH="$EVAL_POS_LOCAL"
        cp -f "$EVAL_NEG_DATA_PATH" "$EVAL_NEG_LOCAL"
        EVAL_NEG_DATA_PATH="$EVAL_NEG_LOCAL"
        echo "Staging complete."
    else
        echo "No local tmp dir found; skip staging."
    fi
fi

cmd=(
    python -u "${SCRIPT_DIR}/train_optical_only.py"
    --config "$args_file"
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
if [[ -n "$OPTICAL_CURVE_DIM" && "$OPTICAL_CURVE_DIM" != "null" ]]; then
    cmd+=(--optical_curve_dim "$OPTICAL_CURVE_DIM")
fi
if [[ -n "$OPTICAL_CURVE_HIDDEN_DIM" && "$OPTICAL_CURVE_HIDDEN_DIM" != "null" ]]; then
    cmd+=(--optical_curve_hidden_dim "$OPTICAL_CURVE_HIDDEN_DIM")
fi
if [[ -n "$NUM_HEADS" && "$NUM_HEADS" != "null" ]]; then
    cmd+=(--num_heads "$NUM_HEADS")
fi
if [[ -n "$K_DIM" && "$K_DIM" != "null" ]]; then
    cmd+=(--k_dim "$K_DIM")
fi
if [[ -n "$MTAN_SNR_S0" && "$MTAN_SNR_S0" != "null" ]]; then
    cmd+=(--mtan_snr_s0 "$MTAN_SNR_S0")
fi
if [[ -n "$MTAN_SNR_BETA" && "$MTAN_SNR_BETA" != "null" ]]; then
    cmd+=(--mtan_snr_beta "$MTAN_SNR_BETA")
fi
if [[ -n "$MTAN_SNR_CLIP_MIN" && "$MTAN_SNR_CLIP_MIN" != "null" ]]; then
    cmd+=(--mtan_snr_clip_min "$MTAN_SNR_CLIP_MIN")
fi
if [[ -n "$MTAN_SNR_CLIP_MAX" && "$MTAN_SNR_CLIP_MAX" != "null" ]]; then
    cmd+=(--mtan_snr_clip_max "$MTAN_SNR_CLIP_MAX")
fi
if [[ -n "$MTAN_SNR_EPS" && "$MTAN_SNR_EPS" != "null" ]]; then
    cmd+=(--mtan_snr_eps "$MTAN_SNR_EPS")
fi
if [[ -n "$MTAN_LUPT_PSFFLUX_ZP" && "$MTAN_LUPT_PSFFLUX_ZP" != "null" ]]; then
    cmd+=(--mtan_lupt_psfflux_zp "$MTAN_LUPT_PSFFLUX_ZP")
fi
if [[ -n "$MTAN_LUPT_K" && "$MTAN_LUPT_K" != "null" ]]; then
    cmd+=(--mtan_lupt_k "$MTAN_LUPT_K")
fi
if [[ -n "$MTAN_LUPT_M5_MAG" && "$MTAN_LUPT_M5_MAG" != "null" ]]; then
    cmd+=(--mtan_lupt_m5_mag "$MTAN_LUPT_M5_MAG")
fi
if [[ -n "$MTAN_PERIOD_RANGE_DAYS" && "$MTAN_PERIOD_RANGE_DAYS" != "null" ]]; then
    cmd+=(--mtan_period_range_days "$MTAN_PERIOD_RANGE_DAYS")
fi
if [[ -n "$MTAN_TIME_SCALE_DIVISOR" && "$MTAN_TIME_SCALE_DIVISOR" != "null" ]]; then
    cmd+=(--mtan_time_scale_divisor "$MTAN_TIME_SCALE_DIVISOR")
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
if is_truthy "$CACHE_IN_MEMORY"; then
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
if is_truthy "$DISABLE_TENSORBOARD"; then
    cmd+=(--disable_tensorboard)
fi
if [[ -n "$RUN_NAME" && "$RUN_NAME" != "null" ]]; then
    cmd+=(--run_name "$RUN_NAME")
fi
if [[ -n "$PREFIX_MIN_DET" && "$PREFIX_MIN_DET" != "null" ]]; then
    cmd+=(--prefix_min_det "$PREFIX_MIN_DET")
fi
if [[ -n "$PREFIX_TRAIN_SAMPLING" && "$PREFIX_TRAIN_SAMPLING" != "null" ]]; then
    cmd+=(--prefix_train_sampling "$PREFIX_TRAIN_SAMPLING")
fi
if [[ -n "$PREFIX_TERMINAL_MIX_PROB" && "$PREFIX_TERMINAL_MIX_PROB" != "null" ]]; then
    cmd+=(--prefix_terminal_mix_prob "$PREFIX_TERMINAL_MIX_PROB")
fi
if [[ -n "$PREFIX_BUCKET_UNIFORM_MIX_WEIGHT" && "$PREFIX_BUCKET_UNIFORM_MIX_WEIGHT" != "null" ]]; then
    cmd+=(--prefix_bucket_uniform_mix_weight "$PREFIX_BUCKET_UNIFORM_MIX_WEIGHT")
fi
if [[ -n "$PREFIX_TERMINAL_MIX_WEIGHT" && "$PREFIX_TERMINAL_MIX_WEIGHT" != "null" ]]; then
    cmd+=(--prefix_terminal_mix_weight "$PREFIX_TERMINAL_MIX_WEIGHT")
fi
if [[ -n "$PREFIX_EVAL_DET_SUPPORT" && "$PREFIX_EVAL_DET_SUPPORT" != "null" ]]; then
    cmd+=(--prefix_eval_det_support "$PREFIX_EVAL_DET_SUPPORT")
fi
if is_truthy "$UNIVERSAL_TRAIN_ENABLE"; then
    cmd+=(--universal_train_enable)
fi
if [[ -n "$UNIVERSAL_STAGE1_EPOCHS" && "$UNIVERSAL_STAGE1_EPOCHS" != "null" ]]; then
    cmd+=(--universal_stage1_epochs "$UNIVERSAL_STAGE1_EPOCHS")
fi
if [[ -n "$UNIVERSAL_STAGE2_EPOCHS" && "$UNIVERSAL_STAGE2_EPOCHS" != "null" ]]; then
    cmd+=(--universal_stage2_epochs "$UNIVERSAL_STAGE2_EPOCHS")
fi
if [[ -n "$UNIVERSAL_STAGE3_EPOCHS" && "$UNIVERSAL_STAGE3_EPOCHS" != "null" ]]; then
    cmd+=(--universal_stage3_epochs "$UNIVERSAL_STAGE3_EPOCHS")
fi
if [[ -n "$UNIVERSAL_STAGE2_MODE" && "$UNIVERSAL_STAGE2_MODE" != "null" ]]; then
    cmd+=(--universal_stage2_mode "$UNIVERSAL_STAGE2_MODE")
fi
if [[ -n "$UNIVERSAL_STAGE3_MODE" && "$UNIVERSAL_STAGE3_MODE" != "null" ]]; then
    cmd+=(--universal_stage3_mode "$UNIVERSAL_STAGE3_MODE")
fi
if [[ -n "$VIEW_KEEP_PROB_MIN" && "$VIEW_KEEP_PROB_MIN" != "null" ]]; then
    cmd+=(--view_keep_prob_min "$VIEW_KEEP_PROB_MIN")
fi
if [[ -n "$VIEW_KEEP_PROB_MAX" && "$VIEW_KEEP_PROB_MAX" != "null" ]]; then
    cmd+=(--view_keep_prob_max "$VIEW_KEEP_PROB_MAX")
fi
if [[ -n "$VIEW_BAND_DROPOUT_MAX" && "$VIEW_BAND_DROPOUT_MAX" != "null" ]]; then
    cmd+=(--view_band_dropout_max "$VIEW_BAND_DROPOUT_MAX")
fi
if [[ -n "$CONSISTENCY_EMBED_WEIGHT" && "$CONSISTENCY_EMBED_WEIGHT" != "null" ]]; then
    cmd+=(--consistency_embed_weight "$CONSISTENCY_EMBED_WEIGHT")
fi
if [[ -n "$CONSISTENCY_PROB_WEIGHT" && "$CONSISTENCY_PROB_WEIGHT" != "null" ]]; then
    cmd+=(--consistency_prob_weight "$CONSISTENCY_PROB_WEIGHT")
fi
if [[ -n "$ADV_DET_WEIGHT" && "$ADV_DET_WEIGHT" != "null" ]]; then
    cmd+=(--adv_det_weight "$ADV_DET_WEIGHT")
fi
if [[ -n "$ADV_BAND_WEIGHT" && "$ADV_BAND_WEIGHT" != "null" ]]; then
    cmd+=(--adv_band_weight "$ADV_BAND_WEIGHT")
fi
if [[ -n "$ADV_SPAN_WEIGHT" && "$ADV_SPAN_WEIGHT" != "null" ]]; then
    cmd+=(--adv_span_weight "$ADV_SPAN_WEIGHT")
fi
if [[ -n "$GRL_LAMBDA" && "$GRL_LAMBDA" != "null" ]]; then
    cmd+=(--grl_lambda "$GRL_LAMBDA")
fi
if [[ -n "$META_FILTER_N_DET_MIN" && "$META_FILTER_N_DET_MIN" != "null" ]]; then
    cmd+=(--meta_filter_n_det_min "$META_FILTER_N_DET_MIN")
fi
if [[ -n "$META_FILTER_N_DET_MAX" && "$META_FILTER_N_DET_MAX" != "null" ]]; then
    cmd+=(--meta_filter_n_det_max "$META_FILTER_N_DET_MAX")
fi
if [[ -n "$META_FILTER_N_BANDS_MAX" && "$META_FILTER_N_BANDS_MAX" != "null" ]]; then
    cmd+=(--meta_filter_n_bands_max "$META_FILTER_N_BANDS_MAX")
fi
if [[ -n "$META_FILTER_T_SPAN_MAX" && "$META_FILTER_T_SPAN_MAX" != "null" ]]; then
    cmd+=(--meta_filter_t_span_max "$META_FILTER_T_SPAN_MAX")
fi
if [[ -n "$META_FILTER_RELAX_T_SPAN_IF_BELOW_ROWS" && "$META_FILTER_RELAX_T_SPAN_IF_BELOW_ROWS" != "null" ]]; then
    cmd+=(--meta_filter_relax_t_span_if_below_rows "$META_FILTER_RELAX_T_SPAN_IF_BELOW_ROWS")
fi

echo "Training Command: ${cmd[*]}"
echo "Optical controls (from config): optical_curve_dim=${OPTICAL_CURVE_DIM:-<default>}, optical_curve_hidden_dim=${OPTICAL_CURVE_HIDDEN_DIM:-<default>}, ref_dim=${REF_DIM:-<default>}, n_ref=${N_REF:-<default>}, num_heads=${NUM_HEADS:-<default>}, k_dim=${K_DIM:-<default>}, mtan_time_scale_divisor=${MTAN_TIME_SCALE_DIVISOR:-<default>}, mtan_lupt_m5_mag=${MTAN_LUPT_M5_MAG:-<default>}, meta_matched_sampling=${META_MATCHED_SAMPLING:-<default>}, meta_match_fallback=${META_MATCH_FALLBACK:-<default>}, meta_filter_n_det=[${META_FILTER_N_DET_MIN:-<default>},${META_FILTER_N_DET_MAX:-<default>}], meta_filter_n_bands_max=${META_FILTER_N_BANDS_MAX:-<default>}, meta_filter_t_span_max=${META_FILTER_T_SPAN_MAX:-<default>}, meta_filter_relax=${META_FILTER_RELAX_T_SPAN_IF_BELOW_ROWS:-<default>}, universal_train_enable=${UNIVERSAL_TRAIN_ENABLE:-<default>}, universal_epochs=${UNIVERSAL_STAGE1_EPOCHS:-<default>}/${UNIVERSAL_STAGE2_EPOCHS:-<default>}/${UNIVERSAL_STAGE3_EPOCHS:-<default>}, universal_modes=single/${UNIVERSAL_STAGE2_MODE:-<default>}/${UNIVERSAL_STAGE3_MODE:-<default>}, view_keep_prob=${VIEW_KEEP_PROB_MIN:-<default>}..${VIEW_KEEP_PROB_MAX:-<default>}, view_band_dropout_max=${VIEW_BAND_DROPOUT_MAX:-<default>}, cons_weights=${CONSISTENCY_EMBED_WEIGHT:-<default>}/${CONSISTENCY_PROB_WEIGHT:-<default>}, adv_weights=${ADV_DET_WEIGHT:-<default>}/${ADV_BAND_WEIGHT:-<default>}/${ADV_SPAN_WEIGHT:-<default>}, grl_lambda=${GRL_LAMBDA:-<default>}, shortcut_audit_enable=${SHORTCUT_AUDIT_ENABLE:-<default>}, shortcut_audit_val_samples=${SHORTCUT_AUDIT_VAL_SAMPLES:-<default>}, prefix_min_det=${PREFIX_MIN_DET:-<default>}, prefix_train_sampling=${PREFIX_TRAIN_SAMPLING:-<default>}, prefix_mix_weights=${PREFIX_BUCKET_UNIFORM_MIX_WEIGHT:-<default>}:${PREFIX_TERMINAL_MIX_WEIGHT:-<default>}"
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
EVAL_PY="${SCRIPT_DIR}/test_evaluate_optical_only.py"

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
if is_truthy "$EVAL_NO_PLOTS"; then
    eval_cmd+=(--no_plots)
fi
if is_truthy "$PREFIX_EVAL_ENABLE"; then
    eval_cmd+=(--prefix_eval_enable)
fi
if [[ -n "$PREFIX_MIN_DET" && "$PREFIX_MIN_DET" != "null" ]]; then
    eval_cmd+=(--prefix_min_det "$PREFIX_MIN_DET")
fi
if [[ -n "$PREFIX_EVAL_DET_SUPPORT" && "$PREFIX_EVAL_DET_SUPPORT" != "null" ]]; then
    eval_cmd+=(--prefix_eval_det_support "$PREFIX_EVAL_DET_SUPPORT")
fi
if [[ -n "$PREFIX_MANIFEST_OUT" && "$PREFIX_MANIFEST_OUT" != "null" ]]; then
    eval_cmd+=(--prefix_manifest_out "$PREFIX_MANIFEST_OUT")
fi
if [[ -n "$META_FILTER_N_DET_MIN" && "$META_FILTER_N_DET_MIN" != "null" ]]; then
    eval_cmd+=(--meta_filter_n_det_min "$META_FILTER_N_DET_MIN")
fi
if [[ -n "$META_FILTER_N_DET_MAX" && "$META_FILTER_N_DET_MAX" != "null" ]]; then
    eval_cmd+=(--meta_filter_n_det_max "$META_FILTER_N_DET_MAX")
fi
if [[ -n "$META_FILTER_N_BANDS_MAX" && "$META_FILTER_N_BANDS_MAX" != "null" ]]; then
    eval_cmd+=(--meta_filter_n_bands_max "$META_FILTER_N_BANDS_MAX")
fi
if [[ -n "$META_FILTER_T_SPAN_MAX" && "$META_FILTER_T_SPAN_MAX" != "null" ]]; then
    eval_cmd+=(--meta_filter_t_span_max "$META_FILTER_T_SPAN_MAX")
fi
if [[ -n "$META_FILTER_RELAX_T_SPAN_IF_BELOW_ROWS" && "$META_FILTER_RELAX_T_SPAN_IF_BELOW_ROWS" != "null" ]]; then
    eval_cmd+=(--meta_filter_relax_t_span_if_below_rows "$META_FILTER_RELAX_T_SPAN_IF_BELOW_ROWS")
fi
if is_truthy "$REGIME_EVAL_ENABLE"; then
    eval_cmd+=(--regime_eval_enable)
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

compat_prefix_support="${SECONDARY_PREFIX_EVAL_DET_SUPPORT:-3,4,5,6,8,10,12}"
if is_truthy "$PREFIX_EVAL_ENABLE" && [[ -n "$compat_prefix_support" && "$compat_prefix_support" != "$PREFIX_EVAL_DET_SUPPORT" ]]; then
    compat_output_dir="${EVAL_OUTPUT_DIR}/compat_prefix_support"
    compat_manifest_out=""
    if [[ -n "$PREFIX_MANIFEST_OUT" && "$PREFIX_MANIFEST_OUT" != "null" ]]; then
        compat_manifest_out="${PREFIX_MANIFEST_OUT%.csv}_compat.csv"
    fi
    compat_eval_cmd=("${eval_cmd[@]}")
    compat_eval_cmd+=(--output_dir "$compat_output_dir" --prefix_eval_det_support "$compat_prefix_support")
    if [[ -n "$compat_manifest_out" ]]; then
        compat_eval_cmd+=(--prefix_manifest_out "$compat_manifest_out")
    fi
    echo "Compatibility Evaluation Command: ${compat_eval_cmd[*]}"
    set +e
    "${compat_eval_cmd[@]}"
    compat_eval_exit_code=$?
    set -e
    if [ $compat_eval_exit_code -ne 0 ]; then
        echo "Compatibility evaluation failed with exit code $compat_eval_exit_code"
        echo "------------------------------------------------"
        echo "End time: $(date)"
        echo "Exit code: $compat_eval_exit_code"
        exit $compat_eval_exit_code
    fi
fi

echo "Evaluation complete. Results saved to: $EVAL_OUTPUT_DIR"
echo "------------------------------------------------"
echo "End time: $(date)"
echo "Exit code: 0"

exit 0
