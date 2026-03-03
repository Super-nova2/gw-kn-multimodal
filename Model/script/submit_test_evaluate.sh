#!/bin/bash

#SBATCH --job-name=ALBEF_test_eval
#SBATCH --output=logs/eval/%x_%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --gres=gpu:1
#SBATCH --time=0:30:00
#SBATCH --partition=gpu
#SBATCH --tmp=80G

set -euo pipefail

extract_version_tag() {
    local path_text="$1"
    if [[ "$path_text" =~ _v([0-9]+) ]]; then
        echo "v${BASH_REMATCH[1]}"
    else
        echo ""
    fi
}

enforce_eval_mapping_checks() {
    local checkpoint_path="$1"
    local output_dir_path="$2"
    local cls_dt_state="$3"
    local cls_dt_force_zero="$4"
    local allow_version_mismatch="${EVAL_ALLOW_VERSION_MISMATCH:-false}"
    local allow_mode_mismatch="${EVAL_ALLOW_MODE_TAG_MISMATCH:-false}"

    local ckpt_ver
    local out_ver
    ckpt_ver="$(extract_version_tag "$checkpoint_path")"
    out_ver="$(extract_version_tag "$output_dir_path")"

    if [[ -n "$ckpt_ver" && -n "$out_ver" && "$ckpt_ver" != "$out_ver" ]]; then
        echo "Version mismatch: checkpoint=$ckpt_ver, output_dir=$out_ver"
        echo "checkpoint: $checkpoint_path"
        echo "output_dir: $output_dir_path"
        if [[ "$allow_version_mismatch" != "true" ]]; then
            echo "Set EVAL_ALLOW_VERSION_MISMATCH=true to bypass."
            exit 1
        fi
        echo "EVAL_ALLOW_VERSION_MISMATCH=true, continuing despite mismatch."
    fi

    local eval_mode="unknown"
    if [[ "$cls_dt_force_zero" == "true" ]]; then
        eval_mode="force_zero"
    elif [[ "$cls_dt_state" == "true" ]]; then
        eval_mode="delta_on"
    elif [[ "$cls_dt_state" == "false" ]]; then
        eval_mode="force_zero"
    fi

    local out_base_lc
    out_base_lc="$(basename "$output_dir_path" | tr '[:upper:]' '[:lower:]')"
    local out_has_no_delta="false"
    if [[ "$out_base_lc" == *"no_delta"* || "$out_base_lc" == *"nodelta"* ]]; then
        out_has_no_delta="true"
    fi

    if [[ "$eval_mode" == "force_zero" && "$out_has_no_delta" != "true" ]]; then
        echo "Output dir naming mismatch: eval mode is force-zero but output_dir lacks no_delta tag."
        echo "output_dir: $output_dir_path"
        if [[ "$allow_mode_mismatch" != "true" ]]; then
            echo "Set EVAL_ALLOW_MODE_TAG_MISMATCH=true to bypass."
            exit 1
        fi
        echo "EVAL_ALLOW_MODE_TAG_MISMATCH=true, continuing despite mismatch."
    fi

    if [[ "$eval_mode" == "delta_on" && "$out_has_no_delta" == "true" ]]; then
        echo "Output dir naming mismatch: eval mode is delta-on but output_dir contains no_delta tag."
        echo "output_dir: $output_dir_path"
        if [[ "$allow_mode_mismatch" != "true" ]]; then
            echo "Set EVAL_ALLOW_MODE_TAG_MISMATCH=true to bypass."
            exit 1
        fi
        echo "EVAL_ALLOW_MODE_TAG_MISMATCH=true, continuing despite mismatch."
    fi
}

args_file=${1:-}
if [[ -z "${args_file}" || ! -f "${args_file}" ]]; then
    echo "Usage: $0 <eval_args.json>"
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

    mkdir -p logs/eval

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

CHECKPOINT=$(jq -r '.checkpoint // empty' "$args_file")
TEST_DATA_PATH=$(jq -r '.test_data_path // empty' "$args_file")
NEG_DATA_PATH=$(jq -r '.neg_data_path // empty' "$args_file")
NEG_GROUP=$(jq -r '.neg_group // empty' "$args_file")
CONFIG_PATH=$(jq -r '.config // empty' "$args_file")
OUTPUT_DIR=$(jq -r '.output_dir // empty' "$args_file")
DEVICE=$(jq -r '.device // "cuda"' "$args_file")
NO_PLOTS=$(jq -r '.no_plots // false' "$args_file")
STAGE_TO_JOBFS=$(jq -r '.stage_to_jobfs // false' "$args_file")

BATCH_SIZE=$(jq -r '.batch_size // empty' "$args_file")
NUM_WORKERS=$(jq -r '.num_workers // empty' "$args_file")
N_NEG_SAMPLES=$(jq -r '.n_neg_samples // empty' "$args_file")
TEST_STEPS=$(jq -r '.test_steps // empty' "$args_file")
GALLERY_SIZES=$(jq -r '.gallery_sizes // empty' "$args_file")
GALLERY_TRIALS=$(jq -r '.gallery_trials // empty' "$args_file")
NEG_TIME_OFFSET_ENABLE=$(jq -r '.neg_time_offset_enable // false' "$args_file")
NEG_OFFSET_DIST_NPZ=$(jq -r '.neg_offset_dist_npz // empty' "$args_file")
NEG_OFFSET_DIST_KEY=$(jq -r '.neg_offset_dist_key // empty' "$args_file")
NEG_OFFSET_EVAL_MODE=$(jq -r '.neg_offset_eval_mode // empty' "$args_file")
NEG_OFFSET_EVAL_QUANTILES=$(jq -r '.neg_offset_eval_quantiles // empty' "$args_file")
NEG_OFFSET_SCALE_DAYS_DIVISOR=$(jq -r '.neg_offset_scale_days_divisor // empty' "$args_file")
CLS_TIME_DELTA_ENABLE=$(jq -r '.cls_time_delta_enable // false' "$args_file")
CLS_TIME_DELTA_ENABLE_STATE=$(jq -r 'if has("cls_time_delta_enable") then (.cls_time_delta_enable | tostring) else "unset" end' "$args_file")
CLS_TIME_DELTA_SCALE_DAYS=$(jq -r '.cls_time_delta_scale_days // empty' "$args_file")
CLS_TIME_DELTA_CLIP=$(jq -r '.cls_time_delta_clip // empty' "$args_file")
CLS_TIME_DELTA_FORCE_ZERO=$(jq -r '.cls_time_delta_force_zero // false' "$args_file")
NONKN_CLS_BASE_FIELD=$(jq -r '.nonkn_cls_base_field // empty' "$args_file")
REPORT_DT_BINS_STATE=$(jq -r 'if has("report_dt_bins") then (.report_dt_bins | tostring) else "unset" end' "$args_file")
DT_BIN_EDGES=$(jq -r '.dt_bin_edges // empty' "$args_file")
REPORT_DT_MACRO_STATE=$(jq -r 'if has("report_dt_macro") then (.report_dt_macro | tostring) else "unset" end' "$args_file")
DT_MATCH_STRATEGY=$(jq -r '.dt_match_strategy // empty' "$args_file")
DT_MATCH_WINDOW_DAYS=$(jq -r '.dt_match_window_days // empty' "$args_file")
DT_MATCH_QUANTILES=$(jq -r '.dt_match_quantiles // empty' "$args_file")
DT_MATCH_TARGET=$(jq -r '.dt_match_target // empty' "$args_file")
DT_MATCH_APPLY_TO=$(jq -r '.dt_match_apply_to // empty' "$args_file")
POS_TIME_OFFSETS_DAYS=$(jq -r '.pos_time_offsets_days // empty' "$args_file")

if [[ -z "$CHECKPOINT" ]]; then
    echo "Required field missing in args: checkpoint"
    exit 1
fi
if [[ -z "$TEST_DATA_PATH" ]]; then
    echo "Required field missing in args: test_data_path"
    exit 1
fi
if [[ ! -f "$CHECKPOINT" ]]; then
    echo "Checkpoint not found: $CHECKPOINT"
    exit 1
fi
if [[ ! -f "$TEST_DATA_PATH" ]]; then
    echo "Test dataset not found: $TEST_DATA_PATH"
    exit 1
fi
if [[ -n "$NEG_DATA_PATH" && "$NEG_DATA_PATH" != "null" && ! -f "$NEG_DATA_PATH" ]]; then
    echo "Negative dataset not found: $NEG_DATA_PATH"
    exit 1
fi
if [[ -n "$CONFIG_PATH" && "$CONFIG_PATH" != "null" && ! -f "$CONFIG_PATH" ]]; then
    echo "Config not found: $CONFIG_PATH"
    exit 1
fi
if [[ "$NEG_TIME_OFFSET_ENABLE" == "true" ]]; then
    if [[ -z "$NEG_OFFSET_DIST_NPZ" || "$NEG_OFFSET_DIST_NPZ" == "null" ]]; then
        echo "neg_time_offset_enable=true requires neg_offset_dist_npz"
        exit 1
    fi
    if [[ ! -f "$NEG_OFFSET_DIST_NPZ" ]]; then
        echo "Negative offset distribution file not found: $NEG_OFFSET_DIST_NPZ"
        exit 1
    fi
fi

if [[ -z "$OUTPUT_DIR" || "$OUTPUT_DIR" == "null" ]]; then
    OUTPUT_DIR="$(dirname "$CHECKPOINT")/eval_results"
fi
enforce_eval_mapping_checks "$CHECKPOINT" "$OUTPUT_DIR" "$CLS_TIME_DELTA_ENABLE_STATE" "$CLS_TIME_DELTA_FORCE_ZERO"
mkdir -p "$OUTPUT_DIR"

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
echo "Evaluation args: $args_file"
echo "Checkpoint: $CHECKPOINT"
echo "Test data: $TEST_DATA_PATH"
echo "Output dir: $OUTPUT_DIR"
echo ""

if [[ "$STAGE_TO_JOBFS" == "true" ]]; then
    JOBFS_DIR="${SLURM_TMPDIR:-${TMPDIR:-${JOBFS:-}}}"
    if [[ -n "$JOBFS_DIR" ]]; then
        echo "Staging files to local disk: $JOBFS_DIR"
        cp -f "$CHECKPOINT" "$JOBFS_DIR"/
        CHECKPOINT="$JOBFS_DIR/$(basename "$CHECKPOINT")"

        cp -f "$TEST_DATA_PATH" "$JOBFS_DIR"/
        TEST_DATA_PATH="$JOBFS_DIR/$(basename "$TEST_DATA_PATH")"

        if [[ -n "$NEG_DATA_PATH" && "$NEG_DATA_PATH" != "null" ]]; then
            cp -f "$NEG_DATA_PATH" "$JOBFS_DIR"/
            NEG_DATA_PATH="$JOBFS_DIR/$(basename "$NEG_DATA_PATH")"
        fi
        if [[ -n "$CONFIG_PATH" && "$CONFIG_PATH" != "null" ]]; then
            cp -f "$CONFIG_PATH" "$JOBFS_DIR"/
            CONFIG_PATH="$JOBFS_DIR/$(basename "$CONFIG_PATH")"
        fi
        if [[ "$NEG_TIME_OFFSET_ENABLE" == "true" && -n "$NEG_OFFSET_DIST_NPZ" && "$NEG_OFFSET_DIST_NPZ" != "null" ]]; then
            NEG_OFFSET_STAGED_NAME="neg_offset_$(basename "$NEG_OFFSET_DIST_NPZ")"
            cp -f "$NEG_OFFSET_DIST_NPZ" "$JOBFS_DIR/$NEG_OFFSET_STAGED_NAME"
            NEG_OFFSET_DIST_NPZ="$JOBFS_DIR/$NEG_OFFSET_STAGED_NAME"
        fi
    else
        echo "No local tmp dir found; skip staging."
    fi
fi

cmd=(
    python -u /fred/oz016/bgao_kn/ML+GW+KN/Model/test_evaluate.py
    --checkpoint "$CHECKPOINT"
    --test_data_path "$TEST_DATA_PATH"
    --output_dir "$OUTPUT_DIR"
    --device "$DEVICE"
)

if [[ -n "$NEG_DATA_PATH" && "$NEG_DATA_PATH" != "null" ]]; then
    cmd+=(--neg_data_path "$NEG_DATA_PATH")
fi
if [[ -n "$NEG_GROUP" && "$NEG_GROUP" != "null" ]]; then
    cmd+=(--neg_group "$NEG_GROUP")
fi
if [[ -n "$CONFIG_PATH" && "$CONFIG_PATH" != "null" ]]; then
    cmd+=(--config "$CONFIG_PATH")
fi
if [[ -n "$BATCH_SIZE" && "$BATCH_SIZE" != "null" ]]; then
    cmd+=(--batch_size "$BATCH_SIZE")
fi
if [[ -n "$NUM_WORKERS" && "$NUM_WORKERS" != "null" ]]; then
    cmd+=(--num_workers "$NUM_WORKERS")
fi
if [[ -n "$N_NEG_SAMPLES" && "$N_NEG_SAMPLES" != "null" ]]; then
    cmd+=(--n_neg_samples "$N_NEG_SAMPLES")
fi
if [[ -n "$TEST_STEPS" && "$TEST_STEPS" != "null" ]]; then
    cmd+=(--test_steps "$TEST_STEPS")
fi
if [[ -n "$GALLERY_SIZES" && "$GALLERY_SIZES" != "null" ]]; then
    cmd+=(--gallery_sizes "$GALLERY_SIZES")
fi
if [[ -n "$GALLERY_TRIALS" && "$GALLERY_TRIALS" != "null" ]]; then
    cmd+=(--gallery_trials "$GALLERY_TRIALS")
fi
if [[ "$NO_PLOTS" == "true" ]]; then
    cmd+=(--no_plots)
fi
if [[ "$NEG_TIME_OFFSET_ENABLE" == "true" ]]; then
    cmd+=(--neg_time_offset_enable)
fi
if [[ -n "$NEG_OFFSET_DIST_NPZ" && "$NEG_OFFSET_DIST_NPZ" != "null" ]]; then
    cmd+=(--neg_offset_dist_npz "$NEG_OFFSET_DIST_NPZ")
fi
if [[ -n "$NEG_OFFSET_DIST_KEY" && "$NEG_OFFSET_DIST_KEY" != "null" ]]; then
    cmd+=(--neg_offset_dist_key "$NEG_OFFSET_DIST_KEY")
fi
if [[ -n "$NEG_OFFSET_EVAL_MODE" && "$NEG_OFFSET_EVAL_MODE" != "null" ]]; then
    cmd+=(--neg_offset_eval_mode "$NEG_OFFSET_EVAL_MODE")
fi
if [[ -n "$NEG_OFFSET_EVAL_QUANTILES" && "$NEG_OFFSET_EVAL_QUANTILES" != "null" ]]; then
    cmd+=(--neg_offset_eval_quantiles "$NEG_OFFSET_EVAL_QUANTILES")
fi
if [[ -n "$NEG_OFFSET_SCALE_DAYS_DIVISOR" && "$NEG_OFFSET_SCALE_DAYS_DIVISOR" != "null" ]]; then
    cmd+=(--neg_offset_scale_days_divisor "$NEG_OFFSET_SCALE_DAYS_DIVISOR")
fi
if [[ "$CLS_TIME_DELTA_ENABLE" == "true" ]]; then
    cmd+=(--cls_time_delta_enable)
fi
if [[ "$CLS_TIME_DELTA_ENABLE_STATE" == "false" ]]; then
    cmd+=(--cls_time_delta_force_zero)
fi
if [[ -n "$CLS_TIME_DELTA_SCALE_DAYS" && "$CLS_TIME_DELTA_SCALE_DAYS" != "null" ]]; then
    cmd+=(--cls_time_delta_scale_days "$CLS_TIME_DELTA_SCALE_DAYS")
fi
if [[ -n "$CLS_TIME_DELTA_CLIP" && "$CLS_TIME_DELTA_CLIP" != "null" ]]; then
    cmd+=(--cls_time_delta_clip "$CLS_TIME_DELTA_CLIP")
fi
if [[ "$CLS_TIME_DELTA_FORCE_ZERO" == "true" ]]; then
    cmd+=(--cls_time_delta_force_zero)
fi
if [[ -n "$NONKN_CLS_BASE_FIELD" && "$NONKN_CLS_BASE_FIELD" != "null" ]]; then
    cmd+=(--nonkn_cls_base_field "$NONKN_CLS_BASE_FIELD")
fi
if [[ "$REPORT_DT_BINS_STATE" == "true" ]]; then
    cmd+=(--report_dt_bins)
fi
if [[ "$REPORT_DT_BINS_STATE" == "false" ]]; then
    cmd+=(--no_report_dt_bins)
fi
if [[ -n "$DT_BIN_EDGES" && "$DT_BIN_EDGES" != "null" ]]; then
    cmd+=(--dt_bin_edges "$DT_BIN_EDGES")
fi
if [[ "$REPORT_DT_MACRO_STATE" == "true" ]]; then
    cmd+=(--report_dt_macro)
fi
if [[ "$REPORT_DT_MACRO_STATE" == "false" ]]; then
    cmd+=(--no_report_dt_macro)
fi
if [[ -n "$DT_MATCH_STRATEGY" && "$DT_MATCH_STRATEGY" != "null" ]]; then
    cmd+=(--dt_match_strategy "$DT_MATCH_STRATEGY")
fi
if [[ -n "$DT_MATCH_WINDOW_DAYS" && "$DT_MATCH_WINDOW_DAYS" != "null" ]]; then
    cmd+=(--dt_match_window_days "$DT_MATCH_WINDOW_DAYS")
fi
if [[ -n "$DT_MATCH_QUANTILES" && "$DT_MATCH_QUANTILES" != "null" ]]; then
    cmd+=(--dt_match_quantiles "$DT_MATCH_QUANTILES")
fi
if [[ -n "$DT_MATCH_TARGET" && "$DT_MATCH_TARGET" != "null" ]]; then
    cmd+=(--dt_match_target "$DT_MATCH_TARGET")
fi
if [[ -n "$DT_MATCH_APPLY_TO" && "$DT_MATCH_APPLY_TO" != "null" ]]; then
    cmd+=(--dt_match_apply_to "$DT_MATCH_APPLY_TO")
fi
if [[ -n "$POS_TIME_OFFSETS_DAYS" && "$POS_TIME_OFFSETS_DAYS" != "null" ]]; then
    cmd+=(--pos_time_offsets_days "$POS_TIME_OFFSETS_DAYS")
fi

echo "Command: ${cmd[*]}"
"${cmd[@]}"
exit_code=$?
if [[ $exit_code -ne 0 ]]; then
    echo "Evaluation script failed with exit code $exit_code"
    exit $exit_code
fi

echo "------------------------------------------------"
echo "End time: $(date)"
