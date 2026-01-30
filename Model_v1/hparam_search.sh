#!/bin/bash

#SBATCH --job-name=HPO_GWOptical
#SBATCH --output=logs/hpo/%x_%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --time=48:00:00
#SBATCH --partition=gpu

set -euo pipefail

# Usage: ./hparam_search.sh <args.json>
# Or with env vars: DATA_PATH=/path/to/data.h5 ./hparam_search.sh

args_file=${1:-}

# If args file provided, load values from JSON
if [[ -n "${args_file}" && -f "${args_file}" ]]; then
    args_dir="$(cd "$(dirname "${args_file}")" && pwd)"
    args_file="${args_dir}/$(basename "${args_file}")"

    # Read values from JSON (use env vars as fallback)
    DATA_PATH="${DATA_PATH:-$(jq -r '.data_path // empty' "$args_file")}"
    NEG_DATA_PATH="${NEG_DATA_PATH:-$(jq -r '.neg_data_path // empty' "$args_file")}"
    NEG_GROUP="${NEG_GROUP:-$(jq -r '.neg_group // "events/optical_data"' "$args_file")}"
    N_TRIALS="${N_TRIALS:-$(jq -r '.n_trials // 100' "$args_file")}"
    MAX_EPOCHS="${MAX_EPOCHS:-$(jq -r '.max_epochs // 50' "$args_file")}"
    STUDY_NAME="${STUDY_NAME:-$(jq -r '.study_name // "gw_optical_fusion_hpo"' "$args_file")}"
    OUTPUT_DIR="${OUTPUT_DIR:-$(jq -r '.output_dir // empty' "$args_file")}"
    RESUME="${RESUME:-$(jq -r '.resume // false' "$args_file")}"
    N_STARTUP_TRIALS="${N_STARTUP_TRIALS:-$(jq -r '.n_startup_trials // 20' "$args_file")}"
    PRUNER_WARMUP="${PRUNER_WARMUP:-$(jq -r '.pruner_warmup // 10' "$args_file")}"
    SEED="${SEED:-$(jq -r '.seed // 42' "$args_file")}"
    TIMEOUT="${TIMEOUT:-$(jq -r '.timeout // empty' "$args_file")}"
fi

# Default values if not set
DATA_PATH="${DATA_PATH:-/fred/oz016/bgao_kn/data/LSST_KN_BNS/combined_dataset_with_neg_gw_pixel.h5}"
NEG_DATA_PATH="${NEG_DATA_PATH:-/fred/oz016/bgao_kn/data/ELASTICC2_TRAIN/negative_dataset.h5}"
NEG_GROUP="${NEG_GROUP:-ELASTICC2_TRAIN/optical_data}"
N_TRIALS="${N_TRIALS:-100}"
MAX_EPOCHS="${MAX_EPOCHS:-50}"
STUDY_NAME="${STUDY_NAME:-gw_optical_fusion_hpo}"
OUTPUT_DIR="${OUTPUT_DIR:-/fred/oz016/bgao_kn/ML+GW+KN/Model_v1/hpo_results}"
RESUME="${RESUME:-false}"
N_STARTUP_TRIALS="${N_STARTUP_TRIALS:-20}"
PRUNER_WARMUP="${PRUNER_WARMUP:-10}"
SEED="${SEED:-42}"
TIMEOUT="${TIMEOUT:-}"

# Check if running inside SLURM or should submit
if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    if ! command -v sbatch >/dev/null 2>&1; then
        echo "sbatch not found; run inside a SLURM allocation or install SLURM tools."
        exit 1
    fi

    script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    script_path="${script_dir}/$(basename "${BASH_SOURCE[0]}")"

    # Create logs directory
    mkdir -p "${script_dir}/logs/hpo"

    sbatch_opts=()
    if [[ -n "${JOB_NAME:-}" ]]; then
        sbatch_opts+=(--job-name="${JOB_NAME}")
    fi
    if [[ -n "${TIME_LIMIT:-}" ]]; then
        sbatch_opts+=(--time="${TIME_LIMIT}")
    fi
    if [[ -n "${PARTITION:-}" ]]; then
        sbatch_opts+=(--partition="${PARTITION}")
    fi
    if [[ -n "${MEM:-}" ]]; then
        sbatch_opts+=(--mem="${MEM}")
    fi

    echo "=============================================="
    echo "Submitting HPO job..."
    echo "=============================================="
    echo "  Data:            $DATA_PATH"
    echo "  Neg Data:        $NEG_DATA_PATH"
    echo "  N Trials:        $N_TRIALS"
    echo "  Max Epochs:      $MAX_EPOCHS"
    echo "  Study Name:      $STUDY_NAME"
    echo "  Output Dir:      $OUTPUT_DIR"
    echo "  Resume:          $RESUME"
    echo "  Startup Trials:  $N_STARTUP_TRIALS"
    echo "  Pruner Warmup:   $PRUNER_WARMUP"
    echo "  Seed:            $SEED"
    [[ -n "$TIMEOUT" ]] && echo "  Timeout:         $TIMEOUT"
    echo "=============================================="
    echo ""

    # Export environment variables for the job
    export DATA_PATH NEG_DATA_PATH NEG_GROUP N_TRIALS MAX_EPOCHS STUDY_NAME OUTPUT_DIR RESUME
    export N_STARTUP_TRIALS PRUNER_WARMUP SEED TIMEOUT

    if [[ -n "${args_file}" ]]; then
        sbatch --export=ALL "${sbatch_opts[@]}" "${script_path}" "${args_file}"
    else
        sbatch --export=ALL "${sbatch_opts[@]}" "${script_path}"
    fi
    exit 0
fi

# Inside SLURM job
echo "=============================================="
echo "HPO Job Started: $(date)"
echo "=============================================="
echo "SLURM Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPUs: ${CUDA_VISIBLE_DEVICES:-not set}"
echo ""

which python
python --version
echo ""

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Build command
cmd=(
    python -u /fred/oz016/bgao_kn/ML+GW+KN/Model_v1/hparam_search.py
    --data_path "$DATA_PATH"
    --n_trials "$N_TRIALS"
    --max_epochs "$MAX_EPOCHS"
    --study_name "$STUDY_NAME"
    --output_dir "$OUTPUT_DIR"
    --n_startup_trials "$N_STARTUP_TRIALS"
    --pruner_warmup "$PRUNER_WARMUP"
    --seed "$SEED"
)

if [[ -n "$NEG_DATA_PATH" && "$NEG_DATA_PATH" != "null" ]]; then
    cmd+=(--neg_data_path "$NEG_DATA_PATH")
fi
if [[ -n "$NEG_GROUP" && "$NEG_GROUP" != "null" ]]; then
    cmd+=(--neg_group "$NEG_GROUP")
fi
if [[ "$RESUME" == "true" ]]; then
    cmd+=(--resume)
fi
if [[ -n "$TIMEOUT" && "$TIMEOUT" != "null" ]]; then
    cmd+=(--timeout "$TIMEOUT")
fi

printf 'Running: %q ' "${cmd[@]}"
echo ""
echo ""

"${cmd[@]}"

echo ""
echo "=============================================="
echo "HPO Job Completed: $(date)"
echo "=============================================="
