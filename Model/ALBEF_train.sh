#!/bin/bash

#SBATCH --job-name=ALBEF_bns_nsbh
#SBATCH --output=logs/train/%x_%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=210G
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --partition=milan-gpu
#SBATCH --tmp=200G

set -euo pipefail

SCRIPT_SUBDIR="Model"
SCRIPT_REL_PATH="Model/ALBEF_train.sh"
REPO_NAME="gw-kn-multimodal"
DEFAULT_CONFIG_REL_PATH="Model/args/defaults/ALBEF_BNS_NSBH_default.json"
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

args_file=${1:-}
default_file=${2:-}
if [[ -z "${args_file}" || ! -f "${args_file}" ]]; then
    echo "Usage: $0 <experiment.json> [default.json]"
    exit 1
fi

args_dir="$(cd "$(dirname "${args_file}")" && pwd)"
args_file="${args_dir}/$(basename "${args_file}")"
if [[ -z "${default_file}" ]]; then
    args_base="$(basename "${args_file}")"
    inferred_default_base="${args_base%.json}_default.json"
    inferred_default="${REPO_ROOT}/Model/args/defaults/${inferred_default_base}"
    if [[ -f "${inferred_default}" ]]; then
        default_file="${inferred_default}"
    elif [[ "${args_base}" =~ ^(.+_v[0-9]+)(_.+)?\.json$ ]]; then
        versioned_default="${REPO_ROOT}/Model/args/defaults/${BASH_REMATCH[1]}_default.json"
        if [[ -f "${versioned_default}" ]]; then
            default_file="${versioned_default}"
        else
            default_file="${REPO_ROOT}/${DEFAULT_CONFIG_REL_PATH}"
        fi
    else
        default_file="${REPO_ROOT}/${DEFAULT_CONFIG_REL_PATH}"
    fi
fi
if [[ ! -f "${default_file}" ]]; then
    echo "Default config file not found: ${default_file}" >&2
    exit 1
fi
default_dir="$(cd "$(dirname "${default_file}")" && pwd)"
default_file="${default_dir}/$(basename "${default_file}")"

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

    echo "Submitting job with: sbatch ${sbatch_opts[*]} ${script_path} ${args_file} ${default_file}"
    (
        cd "${REPO_ROOT}"
        sbatch "${sbatch_opts[@]}" "${script_path}" "${args_file}" "${default_file}"
    )
    exit 0
fi

which python

DATA_PATH=$(jq -r -s '.[0] * .[1] | .data_path' "$default_file" "$args_file")
NEG_DATA_PATH=$(jq -r -s '.[0] * .[1] | .neg_data_path // empty' "$default_file" "$args_file")
CKPT_PATH=$(jq -r -s '.[0] * .[1] | .ckpt_path' "$default_file" "$args_file")
STAGE_TO_JOBFS=$(jq -r -s '.[0] * .[1] | .stage_to_jobfs // false' "$default_file" "$args_file")
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
echo "Default Configuration File: $default_file"
echo "Checkpoint Path: $CKPT_PATH"
echo ""

echo "========================================"
echo "Experiment Config Contents ($(basename "$args_file"))"
echo "========================================"
cat "$args_file"
echo ""

echo "========================================"
echo "Default Config Contents ($(basename "$default_file"))"
echo "========================================"
cat "$default_file"
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
    python -u "${SCRIPT_DIR}/ALBEF_train.py"
    --default_json_config "$default_file"
    --json_config "$args_file"
    --data_path "$DATA_PATH"
    --ckpt_path "$CKPT_PATH"
)

if [[ -n "$NEG_DATA_PATH" && "$NEG_DATA_PATH" != "null" ]]; then
    cmd+=(--neg_data_path "$NEG_DATA_PATH")
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
