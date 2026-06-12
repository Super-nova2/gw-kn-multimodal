#!/bin/bash

#SBATCH --job-name=ALBEF_model_speed
#SBATCH --output=/fred/oz016/bgao_kn/logs/benchmark/%x_%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=0:30:00
#SBATCH --partition=gpu
#SBATCH --tmp=20G

set -euo pipefail

SCRIPT_SUBDIR="Model/script"
SCRIPT_REL_PATH="Model/script/submit_model_speed_benchmark.sh"
PY_REL_PATH="Model/script/benchmark_model_speed.py"
REPO_NAME="gw-kn-multimodal"
WORKSPACE_ROOT_DEFAULT="/fred/oz016/bgao_kn"
DEFAULT_CONFIG_REL="Model/args/eval/model_speed_benchmark.json"
WORKSPACE_ROOT="${WORKSPACE_ROOT:-${WORKSPACE_ROOT_DEFAULT}}"
LOG_DIR="${WORKSPACE_ROOT}/logs/benchmark"
DEFAULT_OUTPUT_LOG="${LOG_DIR}/%x_%j.out"

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

SCRIPT_PATH="${REPO_ROOT}/${SCRIPT_REL_PATH}"
BENCH_SCRIPT="${REPO_ROOT}/${PY_REL_PATH}"
DEFAULT_CONFIG_PATH="${REPO_ROOT}/${DEFAULT_CONFIG_REL}"

if [[ ! -f "${SCRIPT_PATH}" ]]; then
    echo "Resolved script path not found: ${SCRIPT_PATH}" >&2
    exit 1
fi
if [[ ! -f "${BENCH_SCRIPT}" ]]; then
    echo "Benchmark script not found: ${BENCH_SCRIPT}" >&2
    exit 1
fi

config_file=${1:-${DEFAULT_CONFIG_PATH}}
if [[ ! -f "${config_file}" ]]; then
    echo "Config not found: ${config_file}" >&2
    echo "Usage: $0 [model_speed_benchmark.json]" >&2
    exit 1
fi
config_dir="$(cd "$(dirname "${config_file}")" && pwd)"
config_file="${config_dir}/$(basename "${config_file}")"

mkdir -p "${LOG_DIR}"

if ! command -v jq >/dev/null 2>&1; then
    echo "jq not found; required to parse benchmark config." >&2
    exit 1
fi

CHECKPOINT=$(jq -r '.checkpoint // empty' "${config_file}")
DATA_PATH=$(jq -r '.data_path // empty' "${config_file}")
MODEL_CONFIG=$(jq -r '.model_config // empty' "${config_file}")
OUTPUT_DIR=$(jq -r '.output_dir // empty' "${config_file}")
BATCH_SIZES=$(jq -r 'if has("batch_sizes") then (.batch_sizes | join(",")) else "1,8,32,128,256,512,1024" end' "${config_file}")
PRECISION_MODES=$(jq -r 'if has("precision_modes") then (.precision_modes | join(",")) else "fp32,amp" end' "${config_file}")

if [[ -z "${CHECKPOINT}" || "${CHECKPOINT}" == "null" ]]; then
    echo "Required field missing in config: checkpoint" >&2
    exit 1
fi
if [[ -z "${DATA_PATH}" || "${DATA_PATH}" == "null" ]]; then
    echo "Required field missing in config: data_path" >&2
    exit 1
fi
if [[ -z "${OUTPUT_DIR}" || "${OUTPUT_DIR}" == "null" ]]; then
    echo "Required field missing in config: output_dir" >&2
    exit 1
fi
if [[ ! -f "${CHECKPOINT}" ]]; then
    echo "Checkpoint not found: ${CHECKPOINT}" >&2
    exit 1
fi
if [[ ! -f "${DATA_PATH}" ]]; then
    echo "Data path not found: ${DATA_PATH}" >&2
    exit 1
fi
if [[ -n "${MODEL_CONFIG}" && "${MODEL_CONFIG}" != "null" && ! -f "${MODEL_CONFIG}" ]]; then
    echo "Model config not found: ${MODEL_CONFIG}" >&2
    exit 1
fi
mkdir -p "${OUTPUT_DIR}"

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    if ! command -v sbatch >/dev/null 2>&1; then
        echo "sbatch not found; run inside a Slurm allocation or install Slurm tools." >&2
        exit 1
    fi

    sbatch_opts=()
    if [[ -n "${JOB_NAME:-}" ]]; then
        sbatch_opts+=(--job-name="${JOB_NAME}")
    fi
    if [[ -n "${OUTPUT_LOG:-}" ]]; then
        sbatch_opts+=(--output="${OUTPUT_LOG}")
    else
        sbatch_opts+=(--output="${DEFAULT_OUTPUT_LOG}")
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

    echo "Submitting model speed benchmark"
    echo "  config: ${config_file}"
    echo "  checkpoint: ${CHECKPOINT}"
    echo "  data: ${DATA_PATH}"
    echo "  output_dir: ${OUTPUT_DIR}"
    echo "  batch_sizes: ${BATCH_SIZES}"
    echo "  precision_modes: ${PRECISION_MODES}"
    echo "Submitting job with: sbatch ${sbatch_opts[*]} ${SCRIPT_PATH} ${config_file}"
    (
        cd "${WORKSPACE_ROOT}"
        sbatch "${sbatch_opts[@]}" "${SCRIPT_PATH}" "${config_file}"
    )
    exit 0
fi

which python

echo "========================================"
echo "SLURM Job Information"
echo "========================================"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Job Name: ${SLURM_JOB_NAME:-unknown}"
echo "Node: ${SLURMD_NODENAME:-unknown}"
echo "Partition: ${SLURM_JOB_PARTITION:-unknown}"
echo "CPUs: ${SLURM_CPUS_PER_TASK:-unknown}"
echo "Memory: ${SLURM_MEM_PER_NODE:-unknown}MB"
echo "GPUs: ${CUDA_VISIBLE_DEVICES:-unknown}"
echo "Start time: $(date)"
echo "========================================"
echo
echo "Config: ${config_file}"
echo "Checkpoint: ${CHECKPOINT}"
echo "Data path: ${DATA_PATH}"
echo "Model config: ${MODEL_CONFIG:-none}"
echo "Output dir: ${OUTPUT_DIR}"
echo "Batch sizes: ${BATCH_SIZES}"
echo "Precision modes: ${PRECISION_MODES}"
echo "Run mode: pure GPU forward benchmark"
echo

python -u "${BENCH_SCRIPT}" --config "${config_file}"
