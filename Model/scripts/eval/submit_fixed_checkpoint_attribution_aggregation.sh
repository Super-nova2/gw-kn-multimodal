#!/bin/bash

#SBATCH --job-name=MAGIKS_attr_agg
#SBATCH --output=/fred/oz016/bgao_kn/logs/eval/%x_%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --partition=skylake

set -euo pipefail

SCRIPT_SUBDIR="Model/scripts/eval"
REPO_NAME="gw-kn-multimodal"
WORKSPACE_ROOT_DEFAULT="/fred/oz016/bgao_kn"
WORKSPACE_ROOT="${WORKSPACE_ROOT:-${WORKSPACE_ROOT_DEFAULT}}"

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

SCRIPT_PATH="${REPO_ROOT}/${SCRIPT_SUBDIR}/submit_fixed_checkpoint_attribution_aggregation.sh"
AGGREGATE_SCRIPT="${REPO_ROOT}/${SCRIPT_SUBDIR}/aggregate_fixed_checkpoint_attribution.py"
DEFAULT_INPUT_ROOT="${REPO_ROOT}/Model/eval_results/fixed_checkpoint_attribution_v3/three_seed"
DEFAULT_OUTPUT_DIR="${REPO_ROOT}/Model/eval_results/fixed_checkpoint_attribution_v3/aggregation"
LOG_DIR="${WORKSPACE_ROOT}/logs/eval"

input_root="${1:-${DEFAULT_INPUT_ROOT}}"
output_dir="${2:-${DEFAULT_OUTPUT_DIR}}"
bootstrap_replicates="${3:-10000}"
bootstrap_seed="${4:-20260824}"

if [[ ! -f "${SCRIPT_PATH}" || ! -f "${AGGREGATE_SCRIPT}" ]]; then
    echo "Required script not found." >&2
    exit 1
fi
if [[ ! -d "${input_root}" ]]; then
    echo "Input root not found: ${input_root}" >&2
    exit 1
fi
if [[ ! "${bootstrap_replicates}" =~ ^[1-9][0-9]*$ ]]; then
    echo "bootstrap_replicates must be a positive integer." >&2
    exit 2
fi
if [[ ! "${bootstrap_seed}" =~ ^[0-9]+$ ]]; then
    echo "bootstrap_seed must be a non-negative integer." >&2
    exit 2
fi
mkdir -p "${LOG_DIR}"

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
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
    if [[ -n "${MEM_PER_TASK:-}" ]]; then
        sbatch_opts+=(--mem="${MEM_PER_TASK}")
    fi
    echo "Submitting fixed-checkpoint attribution aggregation"
    echo "  input_root: ${input_root}"
    echo "  output_dir: ${output_dir}"
    echo "  bootstrap_replicates: ${bootstrap_replicates}"
    echo "  bootstrap_seed: ${bootstrap_seed}"
    (
        cd "${WORKSPACE_ROOT}"
        sbatch "${sbatch_opts[@]}" "${SCRIPT_PATH}" \
            "${input_root}" "${output_dir}" \
            "${bootstrap_replicates}" "${bootstrap_seed}"
    )
    exit 0
fi

echo "========================================"
echo "Fixed-checkpoint attribution aggregation"
echo "Job: ${SLURM_JOB_ID}"
echo "Node: ${SLURMD_NODENAME:-unknown}"
echo "Partition: ${SLURM_JOB_PARTITION:-unknown}"
echo "Input: ${input_root}"
echo "Output: ${output_dir}"
echo "Bootstrap replicates: ${bootstrap_replicates}"
echo "Bootstrap seed: ${bootstrap_seed}"
echo "========================================"

python -u "${AGGREGATE_SCRIPT}" \
    --input-root "${input_root}" \
    --output-dir "${output_dir}" \
    --expected-trials 10 \
    --bootstrap-replicates "${bootstrap_replicates}" \
    --bootstrap-seed "${bootstrap_seed}" \
    --equivalence-margin 0.01 \
    --overwrite
