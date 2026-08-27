#!/bin/bash
# Event-specific orchestration; retrieval execution remains in the common wrapper.

#SBATCH --job-name=GW170817A_retrieval
#SBATCH --output=/fred/oz016/bgao_kn/logs/eval/%x_%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --gres=gpu:a100:1
#SBATCH --time=12:00:00
#SBATCH --partition=gpu
#SBATCH --tmp=200G

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
SCRIPT_PATH="${SCRIPT_DIR}/submit_gw170817a_retrieval.sh"
CONFIG="${1:-${REPO_ROOT}/Model/args/eval/retrieval_gw170817a_lsst.json}"

if [[ ! -f "${CONFIG}" ]]; then
    echo "Config not found: ${CONFIG}" >&2
    exit 1
fi
if [[ "${DRY_RUN:-false}" == "true" && -z "${SLURM_JOB_ID:-}" ]]; then
    SLURM_JOB_ID=dry-run "${SCRIPT_DIR}/submit_retrieval_comparison.sh" "${CONFIG}"
    exit $?
fi

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    mkdir -p /fred/oz016/bgao_kn/logs/eval
    sbatch_opts=(--job-name="${JOB_NAME:-GW170817A_retrieval}")
    [[ -z "${OUTPUT_LOG:-}" ]] || sbatch_opts+=(--output="${OUTPUT_LOG}")
    [[ -z "${DEPENDENCY:-}" ]] || sbatch_opts+=(--dependency="${DEPENDENCY}")
    sbatch "${sbatch_opts[@]}" "${SCRIPT_PATH}" "${CONFIG}"
    exit 0
fi

"${SCRIPT_DIR}/submit_retrieval_comparison.sh" "${CONFIG}"

TEST_H5="$(jq -r '.test_data_path' "${CONFIG}")"
OUTPUT_DIR="$(jq -r '.output_dir' "${CONFIG}")"
python -u "${SCRIPT_DIR}/analyze_gw170817a_retrieval.py" \
    --outcomes "${OUTPUT_DIR}/retrieval_outcomes.csv.gz" \
    --test-h5 "${TEST_H5}" \
    --output-dir "${OUTPUT_DIR}/scenario_analysis" \
    --reference-model Full \
    --n-bootstrap 10000 \
    --seed 170817
