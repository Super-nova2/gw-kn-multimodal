#!/bin/bash

#SBATCH --job-name=MAGIKS_attr
#SBATCH --output=/fred/oz016/bgao_kn/logs/eval/%x_%A_%a.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=80G
#SBATCH --gres=gpu:a100:1
#SBATCH --time=12:00:00
#SBATCH --partition=gpu
#SBATCH --tmp=200G

set -euo pipefail

SCRIPT_SUBDIR="Model/scripts/eval"
REPO_NAME="gw-kn-multimodal"
WORKSPACE_ROOT_DEFAULT="/fred/oz016/bgao_kn"
DEFAULT_TEMPLATE_REL="Model/args/eval/fixed_checkpoint_attribution_v2.json"

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

SCRIPT_PATH="${REPO_ROOT}/${SCRIPT_SUBDIR}/submit_fixed_checkpoint_attribution.sh"
RUN_SCRIPT="${REPO_ROOT}/${SCRIPT_SUBDIR}/run_fixed_checkpoint_attribution.py"
WORKSPACE_ROOT="${WORKSPACE_ROOT:-${WORKSPACE_ROOT_DEFAULT}}"
LOG_DIR="${WORKSPACE_ROOT}/logs/eval"

template_file="${1:-${REPO_ROOT}/${DEFAULT_TEMPLATE_REL}}"
phase="${2:-smoke}"
manifest_file="${3:-}"

# Slurm does not guarantee that the batch step starts in the caller's current
# directory. Resolve user-supplied relative paths before passing them to sbatch.
if [[ "${template_file}" != /* ]]; then
    if [[ -f "${template_file}" ]]; then
        template_file="$(cd "$(dirname "${template_file}")" && pwd)/$(basename "${template_file}")"
    else
        template_file="${REPO_ROOT}/${template_file}"
    fi
fi
if [[ -n "${manifest_file}" && "${manifest_file}" != /* ]]; then
    if [[ -f "${manifest_file}" ]]; then
        manifest_file="$(cd "$(dirname "${manifest_file}")" && pwd)/$(basename "${manifest_file}")"
    else
        manifest_file="${REPO_ROOT}/${manifest_file}"
    fi
fi

case "${phase}" in
    smoke|seed42|remaining_seeds|three_seed) ;;
    *)
        echo "Invalid phase: ${phase}" >&2
        echo "Expected smoke, seed42, remaining_seeds, or three_seed." >&2
        exit 2
        ;;
esac

for required in "${SCRIPT_PATH}" "${RUN_SCRIPT}" "${template_file}"; do
    if [[ ! -f "${required}" ]]; then
        echo "Required file not found: ${required}" >&2
        exit 1
    fi
done
if ! command -v jq >/dev/null 2>&1; then
    echo "jq is required." >&2
    exit 1
fi
mkdir -p "${LOG_DIR}"

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    if [[ "${DRY_RUN:-false}" == "true" ]]; then
        python -u "${RUN_SCRIPT}" --template "${template_file}" --phase "${phase}" --dry-run
        echo "DRY_RUN=true: no configs written and no jobs submitted."
        exit 0
    fi
    if [[ -z "${manifest_file}" ]]; then
        generated_root=$(jq -r '.generated_config_root' "${template_file}")
        if [[ "${generated_root}" != /* ]]; then
            generated_root="$(cd "$(dirname "${template_file}")" && pwd)/${generated_root}"
        fi
        manifest_file="${generated_root}/${phase}/suite_manifest.json"
        if [[ -f "${manifest_file}" ]]; then
            echo "Reusing existing immutable suite manifest: ${manifest_file}"
        else
            python -u "${RUN_SCRIPT}" --template "${template_file}" --phase "${phase}"
        fi
    fi
    if [[ ! -f "${manifest_file}" ]]; then
        echo "Suite manifest not found: ${manifest_file}" >&2
        exit 1
    fi
    job_count=$(jq -r '.n_jobs' "${manifest_file}")
    if [[ ! "${job_count}" =~ ^[1-9][0-9]*$ ]]; then
        echo "Invalid job count in manifest: ${job_count}" >&2
        exit 1
    fi
    if ! command -v sbatch >/dev/null 2>&1; then
        echo "sbatch not found; configs were generated at ${manifest_file}." >&2
        exit 1
    fi
    max_parallel="${MAX_PARALLEL:-4}"
    sbatch_opts=(--array="0-$((job_count - 1))%${max_parallel}")
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
    echo "Submitting ${job_count} jobs from ${manifest_file} (max_parallel=${max_parallel})"
    (
        cd "${WORKSPACE_ROOT}"
        sbatch "${sbatch_opts[@]}" "${SCRIPT_PATH}" "${template_file}" "${phase}" "${manifest_file}"
    )
    exit 0
fi

if [[ -z "${manifest_file}" || ! -f "${manifest_file}" ]]; then
    echo "Slurm worker requires a valid suite manifest as argument 3." >&2
    exit 1
fi
array_index="${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}"
config_file=$(jq -r --argjson idx "${array_index}" '.jobs[$idx].config // empty' "${manifest_file}")
if [[ -z "${config_file}" || ! -f "${config_file}" ]]; then
    echo "Config for array index ${array_index} not found: ${config_file}" >&2
    exit 1
fi

test_data=$(jq -r '.test_data_path // empty' "${config_file}")
checkpoint=$(jq -r '.models[] | select(.name == "Full") | .checkpoint // empty' "${config_file}")
model_config=$(jq -r '.models[] | select(.name == "Full") | .config // empty' "${config_file}")
output_dir=$(jq -r '.output_dir // empty' "${config_file}")
task=$(jq -r '.task // empty' "${config_file}")
condition=$(jq -r '.condition.name // empty' "${config_file}")
seed=$(jq -r '.seed // empty' "${config_file}")

for required in "${test_data}" "${checkpoint}" "${model_config}"; do
    if [[ -z "${required}" || ! -e "${required}" ]]; then
        echo "Required input not found: ${required}" >&2
        exit 1
    fi
done
if [[ -z "${output_dir}" ]]; then
    echo "output_dir missing from ${config_file}" >&2
    exit 1
fi

echo "========================================"
echo "Fixed-checkpoint attribution worker"
echo "Job: ${SLURM_JOB_ID}; array index: ${array_index}"
echo "Task: ${task}; condition: ${condition}; seed: ${seed}"
echo "Config: ${config_file}"
echo "Output: ${output_dir}"
echo "GPU: ${CUDA_VISIBLE_DEVICES:-unknown}"
echo "========================================"

python -u "${RUN_SCRIPT}" --config "${config_file}"
