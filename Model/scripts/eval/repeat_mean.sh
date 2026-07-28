#!/bin/bash
# One entrypoint for three fixed-seed evaluation and arithmetic-mean aggregation.
set -euo pipefail

CONFIG_ARG="${1:-Model/args/eval/repeat_mean_v1.json}"
ACTION="${2:-dry-run}"
TASK="${3:-}"
SEED="${4:-}"

resolve_repo() {
    git -C "$(dirname "$1")" rev-parse --show-toplevel
}

if [[ "${ACTION}" == "worker-aggregate" ]]; then
    MEAN_CONFIG="$(realpath -e "${CONFIG_ARG}")"
    REPO_ROOT="$(resolve_repo "${MEAN_CONFIG}")"
    ENTRYPOINT="${REPO_ROOT}/Model/scripts/eval/repeat_mean.py"
    if [[ "${WORKER_DRY_RUN:-false}" == "true" ]]; then
        echo "repo_root=${REPO_ROOT} action=worker-aggregate config=${MEAN_CONFIG}"
        exit 0
    fi
    cd "${REPO_ROOT}"
    exec python -u "${ENTRYPOINT}" aggregate --config "${MEAN_CONFIG}"
fi

MASTER_CONFIG="$(realpath -e "${CONFIG_ARG}")"
MASTER_DIR="$(dirname "${MASTER_CONFIG}")"
REPO_ROOT="$(resolve_repo "${MASTER_CONFIG}")"
ENTRYPOINT="${REPO_ROOT}/Model/scripts/eval/repeat_mean.py"
SELF="${REPO_ROOT}/Model/scripts/eval/repeat_mean.sh"
CONFIG_ROOT="$(jq -r '.generated_config_root' "${MASTER_CONFIG}")"
OUTPUT_ROOT="$(jq -r '.output_root' "${MASTER_CONFIG}")"
if [[ "${CONFIG_ROOT}" != /* ]]; then CONFIG_ROOT="${MASTER_DIR}/${CONFIG_ROOT}"; fi
if [[ "${OUTPUT_ROOT}" != /* ]]; then OUTPUT_ROOT="${MASTER_DIR}/${OUTPUT_ROOT}"; fi
mapfile -t SEEDS < <(jq -r '.eval_seeds[]' "${MASTER_CONFIG}")
[[ "${SEEDS[*]}" == "42 123 456" ]] || {
    echo "Repeat-mean workflow requires seeds 42, 123, 456." >&2
    exit 2
}
LOG_DIR="/fred/oz016/bgao_kn/logs/eval/repeat_mean_v1"

require_task() {
    [[ "$1" == "retrieval" || "$1" == "gw170817" ]] || {
        echo "Task must be retrieval or gw170817." >&2
        exit 2
    }
}

require_success() {
    [[ -f "$1/_SUCCESS.json" ]] || {
        echo "Missing successful run: $1" >&2
        exit 3
    }
}

cd "${REPO_ROOT}"
case "${ACTION}" in
    dry-run)
        python "${ENTRYPOINT}" prepare --config "${MASTER_CONFIG}" --dry-run
        bash -n "${SELF}"
        SLURM_ARRAY_TASK_ID=0 WORKER_DRY_RUN=true \
            "${SELF}" "${MASTER_CONFIG}" worker-seed retrieval
        WORKER_DRY_RUN=true \
            "${SELF}" "${CONFIG_ROOT}/retrieval/mean.json" worker-aggregate
        echo "Plan: exactly seeds 42,123,456; per-seed metrics; arithmetic mean only."
        ;;
    prepare)
        python "${ENTRYPOINT}" prepare --config "${MASTER_CONFIG}"
        ;;
    seed)
        require_task "${TASK}"
        INDEX="$(jq -r --argjson seed "${SEED:?seed action requires a seed}" \
            '.eval_seeds | index($seed) // empty' "${MASTER_CONFIG}")"
        [[ -n "${INDEX}" ]] || {
            echo "Seed ${SEED} is not configured." >&2
            exit 2
        }
        [[ ! -e "${OUTPUT_ROOT}/${TASK}/seed_${SEED}" ]] || {
            echo "Refusing to overwrite: ${OUTPUT_ROOT}/${TASK}/seed_${SEED}" >&2
            exit 4
        }
        mkdir -p "${LOG_DIR}"
        if [[ "${TASK}" == "retrieval" ]]; then
            sbatch --array="${INDEX}" --partition=milan-gpu --gres=gpu:a100:1 \
                --cpus-per-task=4 --mem=80G --tmp=200G --time=12:00:00 \
                --chdir="${REPO_ROOT}" --job-name="mean3_ret_s${SEED}" \
                --output="${LOG_DIR}/mean3_ret_s${SEED}_%A_%a.out" \
                "${SELF}" "${MASTER_CONFIG}" worker-seed retrieval
        else
            sbatch --array="${INDEX}" --partition=milan-gpu --gres=gpu:a100:1 \
                --cpus-per-task=4 --mem=80G --time=08:00:00 \
                --chdir="${REPO_ROOT}" --job-name="mean3_gw_s${SEED}" \
                --output="${LOG_DIR}/mean3_gw_s${SEED}_%A_%a.out" \
                "${SELF}" "${MASTER_CONFIG}" worker-seed gw170817
        fi
        ;;
    worker-seed)
        require_task "${TASK}"
        INDEX="${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}"
        if (( INDEX < 0 || INDEX >= ${#SEEDS[@]} )); then
            echo "Array index ${INDEX} is outside seed list." >&2
            exit 2
        fi
        SELECTED_SEED="${SEEDS[${INDEX}]}"
        SEED_CONFIG="${CONFIG_ROOT}/${TASK}/seed_${SELECTED_SEED}.json"
        [[ -f "${SEED_CONFIG}" ]] || {
            echo "Generated config not found: ${SEED_CONFIG}" >&2
            exit 2
        }
        if [[ "${WORKER_DRY_RUN:-false}" == "true" ]]; then
            echo "repo_root=${REPO_ROOT} action=worker-seed task=${TASK} seed=${SELECTED_SEED} config=${SEED_CONFIG}"
            exit 0
        fi
        case "${TASK}" in
            retrieval)
                exec Model/scripts/eval/submit_retrieval_comparison.sh "${SEED_CONFIG}"
                ;;
            gw170817)
                exec Model/scripts/eval/submit_gw170817a_retrieval.sh "${SEED_CONFIG}"
                ;;
        esac
        ;;
    aggregate)
        mkdir -p "${LOG_DIR}"
        for task_name in retrieval gw170817; do
            for seed_value in "${SEEDS[@]}"; do
                require_success "${OUTPUT_ROOT}/${task_name}/seed_${seed_value}"
            done
            [[ ! -e "${OUTPUT_ROOT}/${task_name}/mean3_summary" ]] || {
                echo "Refusing to overwrite mean output for ${task_name}." >&2
                exit 4
            }
            sbatch --partition=milan --cpus-per-task=4 --mem=32G --time=02:00:00 \
                --chdir="${REPO_ROOT}" --job-name="mean3_${task_name}" \
                --output="${LOG_DIR}/mean3_${task_name}_%j.out" \
                "${SELF}" "${CONFIG_ROOT}/${task_name}/mean.json" worker-aggregate
        done
        ;;
    *)
        echo "Usage: $0 CONFIG {dry-run|prepare|seed TASK SEED|aggregate}" >&2
        exit 2
        ;;
esac
