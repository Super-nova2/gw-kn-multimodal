#!/bin/bash
# Batch submit the current module-ablation training jobs.
# Usage: bash Model/script/submit_ablation_train.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
TRAIN_SCRIPT="${REPO_ROOT}/Model/ALBEF_train.sh"
DEFAULT_CONFIG="${REPO_ROOT}/Model/args/defaults/ALBEF_BNS_NSBH_default.json"
ARGS_DIR="${REPO_ROOT}/Model/args"

TIMESTAMP="$(date +"%Y%m%d-%H%M%S")"
LOG_DIR="${REPO_ROOT}/logs/ablation_train_${TIMESTAMP}"
mkdir -p "${LOG_DIR}"

echo "========================================"
echo "Module Ablation Train Submission"
echo "========================================"
echo "Default config: ${DEFAULT_CONFIG}"
echo "Log directory:  ${LOG_DIR}"
echo "Submit time:    ${TIMESTAMP}"
echo ""

declare -A EXPERIMENTS=(
    ["full"]="ALBEF_BNS_NSBH_full.json"
    ["no_cls_loss"]="ALBEF_BNS_NSBH_no_cls_loss.json"
    ["no_gallery_loss"]="ALBEF_BNS_NSBH_no_gallery_loss.json"
    ["no_itc_loss"]="ALBEF_BNS_NSBH_no_itc_loss.json"
    ["no_cross_atten"]="ALBEF_BNS_NSBH_no_cross_atten.json"
    ["no_hard_mining"]="ALBEF_BNS_NSBH_no_hard_mining.json"
    ["no_fusion"]="ALBEF_BNS_NSBH_no_fusion.json"
)

SUBMIT_ORDER=(
    "full"
    "no_cls_loss"
    "no_gallery_loss"
    "no_itc_loss"
    "no_cross_atten"
    "no_hard_mining"
    "no_fusion"
)

if ! command -v sbatch >/dev/null 2>&1; then
    echo "[ERROR] sbatch not found; run this script on a Slurm login node." >&2
    exit 1
fi

if [[ ! -f "${TRAIN_SCRIPT}" ]]; then
    echo "[ERROR] Train script not found: ${TRAIN_SCRIPT}" >&2
    exit 1
fi

if [[ ! -f "${DEFAULT_CONFIG}" ]]; then
    echo "[ERROR] Default config not found: ${DEFAULT_CONFIG}" >&2
    exit 1
fi

for name in "${SUBMIT_ORDER[@]}"; do
    config_file="${ARGS_DIR}/${EXPERIMENTS[$name]}"
    if [[ ! -f "${config_file}" ]]; then
        echo "[ERROR] Config not found: ${config_file}" >&2
        exit 1
    fi
done

for name in "${SUBMIT_ORDER[@]}"; do
    config_file="${ARGS_DIR}/${EXPERIMENTS[$name]}"
    job_name="abl_train_${name}"

    echo "Submitting: ${job_name}"
    sbatch \
        --job-name="${job_name}" \
        --output="${LOG_DIR}/${name}_%j.out" \
        "${TRAIN_SCRIPT}" "${config_file}" "${DEFAULT_CONFIG}"
    echo "  Config: ${EXPERIMENTS[$name]}"
    echo "  Log:    ${LOG_DIR}/${name}_%j.out"
    echo ""
done

echo "All ${#SUBMIT_ORDER[@]} ablation experiments submitted."
echo "Log directory: ${LOG_DIR}"
echo ""
echo "Check status:  squeue -u \$USER"
echo "View logs:     ls ${LOG_DIR}/"
