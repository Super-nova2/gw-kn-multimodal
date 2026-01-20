#!/bin/bash

#SBATCH --job-name=ALBEF_two_stage
#SBATCH --output=logs/train/%x_%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --partition=gpu

set -euo pipefail
which python

pretrain_args=${1:-}
finetune_args=${2:-}

if [[ -z "${pretrain_args}" || -z "${finetune_args}" ]]; then
    echo "Usage: $0 <pretrain_args.json> <finetune_args.json>"
    exit 1
fi

echo "========================================"
echo "Two-stage training job"
echo "========================================"
echo "Pretrain config: ${pretrain_args}"
echo "Finetune config: ${finetune_args}"
echo "Start time: $(date)"
echo "========================================"

bash /fred/oz016/bgao_kn/ML+GW+KN/Model/ALBEF_train.sh "${pretrain_args}"

echo "----------------------------------------"
echo "Stage 1 complete, starting finetune..."
echo "----------------------------------------"

bash /fred/oz016/bgao_kn/ML+GW+KN/Model/ALBEF_train.sh "${finetune_args}"

echo "----------------------------------------"
echo "All stages complete: $(date)"
