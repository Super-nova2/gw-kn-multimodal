#!/bin/bash

set -euo pipefail

if [[ -z "${KN_PIPELINE_ROOT:-}" ]]; then
    echo "KN_PIPELINE_ROOT is not set; submit this launcher through kn-sim" >&2
    exit 2
fi

PIPELINE_ROOT="$(cd "${KN_PIPELINE_ROOT}" && pwd)"
if [[ ! -f "${PIPELINE_ROOT}/src/worker.py" ]]; then
    echo "KN pipeline worker does not exist: ${PIPELINE_ROOT}/src/worker.py" >&2
    exit 2
fi
export PYTHONPATH="${PIPELINE_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONDONTWRITEBYTECODE=1

if [[ "${KN_COMMAND:-work}" == "work" ]]; then
    SCRATCH_BASE="${SLURM_TMPDIR:-${JOBFS:-}}"
    if [[ -z "$SCRATCH_BASE" || ! -d "$SCRATCH_BASE" || ! -w "$SCRATCH_BASE" ]]; then
        echo "A writable SLURM_TMPDIR or JOBFS is required for SNANA work" >&2
        exit 2
    fi
    KN_SCRATCH_DIR="$(mktemp -d "${SCRATCH_BASE%/}/kn_sim_${SLURM_JOB_ID:-local}_${SLURM_ARRAY_TASK_ID:-0}.XXXXXX")"
    cleanup_scratch() {
        local expected_prefix="${SCRATCH_BASE%/}/kn_sim_${SLURM_JOB_ID:-local}_${SLURM_ARRAY_TASK_ID:-0}."
        if [[ -n "${KN_SCRATCH_DIR:-}" && "$KN_SCRATCH_DIR" == "${expected_prefix}"* && -d "$KN_SCRATCH_DIR" ]]; then
            rm -rf -- "$KN_SCRATCH_DIR"
        fi
    }
    trap cleanup_scratch EXIT
    export KN_SCRATCH_DIR
    python "${PIPELINE_ROOT}/src/worker.py"
else
    exec python "${PIPELINE_ROOT}/src/worker.py"
fi
