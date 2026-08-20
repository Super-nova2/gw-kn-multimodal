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
exec python "${PIPELINE_ROOT}/src/worker.py"
