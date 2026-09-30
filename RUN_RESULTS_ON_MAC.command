#!/bin/bash
set -euo pipefail
cd -- "$(dirname -- "$0")"
task_python="${SECURE_RAG_PYTHON:-python3.11}"
if ! command -v "$task_python" >/dev/null 2>&1; then
    task_python=python3
fi
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
"$task_python" -c 'import sys; assert sys.version_info >= (3,11), "Python 3.11 or newer is required"'
"$task_python" -m secure_rag run-review \
    --baseline-predictions outputs/e0_predictions.jsonl
echo "Open outputs/submission/RESULTS.md for the new measured results."
