#!/bin/bash
set -euo pipefail
cd -- "$(dirname -- "$0")"
task_python="${SECURE_RAG_PYTHON:-python3.11}"
if ! command -v "$task_python" >/dev/null 2>&1; then
    task_python=python3
fi
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
"$task_python" -c 'import sys; assert sys.version_info >= (3,11), "Python 3.11 or newer is required"'
if ! "$task_python" -m secure_rag run-kcv-review --help >/dev/null 2>&1; then
    if [ ! -f SECURE_RAG_KCV_Review.patch ]; then
        echo "Put SECURE_RAG_KCV_Review.patch beside this command in the existing project folder."
        exit 1
    fi
    patch --dry-run -p1 < SECURE_RAG_KCV_Review.patch
    patch -p1 < SECURE_RAG_KCV_Review.patch
fi
"$task_python" -m secure_rag run-kcv-review \
    --predictions outputs/submission/evidence_first_repaired_predictions.jsonl \
    "$@"
echo "Open outputs/kcv_review/RESULTS.md for measured results (or your custom output directory)."
