#!/bin/bash
set -euo pipefail
project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$project_dir"
python_bin="${PYTHON_BIN:-python3}"
mode="${1:-quick}"
model="${2:-qwen2.5:3b-instruct}"
case "$mode" in
  quick|run) ;;
  *) echo 'Usage: bash RUN_FRESH_CLAIMS.command [quick|run] [model]' >&2; exit 1 ;;
esac
"$python_bin" -c 'import sys; assert sys.version_info >= (3,11), "Python 3.11+ required"'
export PYTHONPATH="$project_dir/src${PYTHONPATH:+:$PYTHONPATH}"
"$python_bin" -m secure_rag.fresh_data
"$python_bin" -c 'import json; from pathlib import Path; assert json.loads(Path("data/fresh_claims/manifest.json").read_text()) == json.loads(Path("data/manifests/fresh_claims_v3.json").read_text()), "Fresh-data preparation differs from the fixed source manifest"'
set --
if [ "$mode" = quick ]; then
  set -- --max-new-pairs 1
fi
safe_model="${model//[^a-zA-Z0-9._-]/_}"
exec "$python_bin" -m secure_rag.claim_study run --data-dir data/fresh_claims \
  --model "$model" --seed 20261102 \
  --output-dir "outputs/fresh_v3_${safe_model}_20261102" "$@"
