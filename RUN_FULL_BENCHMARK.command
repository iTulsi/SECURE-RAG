#!/bin/bash
set -euo pipefail

project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$project_dir"
python_bin="${PYTHON_BIN:-python3}"
mode="${1:-quick}"
model="${2:-qwen2.5:3b-instruct}"
seed="${3:-20261031}"
upstream_dir="data/raw/SECURE"
upstream_revision="a2412e8ab4b6051ba7381d4f590cb48d9743c7c8"

case "$mode" in
  quick|test|full|development|validation) ;;
  *) echo 'Usage: bash RUN_FULL_BENCHMARK.command [quick|test|full|development|validation] [model] [seed]' >&2; exit 1 ;;
esac
"$python_bin" -c 'import sys; assert sys.version_info >= (3, 11), "Python 3.11+ is required"'
export PYTHONPATH="$project_dir/src${PYTHONPATH:+:$PYTHONPATH}"
# Check the model before fetching/preparing data. Never install or pull weights silently.
"$python_bin" -c 'import sys; from secure_rag.benchmark import model_identity; print(model_identity("http://localhost:11434", sys.argv[1], 10)["model"])' "$model"

if [ ! -d "$upstream_dir/.git" ]; then
  git clone https://github.com/aiforsec/SECURE.git "$upstream_dir"
fi
if [ -n "$(git -C "$upstream_dir" status --porcelain)" ]; then
  echo "The upstream data checkout has local edits. Preserve them before continuing." >&2
  exit 1
fi
git -C "$upstream_dir" checkout --detach "$upstream_revision"
mkdir -p data/full
"$python_bin" -m secure_rag.benchmark prepare --data-dir "$upstream_dir/Dataset" \
  --output-dir data/full > data/full/preparation_summary.json
"$python_bin" -c 'import json; from pathlib import Path; assert json.loads(Path("data/full/manifest.json").read_text()) == json.loads(Path("data/manifests/full_benchmark.json").read_text()), "Prepared split differs from committed protocol"'

split="$mode"
set --
if [ "$mode" = quick ]; then
  split=test
  set -- --max-new-examples 6
fi
safe_model="${model//[^a-zA-Z0-9._-]/_}"
exec "$python_bin" -m secure_rag.benchmark run --model "$model" --split "$split" \
  --seed "$seed" --output-dir "outputs/full_${safe_model}_${split}_${seed}" "$@"
