#!/bin/bash
set -euo pipefail
project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$project_dir"
mode="${1:-quick}"
case "$mode" in
  setup|prepare|quick|run) ;;
  *) echo 'Usage: bash RUN_LORA_TRIAL.command [setup|prepare|quick|run]' >&2; exit 1 ;;
esac
python_base="${PYTHON_BIN:-python3.11}"
if ! command -v "$python_base" >/dev/null; then
  python_base=python3
fi
"$python_base" -c 'import platform, sys; assert sys.version_info >= (3,11), "Python 3.11+ required"; assert platform.system() == "Darwin" and platform.machine() == "arm64", "Native Apple Silicon Python required"'
if [ ! -x .venv-lora/bin/python ]; then
  "$python_base" -m venv .venv-lora
fi
python_bin="$project_dir/.venv-lora/bin/python"
if ! "$python_bin" -c 'from importlib.metadata import version; assert version("mlx-lm") == "0.32.2" and version("mlx") == "0.32.2"' 2>/dev/null; then
  "$python_bin" -m pip install 'mlx-lm[train]==0.32.2' 'mlx==0.32.2'
fi
if [ "$mode" = setup ]; then
  echo 'MLX environment installed. Next: bash RUN_LORA_TRIAL.command quick'
  exit 0
fi
if [ ! -f data/full/manifest.json ]; then
  echo 'Missing data/full/manifest.json. Use the previously prepared six-task benchmark data.' >&2
  exit 1
fi
export PYTHONPATH="$project_dir/src${PYTHONPATH:+:$PYTHONPATH}"
exec "$python_bin" -m secure_rag.lora_trial "$mode"
