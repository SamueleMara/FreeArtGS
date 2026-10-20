#!/bin/bash
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate freeartgs
export CUDA_HOME=/usr/local/cuda-12.8
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
runtime_env=$(python -B utils/local_storage.py --dry-run "$@")
eval "$runtime_env"
export MAX_JOBS="${MAX_JOBS:-2}"
python -B pipelines/train_local_datasets.py "$@"
