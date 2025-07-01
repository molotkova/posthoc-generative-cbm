#!/bin/bash
# Unset PYTHONPATH to avoid any local Python settings
unset PYTHONPATH
# Unset any conda-related environment variables to start fresh
unset CONDA_PREFIX
unset CONDA_DEFAULT_ENV
unset CONDA_PYTHON_EXE
unset CONDA_SHLVL

# Set torch visible gpus environment variable
if command -v nvidia-smi &> /dev/null; then
    GPU_COUNT=$(nvidia-smi --list-gpus | wc -l)
    if [ "$GPU_COUNT" -gt 0 ]; then
        echo "Detected $GPU_COUNT GPUs"
        # Create a comma-separated list of all GPU indices
        GPU_INDICES=$(seq -s, 0 $((GPU_COUNT-1)))
        export CUDA_VISIBLE_DEVICES=$GPU_INDICES
        echo "Setting CUDA_VISIBLE_DEVICES=$GPU_INDICES"
    else
        echo "No GPUs detected"
    fi
else
    echo "nvidia-smi not found, assuming no GPUs available"
fi

path=/home/somo00003/posthoc-generative-cbm  # Project working directory

source ~/miniconda3/etc/profile.d/conda.sh
conda activate posthocgencbm

cd "${path}"
pwd=$(pwd)

export HOME="$(pwd)"
export PYTHONPATH="${path}"
export WANDB_API_KEY="46377815ea349348f24386c0730d943f2da31f72"

./scripts/train_cbae.sh