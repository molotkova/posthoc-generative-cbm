#!/bin/bash
# Unset PYTHONPATH to avoid any local Python settings
unset PYTHONPATH
# Unset any conda-related environment variables
unset CONDA_PREFIX
unset CONDA_DEFAULT_ENV
unset CONDA_PYTHON_EXE

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

eval "$(~/miniconda3/bin/conda shell.bash hook)"
conda activate posthocgencbm

cd "${path}"
pwd=$(pwd)

export HOME="$(pwd)"
export PYTHONPATH="${path}"
export WANDB_API_KEY=""

python eval/eval_intervention_gan.py --dataset celebahq --expt-name "cbae_stygan2_thr90" --tensorboard-name "sup_pl_unk40_cls8" --classes "Male" --additional-concepts "Heavy_Makeup" --concept-value 1
