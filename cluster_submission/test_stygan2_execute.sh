#!/bin/bash
# Unset PYTHONPATH to avoid any local Python settings
unset PYTHONPATH
# Unset any conda-related environment variables
unset CONDA_PREFIX
unset CONDA_DEFAULT_ENV
unset CONDA_PYTHON_EXE

path=/home/molotkova_s/Desktop/mlteam/posthoc-generative-cbm
python_exec=/home/molotkova_s/miniforge3/envs/posthocgencbm/bin/python
cd "${path}"

pwd=$(pwd)
export HOME="$(pwd)"
export PYTHONPATH="${path}"
export WANDB_API_KEY=""

${python_exec} eval/test_stygan2_cpu.py
