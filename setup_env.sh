#!/bin/bash

# load required modules for hpc cluster
module load python/3.10
module load cuda/11.8
module load gcc/10.2.0

# activate conda environment
source ~/.bashrc
conda activate diffdock_ab

# set environment variables
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=0,1,2

# cd to diffdock-ab directory
cd /path/to/DiffDock-Ab

echo "environment setup complete"
echo "python: $(which python)"
echo "cuda available: $(python -c 'import torch; print(torch.cuda.is_available())')"


