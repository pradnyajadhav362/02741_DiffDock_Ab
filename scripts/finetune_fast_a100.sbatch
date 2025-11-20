#!/bin/bash
#SBATCH -N 1
#SBATCH -t 48:00:00
#SBATCH -J aadam_fast
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=your.email@university.edu
#SBATCH --mem=400G
#SBATCH --cluster=GPU
#SBATCH --gres=gpu:3
#SBATCH --partition=a100
#SBATCH --output=/path/to/outputs/finetune_full_aadam/training_%j.out
#SBATCH --error=/path/to/outputs/finetune_full_aadam/training_%j.err

source /path/to/setup_env.sh

echo "=========================================="
echo "FAST TRAINING: 3x A100 GPUs"
echo "Job started: $(date)"
echo "Node: $(hostname)"
echo "=========================================="
nvidia-smi
echo "=========================================="

mkdir -p /path/to/outputs/finetune_full_aadam/ckpts

python src/main.py \
    --mode train \
    --config_file /path/to/config/finetune_full_aadam.yaml \
    --checkpoint_path /path/to/pretrained/model_best_338669_140_31.084_30.347.pth \
    --run_name finetune_full_aadam \
    --save_path /path/to/outputs/finetune_full_aadam/ckpts \
    --batch_size 24 \
    --num_folds 1 \
    --num_gpu 3 \
    --gpu 0 \
    --seed 0 \
    --logger wandb \
    --project "DiffDock-Ab AADaM"

echo "=========================================="
echo "Job finished: $(date)"
echo "=========================================="


