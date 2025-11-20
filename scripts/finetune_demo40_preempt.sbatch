#!/bin/bash
#SBATCH -N 1
#SBATCH -t 2:00:00
#SBATCH -J demo40_preempt
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=your.email@university.edu
#SBATCH --mem=64G
#SBATCH --cluster=GPU
#SBATCH --gres=gpu:1
#SBATCH --partition=preempt
#SBATCH --output=/path/to/outputs/demo_40/training_preempt_%j.out
#SBATCH --error=/path/to/outputs/demo_40/training_preempt_%j.err

source /path/to/setup_env.sh

echo "=========================================="
echo "demo training on preempt - 40 structures"
echo "job started: $(date)"
echo "node: $(hostname)"
echo "=========================================="
nvidia-smi
echo "=========================================="

mkdir -p /path/to/outputs/demo_40/ckpts_preempt

python src/main.py \
    --mode train \
    --config_file /path/to/config/demo_40_structures.yaml \
    --checkpoint_path /path/to/pretrained/model_best_338669_140_31.084_30.347.pth \
    --run_name demo_40_preempt \
    --save_path /path/to/outputs/demo_40/ckpts_preempt \
    --num_folds 1 \
    --num_gpu 1 \
    --gpu 0 \
    --seed 0 \
    --logger wandb \
    --project "DiffDock-Ab AADaM Demo" \
    --batch_size 8

echo "=========================================="
echo "job finished: $(date)"
echo "=========================================="


