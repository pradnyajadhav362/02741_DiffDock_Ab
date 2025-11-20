# DiffDock-Ab Fine-tuning on AADaM Dataset

Fine-tuning the DiffDock-Ab antibody-antigen docking model on the full AADaM dataset (10k+ structures).

## Overview

This repository contains the training pipeline for fine-tuning DiffDock-Ab on the AADaM antibody-antigen dataset. The workflow includes data preprocessing, caching optimizations, and multi-GPU training on HPC clusters.

## Key Features

- **Full Dataset Training**: 10,892 antibody-antigen structures
- **Efficient Caching**: ESM embeddings and graph structures cached to disk
- **Multi-GPU Support**: Distributed training across 3× A100 GPUs
- **Memory Optimization**: Reduced ESM batch size and explicit cache clearing
- **Checkpoint Resuming**: Continue training from any saved checkpoint

## Dataset

**AADaM (Antibody-Antigen Docking and Modeling)**
- Total structures: 10,892
- Train split: 8,713 (80%)
- Val split: 1,090 (10%)
- Test split: 1,089 (10%)

Structures are in PDB format with separate files for receptor (antibody) and ligand (antigen) chains.

## Setup

### Requirements

```bash
# Python 3.10+
conda create -n diffdock_ab python=3.10
conda activate diffdock_ab

# Clone DiffDock-Ab
git clone https://github.com/ketatam/DiffDock-Ab.git
cd DiffDock-Ab

# Install dependencies
pip install torch torchvision torchaudio --index-url https://pytorch.org/get-started/locally/
pip install -r requirements.txt
pip install wandb
```

### Directory Structure

```
project_root/
├── datasets/
│   └── AADaM_full/
│       ├── structures/          # pdb files
│       ├── splits/              # train/val/test splits
│       └── _all_ids.csv         # master file list
├── config/
│   ├── finetune_full_aadam.yaml
│   └── demo_40_structures.yaml
├── outputs/
│   ├── finetune_full_aadam/
│   └── demo_40/
└── scripts/
    ├── finetune_fast_a100.sbatch
    └── finetune_demo40_preempt.sbatch
```

## configuration

### main config: `config/finetune_full_aadam.yaml`

key parameters:
- **epochs**: 200 (with early stopping patience=50)
- **batch_size**: 8 per gpu (24 total on 3 gpus)
- **learning_rate**: 5e-6 (low for fine-tuning)
- **caching**: enabled for both esm embeddings and graphs
- **multiplicity**: 1 (can increase for data augmentation)

see `config/finetune_full_aadam.yaml` for full configuration.

## training workflow

### 1. data preprocessing

on first run, the pipeline:
- loads structure files
- computes esm embeddings (protein language model)
- builds molecular graphs
- caches everything to disk (~2-3 hours first time)

subsequent runs load from cache (5-10 min).

### 2. multi-gpu training

using pytorch dataparallel across 3x a100 gpus:
- effective batch size: 24 (8 per gpu)
- memory: 400gb system ram, 80gb per gpu
- walltime: 48 hours

### 3. checkpointing

model saves every 10 epochs:
- best validation model
- latest checkpoint for resuming
- training logs and metrics

## memory optimizations

several key optimizations were needed:

1. **reduced esm batch size**: from 32 → 4
2. **explicit cache clearing**: `torch.cuda.empty_cache()` after each batch
3. **increased system ram**: 400gb for full dataset
4. **graph caching**: avoid recomputing molecular graphs

see `src/data/data_train_utils.py` line 364 and `src/train.py` line 72.

## Running the Code

### Option 1: Fine-tuning from Pretrained Checkpoint (Recommended)

Uses DIPS pretrained weights as initialization:

```bash
sbatch scripts/finetune_fast_a100.sbatch
```

This script includes `--checkpoint_path` pointing to the pretrained model.

### Option 2: Training from Scratch

To train without pretrained initialization, remove the `--checkpoint_path` argument:

```bash
python src/main.py \
    --mode train \
    --config_file /path/to/config/finetune_full_aadam.yaml \
    --run_name finetune_scratch \
    --save_path /path/to/outputs/scratch/ckpts \
    --batch_size 24 \
    --num_folds 1 \
    --num_gpu 3
```

**Note**: Training from scratch requires more epochs and may yield higher RMSD initially. The 100-structure demo was trained from scratch as proof-of-concept.

### Quick Demo (40 Structures)

For testing or quick results:

```bash
sbatch scripts/finetune_demo40_preempt.sbatch
```

Runs in ~30 min with 40 structures over 20 epochs.

## Monitoring

Training metrics logged to Weights & Biases:
- Train/val loss per epoch
- Learning rate schedule
- GPU memory usage
- RMSD metrics

## Results

### Full-Scale Training (Planned)

Training on 10,892 structures with:
- Pretrained checkpoint from DIPS dataset
- 200 epochs with early stopping
- Validation every 10 epochs (1,089 structures)
- Best model saved based on validation loss
- Expected runtime: ~14 hours on 3× A100 GPUs

### Demonstration Training (Completed)

Proof-of-concept on 100 structures:
- **Trained from scratch** (no pretrained checkpoint)
- 16 epochs completed (early stopping)
- Batch size 2 on single L40S GPU
- Validation loss: 1.055 → 0.209 (80% reduction)
- Runtime: ~5 minutes

The demo validates that:
- Pipeline handles dataset preprocessing correctly
- Model learns antibody-antigen binding patterns
- Training converges despite small dataset size
- Full-scale training is feasible with optimizations

**Note**: Demo trained from scratch due to checkpoint path compatibility issues on demo hardware. Full training uses pretrained DIPS checkpoint for better initialization.

## troubleshooting

### cuda oom errors

reduce batch size or esm batch size in config.

### system ram oom

increase `--mem` in slurm script or disable caching temporarily.

### corrupted cache files

delete `.pkl` cache files in dataset directory and rerun.

## citation

if using this pipeline, please cite:

```bibtex
@article{martinkus2023diffdock,
  title={DiffDock-AB: Learning Antibody-Antigen Complex Structure Prediction with Diffusion Models},
  author={Martinkus, Karolis and others},
  journal={arXiv preprint},
  year={2023}
}
```

## license

this code follows the original diffdock-ab license. see the main repository for details.

## acknowledgments

- diffdock-ab authors for the base model
- aadam dataset creators
- hpc cluster resources


