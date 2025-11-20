# DiffDock-Ab Fine-tuning on AADaM Dataset

fine-tuning the diffdock-ab antibody-antigen docking model on the full aadam dataset (10k+ structures)

## overview

this repo contains the training pipeline for fine-tuning diffdock-ab on the aadam antibody-antigen dataset. the workflow includes data preprocessing, caching optimizations, and multi-gpu training on hpc clusters.

## key features

- **full dataset training**: 10,892 antibody-antigen structures
- **efficient caching**: esm embeddings and graph structures cached to disk
- **multi-gpu support**: distributed training across 3x a100 gpus
- **memory optimization**: reduced esm batch size and explicit cache clearing
- **checkpoint resuming**: continue training from any saved checkpoint

## dataset

**aadam (antibody-antigen docking and modeling)**
- total structures: 10,892
- train split: 8,713 (80%)
- val split: 1,090 (10%)
- test split: 1,089 (10%)

structures are in pdb format with separate files for receptor (antibody) and ligand (antigen) chains.

## setup

### requirements

```bash
# python 3.10+
conda create -n diffdock_ab python=3.10
conda activate diffdock_ab

# clone diffdock-ab
git clone https://github.com/ketatam/DiffDock-Ab.git
cd DiffDock-Ab

# install dependencies
pip install torch torchvision torchaudio --index-url https://pytorch.org/get-started/locally/
pip install -r requirements.txt
pip install wandb
```

### directory structure

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

## running the code

### option 1: fine-tuning from pretrained checkpoint (recommended)

uses dips pretrained weights as initialization:

```bash
sbatch scripts/finetune_fast_a100.sbatch
```

this script includes `--checkpoint_path` pointing to the pretrained model.

### option 2: training from scratch

to train without pretrained initialization, remove the `--checkpoint_path` argument:

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

**note**: training from scratch requires more epochs and may yield higher rmsd initially. the 100-structure demo was trained from scratch as proof-of-concept.

### quick demo (40 structures)

for testing or quick results:

```bash
sbatch scripts/finetune_demo40_preempt.sbatch
```

runs in ~30 min with 40 structures over 20 epochs.

## monitoring

training metrics logged to weights & biases:
- train/val loss per epoch
- learning rate schedule
- gpu memory usage
- rmsd metrics

## results

### full-scale training (planned)

training on 10,892 structures with:
- pretrained checkpoint from dips dataset
- 200 epochs with early stopping
- validation every 10 epochs (1,089 structures)
- best model saved based on validation loss
- expected runtime: ~14 hours on 3x a100 gpus

### demonstration training (completed)

proof-of-concept on 100 structures:
- **trained from scratch** (no pretrained checkpoint)
- 16 epochs completed (early stopping)
- batch size 2 on single l40s gpu
- validation loss: 1.055 → 0.209 (80% reduction)
- runtime: ~5 minutes

the demo validates that:
- pipeline handles dataset preprocessing correctly
- model learns antibody-antigen binding patterns
- training converges despite small dataset size
- full-scale training is feasible with optimizations

**note**: demo trained from scratch due to checkpoint path compatibility issues on demo hardware. full training uses pretrained dips checkpoint for better initialization.

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


