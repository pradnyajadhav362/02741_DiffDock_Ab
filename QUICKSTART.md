# quick start guide

## 1. clone and setup

```bash
# clone diffdock-ab
git clone https://github.com/ketatam/DiffDock-Ab.git
cd DiffDock-Ab

# create environment
conda create -n diffdock_ab python=3.10
conda activate diffdock_ab

# install dependencies
pip install torch torchvision torchaudio --index-url https://pytorch.org/get-started/locally/
pip install biopython numpy pandas pyyaml tqdm wandb
pip install e3nn==0.5.1
pip install transformers
```

## 2. download pretrained model (optional but recommended)

**for fine-tuning from pretrained weights:**

```bash
# download dips pretrained checkpoint from diffdock-ab repository
wget https://zenodo.org/record/XXXXX/files/large_model_dips.tar.gz
tar -xzf large_model_dips.tar.gz
mv large_model_dips checkpoints/
```

**for training from scratch:** skip this step and remove `--checkpoint_path` from training commands.

**note**: pretrained weights provide better initialization and faster convergence. training from scratch requires more epochs and data.

## 3. prepare aadam dataset

```bash
# organize structure files
mkdir -p datasets/AADaM_full/structures
mkdir -p datasets/AADaM_full/splits

# copy pdb files to structures/
# create split files (train/val/test txt files with structure ids)
```

## 4. update config paths

edit `config/finetune_full_aadam.yaml`:
- set `data_file` to your csv file path
- set `root` and `structures_dir` to your structures folder
- set `train_ids`, `val_ids`, `test_ids` to your split files
- set `out_dir` to desired output location

## 5. update slurm script

edit `scripts/finetune_fast_a100.sbatch`:
- set email address
- update all `/path/to/...` with actual paths
- adjust memory/gpu settings for your cluster
- update partition name if needed

## 6. run training

```bash
# full training
sbatch scripts/finetune_fast_a100.sbatch

# or demo training (40 structures)
python scripts/sample_40_structures.py
sbatch scripts/finetune_demo40_preempt.sbatch
```

## 7. monitor training

```bash
# check job status
squeue -u $USER

# watch logs
tail -f outputs/finetune_full_aadam/training_*.out

# or use wandb dashboard
wandb login
# then visit wandb.ai to see live plots
```

## expected timeline

**first run** (with data processing):
- esm embedding generation: 1-2 hours
- graph construction: 30-60 min
- training epoch 1: 30-40 min
- subsequent epochs: 30-40 min each

**subsequent runs** (from cache):
- data loading: 5-10 min
- training epoch 1: 30-40 min
- subsequent epochs: 30-40 min each

**full 200 epochs**: 5-6 days on 3x a100 gpus

## troubleshooting

see `MEMORY_OPTIMIZATIONS.md` for common issues and fixes.

## next steps

after training completes:
1. evaluate on test set
2. run inference on new antibody-antigen pairs
3. analyze rmsd metrics
4. visualize predicted structures

see main diffdock-ab documentation for evaluation and inference scripts.


