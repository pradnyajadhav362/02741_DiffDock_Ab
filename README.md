# 02741_DiffDock_Ab

# Adapting DiffDock-PP for Antibody–Antigen Docking with Structural Confidence Calibration

This repository contains code, configurations, and scripts for adapting the generative diffusion docking model **DiffDock-PP** to antibody–antigen (Ab–Ag) docking. The project evaluates both structural accuracy and model confidence calibration.

---

## Overview

Antibody–antigen docking is a challenging problem due to the structural flexibility of complementarity-determining regions (CDRs) and diverse antigen interfaces. This project fine-tunes **DiffDock-PP** on curated Ab–Ag complexes and examines whether diffusion-based generative models can be extended to reliably capture antibody-specific binding.

We additionally evaluate calibration—how well the model’s confidence scores correlate with structural correctness—by measuring correlations between model confidence, DockQ, and interface RMSD (iRMSD).

---

## Features

- Fine-tuning pipeline for **DiffDock-PP** on antibody–antigen datasets  
- Lightweight configuration system for training and evaluation  
- Confidence model training for pose-level calibration  
- Evaluation metrics including DockQ, iRMSD, and calibration plots  
- Compatible with high-performance computing (HPC) environments (e.g., PSC Bridges2)

---

## Installation

### Step 1: Clone the repository

```bash
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>
Step 2: Create and activate the environment
module load anaconda3/2024.10-1
conda create -p /ocean/projects/<project-id>/<username>/.conda/envs/diffdockab python=3.10 -y
conda activate /ocean/projects/<project-id>/<username>/.conda/envs/diffdockab

Step 3: Install dependencies
# Core libraries
conda install pytorch=1.13.0 pytorch-cuda=11.6 -c pytorch -c nvidia -y

# PyTorch Geometric dependencies
pip install torch-scatter==2.0.9 torch-sparse==0.6.15 torch-cluster==1.6.0 torch-spline-conv==1.2.2 \
  -f https://data.pyg.org/whl/torch-1.13.0+cu116.html
pip install torch-geometric==2.3.1

# Supporting libraries
pip install numpy scipy pandas pyyaml tqdm dill biopython biopandas scikit-learn e3nn wandb tensorboard matplotlib

Step 4: Verify installation
python -c "import torch; import torch_geometric; print('PyTorch:', torch.__version__, 'CUDA:', torch.cuda.is_available())"

Directory Structure
DiffDock-Ab/
│
├── config/                 # YAML configuration files
│   ├── abag_score_pilot.yaml
│   ├── abag_conf_pilot.yaml
│
├── src/                    # Training and inference scripts
│   ├── train.sh
│   ├── generate_samples.sh
│   ├── train_confidence.sh
│
├── datasets/
│   └── AADaM_notstrict/
│       ├── structures/     # PDB files for Ab–Ag complexes
│       ├── splits/         # Train/val/test ID lists
│
├── outputs/
│   ├── poses_abag/         # Generated docking poses
│   ├── metrics/            # Evaluation results
│   └── figs/               # Calibration plots
│
├── tools/                  # Evaluation scripts
│   ├── compute_metrics.py
│   ├── plot_calibration.py
│
├── requirements.txt
└── README.md

Usage
1. Prepare Dataset

Place antibody–antigen complex files (.pdb) under:

datasets/AADaM_notstrict/structures/


Then create splits:

ls datasets/AADaM_notstrict/structures/*.pdb | sed 's#.*/##' | cut -d'_' -f1 | sort -u | head -n 10 > datasets/AADaM_notstrict/splits/all_ids.txt
head -n 6 datasets/AADaM_notstrict/splits/all_ids.txt > datasets/AADaM_notstrict/splits/aadam_train.txt
tail -n +7 datasets/AADaM_notstrict/splits/all_ids.txt | head -n 2 > datasets/AADaM_notstrict/splits/aadam_val.txt
tail -n 2 datasets/AADaM_notstrict/splits/all_ids.txt > datasets/AADaM_notstrict/splits/aadam_test.txt

2. Train the Model
bash src/train.sh config/abag_score_pilot.yaml


Checkpoints and logs will be saved under checkpoints/.

3. Generate Docking Poses
bash src/generate_samples.sh \
  --config config/abag_score_pilot.yaml \
  --ckpt checkpoints/abag_score/best.pt \
  --ids_file datasets/AADaM_notstrict/splits/aadam_test.txt \
  --num_samples 20 \
  --out_dir outputs/poses_abag/test

4. Train Confidence Model
bash src/train_confidence.sh \
  --config config/abag_conf_pilot.yaml \
  --poses_train outputs/poses_abag/train \
  --poses_val outputs/poses_abag/val

5. Evaluate and Visualize
python tools/compute_metrics.py \
  --native_dir datasets/AADaM_notstrict/structures \
  --pred_dir outputs/poses_abag/test \
  --out_csv outputs/test_metrics.csv

python tools/plot_calibration.py \
  --in outputs/test_metrics.csv \
  --out figs_pilot/


Results and plots (e.g., calibration curves) will be saved in outputs/figs_pilot/.
