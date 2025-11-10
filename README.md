# 02741_DiffDock_Ab

## Adapting DiffDock-PP for Antibody–Antigen Docking with Structural Confidence Calibration

This repository contains code, configurations, and scripts for adapting the generative diffusion docking model **DiffDock-PP** to antibody–antigen (Ab–Ag) docking.  
The project evaluates both structural accuracy and model confidence calibration.

---

## Overview

Antibody–antigen docking is a challenging problem due to the structural flexibility of complementarity-determining regions (CDRs) and diverse antigen interfaces.  
This project fine-tunes **DiffDock-PP** on curated Ab–Ag complexes and examines whether diffusion-based generative models can be extended to reliably capture antibody-specific binding.

We additionally evaluate calibration — how well the model’s confidence scores correlate with structural correctness — by measuring correlations between model confidence, DockQ, and interface RMSD (iRMSD).

---

## Features

- Fine-tuning pipeline for **DiffDock-PP** on antibody–antigen datasets  
- Lightweight configuration system for training and evaluation  
- Confidence model training for pose-level calibration  
- Evaluation metrics including DockQ, iRMSD, and calibration plots  
- Compatible with high-performance computing (HPC) environments (e.g., PSC Bridges2)

---

## Installation

```bash
# Step 1: Clone the repository
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>

# Step 2: Create and activate the environment
module load anaconda3/2024.10-1
conda create -p /ocean/projects/<project-id>/<username>/.conda/envs/diffdockab python=3.10 -y
conda activate /ocean/projects/<project-id>/<username>/.conda/envs/diffdockab

# Step 3: Install dependencies
# Core libraries
conda install pytorch=1.13.0 pytorch-cuda=11.6 -c pytorch -c nvidia -y

# PyTorch Geometric dependencies
pip install torch-scatter==2.0.9 torch-sparse==0.6.15 torch-cluster==1.6.0 torch-spline-conv==1.2.2 \
  -f https://data.pyg.org/whl/torch-1.13.0+cu116.html
pip install torch-geometric==2.3.1

# Supporting libraries
pip install numpy scipy pandas pyyaml tqdm dill biopython biopandas scikit-learn e3nn wandb tensorboard matplotlib

# Step 4: Verify installation
python -c "import torch; import torch_geometric; print('PyTorch:', torch.__version__, 'CUDA:', torch.cuda.is_available())"

