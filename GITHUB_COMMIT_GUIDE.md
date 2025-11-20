# github commit guide

## ready to commit

all files are sanitized and ready for public github repo. no personal info included.

## what to commit

```
github/
├── README.md                    # main documentation
├── QUICKSTART.md               # setup and run instructions
├── MEMORY_OPTIMIZATIONS.md     # key code modifications
├── DATA_FORMAT.md              # data structure specification
├── requirements.txt            # python dependencies
├── .gitignore                  # ignore patterns
├── setup_env.sh                # environment setup template
├── config/
│   ├── finetune_full_aadam.yaml
│   └── demo_40_structures.yaml
└── scripts/
    ├── finetune_fast_a100.sbatch
    ├── finetune_demo40_preempt.sbatch
    └── sample_40_structures.py
```

## steps to commit

```bash
cd /path/to/your/repo

# copy github folder contents
cp -r /path/to/github/* .

# initialize git (if new repo)
git init
git add .
git commit -m "initial commit: diffdock-ab aadam fine-tuning pipeline"

# or add to existing repo
git add .
git commit -m "add diffdock-ab training pipeline"

# push to github
git remote add origin https://github.com/yourusername/repo-name.git
git branch -M main
git push -u origin main
```

## what was sanitized

replaced with generic placeholders:
- `/ix/djishnu/Pradnya/...` → `/path/to/...`
- `jadhavpr@pitt.edu` → `your.email@university.edu`
- specific cluster names kept generic
- no wandb api keys or credentials

## recommended repo name

suggestions:
- `diffdock-ab-aadam-finetuning`
- `antibody-docking-training`
- `diffdock-ab-pipeline`

## repo description

"fine-tuning pipeline for diffdock-ab on the aadam antibody-antigen dataset with memory optimizations for large-scale training on hpc clusters"

## topics/tags

- antibody-docking
- protein-structure-prediction
- diffusion-models
- computational-biology
- machine-learning
- pytorch
- hpc-computing

## license

consider adding mit or apache 2.0 license if sharing publicly.

## what NOT to commit

(already in .gitignore):
- actual pdb structure files
- cache files (.pkl)
- model checkpoints (.pth)
- wandb logs
- slurm output files
- your actual config files with real paths

keep those in your private working directory only.
