# data format specification

## directory structure

```
datasets/AADaM_full/
├── structures/           # pdb files for all complexes
│   ├── 1a2y_A_B_l_b.pdb # ligand (antigen) chain
│   ├── 1a2y_A_B_r_b.pdb # receptor (antibody) chain
│   ├── ...
├── splits/              # train/val/test splits
│   ├── aadam_train.txt
│   ├── aadam_val.txt
│   └── aadam_test.txt
└── _all_ids.csv         # master file list
```

## file naming convention

structure files follow pattern: `{pdb_id}_{chain1}_{chain2}_{type}_b.pdb`

examples:
- `1a2y_A_B_r_b.pdb` - receptor (antibody)
- `1a2y_A_B_l_b.pdb` - ligand (antigen)

components:
- `1a2y` - pdb id
- `A_B` - chain identifiers
- `r` or `l` - receptor or ligand
- `b` - bound structure

## _all_ids.csv format

csv file with structure identifiers and splits:

```csv
path,split
1a2y_A_B,train
1b2c_C_D,train
1c3d_E_F,val
1d4e_G_H,test
```

columns:
- `path`: base name (without _r_b.pdb or _l_b.pdb suffix)
- `split`: train, val, or test

## split files format

plain text files with one structure id per line:

```
1a2y_A_B
1b2c_C_D
1c3d_E_F
```

## pdb structure requirements

each pdb file must contain:
- atom coordinates
- residue information
- chain identifiers
- standard amino acid codes

cleaned and preprocessed:
- remove waters
- remove heteroatoms (unless part of protein)
- single model (no nmd ensembles)

## data splits

recommended split ratios:
- train: 80% (8,713 structures)
- val: 10% (1,090 structures)
- test: 10% (1,089 structures)

ensure no overlap between splits at the pdb id level to prevent data leakage.

## cache files (auto-generated)

after first run, cache files created:

```
datasets/AADaM_full/
├── _all_ids_esm_b.pkl      # esm embeddings for all structures
├── cache_residue/          # cached graph structures
│   ├── 1a2y_A_B.pkl
│   └── ...
```

these are large files (several gb total) but save hours of preprocessing time on subsequent runs.

## minimal example

to create a minimal dataset for testing:

1. create directory structure
```bash
mkdir -p datasets/test_data/{structures,splits}
```

2. add 5-10 pdb files to structures/

3. create _all_ids.csv:
```csv
path,split
structure1,train
structure2,train
structure3,val
```

4. create split files:
```bash
echo "structure1" > splits/train.txt
echo "structure2" >> splits/train.txt
echo "structure3" > splits/val.txt
```

5. update config yaml to point to test_data/

this minimal setup lets you test the pipeline before running on full dataset.


