# memory optimizations for large-scale training

this document describes key code modifications needed to train on 10k+ structures without running out of memory.

## 1. reduce esm batch size

**file**: `src/data/data_train_utils.py`  
**line**: 364

**original**:
```python
batch_size = find_largest_diviser_smaller_than(len(seqs), 32)
```

**modified**:
```python
batch_size = find_largest_diviser_smaller_than(len(seqs), 4)
```

**reason**: esm (protein language model) was trying to process 32 sequences at once, causing cuda oom. reducing to 4 fits in 80gb a100 memory.

## 2. explicit gpu cache clearing

**file**: `src/train.py`  
**line**: 72 (after optimizer.zero_grad())

**add this line**:
```python
torch.cuda.empty_cache()
```

**reason**: pytorch doesn't always release gpu memory immediately after backward pass. explicit clearing prevents accumulation over batches.

**full context**:
```python
for batch in train_loader:
    optimizer.zero_grad()
    torch.cuda.empty_cache()
    
    loss = model(batch)
    loss.backward()
    optimizer.step()
```

## 3. enable full caching

**file**: `config/finetune_full_aadam.yaml`

**setting**:
```yaml
data:
    no_graph_cache: False
```

**reason**: 
- first run: processes all structures and saves to disk (~2-3 hours)
- subsequent runs: loads from cache (~5-10 min)
- esm embeddings saved as `.pkl` files
- graph structures cached separately

**cache files**:
- `datasets/AADaM_full/_all_ids_esm_b.pkl` - esm embeddings
- `datasets/AADaM_full/cache_*` - graph structures

## 4. system ram allocation

**file**: `scripts/finetune_fast_a100.sbatch`

**slurm parameter**:
```bash
#SBATCH --mem=400G
```

**reason**: loading 10k structures + esm embeddings + graphs requires substantial system ram. started with 100g, incrementally increased to 400g.

## 5. batch size tuning

**single gpu**: batch_size=8  
**3x gpu**: batch_size=24 (8 per gpu)

effective batch size = batch_size × num_gpu

larger batches = more stable gradients but require more memory.

## troubleshooting guide

### cuda out of memory during esm processing
- reduce esm batch size further (try 2 or 1)
- use fewer structures for testing

### cuda out of memory during training
- reduce batch_size in config
- use fewer gpus
- add more `torch.cuda.empty_cache()` calls

### system ram out of memory
- increase `--mem` in slurm script
- temporarily disable caching (set `no_graph_cache: True`)
- process data in smaller chunks

### corrupted cache files
```bash
rm datasets/AADaM_full/_all_ids_esm_b.pkl
rm -rf datasets/AADaM_full/cache_*
```
then rerun to regenerate cache.

## performance impact

with these optimizations:
- training throughput: ~150 structures/hour on 3x a100
- first epoch: 2-3 hours (with data loading)
- subsequent epochs: 30-40 min each
- total training time (200 epochs): ~5-6 days

without caching: would need 2-3 hours per epoch = 400-600 hours total.


