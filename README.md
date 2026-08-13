# AFS

AFS is a two-stage training framework built on the repository's Self-Forcing
Wan2.1 few-step autoregressive video generation implementation.

## Stage 1: Semantic Preprocessing

Configure local paths in `configs/afs_stage1_semantics.yaml`, then run:

```bash
python scripts/afs_stage1_prepare_semantics.py \
  --config configs/afs_stage1_semantics.yaml
```

Stage 1 aligns GT video chunks to Self-Forcing temporal boundaries, obtains one
natural-language description per chunk through a local Qwen3-VL adapter, and
encodes global and chunk captions with the frozen Self-Forcing UMT5 encoder.

## Stage 2: Training

Configure local paths and inherited Self-Forcing hyperparameters in
`configs/afs_stage2_training.yaml`, then run:

```bash
torchrun --nproc_per_node=${NUM_GPUS} \
  scripts/afs_stage2_train.py \
  --config configs/afs_stage2_training.yaml
```

The Student performs the original few-step on-policy streaming rollout under
the global text condition. The EMA Teacher evaluates the same Student noisy
states with target-chunk text and GT-assisted history. Only Student parameters
or Student LoRA are optimized.

See [AFS_TRAINING.md](AFS_TRAINING.md) for data contracts, cache semantics,
local resource requirements, and remaining integration work.
