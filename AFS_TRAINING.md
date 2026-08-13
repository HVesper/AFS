# AFS Training Architecture

AFS extends the repository's Self-Forcing Wan2.1 streaming implementation. It
does not change the inference path or duplicate the DiT, scheduler, or KV cache.

## Upstream Components

- `model/afs_model.py`: Wan generator, frozen UMT5 and VAE loading, scheduler setup.
- `pipeline/afs_streaming_training.py`: native few-step chunk execution.
- `model/afs_streaming_training.py`: autoregressive rollout and visual/text KV caches.
- `utils/wan_wrapper.py`: Wan forward and prompt embedding contract.
- `wan/modules/causal_model.py`: attention cache implementation.
- `trainer/afs_trainer.py`: FSDP, optimizer, LoRA, EMA, loss/backward loop.
- `train.py`: original distributed training entrypoint.

## Stage 1

Run:

```bash
python scripts/afs_stage1_prepare_semantics.py --config configs/afs_stage1_semantics.yaml
```

The generic JSONL input contains `sample_id`, `video_path`, and
`global_caption`. `AFSChunkBoundaryResolver` reads `num_frame_per_block` from
the selected Self-Forcing config, combines it with the configured VAE temporal
ratio, FPS, and target frame count, and creates the single boundary list used
for captioning and output indexing.

The intended local-only flow is GT video chunk -> Qwen3-VL English caption ->
frozen Self-Forcing UMT5 embedding. Caption and encode phases can run
separately so both large models need not share GPU memory. The Qwen processor
adapter remains an explicit cluster-version TODO and raises instead of
fabricating captions.

Each output manifest record stores captions, boundaries, status/error, and a
semantic cache path. Each `.safetensors` cache stores:

- `global_text_embedding`: `[L,D]`
- `chunk_text_embeddings`: `[N,L,D]`
- `global_text_mask`: `[L]`
- `chunk_text_masks`: `[N,L]`

Manifest updates are atomic per sample. Completed samples can be skipped and
input records are deterministically sharded with `index % num_shards`.

## Stage 2

Run locally or under torchrun:

```bash
python scripts/afs_stage2_train.py --config configs/afs_stage2_training.yaml
torchrun --nproc_per_node=$NUM_GPUS scripts/afs_stage2_train.py --config configs/afs_stage2_training.yaml
```

The launcher validates all local model, checkpoint, manifest, and latent-cache
paths before constructing `AFSTrainer`. No loader has a remote or random
fallback. `AFSStage2Dataset` joins complete Stage 1 records to precomputed
`gt_chunk_latents` and checks that captions, boundaries, embeddings, and
latents use the same chunk count.

`AFSTrainingBatch` uses `[B,L,D]` global embeddings, `[B,N,L,D]` chunk
embeddings, `[B,N,C,T,H,W]` latents and optional masks. Student always uses
global text and generated history. Teacher uses EMA weights and either global
or target-chunk text, but evaluates the exact Student noisy latent/timestep.
Teacher text K/V is reset per chunk. Its visual history keeps the latest
Student chunk and replaces only older history with GT; current target GT is
never cached. Dense velocity MSE is the only AFS objective, optimizer updates
Student/Student LoRA, and EMA updates immediately after optimizer step.

## Remaining TODOs

- Adapt Qwen3-VL video processor invocation to the cluster's installed local
  `transformers` version; no caption operation is run in this repository state.
- Provide local model/checkpoint/data paths and concrete inherited training
  hyperparameters in both AFS configs.
- Implement the optional online GT-video VAE dataset adapter. The current
  Stage 2 path supports precomputed GT latent caches.
