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

## Unified Launch

One configuration and command run semantic preprocessing followed by training:

```bash
torchrun --nproc_per_node=$NUM_GPUS scripts/afs_train.py --config configs/afs_training.yaml
```

For a full cluster bootstrap, edit the small configuration block at the top of
`run_afs.sh` and run `bash run_afs.sh`. It downloads the selected public model
and dataset assets, prepares GT latent caches, generates the internal unified
configuration, and invokes the command above. Hugging Face and preprocessing
caches support resume after an interrupted run. After successful comparison
inference, `CLEANUP_AFTER_SUCCESS=true` removes model weights, source/training
data, and caches while retaining the AFS checkpoint with optimizer/EMA state,
comparison artifacts, WorldScore prompt metadata, and the timestamped terminal
log.

The bootstrap dataset is Sekai-Real-Walking-HQ. Its annotations come from the
`Lixsp11/Sekai` Hugging Face dataset and its exact annotated frame intervals are
downloaded from the original YouTube sources with `yt-dlp`. The source clips
are sampled at the configured target FPS without compressing the full annotated
duration into the shorter AFS training window. Sekai permits non-commercial
research use only, and the launcher prints that restriction before downloading.

Comparison inference uses two official WorldScore dynamic T2V prompts downloaded
from the Hugging Face dataset viewer API. At step 300, training saves the
LoRA/optimizer/EMA state and exits without duplicate periodic generation.
Standalone inference then runs both prompts through the original Self-Forcing
model and final AFS Student LoRA with the same seed policy, sampler, resolution,
frame count, and cache policy. It also runs the two held-out Sekai captions
through both models. Each model produces four minute-long videos.

The generic JSONL input contains `sample_id`, `video_path`, and
`global_caption`. `AFSChunkBoundaryResolver` reads `num_frame_per_block` from
the selected Self-Forcing config, combines it with the configured VAE temporal
ratio, FPS, and target frame count, and creates the single boundary list used
for captioning and output indexing.

The local-only flow is GT video chunk -> Qwen3-VL English caption -> frozen
Self-Forcing UMT5 embedding. The unified launcher releases Qwen3-VL before it
loads UMT5, so both large models do not share GPU memory.

Each output manifest record stores captions, boundaries, status/error, and a
semantic cache path. Each `.safetensors` cache stores:

- `global_text_embedding`: `[L,D]`
- `chunk_text_embeddings`: `[N,L,D]`
- `global_text_mask`: `[L]`
- `chunk_text_masks`: `[N,L]`

Manifest updates are atomic per sample. Completed samples can be skipped and
input records are deterministically sharded with `index % num_shards`.

Each distributed rank runs Qwen3-VL-32B and UMT5 on a disjoint sample shard and
writes a rank-specific manifest. Rank 0 waits for every shard marker, merges
the manifests in source order, validates every cache, and atomically publishes
the global readiness marker. A shard failure prevents training from starting
with incomplete caches.

The launcher validates all local model, checkpoint, manifest, and latent-cache
paths before constructing `AFSTrainer`. No loader has a remote or random
fallback. `AFSTrainingDataset` joins complete semantic preprocessing records to precomputed
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

- Provide local model/checkpoint/data paths and concrete inherited training
  hyperparameters in `configs/afs_training.yaml`.
- Implement the optional online GT-video VAE dataset adapter. The current
  AFS training path supports precomputed GT latent caches.
