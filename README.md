# AFS

AFS is a unified training pipeline built on the repository's Self-Forcing
Wan2.1 few-step autoregressive video generation implementation.

## Run

Edit the local paths and runtime settings at the top of `run_afs.sh`, then run
the entire pipeline with one command:

```bash
bash run_afs.sh
```

The script can bootstrap a fresh cluster workspace: it installs Python
dependencies, resumes local Hugging Face downloads for Wan2.1-T2V-1.3B,
Self-Forcing DMD, Qwen3-VL-32B-Instruct and selected Sekai-Real-Walking-HQ
clips, builds the manifest, normalizes frames, creates Wan VAE chunk latents,
runs semantic preprocessing, and finally starts training. Sekai annotations are
downloaded from Hugging Face, while exact annotated video intervals are fetched
from YouTube with `yt-dlp`; use of the data remains subject to Sekai's
non-commercial research license. Downloads and generated assets are stored
under `WORK_ROOT` while the pipeline is running.
The default selects 66 clips; setting `MAX_DATASET_SAMPLES=0` requests the full
18,208-sample HQ split and can consume roughly 600 GB before derived caches.
Every invocation writes its complete terminal output to one timestamped file,
such as `logs/afs_20260814_153000.log`, while continuing to display it
interactively.

All eight ranks process disjoint video shards with one local Qwen3-VL-32B worker
per GPU, release Qwen3-VL, and encode global and chunk captions with frozen
UMT5. Rank 0 merges the eight semantic manifests in source order. Training
starts only after every sample has a complete semantic cache; any shard failure
stops the entire run before optimizer initialization.

The Student performs the original few-step on-policy streaming rollout under
the global text condition. The EMA Teacher evaluates the same Student noisy
states with target-chunk text and GT-assisted history. Only Student parameters
or Student LoRA are optimized.

Sekai-Real-Walking-HQ samples are deterministically separated into train and
eval splits. The launcher fetches two official dynamic T2V records from the
WorldScore dataset API and caches their exact camera-conditioned prompts. The
default 300-step smoke run saves `checkpoint_model_000300/model.pt` without
running a duplicate in-training video evaluation.

After training, both WorldScore prompts and both held-out Sekai captions are
run through the original Self-Forcing checkpoint and the final AFS Student
LoRA with the same random-seed policy. Each model therefore generates four MP4
files, for eight videos total. The videos and their resolved inference
configurations are stored under
`WORK_ROOT/worldscore_t2v/comparisons/checkpoint_model_XXXXXX/`. The default
trial run stops at 300 optimizer steps and uses its step-300 checkpoint for the
comparison before automatic cleanup.
Final comparisons use 243 latent frames, producing approximately one minute
per video at 832 x 480 and 16 FPS. These values remain editable in
the first block of `run_afs.sh`.

After all eight comparison videos finish successfully, the default
`CLEANUP_AFTER_SUCCESS=true` removes downloaded model weights, Sekai data,
Hugging Face caches, and semantic/latent caches. It keeps the AFS training
checkpoint (including optimizer and EMA state), WorldScore prompts, metadata,
comparison videos/configurations, and the timestamped terminal log. Failed or
interrupted runs are left intact for diagnosis and resume.

See [AFS_TRAINING.md](AFS_TRAINING.md) for data contracts, cache semantics,
local resource requirements, and remaining integration work.
