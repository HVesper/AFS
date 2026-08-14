#!/usr/bin/env bash
set -euo pipefail

# =============================================================================
# Edit only this section.
# =============================================================================
NUM_GPUS=8
WORK_ROOT="${AFS_WORK_ROOT:-$HOME/afs_runtime}"

# Sekai is restricted to non-commercial research. Running this downloader must
# comply with https://github.com/Lixsp11/sekai-codebase/blob/main/LICENSE.
# The HQ split contains 18,208 approximately one-minute samples (~600 GB).
MAX_DATASET_SAMPLES=66         # 64 train + 2 eval for the 8-GPU smoke run.
EVAL_DATASET_SAMPLES=2
SEKAI_SOURCE_FPS=60
SEKAI_MAX_DOWNLOAD_FAILURES=32

AUTO_INSTALL_DEPENDENCIES=true
INSTALL_FLASH_ATTN=true
MIN_FREE_DISK_GB=300

TARGET_FPS=16
TARGET_FRAME_COUNT=957         # 240 Wan latent frames, approximately 60 seconds.
TARGET_HEIGHT=480
TARGET_WIDTH=832
VAE_TEMPORAL_COMPRESSION_RATIO=4
LATENT_CHUNK_SIZE=3
QWEN_MAX_NEW_TOKENS=96

EMA_DECAY=0.9999
SHARED_PREFIX_CHUNKS=1
MAX_TRAIN_STEPS=300
SAVE_INTERVAL_STEPS=300
MAX_CHECKPOINTS=0             # Keep checkpoints until final comparison cleanup.
SEMANTIC_RESUME=true
SEMANTIC_OVERWRITE=false
TRAINING_RESUME=false          # Run this 300-step smoke test from step 0.
SEMANTIC_WAIT_TIMEOUT_SEC=86400
SEMANTIC_POLL_INTERVAL_SEC=10
# Periodic in-training generation is disabled. After the final checkpoint, the
# launcher evaluates two prompts with both original Self-Forcing and AFS.
EVAL_INTERVAL_STEPS=0
EVAL_NUM_SAMPLES=2
EVAL_NUM_LATENT_FRAMES=243
WORLDSCORE_T2V_ROW_INDICES=(0 1)
WORLDSCORE_EVAL_SEED=12345
CLEANUP_AFTER_SUCCESS=false    # Keep all assets and outputs for this first run.

# Optional Hugging Face token for authenticated/rate-limited environments.
# Prefer exporting HF_TOKEN before launching instead of writing it here.
HF_TOKEN="${HF_TOKEN:-}"
EXTRA_ARGS=()

# =============================================================================
# Everything below is automatic.
# =============================================================================
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="${REPO_ROOT}/logs"
RUN_TIMESTAMP="$(date '+%Y%m%d_%H%M%S')"
RUN_HOST="$(hostname -s 2>/dev/null || hostname)"
RUN_LOG="${LOG_DIR}/afs_${RUN_TIMESTAMP}.log"
mkdir -p "${LOG_DIR}"

# Record this execution's complete stdout/stderr stream in one timestamped file.
exec > >(tee "${RUN_LOG}") 2>&1
printf '\n================================================================================\n'
printf '[AFS] Run started at %s | host=%s | pid=%s\n' "$(date '+%Y-%m-%d %H:%M:%S %z')" "${RUN_HOST}" "$$"
printf '================================================================================\n'

ASSET_ROOT="${WORK_ROOT}/assets"
DATA_ROOT="${WORK_ROOT}/sekai_real_walking_hq"
DOWNLOAD_ROOT="${DATA_ROOT}/download"
RAW_VIDEO_ROOT="${DATA_ROOT}/raw_videos"
PROCESSED_ROOT="${DATA_ROOT}/processed"
GT_LATENT_CACHE_ROOT="${DATA_ROOT}/gt_latents"
SOURCE_MANIFEST="${DATA_ROOT}/sekai_source.jsonl"
INPUT_MANIFEST="${DATA_ROOT}/afs_input.jsonl"
SEMANTIC_MANIFEST_PATH="${DATA_ROOT}/afs_semantics.jsonl"
SEMANTIC_CACHE_ROOT="${DATA_ROOT}/semantic_cache"
SEKAI_EVAL_PROMPT_PATH="${DATA_ROOT}/sekai_eval_prompts.txt"
OUTPUT_DIR="${WORK_ROOT}/outputs"
WORLDSCORE_ROOT="${WORK_ROOT}/worldscore_t2v"
WORLDSCORE_ROW_KEY="$(IFS=_; printf '%s' "${WORLDSCORE_T2V_ROW_INDICES[*]}")"
WORLDSCORE_PROMPT_PATH="${WORLDSCORE_ROOT}/prompts_${WORLDSCORE_ROW_KEY}.txt"
WORLDSCORE_METADATA_PATH="${WORLDSCORE_ROOT}/metadata_${WORLDSCORE_ROW_KEY}.json"
WORLDSCORE_PROMPT_CACHE_PATH="${WORLDSCORE_ROOT}/prompt_embeddings_${WORLDSCORE_ROW_KEY}.safetensors"

WAN_MODEL_PATH="${ASSET_ROOT}/Wan2.1-T2V-1.3B"
QWEN3_VL_MODEL_PATH="${ASSET_ROOT}/Qwen3-VL-32B-Instruct"
SELF_FORCING_ROOT="${ASSET_ROOT}/Self-Forcing"
SELF_FORCING_CHECKPOINT_PATH="${SELF_FORCING_ROOT}/checkpoints/self_forcing_dmd.pt"
SELF_FORCING_CONFIG_PATH="${REPO_ROOT}/configs/afs_self_forcing_lora.yaml"
TEXT_ENCODER_MODEL_PATH="${WAN_MODEL_PATH}"
SEKAI_CSV="${DOWNLOAD_ROOT}/train/sekai-real-walking-hq.csv"
GENERATED_CONFIG="${REPO_ROOT}/.afs_runtime_config.yaml"

export HF_HOME="${HF_HOME:-${WORK_ROOT}/huggingface_cache}"
export HF_HUB_DISABLE_TELEMETRY=1
[[ -n "${HF_TOKEN}" ]] && export HF_TOKEN

log() { printf '[AFS] %s\n' "$*"; }
die() { printf '[AFS] ERROR: %s\n' "$*" >&2; exit 2; }
log "This execution's complete terminal output is being recorded at ${RUN_LOG}"
command -v python3 >/dev/null || die "python3 is required"
log "Sekai data is licensed for non-commercial research use only"
[[ "${NUM_GPUS}" =~ ^[1-9][0-9]*$ ]] || die "NUM_GPUS must be a positive integer"
[[ "${MAX_DATASET_SAMPLES}" =~ ^[0-9]+$ ]] || die "MAX_DATASET_SAMPLES must be a non-negative integer"
[[ "${EVAL_DATASET_SAMPLES}" =~ ^[1-9][0-9]*$ ]] || die "EVAL_DATASET_SAMPLES must be at least 1"
[[ "${MAX_TRAIN_STEPS}" =~ ^[1-9][0-9]*$ ]] || die "MAX_TRAIN_STEPS must be a positive integer"
[[ "${SAVE_INTERVAL_STEPS}" =~ ^[1-9][0-9]*$ ]] || die "SAVE_INTERVAL_STEPS must be a positive integer"
[[ "${EVAL_INTERVAL_STEPS}" =~ ^[0-9]+$ ]] || die "EVAL_INTERVAL_STEPS must be a non-negative integer"
[[ "${EVAL_NUM_SAMPLES}" -eq 2 ]] || die "EVAL_NUM_SAMPLES must remain 2 for comparison generation"
[[ "${CLEANUP_AFTER_SUCCESS}" == true || "${CLEANUP_AFTER_SUCCESS}" == false ]] || \
  die "CLEANUP_AFTER_SUCCESS must be true or false"
TRAIN_NUM_LATENT_FRAMES=$(( (TARGET_FRAME_COUNT - 1) / VAE_TEMPORAL_COMPRESSION_RATIO + 1 ))
(( (TARGET_FRAME_COUNT - 1) % VAE_TEMPORAL_COMPRESSION_RATIO == 0 )) || \
  die "TARGET_FRAME_COUNT must satisfy (frames - 1) % VAE_TEMPORAL_COMPRESSION_RATIO == 0"
(( TRAIN_NUM_LATENT_FRAMES % LATENT_CHUNK_SIZE == 0 )) || \
  die "Derived training latent frames must be divisible by LATENT_CHUNK_SIZE"
if (( MAX_DATASET_SAMPLES > 0 && EVAL_DATASET_SAMPLES >= MAX_DATASET_SAMPLES )); then
  die "MAX_DATASET_SAMPLES must leave at least one train sample after the eval split"
fi

mkdir -p \
  "${ASSET_ROOT}" "${DOWNLOAD_ROOT}" "${RAW_VIDEO_ROOT}" \
  "${PROCESSED_ROOT}" "${GT_LATENT_CACHE_ROOT}" \
  "${SEMANTIC_CACHE_ROOT}" "${OUTPUT_DIR}" "${WORLDSCORE_ROOT}" "${HF_HOME}"

free_gb=$(df -Pk "${WORK_ROOT}" | awk 'NR==2 {print int($4/1024/1024)}')
(( free_gb >= MIN_FREE_DISK_GB )) || die "Only ${free_gb} GiB free under ${WORK_ROOT}; ${MIN_FREE_DISK_GB} GiB required"

if [[ "${AUTO_INSTALL_DEPENDENCIES}" == true ]]; then
  log "Installing/updating Python dependencies"
  python3 -m pip install --upgrade pip
  python3 -m pip install -r "${REPO_ROOT}/requirements.txt"
  if [[ "${INSTALL_FLASH_ATTN}" == true ]]; then
    python3 -m pip install flash-attn --no-build-isolation
  fi
fi

command -v hf >/dev/null || die "The Hugging Face 'hf' CLI was not installed"
if ! command -v ffmpeg >/dev/null; then
  log "System ffmpeg was not found; exposing the imageio-ffmpeg bundled binary"
  FFMPEG_EXE="$(python3 -c 'import imageio_ffmpeg; print(imageio_ffmpeg.get_ffmpeg_exe())')"
  [[ -x "${FFMPEG_EXE}" ]] || die "No usable ffmpeg binary was found"
  mkdir -p "${WORK_ROOT}/bin"
  ln -sf "${FFMPEG_EXE}" "${WORK_ROOT}/bin/ffmpeg"
  export PATH="${WORK_ROOT}/bin:${PATH}"
fi

log "Fetching and caching two official WorldScore dynamic T2V prompts"
python3 "${REPO_ROOT}/data_processing/prepare_worldscore_t2v_prompt.py" \
  --output-prompt "${WORLDSCORE_PROMPT_PATH}" \
  --output-metadata "${WORLDSCORE_METADATA_PATH}" \
  --row-indices "${WORLDSCORE_T2V_ROW_INDICES[@]}"

hf_download_model() {
  local repo_id="$1"
  local destination="$2"
  log "Downloading or resuming ${repo_id}"
  hf download "${repo_id}" --local-dir "${destination}"
}

hf_download_model "Wan-AI/Wan2.1-T2V-1.3B" "${WAN_MODEL_PATH}"
hf_download_model "Qwen/Qwen3-VL-32B-Instruct" "${QWEN3_VL_MODEL_PATH}"
log "Downloading or resuming Self-Forcing DMD checkpoint"
hf download gdhe17/Self-Forcing \
  checkpoints/self_forcing_dmd.pt \
  --local-dir "${SELF_FORCING_ROOT}"

log "Downloading Sekai-Real-Walking-HQ annotations"
hf download Lixsp11/Sekai \
  --repo-type dataset \
  train/sekai-real-walking-hq.csv \
  --local-dir "${DOWNLOAD_ROOT}"

log "Downloading selected Sekai clips and building the source manifest"
python3 "${REPO_ROOT}/data_processing/prepare_sekai_manifest.py" \
  --csv "${SEKAI_CSV}" \
  --video-dir "${RAW_VIDEO_ROOT}" \
  --output "${SOURCE_MANIFEST}" \
  --max-samples "${MAX_DATASET_SAMPLES}" \
  --eval-samples "${EVAL_DATASET_SAMPLES}" \
  --source-fps "${SEKAI_SOURCE_FPS}" \
  --max-download-failures "${SEKAI_MAX_DOWNLOAD_FAILURES}"

log "Normalizing video frames for Qwen3-VL and Wan VAE"
python3 "${REPO_ROOT}/data_processing/prepare_videos_and_prompts.py" \
  --manifest "${SOURCE_MANIFEST}" \
  --output_root "${PROCESSED_ROOT}" \
  --num_frames "${TARGET_FRAME_COUNT}" \
  --height "${TARGET_HEIGHT}" \
  --width "${TARGET_WIDTH}" \
  --fps "${TARGET_FPS}" \
  --write_preview_mp4

python3 "${REPO_ROOT}/data_processing/build_afs_manifest.py" \
  --source-manifest "${SOURCE_MANIFEST}" \
  --processed-root "${PROCESSED_ROOT}" \
  --output "${INPUT_MANIFEST}"

python3 "${REPO_ROOT}/data_processing/export_split_prompts.py" \
  --manifest "${SOURCE_MANIFEST}" \
  --split eval \
  --output "${SEKAI_EVAL_PROMPT_PATH}"

log "Encoding or resuming GT Wan latents"
if [[ "${NUM_GPUS}" -eq 1 ]]; then
  python3 "${REPO_ROOT}/data_processing/compute_vae_latents.py" \
    --video_pt_dir "${PROCESSED_ROOT}/processed_video" \
    --output_latent_folder "${GT_LATENT_CACHE_ROOT}" \
    --model_root "${WAN_MODEL_PATH}" \
    --afs_chunk_size "${LATENT_CHUNK_SIZE}" \
    --resume
else
  torchrun --standalone --nproc_per_node="${NUM_GPUS}" \
    "${REPO_ROOT}/data_processing/compute_vae_latents.py" \
    --video_pt_dir "${PROCESSED_ROOT}/processed_video" \
    --output_latent_folder "${GT_LATENT_CACHE_ROOT}" \
    --model_root "${WAN_MODEL_PATH}" \
    --afs_chunk_size "${LATENT_CHUNK_SIZE}" \
    --resume
fi

yaml_quote() {
  local value="$1"
  value=${value//\\/\\\\}
  value=${value//\"/\\\"}
  printf '"%s"' "${value}"
}

cat >"${GENERATED_CONFIG}" <<EOF
method:
  name: afs
data:
  input_manifest: $(yaml_quote "${INPUT_MANIFEST}")
  video_root: $(yaml_quote "${PROCESSED_ROOT}/processed_video")
  semantic_manifest_path: $(yaml_quote "${SEMANTIC_MANIFEST_PATH}")
  semantic_cache_root: $(yaml_quote "${SEMANTIC_CACHE_ROOT}")
  gt_latent_mode: precomputed
  gt_latent_cache_root: $(yaml_quote "${GT_LATENT_CACHE_ROOT}")
self_forcing:
  config_path: $(yaml_quote "${SELF_FORCING_CONFIG_PATH}")
model:
  backend: self_forcing
  wan_model_path: $(yaml_quote "${WAN_MODEL_PATH}")
  self_forcing_checkpoint_path: $(yaml_quote "${SELF_FORCING_CHECKPOINT_PATH}")
  train_mode: lora
qwen3_vl:
  model_path: $(yaml_quote "${QWEN3_VL_MODEL_PATH}")
  dtype: bfloat16
  device_map: auto
  attn_implementation: flash_attention_2
  max_new_tokens: ${QWEN_MAX_NEW_TOKENS}
  video_fps: ${TARGET_FPS}
text_encoder:
  backend: self_forcing_umt5
  model_path: $(yaml_quote "${TEXT_ENCODER_MODEL_PATH}")
  dtype: bfloat16
chunking:
  inherit_from_self_forcing: true
  target_fps: ${TARGET_FPS}
  target_frame_count: ${TARGET_FRAME_COUNT}
  vae_temporal_compression_ratio: ${VAE_TEMPORAL_COMPRESSION_RATIO}
teacher:
  type: ema
  text_condition_mode: chunk_replace
  use_gt_assisted_visual_cache: true
  ema_decay: ${EMA_DECAY}
training:
  max_steps: ${MAX_TRAIN_STEPS}
  save_interval_steps: ${SAVE_INTERVAL_STEPS}
  max_checkpoints: ${MAX_CHECKPOINTS}
  shared_prefix_chunks: ${SHARED_PREFIX_CHUNKS}
  detach_rollout_transition: true
  dense_velocity_matching: true
  train_num_latent_frames: ${TRAIN_NUM_LATENT_FRAMES}
  eval_interval_steps: ${EVAL_INTERVAL_STEPS}
  eval_num_samples: ${EVAL_NUM_SAMPLES}
  eval_num_latent_frames: ${EVAL_NUM_LATENT_FRAMES}
  eval_seed: ${WORLDSCORE_EVAL_SEED}
  eval_prompt_path: $(yaml_quote "${WORLDSCORE_PROMPT_PATH}")
  eval_prompt_cache_path: $(yaml_quote "${WORLDSCORE_PROMPT_CACHE_PATH}")
runtime:
  semantic_resume: ${SEMANTIC_RESUME}
  semantic_overwrite: ${SEMANTIC_OVERWRITE}
  semantic_wait_timeout_sec: ${SEMANTIC_WAIT_TIMEOUT_SEC}
  semantic_poll_interval_sec: ${SEMANTIC_POLL_INTERVAL_SEC}
  training_resume: ${TRAINING_RESUME}
  output_dir: $(yaml_quote "${OUTPUT_DIR}")
  mixed_precision: bf16
EOF

cd "${REPO_ROOT}"
log "Starting Qwen3-VL chunk semantics, followed automatically by AFS training"
if [[ "${NUM_GPUS}" -eq 1 ]]; then
  python3 scripts/afs_train.py --config "${GENERATED_CONFIG}" "${EXTRA_ARGS[@]}"
else
  torchrun --standalone --nproc_per_node="${NUM_GPUS}" \
    scripts/afs_train.py --config "${GENERATED_CONFIG}" "${EXTRA_ARGS[@]}"
fi

LATEST_AFS_CHECKPOINT="$({ find "${OUTPUT_DIR}" -maxdepth 2 -path '*/checkpoint_model_*/model.pt' -type f -print || true; } | sort -V | tail -n 1)"
[[ -n "${LATEST_AFS_CHECKPOINT}" ]] || die "Training completed without an AFS checkpoint under ${OUTPUT_DIR}"
log "Using final AFS checkpoint for WorldScore comparison: ${LATEST_AFS_CHECKPOINT}"
FINAL_STEP="$(basename "$(dirname "${LATEST_AFS_CHECKPOINT}")")"
COMPARISON_ROOT="${WORLDSCORE_ROOT}/comparisons/${FINAL_STEP}"
BASELINE_INFERENCE_CONFIG="${COMPARISON_ROOT}/self_forcing_original.yaml"
AFS_INFERENCE_CONFIG="${COMPARISON_ROOT}/afs_final.yaml"
mkdir -p "${COMPARISON_ROOT}"

cat >"${BASELINE_INFERENCE_CONFIG}" <<EOF
generator_ckpt: $(yaml_quote "${SELF_FORCING_CHECKPOINT_PATH}")
denoising_step_list: [1000, 750, 500, 250]
warp_denoising_step: true
num_frame_per_block: 3
model_name: Wan2.1-T2V-1.3B
model_kwargs:
  model_root: $(yaml_quote "${WAN_MODEL_PATH}")
  local_attn_size: 12
  timestep_shift: 5.0
  sink_size: 3
context_noise: 0
use_ema: false
EOF

cat >"${AFS_INFERENCE_CONFIG}" <<EOF
generator_ckpt: $(yaml_quote "${SELF_FORCING_CHECKPOINT_PATH}")
lora_ckpt: $(yaml_quote "${LATEST_AFS_CHECKPOINT}")
denoising_step_list: [1000, 750, 500, 250]
warp_denoising_step: true
num_frame_per_block: 3
model_name: Wan2.1-T2V-1.3B
model_kwargs:
  model_root: $(yaml_quote "${WAN_MODEL_PATH}")
  local_attn_size: 12
  timestep_shift: 5.0
  sink_size: 3
adapter:
  type: lora
  rank: 256
  alpha: 256
  dropout: 0.0
  dtype: bfloat16
  verbose: false
  init_lora_weights: gaussian
context_noise: 0
use_ema: false
EOF

log "Generating the original Self-Forcing WorldScore T2V comparison video"
python3 inference.py \
  --config_path "${BASELINE_INFERENCE_CONFIG}" \
  --data_path "${WORLDSCORE_PROMPT_PATH}" \
  --output_folder "${COMPARISON_ROOT}/self_forcing_original" \
  --num_output_frames "${EVAL_NUM_LATENT_FRAMES}" \
  --seed "${WORLDSCORE_EVAL_SEED}" \
  --per_prompt_seed \
  --use_lmdb_pipeline \
  --lmdb_cache_update_source generated \
  --lmdb_use_relative_sink

log "Generating the final AFS WorldScore T2V comparison video"
python3 inference.py \
  --config_path "${AFS_INFERENCE_CONFIG}" \
  --data_path "${WORLDSCORE_PROMPT_PATH}" \
  --output_folder "${COMPARISON_ROOT}/afs_final" \
  --num_output_frames "${EVAL_NUM_LATENT_FRAMES}" \
  --seed "${WORLDSCORE_EVAL_SEED}" \
  --per_prompt_seed \
  --use_lmdb_pipeline \
  --lmdb_cache_update_source generated \
  --lmdb_use_relative_sink

log "Generating original Self-Forcing videos for the held-out Sekai prompts"
python3 inference.py \
  --config_path "${BASELINE_INFERENCE_CONFIG}" \
  --data_path "${SEKAI_EVAL_PROMPT_PATH}" \
  --output_folder "${COMPARISON_ROOT}/sekai_eval/self_forcing_original" \
  --num_output_frames "${EVAL_NUM_LATENT_FRAMES}" \
  --seed "${WORLDSCORE_EVAL_SEED}" \
  --per_prompt_seed \
  --use_lmdb_pipeline \
  --lmdb_cache_update_source generated \
  --lmdb_use_relative_sink

log "Generating final AFS videos for the held-out Sekai prompts"
python3 inference.py \
  --config_path "${AFS_INFERENCE_CONFIG}" \
  --data_path "${SEKAI_EVAL_PROMPT_PATH}" \
  --output_folder "${COMPARISON_ROOT}/sekai_eval/afs_final" \
  --num_output_frames "${EVAL_NUM_LATENT_FRAMES}" \
  --seed "${WORLDSCORE_EVAL_SEED}" \
  --per_prompt_seed \
  --use_lmdb_pipeline \
  --lmdb_cache_update_source generated \
  --lmdb_use_relative_sink

if [[ "${CLEANUP_AFTER_SUCCESS}" == true ]]; then
  log "Comparison complete; removing downloaded models, training data, and derived caches"
  rm -rf -- \
    "${ASSET_ROOT}" \
    "${DATA_ROOT}" \
    "${HF_HOME}" \
    "${WORK_ROOT}/bin"
  rm -f -- "${WORLDSCORE_PROMPT_CACHE_PATH}" "${GENERATED_CONFIG}"
  log "Cleanup complete; retained checkpoint at $(dirname "${LATEST_AFS_CHECKPOINT}")"
  log "Retained evaluation results at ${COMPARISON_ROOT} and log at ${RUN_LOG}"
fi

log "AFS pipeline complete. WorldScore and held-out Sekai comparisons: ${COMPARISON_ROOT}"
