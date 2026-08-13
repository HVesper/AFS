import gc
import json
import os
from dataclasses import asdict
from pathlib import Path
from typing import Iterable

from omegaconf import OmegaConf

from .types import AFSChunkBoundary


CHUNK_CAPTION_PROMPT = """You are describing one temporal chunk from a text-to-video training video.

The original full-video caption is:
{global_caption}

Describe only what is visibly happening during the provided video chunk.
Focus on the main subjects, observable action and state change, interactions, motion direction, and persistent attributes.
Use the global caption only to resolve ambiguity. Do not invent or describe events outside this chunk.
Return one concise English sentence without JSON, Markdown, reasoning, confidence scores, or alternatives."""


def require_local_path(value, label: str, kind: str = "file") -> Path:
    if not value:
        raise ValueError(f"{label} must be configured with a local path")
    path = Path(os.path.expanduser(str(value))).resolve()
    exists = path.is_file() if kind == "file" else path.is_dir()
    if not exists:
        raise FileNotFoundError(f"{label} local {kind} does not exist: {path}")
    return path


def load_jsonl(path: Path) -> list[dict]:
    records = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            record = json.loads(line)
            missing = {"sample_id", "video_path", "global_caption"} - record.keys()
            if missing:
                raise ValueError(f"{path}:{line_number} missing fields: {sorted(missing)}")
            records.append(record)
    return records


def write_jsonl_atomic(path: Path, records: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


class AFSChunkBoundaryResolver:
    """Maps Self-Forcing latent blocks to inclusive source pixel-frame bounds."""

    def __init__(self, self_forcing_config, target_fps: float, vae_temporal_ratio: int):
        if not bool(self_forcing_config.get("causal", True)):
            raise ValueError("AFS Stage 1 requires a causal Self-Forcing configuration")
        self.latent_frames_per_chunk = int(self_forcing_config.num_frame_per_block)
        self.target_fps = float(target_fps)
        self.vae_temporal_ratio = int(vae_temporal_ratio)
        if min(self.latent_frames_per_chunk, self.target_fps, self.vae_temporal_ratio) <= 0:
            raise ValueError("FPS, VAE temporal ratio, and latent chunk size must be positive")

    @property
    def pixel_frames_per_chunk(self) -> int:
        return self.latent_frames_per_chunk * self.vae_temporal_ratio

    def resolve(self, target_frame_count: int) -> list[AFSChunkBoundary]:
        boundaries = []
        for index, start in enumerate(range(0, int(target_frame_count), self.pixel_frames_per_chunk)):
            end_exclusive = min(start + self.pixel_frames_per_chunk, int(target_frame_count))
            boundaries.append(AFSChunkBoundary(
                chunk_index=index,
                frame_start=start,
                frame_end=end_exclusive - 1,
                time_start_sec=start / self.target_fps,
                time_end_sec=end_exclusive / self.target_fps,
            ))
        return boundaries


class LocalQwen3VLCaptioner:
    """Lazy local-only Qwen3-VL adapter. No remote fallback is permitted."""

    def __init__(self, config):
        self.config = config
        self.model_path = require_local_path(config.model_path, "qwen3_vl.model_path", "directory")
        self.model = None
        self.processor = None

    def load(self):
        try:
            from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
        except ImportError as exc:
            raise RuntimeError("Stage 1 caption phase requires a local transformers installation with Qwen3-VL support") from exc
        self.processor = AutoProcessor.from_pretrained(self.model_path, local_files_only=True)
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            self.model_path,
            local_files_only=True,
            dtype=self.config.dtype,
            device_map=self.config.device_map,
            attn_implementation=self.config.attn_implementation,
        ).eval()
        self.model.requires_grad_(False)

    def caption(self, video_path: Path, boundary: AFSChunkBoundary, global_caption: str) -> str:
        # TODO(cluster): adapt the local Qwen3-VL processor's exact video input
        # schema/version here. Boundary values are the single source of truth.
        raise NotImplementedError(
            "Qwen3-VL local video processor integration requires the cluster's installed transformers version; "
            f"requested {video_path}, frames {boundary.frame_start}:{boundary.frame_end + 1}"
        )

    def close(self):
        self.model = None
        self.processor = None
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass


class SelfForcingUMT5Encoder:
    def __init__(self, model_root: Path):
        from utils.wan_wrapper import WanTextEncoder
        self.encoder = WanTextEncoder(model_root=str(model_root)).eval().requires_grad_(False)

    def encode(self, captions: list[str]):
        import torch
        with torch.no_grad():
            encoded = self.encoder(text_prompts=captions)
        return {
            "embeddings": encoded["prompt_embeds"].detach().cpu(),
            "masks": encoded["prompt_mask"].detach().cpu(),
        }

    def close(self):
        self.encoder = None
        gc.collect()


class AFSStage1SemanticPreprocessor:
    def __init__(self, config):
        self.config = config
        sf_config_path = require_local_path(config.self_forcing.config_path, "self_forcing.config_path")
        self.sf_config = OmegaConf.load(sf_config_path)
        self.input_manifest = require_local_path(config.data.input_manifest, "data.input_manifest")
        if not config.data.output_manifest or not config.data.semantic_cache_root:
            raise ValueError("data.output_manifest and data.semantic_cache_root must be configured")
        self.output_manifest = Path(config.data.output_manifest).expanduser().resolve()
        self.cache_root = Path(config.data.semantic_cache_root).expanduser().resolve()
        self.cache_root.mkdir(parents=True, exist_ok=True)
        self.boundaries = AFSChunkBoundaryResolver(
            self.sf_config,
            target_fps=config.chunking.target_fps,
            vae_temporal_ratio=config.chunking.vae_temporal_compression_ratio,
        ).resolve(config.chunking.target_frame_count)

    def _sharded_records(self):
        records = load_jsonl(self.input_manifest)
        shard_id = int(self.config.runtime.shard_id)
        num_shards = int(self.config.runtime.num_shards)
        if num_shards <= 0 or not 0 <= shard_id < num_shards:
            raise ValueError("runtime shard_id must satisfy 0 <= shard_id < num_shards")
        return [record for index, record in enumerate(records) if index % num_shards == shard_id]

    def run(self):
        phase = str(self.config.runtime.phase)
        if phase not in {"all", "caption", "encode"}:
            raise ValueError("runtime.phase must be all, caption, or encode")
        existing = {}
        if self.output_manifest.exists() and bool(self.config.runtime.resume):
            existing = {item["sample_id"]: item for item in load_jsonl(self.output_manifest)}
        records = self._sharded_records()
        captioner = None
        if phase in {"all", "caption"}:
            captioner = LocalQwen3VLCaptioner(self.config.qwen3_vl)
            captioner.load()
        try:
            for record in records:
                previous = existing.get(record["sample_id"])
                if previous and previous.get("status") == "complete" and not bool(self.config.runtime.overwrite):
                    continue
                result = self._caption_record(record, captioner, previous) if phase != "encode" else dict(previous or record)
                existing[record["sample_id"]] = result
                write_jsonl_atomic(self.output_manifest, existing.values())
        finally:
            if captioner is not None:
                captioner.close()
        if phase in {"all", "encode"}:
            self._encode_records(existing, records)

    def _caption_record(self, record, captioner, previous):
        result = dict(record)
        result["chunk_boundaries"] = [asdict(boundary) for boundary in self.boundaries]
        result["status"] = "captioning"
        captions = []
        try:
            video_path = Path(record["video_path"])
            if not video_path.is_absolute() and self.config.data.video_root:
                video_path = Path(self.config.data.video_root) / video_path
            require_local_path(video_path, f"video for {record['sample_id']}")
            for boundary in self.boundaries:
                captions.append(captioner.caption(video_path, boundary, record["global_caption"]))
            if len(captions) != len(self.boundaries):
                raise RuntimeError("Caption count does not match Self-Forcing chunk boundaries")
            result.update(chunk_captions=captions, status="caption_complete")
        except Exception as exc:
            result.update(status="failed", error=str(exc))
        return result

    def _encode_records(self, existing, input_records):
        model_root = require_local_path(self.config.text_encoder.model_path, "text_encoder.model_path", "directory")
        encoder = SelfForcingUMT5Encoder(model_root)
        try:
            for source in input_records:
                result = existing.get(source["sample_id"])
                if not result or result.get("status") not in {"caption_complete", "complete"}:
                    continue
                if result.get("status") == "complete" and not bool(self.config.runtime.overwrite):
                    continue
                try:
                    captions = result["chunk_captions"]
                    if len(captions) != len(result["chunk_boundaries"]):
                        raise ValueError("chunk captions and boundaries are not aligned")
                    encoded = encoder.encode([result["global_caption"], *captions])
                    embeddings = encoded["embeddings"]
                    masks = encoded["masks"]
                    cache_path = self.cache_root / f"{result['sample_id']}.safetensors"
                    try:
                        from safetensors.torch import save_file
                    except ImportError as exc:
                        raise RuntimeError("safetensors is required to write Stage 1 embedding caches") from exc
                    save_file({
                        "global_text_embedding": embeddings[0],
                        "global_text_mask": masks[0],
                        "chunk_text_embeddings": embeddings[1:],
                        "chunk_text_masks": masks[1:],
                    }, str(cache_path))
                    result.update(semantic_cache_path=str(cache_path), status="complete")
                except Exception as exc:
                    result.update(status="failed", error=str(exc))
                write_jsonl_atomic(self.output_manifest, existing.values())
        finally:
            encoder.close()
