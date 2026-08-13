from dataclasses import dataclass, field
from typing import Optional

import torch


@dataclass(frozen=True)
class AFSChunkBoundary:
    chunk_index: int
    frame_start: int
    frame_end: int
    time_start_sec: float
    time_end_sec: float


@dataclass
class AFSTextCondition:
    embeddings: torch.Tensor
    attention_mask: Optional[torch.Tensor] = None

    def as_model_dict(self) -> dict:
        result = {"prompt_embeds": self.embeddings}
        if self.attention_mask is not None:
            result["prompt_mask"] = self.attention_mask
        return result


@dataclass
class AFSTrainingBatch:
    sample_ids: list[str]
    global_text_embeddings: torch.Tensor
    global_text_masks: Optional[torch.Tensor]
    chunk_text_embeddings: torch.Tensor
    chunk_text_masks: Optional[torch.Tensor]
    gt_chunk_latents: torch.Tensor
    valid_chunk_mask: Optional[torch.Tensor] = None

    def validate(self) -> None:
        batch_size = len(self.sample_ids)
        if self.global_text_embeddings.ndim != 3 or self.global_text_embeddings.shape[0] != batch_size:
            raise ValueError("global_text_embeddings must be [B, L, D] and match sample_ids")
        if self.chunk_text_embeddings.ndim != 4 or self.chunk_text_embeddings.shape[0] != batch_size:
            raise ValueError("chunk_text_embeddings must be [B, N, L, D]")
        if self.gt_chunk_latents.ndim != 6 or self.gt_chunk_latents.shape[:2] != self.chunk_text_embeddings.shape[:2]:
            raise ValueError("gt_chunk_latents must be [B, N, C, T, H, W] with matching B,N")
        if self.valid_chunk_mask is not None and self.valid_chunk_mask.shape != self.chunk_text_embeddings.shape[:2]:
            raise ValueError("valid_chunk_mask must be [B, N]")

    def frame_major_latents(self) -> torch.Tensor:
        return self.gt_chunk_latents.permute(0, 1, 3, 2, 4, 5).flatten(1, 2)

    def to(self, device: torch.device, dtype: torch.dtype) -> "AFSTrainingBatch":
        def move(value, cast=False):
            return None if value is None else value.to(device=device, dtype=dtype if cast else value.dtype)

        return AFSTrainingBatch(
            sample_ids=self.sample_ids,
            global_text_embeddings=move(self.global_text_embeddings, True),
            global_text_masks=move(self.global_text_masks),
            chunk_text_embeddings=move(self.chunk_text_embeddings, True),
            chunk_text_masks=move(self.chunk_text_masks),
            gt_chunk_latents=move(self.gt_chunk_latents, True),
            valid_chunk_mask=move(self.valid_chunk_mask),
        )


@dataclass
class AFSDenoisingState:
    chunk_index: int
    step_index: int
    noisy_latent: torch.Tensor
    timestep: torch.Tensor
    student_velocity: torch.Tensor


@dataclass
class AFSStudentChunkRollout:
    chunk_index: int
    denoising_states: list[AFSDenoisingState] = field(default_factory=list)
    clean_chunk: Optional[torch.Tensor] = None
