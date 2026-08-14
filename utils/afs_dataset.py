import json
from pathlib import Path

import torch
from torch.utils.data import Dataset


class AFSTrainingDataset(Dataset):
    """Local-only adapter joining semantic preprocessing caches with precomputed GT latents."""

    def __init__(self, manifest_path: str, gt_latent_cache_root: str, split: str = "train"):
        self.manifest_path = Path(manifest_path).expanduser().resolve()
        self.latent_root = Path(gt_latent_cache_root).expanduser().resolve()
        if not self.manifest_path.is_file():
            raise FileNotFoundError(f"AFS semantic preprocessing manifest does not exist: {self.manifest_path}")
        if not self.latent_root.is_dir():
            raise FileNotFoundError(f"AFS GT latent cache root does not exist: {self.latent_root}")
        with self.manifest_path.open("r", encoding="utf-8") as handle:
            self.records = [json.loads(line) for line in handle if line.strip()]
        self.records = [
            record for record in self.records
            if record.get("status") == "complete" and record.get("split", "train") == split
        ]
        if not self.records:
            raise ValueError(f"AFS semantic preprocessing manifest contains no complete {split} samples")

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        semantic_path = Path(record["semantic_cache_path"])
        if not semantic_path.is_file():
            raise FileNotFoundError(f"Semantic cache missing for {record['sample_id']}: {semantic_path}")
        try:
            from safetensors.torch import load_file
        except ImportError as exc:
            raise RuntimeError("safetensors is required to read AFS semantic preprocessing caches") from exc
        semantic = load_file(str(semantic_path), device="cpu")
        latent_path = self.latent_root / f"{record['sample_id']}.safetensors"
        if not latent_path.is_file():
            raise FileNotFoundError(f"Precomputed GT latent cache missing: {latent_path}")
        latents = load_file(str(latent_path), device="cpu")
        if "gt_chunk_latents" not in latents:
            raise KeyError(f"{latent_path} must contain gt_chunk_latents [N,C,T,H,W]")
        chunk_count = len(record["chunk_boundaries"])
        chunks = semantic["chunk_text_embeddings"]
        gt_chunks = latents["gt_chunk_latents"]
        if chunks.shape[0] != chunk_count or gt_chunks.shape[0] != chunk_count:
            raise ValueError(f"Chunk alignment mismatch for {record['sample_id']}")
        output = {
            "sample_ids": record["sample_id"],
            "prompts": record["global_caption"],
            "global_text_embeddings": semantic["global_text_embedding"],
            "chunk_text_embeddings": chunks,
            "gt_chunk_latents": gt_chunks,
            "valid_chunk_mask": torch.ones(chunk_count, dtype=torch.bool),
        }
        for source, target in (
            ("global_text_mask", "global_text_masks"),
            ("chunk_text_masks", "chunk_text_masks"),
        ):
            if source in semantic:
                output[target] = semantic[source]
        return output
