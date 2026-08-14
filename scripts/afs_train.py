#!/usr/bin/env python3
import argparse
import os
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))


def require_local(value, label, directory=False):
    if not value:
        raise ValueError(f"{label} must be configured with a local path")
    path = Path(value).expanduser().resolve()
    valid = path.is_dir() if directory else path.is_file()
    if not valid:
        raise FileNotFoundError(f"{label} does not exist locally: {path}")
    return path


def build_semantic_config(config, OmegaConf, rank=0, world_size=1, output_manifest=None):
    qwen_config = OmegaConf.to_container(config.qwen3_vl, resolve=True)
    if world_size > 1:
        local_rank = int(os.environ.get("LOCAL_RANK", rank))
        qwen_config["device_map"] = f"cuda:{local_rank}"
    return OmegaConf.create({
        "data": {
            "input_manifest": config.data.input_manifest,
            "video_root": config.data.video_root,
            "output_manifest": output_manifest or config.data.semantic_manifest_path,
            "semantic_cache_root": config.data.semantic_cache_root,
        },
        "self_forcing": {"config_path": config.self_forcing.config_path},
        "qwen3_vl": qwen_config,
        "text_encoder": OmegaConf.to_container(config.text_encoder, resolve=True),
        "chunking": OmegaConf.to_container(config.chunking, resolve=True),
        "evaluation": {
            "prompt_path": config.training.eval_prompt_path if rank == 0 else None,
            "prompt_cache_path": config.training.eval_prompt_cache_path if rank == 0 else None,
        },
        "runtime": {
            "phase": "all",
            "resume": config.runtime.semantic_resume,
            "overwrite": config.runtime.semantic_overwrite,
            "shard_id": rank,
            "num_shards": world_size,
        },
    })


def validate_semantic_completion(config):
    from afs.semantics import load_jsonl
    source = load_jsonl(require_local(config.data.input_manifest, "data.input_manifest"))
    output = load_jsonl(require_local(config.data.semantic_manifest_path, "data.semantic_manifest_path"))
    statuses = {record["sample_id"]: record for record in output}
    incomplete = [
        record["sample_id"] for record in source
        if statuses.get(record["sample_id"], {}).get("status") != "complete"
    ]
    if incomplete:
        preview = ", ".join(incomplete[:10])
        raise RuntimeError(
            f"AFS semantic preprocessing is incomplete for {len(incomplete)} samples: {preview}. "
            "Training was not started."
        )
    require_local(config.training.eval_prompt_path, "training.eval_prompt_path")
    require_local(config.training.eval_prompt_cache_path, "training.eval_prompt_cache_path")


def prepare_semantics(config, OmegaConf, rank):
    ready_path = Path(str(config.data.semantic_manifest_path) + ".ready")
    init_path = Path(str(config.data.semantic_manifest_path) + ".shards_initialized")
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    shard_paths = [
        Path(str(config.data.semantic_manifest_path) + f".rank_{index}.jsonl")
        for index in range(world_size)
    ]
    shard_ready_paths = [Path(str(path) + ".ready") for path in shard_paths]
    shard_failed_paths = [Path(str(path) + ".failed") for path in shard_paths]

    if rank == 0:
        for stale_path in [ready_path, init_path, *shard_ready_paths, *shard_failed_paths]:
            if stale_path.exists():
                stale_path.unlink()
        init_path.parent.mkdir(parents=True, exist_ok=True)
        init_path.write_text("ready\n", encoding="utf-8")

    timeout = float(config.runtime.semantic_wait_timeout_sec)
    deadline = time.monotonic() + timeout
    while not init_path.is_file():
        if time.monotonic() >= deadline:
            raise TimeoutError(f"Timed out waiting for semantic shard initialization: {init_path}")
        time.sleep(float(config.runtime.semantic_poll_interval_sec))

    shard_path = shard_paths[rank]
    try:
        from afs.semantics import AFSSemanticPreprocessor

        AFSSemanticPreprocessor(
            build_semantic_config(
                config,
                OmegaConf,
                rank=rank,
                world_size=world_size,
                output_manifest=str(shard_path),
            )
        ).run()
        shard_ready_paths[rank].write_text("complete\n", encoding="utf-8")
    except Exception as exc:
        shard_failed_paths[rank].write_text(
            f"{type(exc).__name__}: {exc}\n", encoding="utf-8"
        )
        raise

    while not all(path.is_file() for path in shard_ready_paths):
        failed = next((path for path in shard_failed_paths if path.is_file()), None)
        if failed is not None:
            raise RuntimeError(f"AFS semantic shard failed: {failed.read_text(encoding='utf-8').strip()}")
        if time.monotonic() >= deadline:
            raise TimeoutError("Timed out waiting for all semantic preprocessing shards")
        time.sleep(float(config.runtime.semantic_poll_interval_sec))

    if rank == 0:
        from afs.semantics import load_jsonl, write_jsonl_atomic

        source_records = load_jsonl(Path(config.data.input_manifest))
        merged = {}
        for path in shard_paths:
            for record in load_jsonl(path):
                merged[record["sample_id"]] = record
        ordered = [merged[record["sample_id"]] for record in source_records]
        write_jsonl_atomic(Path(config.data.semantic_manifest_path), ordered)
        validate_semantic_completion(config)
        ready_path.write_text("complete\n", encoding="utf-8")
    else:
        while not ready_path.is_file():
            if time.monotonic() >= deadline:
                raise TimeoutError(f"Timed out waiting for merged semantic manifest: {ready_path}")
            time.sleep(float(config.runtime.semantic_poll_interval_sec))
        validate_semantic_completion(config)


def build_training_config(config, args, OmegaConf):
    sf_config = require_local(config.self_forcing.config_path, "self_forcing.config_path")
    wan_root = require_local(config.model.wan_model_path, "model.wan_model_path", directory=True)
    checkpoint = require_local(
        config.model.self_forcing_checkpoint_path, "model.self_forcing_checkpoint_path"
    )
    manifest = require_local(config.data.semantic_manifest_path, "data.semantic_manifest_path")
    if config.data.gt_latent_mode != "precomputed":
        raise NotImplementedError("The unified AFS launcher currently requires gt_latent_mode=precomputed")
    latent_root = require_local(
        config.data.gt_latent_cache_root, "data.gt_latent_cache_root", directory=True
    )
    if config.teacher.ema_decay is None:
        raise ValueError("teacher.ema_decay must be configured")
    output_dir = args.output_dir or config.runtime.output_dir
    if not output_dir:
        raise ValueError("runtime.output_dir or --output-dir must be configured")
    upstream = OmegaConf.load(sf_config)
    return OmegaConf.merge(upstream, OmegaConf.create({
        "trainer": "afs_training",
        "distribution_loss": "afs_training",
        "generator_ckpt": str(checkpoint),
        "model_kwargs": {"model_root": str(wan_root)},
        "teacher_text_condition_mode": config.teacher.text_condition_mode,
        "shared_prefix_chunks": int(config.training.shared_prefix_chunks),
        "afs_use_gt_first_chunk": int(config.training.shared_prefix_chunks) == 1,
        "use_gt_assisted_teacher_cache": bool(config.teacher.use_gt_assisted_visual_cache),
        "detach_rollout_transition": bool(config.training.detach_rollout_transition),
        "afs_loss_step_mode": "all" if config.training.dense_velocity_matching else "single",
        "afs_loss_frames": int(config.training.train_num_latent_frames),
        "afs_max_loss_frames": int(config.training.train_num_latent_frames),
        "afs_eval_interval": int(config.training.eval_interval_steps),
        "afs_eval_num_samples": int(config.training.eval_num_samples),
        "afs_eval_num_frames": int(config.training.eval_num_latent_frames),
        "afs_eval_seed": int(config.training.eval_seed),
        "afs_eval_prompt_path": str(
            require_local(config.training.eval_prompt_path, "training.eval_prompt_path")
        ),
        "afs_eval_prompt_cache_path": str(
            require_local(config.training.eval_prompt_cache_path, "training.eval_prompt_cache_path")
        ),
        "max_iters": int(config.training.max_steps),
        "save_iters": int(config.training.save_interval_steps),
        "max_checkpoints": int(config.training.max_checkpoints),
        "ema_weight": config.teacher.ema_decay,
        "afs_semantic_manifest_path": str(manifest),
        "gt_latent_mode": "precomputed",
        "gt_latent_cache_root": str(latent_root),
        "logdir": str(Path(output_dir).expanduser().resolve()),
        "no_save": args.no_save,
        "disable_wandb": args.disable_logging,
        "wandb_save_dir": "",
        "auto_resume": bool(config.runtime.training_resume),
        "image_or_video_shape": [
            1,
            int(config.training.train_num_latent_frames),
            16,
            60,
            104,
        ],
    }))


def main():
    parser = argparse.ArgumentParser(
        description="AFS unified semantic preprocessing and distributed training"
    )
    parser.add_argument("--config", default="configs/afs_training.yaml")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--no-save", action="store_true")
    parser.add_argument("--disable-logging", action="store_true")
    args = parser.parse_args()
    try:
        from omegaconf import OmegaConf
    except ImportError as exc:
        parser.error(f"omegaconf is required after argument parsing: {exc}")
    config_path = require_local(args.config, "--config")
    config = OmegaConf.load(config_path)
    if config.method.name != "afs":
        parser.error("configuration must select method.name=afs")
    if config.model.train_mode != "lora":
        parser.error("the unified launcher currently supports model.train_mode=lora")
    if config.runtime.mixed_precision != "bf16":
        parser.error("the Self-Forcing backend currently supports runtime.mixed_precision=bf16")
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if "LOCAL_RANK" in os.environ:
        import torch

        torch.cuda.set_device(local_rank)
    prepare_semantics(config, OmegaConf, rank)
    training_config = build_training_config(config, args, OmegaConf)
    Path(training_config.logdir).mkdir(parents=True, exist_ok=True)
    if rank == 0:
        OmegaConf.save(training_config, Path(training_config.logdir) / "afs_resolved_config.yaml")
    from trainer import AFSTrainer
    AFSTrainer(training_config).train()


if __name__ == "__main__":
    main()
