#!/usr/bin/env python3
import argparse
import os
import sys
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


def build_upstream_config(config, cli_args):
    from omegaconf import OmegaConf
    if config.method.name != "afs" or config.stage.name != "afs_training":
        raise ValueError("configuration must select method=afs and stage=afs_training")
    sf_config_path = require_local(config.self_forcing.config_path, "self_forcing.config_path")
    wan_root = require_local(config.model.wan_model_path, "model.wan_model_path", directory=True)
    checkpoint = require_local(
        config.model.self_forcing_checkpoint_path,
        "model.self_forcing_checkpoint_path",
    )
    manifest = require_local(config.data.stage1_manifest_path, "data.stage1_manifest_path")
    if config.data.gt_latent_mode == "precomputed":
        latent_root = require_local(config.data.gt_latent_cache_root, "data.gt_latent_cache_root", directory=True)
    elif config.data.gt_latent_mode == "online":
        require_local(config.data.gt_video_root, "data.gt_video_root", directory=True)
        latent_root = Path(config.data.gt_video_root).expanduser().resolve()
    else:
        raise ValueError("data.gt_latent_mode must be precomputed or online")

    upstream = OmegaConf.load(sf_config_path)
    overrides = OmegaConf.create({
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
        "ema_weight": config.teacher.ema_decay,
        "afs_stage1_manifest_path": str(manifest),
        "gt_latent_mode": config.data.gt_latent_mode,
        "gt_latent_cache_root": str(latent_root),
        "logdir": str(Path(cli_args.output_dir or config.runtime.output_dir).expanduser().resolve()),
        "no_save": cli_args.no_save,
        "disable_wandb": cli_args.disable_logging,
        "wandb_save_dir": "",
        "auto_resume": config.runtime.resume_checkpoint is not None,
    })
    if config.model.train_mode != "lora":
        raise ValueError("Current AFS Stage 2 entrypoint supports model.train_mode=lora")
    if config.runtime.mixed_precision != "bf16":
        raise ValueError("Current Self-Forcing wrapper supports runtime.mixed_precision=bf16")
    if config.teacher.ema_decay is None:
        raise ValueError("teacher.ema_decay must inherit a concrete value from the cluster training config")
    return OmegaConf.merge(upstream, overrides)


def main():
    parser = argparse.ArgumentParser(description="AFS Stage 2: distributed on-policy training")
    parser.add_argument("--config", required=True, help="Path to the AFS Stage 2 YAML configuration")
    parser.add_argument("--output-dir", default=None, help="Override runtime.output_dir")
    parser.add_argument("--no-save", action="store_true")
    parser.add_argument("--disable-logging", action="store_true")
    args = parser.parse_args()
    try:
        from omegaconf import OmegaConf
    except ImportError as exc:
        parser.error(f"omegaconf is required after argument parsing: {exc}")
    config_path = Path(args.config).expanduser().resolve()
    if not config_path.is_file():
        parser.error(f"configuration does not exist: {config_path}")
    config = OmegaConf.load(config_path)
    if not (args.output_dir or config.runtime.output_dir):
        parser.error("runtime.output_dir or --output-dir is required")
    upstream = build_upstream_config(config, args)
    os.makedirs(upstream.logdir, exist_ok=True)
    OmegaConf.save(upstream, Path(upstream.logdir) / "afs_resolved_config.yaml")
    from trainer import AFSTrainer
    AFSTrainer(upstream).train()


if __name__ == "__main__":
    main()
