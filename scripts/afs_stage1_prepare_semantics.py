#!/usr/bin/env python3
import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="AFS Stage 1: local chunk semantic preprocessing")
    parser.add_argument("--config", required=True, help="Path to the Stage 1 YAML configuration")
    args = parser.parse_args()
    try:
        from omegaconf import OmegaConf
    except ImportError as exc:
        parser.error(f"omegaconf is required after argument parsing: {exc}")
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from afs.stage1_semantics import AFSStage1SemanticPreprocessor
    config_path = Path(args.config).expanduser().resolve()
    if not config_path.is_file():
        parser.error(f"configuration does not exist: {config_path}")
    config = OmegaConf.load(config_path)
    if config.method.name != "afs" or config.stage.name != "semantic_preprocessing":
        parser.error("configuration must select method=afs and stage=semantic_preprocessing")
    AFSStage1SemanticPreprocessor(config).run()


if __name__ == "__main__":
    main()
