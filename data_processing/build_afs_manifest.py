#!/usr/bin/env python3
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Point AFS at normalized dataset preview videos")
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--processed-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    records = []
    with args.source_manifest.open("r", encoding="utf-8") as handle:
        for line in handle:
            source = json.loads(line)
            sample_id = str(source["id"])
            video_path = (args.processed_root / "processed_video" / f"{sample_id}_video.mp4").resolve()
            if not video_path.is_file():
                raise FileNotFoundError(video_path)
            records.append({
                "sample_id": sample_id,
                "video_path": str(video_path),
                "global_caption": str(source["prompt"]),
                "split": source.get("split", "train"),
            })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    temporary.replace(args.output)


if __name__ == "__main__":
    main()
