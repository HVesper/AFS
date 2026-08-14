#!/usr/bin/env python3
"""Download selected Sekai-Real-Walking-HQ clips and build an AFS manifest."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
from pathlib import Path


def _clip_parts(filename: str) -> tuple[str, int, int]:
    stem = Path(filename).stem
    try:
        video_id, start_frame, end_frame = stem.rsplit("_", 2)
        return video_id, int(start_frame), int(end_frame)
    except (ValueError, TypeError) as exc:
        raise ValueError(f"Invalid Sekai clip filename: {filename}") from exc


def _existing_clip(video_dir: Path, clip_stem: str) -> Path | None:
    video_suffixes = {".mp4", ".mkv", ".webm", ".mov"}
    matches = sorted(
        path
        for path in video_dir.glob(f"{clip_stem}.*")
        if path.is_file() and path.suffix.lower() in video_suffixes
    )
    return matches[0].resolve() if matches else None


def _download_clip(row: dict[str, str], video_dir: Path, source_fps: float) -> Path:
    filename = row["videoFile"].strip()
    video_id, start_frame, end_frame = _clip_parts(filename)
    clip_stem = Path(filename).stem
    existing = _existing_clip(video_dir, clip_stem)
    if existing is not None:
        return existing

    start_seconds = start_frame / source_fps
    end_seconds = end_frame / source_fps
    output_template = str(video_dir / f"{clip_stem}.%(ext)s")
    command = [
        "yt-dlp",
        "--no-playlist",
        "--no-overwrites",
        "--download-sections",
        f"*{start_seconds:.6f}-{end_seconds:.6f}",
        "--force-keyframes-at-cuts",
        "--merge-output-format",
        "mp4",
        "--retries",
        "5",
        "-f",
        "299/bestvideo[height>=720][height<=1080][fps>=50]/"
        "bestvideo[height>=720][height<=1080]/bestvideo[height<=1080]/bestvideo/best",
        "-o",
        output_template,
        f"https://www.youtube.com/watch?v={video_id}",
    ]
    print(f"Downloading Sekai clip {clip_stem} ({start_seconds:.2f}s-{end_seconds:.2f}s)", flush=True)
    subprocess.run(command, check=True)
    downloaded = _existing_clip(video_dir, clip_stem)
    if downloaded is None:
        raise RuntimeError(f"yt-dlp completed without producing {clip_stem}")
    return downloaded


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download Sekai-Real-Walking-HQ clips and build an AFS source manifest"
    )
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--video-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--eval-samples", type=int, default=2)
    parser.add_argument("--source-fps", type=float, default=60.0)
    parser.add_argument(
        "--max-download-failures",
        type=int,
        default=32,
        help="Abort after this many unavailable/private YouTube clips.",
    )
    args = parser.parse_args()

    if shutil.which("yt-dlp") is None:
        raise RuntimeError("yt-dlp is required to download Sekai source clips")
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg is required for precise Sekai clip extraction")
    if args.source_fps <= 0:
        raise ValueError("--source-fps must be positive")
    if args.max_samples < 0:
        raise ValueError("--max-samples cannot be negative")
    if args.eval_samples < 1:
        raise ValueError("--eval-samples must be at least 1")
    if args.max_download_failures < 1:
        raise ValueError("--max-download-failures must be at least 1")

    args.video_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, str]] = []
    failures = 0
    with args.csv.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"videoFile", "caption"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"Sekai CSV must contain {sorted(required)}; found {reader.fieldnames}")
        for row in reader:
            filename = row["videoFile"].strip()
            caption = row["caption"].strip()
            if not filename or not caption:
                continue
            try:
                video_path = _download_clip(row, args.video_dir, args.source_fps)
            except (subprocess.CalledProcessError, RuntimeError) as exc:
                failures += 1
                print(f"Warning: skipping unavailable Sekai clip {filename}: {exc}", flush=True)
                if failures >= args.max_download_failures:
                    raise RuntimeError(
                        f"Aborting after {failures} Sekai download failures; YouTube may be blocked"
                    ) from exc
                continue

            record = {
                "id": Path(filename).stem,
                "video": str(video_path),
                "prompt": caption,
                "location": (row.get("location") or "").strip(),
                "scene": (row.get("scene") or "").strip(),
                "crowd_density": (row.get("crowdDensity") or "").strip(),
                "weather": (row.get("weather") or "").strip(),
                "time_of_day": (row.get("timeOfDay") or "").strip(),
            }
            records.append(record)
            if args.max_samples > 0 and len(records) >= args.max_samples:
                break

    if not records:
        raise RuntimeError("No Sekai-Real-Walking-HQ clips could be prepared")
    if args.max_samples > 0 and len(records) < args.max_samples:
        raise RuntimeError(f"Requested {args.max_samples} samples but prepared only {len(records)}")

    eval_count = min(max(0, args.eval_samples), max(0, len(records) - 1))
    split_index = len(records) - eval_count
    for index, record in enumerate(records):
        record["split"] = "eval" if index >= split_index else "train"

    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    temporary.replace(args.output)
    print(
        f"Prepared {split_index} train and {eval_count} eval Sekai samples "
        f"({failures} unavailable clips skipped): {args.output}"
    )


if __name__ == "__main__":
    main()
