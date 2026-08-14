#!/usr/bin/env python3
"""Cache official WorldScore dynamic T2V prompts for comparison inference."""

from __future__ import annotations

import argparse
import json
import os
import time
import urllib.request
from pathlib import Path


DATASET_API = (
    "https://datasets-server.huggingface.co/first-rows"
    "?dataset=Howieeeee/WorldScore&config=dynamic&split=train"
)

CAMERA_PROMPTS = {
    "fixed": "fixed",
    "push_in": "push in",
    "pull_out": "pull out",
    "move_left": "move left",
    "move_right": "move right",
    "orbit_left": "orbit left",
    "orbit_right": "orbit right",
    "pan_left": "pan left",
    "pan_right": "pan right",
    "pull_left": "move left, pull out, then pan left",
    "pull_right": "move right, pull out, then pan right",
}


def _worldscore_prompt(row: dict) -> str:
    prompt = str(row["prompt"]).strip()
    style = str(row.get("style", "")).strip()
    camera_path = row.get("camera_path") or ["fixed"]
    camera = str(camera_path[0])
    camera_instruction = CAMERA_PROMPTS.get(camera, camera.replace("_", " "))
    if style:
        prompt = f"{prompt.rstrip('.')}. {style}"
    return f"Camera {camera_instruction}. {prompt}".strip()


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare official WorldScore T2V prompts")
    parser.add_argument("--output-prompt", type=Path, required=True)
    parser.add_argument("--output-metadata", type=Path, required=True)
    parser.add_argument("--row-indices", type=int, nargs="+", default=[0, 1])
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if any(index < 0 for index in args.row_indices):
        raise ValueError("--row-indices cannot contain negative values")
    if len(set(args.row_indices)) != len(args.row_indices):
        raise ValueError("--row-indices must be unique")
    if args.output_prompt.is_file() and args.output_metadata.is_file() and not args.overwrite:
        print(f"Reusing cached WorldScore T2V prompt: {args.output_prompt}")
        return

    payload = None
    last_error = None
    for attempt in range(3):
        try:
            with urllib.request.urlopen(DATASET_API, timeout=120) as response:
                payload = json.load(response)
            break
        except Exception as exc:
            last_error = exc
            if attempt < 2:
                time.sleep(5 * (attempt + 1))
    if payload is None:
        raise RuntimeError("Failed to fetch the official WorldScore T2V prompt") from last_error
    rows = payload.get("rows", [])
    rows_by_index = {int(item["row_idx"]): item["row"] for item in rows}
    records = []
    for row_index in args.row_indices:
        if row_index not in rows_by_index:
            raise IndexError(
                f"WorldScore row {row_index} was not returned by the official first-rows API"
            )
        row = rows_by_index[row_index]
        if row.get("visual_movement") != "dynamic":
            raise ValueError(f"WorldScore row {row_index} is not a dynamic T2V record")
        records.append(
            {
                "row_index": row_index,
                "worldscore_prompt": _worldscore_prompt(row),
                "record": row,
            }
        )

    args.output_prompt.parent.mkdir(parents=True, exist_ok=True)
    prompt_tmp = args.output_prompt.with_suffix(args.output_prompt.suffix + ".tmp")
    metadata_tmp = args.output_metadata.with_suffix(args.output_metadata.suffix + ".tmp")
    prompt_tmp.write_text(
        "\n".join(record["worldscore_prompt"] for record in records) + "\n",
        encoding="utf-8",
    )
    metadata_tmp.write_text(
        json.dumps(
            {
                "dataset": "Howieeeee/WorldScore",
                "config": "dynamic",
                "split": "train",
                "row_indices": args.row_indices,
                "source_api": DATASET_API,
                "samples": records,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    os.replace(prompt_tmp, args.output_prompt)
    os.replace(metadata_tmp, args.output_metadata)
    print(f"Prepared {len(records)} WorldScore dynamic T2V prompts: {args.row_indices}")


if __name__ == "__main__":
    main()
