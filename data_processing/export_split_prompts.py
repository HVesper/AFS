#!/usr/bin/env python3
"""Export captions for one manifest split as an inference prompt file."""

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--split", default="eval")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    prompts = []
    with args.manifest.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            if record.get("split", "train") != args.split:
                continue
            prompt = " ".join(str(record["prompt"]).split()).strip()
            if not prompt:
                raise ValueError(f"Empty prompt for {record.get('id', '<unknown>')}")
            prompts.append(prompt)

    if not prompts:
        raise ValueError(f"No {args.split!r} prompts found in {args.manifest}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text("\n".join(prompts) + "\n", encoding="utf-8")
    temporary.replace(args.output)
    print(f"Exported {len(prompts)} {args.split} prompts to {args.output}")


if __name__ == "__main__":
    main()
