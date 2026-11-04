#!/usr/bin/env python3
"""Create an exact Hugging Face LFS manifest for a pinned model revision."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from huggingface_hub import HfApi


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--suffix", action="append", default=[".safetensors"],
        help="Include LFS files ending in this suffix; repeatable.",
    )
    args = parser.parse_args()

    info = HfApi().model_info(args.repo, revision=args.revision, files_metadata=True)
    rows = []
    for sibling in info.siblings or []:
        lfs = sibling.lfs
        if lfs is None or not any(sibling.rfilename.endswith(s) for s in args.suffix):
            continue
        rows.append({
            "path": sibling.rfilename,
            "size": int(sibling.size),
            "sha256": lfs.sha256,
        })
    rows.sort(key=lambda row: row["path"])
    payload = {
        "repo": args.repo,
        "revision": info.sha,
        "requested_revision": args.revision,
        "license": (info.card_data or {}).get("license"),
        "suffixes": args.suffix,
        "file_count": len(rows),
        "total_bytes": sum(row["size"] for row in rows),
        "files": rows,
    }
    if info.sha != args.revision:
        raise SystemExit(f"revision mismatch: expected {args.revision}, got {info.sha}")
    if not rows:
        raise SystemExit("no matching LFS files")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: payload[key] for key in (
        "repo", "revision", "license", "file_count", "total_bytes",
    )}, indent=2))


if __name__ == "__main__":
    main()
