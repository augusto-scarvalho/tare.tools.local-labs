#!/usr/bin/env python3
"""Verify a pinned Hugging Face weight manifest and emit a JSON receipt."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256_file(path: Path, chunk_size: int = 16 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    rows: list[dict[str, object]] = []
    all_valid = True
    for expected in manifest["files"]:
        path = args.root / expected["path"]
        exists = path.is_file()
        size = path.stat().st_size if exists else None
        digest = sha256_file(path) if exists else None
        valid = (
            exists
            and size == expected["size"]
            and digest == expected["sha256"]
        )
        all_valid = all_valid and valid
        rows.append(
            {
                "path": expected["path"],
                "exists": exists,
                "expected_size": expected["size"],
                "actual_size": size,
                "expected_sha256": expected["sha256"],
                "actual_sha256": digest,
                "valid": valid,
            }
        )

    actual_total = sum(int(row["actual_size"] or 0) for row in rows)
    receipt = {
        "repo": manifest["repo"],
        "revision": manifest["revision"],
        "root": str(args.root.resolve()),
        "expected_total_bytes": manifest["total_bytes"],
        "actual_total_bytes": actual_total,
        "file_count": len(rows),
        "all_valid": all_valid and actual_total == manifest["total_bytes"],
        "files": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: receipt[key] for key in (
        "repo", "revision", "file_count", "expected_total_bytes",
        "actual_total_bytes", "all_valid",
    )}, indent=2))
    return 0 if receipt["all_valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
