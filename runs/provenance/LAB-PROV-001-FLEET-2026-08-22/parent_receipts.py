#!/usr/bin/env python3
"""Verify the original Hugging Face shard receipts for the authorial merge parents."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


PARENTS = {
    "base": {
        "repo": "Qwen/Qwen3.6-27B",
        "revision": "6a9e13bd6fc8f0983b9b99948120bc37f49c13e9",
    },
    "tc": {
        "repo": "bottlecapai/ThinkingCap-Qwen3.6-27B",
        "revision": "0c5557fdf61f7485bbf8395144cbb7fb775ff344",
    },
    "fable": {
        "repo": "DavidAU/Qwen3.6-27B-Fable-Fusion-711-Uncensored-Heretic-NM-DAU-MTP",
        "revision": "b7676ecef7d1adcabfdc1b42a389f8643c7723fb",
    },
}


def canonical_digest(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def main() -> int:
    result = {"schema": 1, "parents": {}}
    aggregate_rows = []
    for name, identity in PARENTS.items():
        root = Path("/home/augus/models/fp16") / name
        revision = identity["revision"]
        tree_path = root / ".cache/huggingface/trees" / f"{revision}.json"
        tree = json.loads(tree_path.read_text(encoding="utf-8"))
        shards = []
        for path in sorted(root.glob("*.safetensors")):
            entry = tree["files"].get(path.name)
            if not entry or not entry.get("lfs_sha256"):
                raise RuntimeError(f"missing LFS receipt for {path}")
            metadata_path = root / ".cache/huggingface/download" / f"{path.name}.metadata"
            metadata = metadata_path.read_text(encoding="utf-8").splitlines()
            if metadata[0] != revision or metadata[1] != entry["lfs_sha256"]:
                raise RuntimeError(f"metadata/tree mismatch for {path}")
            if path.stat().st_size != entry["lfs_size"]:
                raise RuntimeError(f"local size/tree mismatch for {path}")
            row = {
                "file": path.name,
                "bytes": entry["lfs_size"],
                "lfs_sha256": entry["lfs_sha256"],
            }
            shards.append(row)
            aggregate_rows.append({"parent": name, **identity, **row})
        if not shards:
            raise RuntimeError(f"no local safetensor shards found for {name}")
        result["parents"][name] = {
            **identity,
            "tree_receipt": str(tree_path),
            "shards": shards,
            "shard_count": len(shards),
            "total_bytes": sum(row["bytes"] for row in shards),
            "weight_manifest_sha256": canonical_digest(shards),
        }
    result["aggregate_weight_manifest_sha256"] = canonical_digest(aggregate_rows)
    output = Path(__file__).with_name("parent_receipts.json")
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({
        name: {
            "revision": value["revision"],
            "shards": value["shard_count"],
            "bytes": value["total_bytes"],
            "weight_manifest_sha256": value["weight_manifest_sha256"],
        }
        for name, value in result["parents"].items()
    }, indent=2))
    print(f"aggregate_weight_manifest_sha256={result['aggregate_weight_manifest_sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
