#!/usr/bin/env python3
"""Inventory local model artifacts without guessing provenance from filenames.

Run inside WSL. Hugging Face download metadata supplies a pinned revision and expected
LFS digest, but those are kept distinct from a locally recomputed content hash. Source
repositories are populated only from explicit project receipts listed below.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
from datetime import datetime, timezone

from gguf_meta import read_meta


MODELS_ROOT = pathlib.Path("/home/augus/models")
REPO_ROOT = pathlib.Path("/mnt/c/projects/tare.tools.local-labs")

# Every mapping has an explicit workspace receipt or an exact Hub revision/digest check;
# an attractive directory name is not evidence.
SOURCE_ROOTS = {
    "/home/augus/models/qwen38-27b/unsloth": "unsloth/Qwen3.8-27B-GGUF",
    "/home/augus/models/qwen38-27b/unsloth-4ca72078": "unsloth/Qwen3.8-27B-GGUF",
    "/home/augus/models/qwen38-27b/bartowski": "bartowski/Qwen3.8-27B-GGUF",
    "/home/augus/models/qwen38-27b/cold-fusion-27a5cb2c":
        "DavidAU/Qwen3.8-27B-Cold-Fusion-GAIN-V1.1-NM-DAU-NEO-MAX-MTP-GGUF",
    "/home/augus/models/muse-glimmer-30b/meta-70bf1b61": "meta-models/Muse-Glimmer-30B-GGUF",
    "/home/augus/models/falcon-h1r-7b": "tiiuae/Falcon-H1R-7B-GGUF",
    "/home/augus/models/fable-fusion-711":
        "DavidAU/Qwen3.6-27B-Fable-Fusion-711-Uncensored-Heretic-NM-DAU-NEO-MAX-MTP-GGUF",
    "/home/augus/models/qwen36-27b-dense": "bartowski/Qwen_Qwen3.6-27B-GGUF",
    "/home/augus/models/qwen36-27b-mtp": "unsloth/Qwen3.6-27B-MTP-GGUF",
    "/home/augus/models/qwen36-35b-a3b": "unsloth/Qwen3.6-35B-A3B-GGUF",
    "/home/augus/models/qwen36-35b-a3b-mtp": "unsloth/Qwen3.6-35B-A3B-MTP-GGUF",
    "/home/augus/models/thinkingcap-27b":
        "bartowski/bottlecapai_ThinkingCap-Qwen3.6-27B-GGUF",
    "/home/augus/models/rwkv7-1.5b/official-d2d414f": "RWKV/RWKV7-1.5B-20260805",
    "/home/augus/models/fp16/base": "Qwen/Qwen3.6-27B",
    "/home/augus/models/fp16/tc": "bottlecapai/ThinkingCap-Qwen3.6-27B",
    "/home/augus/models/fp16/fable":
        "DavidAU/Qwen3.6-27B-Fable-Fusion-711-Uncensored-Heretic-NM-DAU-MTP",
    "/home/augus/models/thinkingcap-27b-mtp": "protoLabsAI/ThinkingCap-Qwen3.6-27B-MTP-GGUF",
    "/home/augus/models/gemma-4-12b-vision/gemma-4-12B-it-qat-assistant-MTP-Q8_0.gguf":
        "Janvitos/gemma-4-12B-it-qat-assistant-MTP-Q8_0-GGUF",
    "/home/augus/models/mistral-small-24b-heretic":
        "mradermacher/Mistral-Small-3.2-24B-Instruct-2506-Heretic-v1.2-2-i1-GGUF",
    "/home/augus/models/gemma4-26b-heretic":
        "mradermacher/Gemma-4-26B-A4B-it-heretic-antislop-i1-GGUF",
    "/home/augus/models/thinkingcap-lora":
        "signsur4739379373/Qwen3.6-27B-ThinkingCap-LoRA",
    "/home/augus/models/ornith-1.5-35b-a3b-bartowski":
        "bartowski/Ornith-1.5-35B-A3B-GGUF",
}

# Exact Hub file receipts collected from the repository API/file viewer. These fill
# gaps left by downloads that pre-date the local Hugging Face metadata cache. They
# do not count as a local content verification until --hash recomputes the file.
EXTERNAL_RECEIPTS = {
    "/home/augus/models/mistral-small-24b-heretic/Mistral-Small-3.2-24B-Instruct-2506-Heretic-v1.2-2.i1-Q4_K_M.gguf": {
        "source_repo": "mradermacher/Mistral-Small-3.2-24B-Instruct-2506-Heretic-v1.2-2-i1-GGUF",
        "source_revision": "87199b98e64c6cd63c0814600ad348f495f5e9f4",
        "expected_sha256": "5079999cca0823bbbddadbf905564311930901b155919386933e5143623da7cf",
        "size": 14333923776,
        "source_url": "https://huggingface.co/mradermacher/Mistral-Small-3.2-24B-Instruct-2506-Heretic-v1.2-2-i1-GGUF/blob/87199b98e64c6cd63c0814600ad348f495f5e9f4/Mistral-Small-3.2-24B-Instruct-2506-Heretic-v1.2-2.i1-Q4_K_M.gguf",
    },
    "/home/augus/models/gemma4-26b-heretic/Gemma-4-26B-A4B-it-heretic-antislop.i1-Q4_K_M.gguf": {
        "source_repo": "mradermacher/Gemma-4-26B-A4B-it-heretic-antislop-i1-GGUF",
        "source_revision": "84775f5b3e286fe1b95251cd6ee79a08a69e1254",
        "expected_sha256": "13cfcadee358e54c3246ecf9b8a528633d1d4444e17177cdaadeec54955eb5ae",
        "size": 16796015904,
        "source_url": "https://huggingface.co/mradermacher/Gemma-4-26B-A4B-it-heretic-antislop-i1-GGUF/blob/84775f5b3e286fe1b95251cd6ee79a08a69e1254/Gemma-4-26B-A4B-it-heretic-antislop.i1-Q4_K_M.gguf",
    },
    "/home/augus/models/thinkingcap-lora/qwen36-27b-thinkingcap-lora-rank64.gguf": {
        "source_repo": "signsur4739379373/Qwen3.6-27B-ThinkingCap-LoRA",
        "source_revision": "b60237efa4cafd0cc5b6426c494de2a3b4336e26",
        "expected_sha256": "59368a0c7127ba58c2947a59f33a14be7a59535943ef2224e6548ab570a3b836",
        "size": 141038144,
        "source_url": "https://huggingface.co/signsur4739379373/Qwen3.6-27B-ThinkingCap-LoRA/blob/b60237efa4cafd0cc5b6426c494de2a3b4336e26/qwen36-27b-thinkingcap-lora-rank64.gguf",
    },
    "/home/augus/models/thinkingcap-27b-mtp/ThinkingCap-Qwen3.6-27B-Q4_K_M-MTP.gguf": {
        "source_repo": "protoLabsAI/ThinkingCap-Qwen3.6-27B-MTP-GGUF",
        "source_revision": "f015d8b219c68de4a9554832842675afc08ae577",
        "expected_sha256": "0ba445d2d0ca3ec32f429d83701b42f2ea828c934fc6378b836ffaf1b0760c75",
        "size": 16810713408,
        "source_url": "https://huggingface.co/protoLabsAI/ThinkingCap-Qwen3.6-27B-MTP-GGUF/blob/f015d8b219c68de4a9554832842675afc08ae577/ThinkingCap-Qwen3.6-27B-Q4_K_M-MTP.gguf",
    },
    "/home/augus/models/gemma-4-12b-vision/gemma-4-12B-it-qat-assistant-MTP-Q8_0.gguf": {
        "source_repo": "Janvitos/gemma-4-12B-it-qat-assistant-MTP-Q8_0-GGUF",
        "source_revision": "7a977bf4406b1c60d29ad602c6be3d9da05ebae2",
        "expected_sha256": "13331068b6af643c3dc75e619373b674c1f75a1958e7c82e2020d96a17c63809",
        "size": 465127040,
        "source_url": "https://huggingface.co/Janvitos/gemma-4-12B-it-qat-assistant-MTP-Q8_0-GGUF/blob/7a977bf4406b1c60d29ad602c6be3d9da05ebae2/gemma-4-12B-it-qat-assistant-MTP-Q8_0.gguf",
    },
    "/home/augus/models/gemma-4-26b-a4b/gemma-4-26B_q4_0-it.gguf": {
        "source_repo": "google/gemma-4-26B-A4B-it-qat-q4_0-gguf",
        "source_revision": "d1c082be9cf3c8a514acf63b8761f4b41935842e",
        "expected_sha256": "3eca3b8f6d7baf218a7dd6bba5fb59a56ee25fe2d567b6f5f589b4f697eca51d",
        "size": 14439363584,
        "source_url": "https://huggingface.co/google/gemma-4-26B-A4B-it-qat-q4_0-gguf/blob/d1c082be9cf3c8a514acf63b8761f4b41935842e/gemma-4-26B_q4_0-it.gguf",
    },
    "/home/augus/models/ornith-1.5-35b-a3b-bartowski/Ornith-1.5-35B-A3B-IQ4_XS.gguf": {
        "source_repo": "bartowski/Ornith-1.5-35B-A3B-GGUF",
        "source_revision": "64b0493d34a5ca4c1b4ad67bb99b41d74b4f07d6",
        "expected_sha256": "d6aef57fa948e9bba3ca4959b3c237ed898c605471f48c73a32cedbd24aabe70",
        "size": 19278554784,
        "source_url": "https://huggingface.co/bartowski/Ornith-1.5-35B-A3B-GGUF/blob/64b0493d34a5ca4c1b4ad67bb99b41d74b4f07d6/Ornith-1.5-35B-A3B-IQ4_XS.gguf",
    },
    "/home/augus/models/gemma-4-12b-vision/gemma-4-12B-it-Q4_0.gguf": {
        "source_repo": "ggml-org/gemma-4-12B-it-GGUF",
        "source_revision": "f6de384f112605b5b635b1a987ee6f842f81919f",
        "expected_sha256": "3712b9bd32cae83a22f67ee7a4466d8d7a4f21646ac8a07d19bf9418e8767a70",
        "size": 7219673216,
        "source_url": "https://huggingface.co/ggml-org/gemma-4-12B-it-GGUF/blob/f6de384f112605b5b635b1a987ee6f842f81919f/gemma-4-12B-it-Q4_0.gguf",
    },
    "/home/augus/models/gemma-4-12b-vision/mmproj-gemma-4-12B-it-Q8_0.gguf": {
        "source_repo": "ggml-org/gemma-4-12B-it-GGUF",
        "source_revision": "f6de384f112605b5b635b1a987ee6f842f81919f",
        "expected_sha256": "59e62255435dda870e2d1de97cc031330b31a898bac12b38a182cecff9cd3738",
        "size": 158987616,
        "source_url": "https://huggingface.co/ggml-org/gemma-4-12B-it-GGUF/blob/f6de384f112605b5b635b1a987ee6f842f81919f/mmproj-gemma-4-12B-it-Q8_0.gguf",
    },
    "/home/augus/models/embedding/nomic-embed-text-v1.5.Q8_0.gguf": {
        "source_repo": "nomic-ai/nomic-embed-text-v1.5-GGUF",
        "source_revision": "18d1044f4866e224159fce8c6fc5c4f3920176e7",
        "expected_sha256": "3e24342164b3d94991ba9692fdc0dd08e3fd7362e0aacc396a9a5c54a544c3b7",
        "size": 146146432,
        "source_url": "https://huggingface.co/nomic-ai/nomic-embed-text-v1.5-GGUF/blob/18d1044f4866e224159fce8c6fc5c4f3920176e7/nomic-embed-text-v1.5.Q8_0.gguf",
    },
    "/home/augus/models/gpt-oss-20b/gpt-oss-20b-Q4_K_M.gguf": {
        "source_repo": "unsloth/gpt-oss-20b-GGUF",
        "source_revision": "ce6ba6163271f5d73dbe2a20b85e66d79126e942",
        "expected_sha256": "c27536640e410032865dc68781d80a08b98f8db5e93575919af8ccc0568aeb4f",
        "size": 11624759488,
        "source_url": "https://huggingface.co/unsloth/gpt-oss-20b-GGUF/blob/ce6ba6163271f5d73dbe2a20b85e66d79126e942/gpt-oss-20b-Q4_K_M.gguf",
    },
}

LOCAL_DERIVATIONS = {
    "/home/augus/models/merges/fable-tc-l1.0-Q4_K_M.gguf": {
        "kind": "authorial_full_rank_task_arithmetic_merge_then_gguf_quantization",
        "formula": "W = Fable + 1.0 * (ThinkingCap - Qwen3.6-27B base)",
        "receipt": "docs/campaigns/a2-ablation-merging/A2_STAGE1_CONCISE_FABLE.md",
        "parent_content_digests":
            "runs/provenance/LAB-PROV-001-FLEET-2026-08-22/parent_receipts.json",
        "parent_weight_manifest_sha256":
            "ee9d33a75716bcec5edd3c69bfdd3cc5d431d7a4a745098228566463eb5eec25",
        "quantizer_revision": "068764d927ecd6d39665a46d31b1ee533eedabe7",
        "quantizer_binary_sha256":
            "279e1a5e934fb16b558eeda6d183829a4fc214c9ee8a63fa5e062bf71cd0b4d6",
        "converter_sha256":
            "8f1bed9466221e57e434caa7ee720abe1569deb6bc2fe5a65da950ea66c8e737",
        "imatrix": "none",
    }
}

# Previously recomputed full hashes retained in the first LAB-PROV-001 result. These are
# explicit receipts, not values inferred from filenames or upstream metadata.
RECORDED_LOCAL_HASHES = {
    "/home/augus/models/embedding/nomic-embed-text-v1.5.Q8_0.gguf": {
        "sha256": "3e24342164b3d94991ba9692fdc0dd08e3fd7362e0aacc396a9a5c54a544c3b7",
        "receipt": "runs/provenance/LAB-PROV-001-FLEET-2026-08-22/RESULT.md",
    },
    "/home/augus/models/gemma-4-12b-vision/mmproj-gemma-4-12B-it-Q8_0.gguf": {
        "sha256": "59e62255435dda870e2d1de97cc031330b31a898bac12b38a182cecff9cd3738",
        "receipt": "runs/provenance/LAB-PROV-001-FLEET-2026-08-22/RESULT.md",
    },
    "/home/augus/models/gemma-4-12b-vision/gemma-4-12B-it-qat-assistant-MTP-Q8_0.gguf": {
        "sha256": "13331068b6af643c3dc75e619373b674c1f75a1958e7c82e2020d96a17c63809",
        "receipt": "runs/provenance/LAB-PROV-001-FLEET-2026-08-22/RESULT.md",
    },
    "/home/augus/models/thinkingcap-lora/qwen36-27b-thinkingcap-lora-rank64.gguf": {
        "sha256": "59368a0c7127ba58c2947a59f33a14be7a59535943ef2224e6548ab570a3b836",
        "receipt": "runs/provenance/LAB-PROV-001-FLEET-2026-08-22/RESULT.md",
    },
    "/home/augus/models/merges/fable-tc-l1.0-Q4_K_M.gguf": {
        "sha256": "052c08ca13d75d8d88c9cc3f201d7bfa9167e2a1e69ad3e1e1f26ff73c1b390b",
        "receipt": "runs/provenance/LAB-PROV-001-FLEET-2026-08-22/RESULT.md",
    },
    "/home/augus/models/qwen38-27b/unsloth/Qwen3.8-27B-UD-Q4_K_XL.gguf": {
        "sha256": "bee238bbeb3dc0a34bde4d0dedbaee1f98c009e8bb4226f03070054c12fb1372",
        "receipt": "runs/provenance/LAB-PROV-001-FLEET-2026-08-22/inventory.json",
    },
    "/home/augus/models/mistral-small-24b-heretic/Mistral-Small-3.2-24B-Instruct-2506-Heretic-v1.2-2.i1-Q4_K_M.gguf": {
        "sha256": "5079999cca0823bbbddadbf905564311930901b155919386933e5143623da7cf",
        "receipt": "runs/requalification/MISTRAL-SMALL-24B-HERETIC-2026-08-22/RESULT.md",
    },
    "/home/augus/models/gemma4-26b-heretic/Gemma-4-26B-A4B-it-heretic-antislop.i1-Q4_K_M.gguf": {
        "sha256": "13cfcadee358e54c3246ecf9b8a528633d1d4444e17177cdaadeec54955eb5ae",
        "receipt": "runs/requalification/GEMMA4-26B-HERETIC-2026-08-22/RESULT.md",
    },
    "/home/augus/models/gpt-oss-20b/gpt-oss-20b-Q4_K_M.gguf": {
        "sha256": "c27536640e410032865dc68781d80a08b98f8db5e93575919af8ccc0568aeb4f",
        "receipt": "runs/requalification/GPT-OSS-20B-2026-08-22/RESULT.md",
    },
    "/home/augus/models/gemma-4-26b-a4b/gemma-4-26B_q4_0-it.gguf": {
        "sha256": "3eca3b8f6d7baf218a7dd6bba5fb59a56ee25fe2d567b6f5f589b4f697eca51d",
        "receipt": "runs/requalification/GEMMA4-26B-OFFICIAL-2026-08-22/RESULT.md",
    },
    "/home/augus/models/ornith-1.5-35b-a3b-bartowski/Ornith-1.5-35B-A3B-IQ4_XS.gguf": {
        "sha256": "d6aef57fa948e9bba3ca4959b3c237ed898c605471f48c73a32cedbd24aabe70",
        "receipt": "runs/requalification/ORNITH-1.5-35B-A3B-2026-08-22/RESULT.md",
    },
}


def sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def explicit_source(path: pathlib.Path) -> str:
    value = str(path)
    matches = [(root, repo) for root, repo in SOURCE_ROOTS.items()
               if value == root or value.startswith(root.rstrip("/") + "/")]
    return max(matches, key=lambda item: len(item[0]))[1] if matches else "UNKNOWN"


def hf_receipt(path: pathlib.Path) -> dict:
    for base in (path.parent, *path.parents):
        if base == MODELS_ROOT.parent:
            break
        cache = base / ".cache" / "huggingface"
        if not cache.is_dir():
            continue
        try:
            relative = path.relative_to(base)
        except ValueError:
            continue
        metadata = cache / "download" / pathlib.Path(str(relative) + ".metadata")
        if not metadata.is_file():
            continue
        lines = metadata.read_text(encoding="utf-8", errors="replace").splitlines()
        revision = lines[0].strip() if lines else "UNKNOWN"
        expected = lines[1].strip() if len(lines) > 1 else "UNKNOWN"
        tree_path = cache / "trees" / f"{revision}.json"
        tree_entry = None
        if tree_path.is_file():
            tree = json.loads(tree_path.read_text(encoding="utf-8"))
            tree_entry = tree.get("files", {}).get(relative.as_posix())
        return {
            "metadata_path": str(metadata), "download_root": str(base),
            "source_revision": revision, "expected_sha256": expected,
            "tree_path": str(tree_path) if tree_path.is_file() else None,
            "tree_entry": tree_entry,
        }
    external = EXTERNAL_RECEIPTS.get(str(path))
    if external:
        return {"metadata_path": None, "download_root": None,
                "source_revision": external["source_revision"],
                "expected_sha256": external["expected_sha256"],
                "tree_path": None,
                "tree_entry": {"lfs_size": external["size"],
                               "lfs_sha256": external["expected_sha256"]},
                "receipt_type": "exact_hub_file_receipt",
                "source_url": external["source_url"]}
    return {"metadata_path": None, "download_root": None, "source_revision": "UNKNOWN",
            "expected_sha256": "UNKNOWN", "tree_path": None, "tree_entry": None,
            "receipt_type": None, "source_url": None}


def identity_index() -> dict[str, dict]:
    result: dict[str, dict] = {path: dict(receipt) for path, receipt in RECORDED_LOCAL_HASHES.items()}
    # Fleet-wide recomputation receipts are immutable evidence just like benchmark
    # identity sidecars. Loading them keeps the expensive full-file reads durable
    # across later inventory refreshes without treating an upstream mismatch as a
    # successful pin.
    for receipt_path in (REPO_ROOT / "runs" / "provenance").glob(
            "**/inventory-full-hash.json"):
        try:
            report = json.loads(receipt_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        for artifact in report.get("artifacts", []):
            model_path = artifact.get("path")
            model_hash = artifact.get("local_sha256")
            if model_path and model_hash:
                result[model_path] = {
                    "sha256": model_hash,
                    "receipt": str(receipt_path),
                    "source_repo": artifact.get("source_repo", "UNKNOWN"),
                    "source_revision": artifact.get("source_revision", "UNKNOWN"),
                }
    for path in (REPO_ROOT / "runs").rglob("identity.json"):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        model_path = data.get("model_path")
        model_hash = data.get("model_sha256")
        if model_path and model_hash and model_hash != "UNKNOWN":
            result.setdefault(model_path, {
                "sha256": model_hash,
                "receipt": str(path),
                "source_repo": data.get("source_repo", "UNKNOWN"),
                "source_revision": data.get("source_revision", "UNKNOWN"),
            })
    return result


def inspect(path: pathlib.Path, hashes: set[str], identities: dict[str, dict]) -> dict:
    receipt = hf_receipt(path)
    source_repo = explicit_source(path)
    external = EXTERNAL_RECEIPTS.get(str(path))
    derivation = LOCAL_DERIVATIONS.get(str(path))
    if source_repo == "UNKNOWN" and external:
        source_repo = external["source_repo"]
    identity = identities.get(str(path))
    local_hash = None
    hash_receipt = None
    if str(path) in hashes:
        local_hash = sha256(path)
        hash_receipt = "recomputed_by_fleet_provenance"
    elif identity:
        local_hash = identity["sha256"]
        hash_receipt = identity["receipt"]
        if source_repo == "UNKNOWN" and identity.get("source_repo", "UNKNOWN") != "UNKNOWN":
            source_repo = identity["source_repo"]
    size = path.stat().st_size
    tree = receipt["tree_entry"] or {}
    tree_size = tree.get("lfs_size", tree.get("size"))
    tree_sha = tree.get("lfs_sha256")
    expected = receipt["expected_sha256"]
    digest_matches = bool(local_hash and expected != "UNKNOWN" and local_hash == expected)
    size_matches = bool(tree_size is not None and tree_size == size)
    tree_digest_matches = bool(tree_sha and expected != "UNKNOWN" and tree_sha == expected)
    try:
        gguf = read_meta(str(path))
        imatrix = {key: value for key, value in gguf.items()
                   if key.startswith("quantize.imatrix.")}
        quantization = {
            "version": gguf.get("general.quantization_version"),
            "file_type": gguf.get("general.file_type"),
            "quantizer_revision": (derivation or {}).get("quantizer_revision", "UNKNOWN"),
            "imatrix_metadata_present": bool(imatrix),
            "imatrix": imatrix,
        }
    except Exception as exc:  # preserve the artifact row and expose parser failure
        quantization = {"metadata_error": f"{type(exc).__name__}: {exc}",
                        "quantizer_revision": "UNKNOWN",
                        "imatrix_metadata_present": None, "imatrix": {}}
    if (source_repo != "UNKNOWN" and receipt["source_revision"] != "UNKNOWN"
            and digest_matches):
        status = "FULLY_PINNED"
    elif (source_repo != "UNKNOWN" and receipt["source_revision"] != "UNKNOWN"
          and local_hash and expected != "UNKNOWN"):
        status = "REVISION_PINNED_LOCAL_DIGEST_MISMATCH"
    elif source_repo != "UNKNOWN" and receipt["source_revision"] != "UNKNOWN" and size_matches:
        status = "REVISION_DIGEST_SIZE_PINNED_LOCAL_HASH_PENDING"
    elif source_repo != "UNKNOWN" and receipt["source_revision"] != "UNKNOWN":
        status = "REVISION_DIGEST_METADATA_PINNED_LOCAL_HASH_PENDING"
    elif receipt["source_revision"] != "UNKNOWN" and size_matches:
        status = "REVISION_DIGEST_SIZE_NO_REPO"
    elif receipt["source_revision"] != "UNKNOWN":
        status = "REVISION_DIGEST_METADATA_NO_REPO"
    elif derivation and local_hash:
        status = "LOCAL_DERIVATION_CONTENT_PINNED"
    elif local_hash:
        status = "LOCAL_CONTENT_ONLY"
    else:
        status = "UNKNOWN"
    return {
        "path": str(path), "bytes": size, "source_repo": source_repo,
        "source_revision": receipt["source_revision"], "expected_sha256": expected,
        "local_sha256": local_hash, "local_hash_receipt": hash_receipt,
        "metadata_path": receipt["metadata_path"], "tree_path": receipt["tree_path"],
        "receipt_type": ("local_huggingface_metadata+exact_hub_file_receipt"
                         if receipt["metadata_path"] and external else
                         receipt.get("receipt_type") or "local_huggingface_metadata"),
        "source_url": external.get("source_url") if external else None,
        "local_derivation": derivation,
        "tree_size_matches": size_matches, "tree_digest_matches_metadata": tree_digest_matches,
        "local_digest_matches_metadata": digest_matches, "status": status,
        "quantization": quantization,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--hash", action="append", default=[], metavar="WSL_PATH")
    parser.add_argument("--max-depth", type=int, default=4)
    args = parser.parse_args()
    identities = identity_index()
    artifacts = []
    for path in sorted(MODELS_ROOT.rglob("*.gguf")):
        if len(path.relative_to(MODELS_ROOT).parts) > args.max_depth:
            continue
        artifacts.append(inspect(path, set(args.hash), identities))
    counts: dict[str, int] = {}
    for item in artifacts:
        counts[item["status"]] = counts.get(item["status"], 0) + 1
    report = {"campaign": "LAB-PROV-001", "timestamp": datetime.now(timezone.utc).isoformat(),
              "boundary": "GGUF files under /home/augus/models at depth <= 4",
              "classification_counts": counts, "artifacts": artifacts}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(counts, indent=2))
    print(f"artifacts={len(artifacts)} evidence={args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
