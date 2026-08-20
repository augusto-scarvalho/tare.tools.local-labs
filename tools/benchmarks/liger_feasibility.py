#!/usr/bin/env python3
"""Fail-closed provenance and state-transfer gate for the pinned Liger source."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback
from typing import Any


SCHEMA = "liger-feasibility-v1"
EXPECTED_REVISIONS = {
    ".": "0b364eb81d2159cc0fd9818b95d2d07d75522043",
    "third_party/flash-linear-attention": "72aa949f27dba47767f13226c45de29600d77312",
    "third_party/lm-evaluation-harness": "1ba35e623b9bd9ca48df926f1a028043e159a6f2",
}
VERSION_PACKAGES = [
    "torch",
    "triton",
    "transformers",
    "datasets",
    "peft",
    "evaluate",
    "omegaconf",
    "einops",
    "flash-attn",
    "flash-linear-attention",
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git(path: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(path), *args],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return result.stdout.strip()


def provenance(source_root: Path) -> dict[str, Any]:
    repos: dict[str, Any] = {}
    for relative, expected in EXPECTED_REVISIONS.items():
        path = source_root if relative == "." else source_root / relative
        actual = git(path, "rev-parse", "HEAD")
        status = git(path, "status", "--porcelain")
        repos[relative] = {
            "path": str(path),
            "expected_head": expected,
            "actual_head": actual,
            "head_matches": actual == expected,
            "clean": status == "",
            "status": status,
        }
    return {
        "repositories": repos,
        "pass": all(row["head_matches"] and row["clean"] for row in repos.values()),
    }


def installed_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for package in VERSION_PACKAGES:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return versions


def transfer_decision(missing: list[str], unexpected: list[str], shape_mismatches: list[dict[str, Any]]) -> dict[str, Any]:
    passed = not missing and not unexpected and not shape_mismatches
    return {
        "pass": passed,
        "missing_count": len(missing),
        "unexpected_count": len(unexpected),
        "shape_mismatch_count": len(shape_mismatches),
        "stop_reason": None if passed else "unexplained state-transfer differences would leave non-base tensors initialized",
    }


def model_pair(source_root: Path, architecture: str):
    sys.path.insert(0, str(source_root))
    os.chdir(source_root)

    import torch

    common = {
        "vocab_size": 256,
        "hidden_size": 128,
        "intermediate_size": 256,
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 32,
        "max_position_embeddings": 256,
        "attention_bias": False,
        "use_cache": True,
        "tie_word_embeddings": False,
        "rope_theta": 1_000_000,
    }
    if architecture == "qwen3":
        from transformers.models.qwen3.configuration_qwen3 import Qwen3Config
        from transformers.models.qwen3.modeling_qwen3 import Qwen3ForCausalLM
        from liger.models.liger_qwen3_gla import LigerQwen3GLAConfig, LigerQwen3GLAForCausalLM

        base_config = Qwen3Config(**common)
        candidate_config = LigerQwen3GLAConfig(**common)
        base_class = Qwen3ForCausalLM
        candidate_class = LigerQwen3GLAForCausalLM
    elif architecture == "llama":
        from transformers.models.llama.configuration_llama import LlamaConfig
        from transformers.models.llama.modeling_llama import LlamaForCausalLM
        from liger.models.liger_gla import LigerGLAConfig, LigerGLAForCausalLM

        base_config = LlamaConfig(**common)
        candidate_config = LigerGLAConfig(**common)
        base_class = LlamaForCausalLM
        candidate_class = LigerGLAForCausalLM
    else:  # argparse prevents this; keep library use fail-closed
        raise ValueError(f"unsupported architecture: {architecture}")

    torch.manual_seed(20260820)
    base = base_class(base_config)
    torch.manual_seed(20260820)
    candidate = candidate_class(candidate_config)
    return torch, common, base, candidate


def state_transfer_gate(source_root: Path, architecture: str) -> dict[str, Any]:
    _torch, common, base, candidate = model_pair(source_root, architecture)

    base_state = base.state_dict()
    candidate_state = candidate.state_dict()
    base_keys = set(base_state)
    candidate_keys = set(candidate_state)
    missing = sorted(candidate_keys - base_keys)
    unexpected = sorted(base_keys - candidate_keys)
    shape_mismatches = [
        {
            "name": name,
            "base_shape": list(base_state[name].shape),
            "candidate_shape": list(candidate_state[name].shape),
        }
        for name in sorted(base_keys & candidate_keys)
        if base_state[name].shape != candidate_state[name].shape
    ]

    load_missing: list[str] = []
    load_unexpected: list[str] = []
    load_error: str | None = None
    if not shape_mismatches:
        try:
            incompatible = candidate.load_state_dict(base_state, strict=False)
            load_missing = sorted(incompatible.missing_keys)
            load_unexpected = sorted(incompatible.unexpected_keys)
        except Exception as exc:  # evidence, not recovery
            load_error = f"{type(exc).__name__}: {exc}"

    decision = transfer_decision(missing, unexpected, shape_mismatches)
    if load_error or load_missing != missing or load_unexpected != unexpected:
        decision = {
            **decision,
            "pass": False,
            "stop_reason": "load_state_dict behavior did not match the enumerated transfer surface",
        }

    return {
        "architecture": architecture,
        "config": common,
        "base_parameter_count": sum(parameter.numel() for parameter in base.parameters()),
        "candidate_parameter_count": sum(parameter.numel() for parameter in candidate.parameters()),
        "candidate_only_keys": missing,
        "base_only_keys": unexpected,
        "shape_mismatches": shape_mismatches,
        "load_missing_keys": load_missing,
        "load_unexpected_keys": load_unexpected,
        "load_error": load_error,
        "decision": decision,
    }


def llama_tensor_gates(source_root: Path) -> dict[str, Any]:
    torch, _common, base, candidate = model_pair(source_root, "llama")
    candidate.load_state_dict(base.state_dict(), strict=True)
    device = torch.device("cuda")
    ids = torch.tensor([[3, 17, 29, 41, 53, 67, 79, 91]], dtype=torch.long, device=device)
    candidate = candidate.to(device=device, dtype=torch.bfloat16)

    construction: dict[str, Any]
    try:
        torch.manual_seed(20260820)
        candidate.train()
        candidate.zero_grad(set_to_none=True)
        output = candidate(input_ids=ids, labels=ids, use_cache=False)
        output.loss.backward()
        finite_gradients = all(
            torch.isfinite(parameter.grad).all().item()
            for parameter in candidate.parameters()
            if parameter.grad is not None
        )
        construction = {
            "pass": bool(
                list(output.logits.shape) == [1, 8, 256]
                and torch.isfinite(output.logits).all().item()
                and torch.isfinite(output.loss).item()
                and finite_gradients
            ),
            "logits_shape": list(output.logits.shape),
            "loss": float(output.loss.detach().float().cpu()),
            "finite_logits": bool(torch.isfinite(output.logits).all().item()),
            "finite_loss": bool(torch.isfinite(output.loss).item()),
            "finite_gradients": finite_gradients,
            "error": None,
        }
    except Exception as exc:
        construction = {
            "pass": False,
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
        }

    if not construction["pass"]:
        return {"construction": construction, "recurrence": "not_run_fail_closed"}

    recurrence: dict[str, Any]
    try:
        candidate.eval()
        with torch.no_grad():
            full = candidate(input_ids=ids, use_cache=True).logits.float()
            past = None
            token_logits = []
            cache_lengths = []
            for index in range(ids.shape[1]):
                step = candidate(
                    input_ids=ids[:, index : index + 1],
                    past_key_values=past,
                    use_cache=True,
                )
                token_logits.append(step.logits.float())
                past = step.past_key_values
                cache_lengths.append(int(past.get_seq_length()))
            incremental = torch.cat(token_logits, dim=1)
        absolute = (full - incremental).abs()
        relative = absolute / full.abs().clamp_min(1e-6)
        max_abs = float(absolute.max().cpu())
        max_rel = float(relative.max().cpu())
        recurrence = {
            "pass": bool(max_abs <= 5e-2 and max_rel <= 5e-2 and cache_lengths == list(range(1, 9))),
            "full_shape": list(full.shape),
            "incremental_shape": list(incremental.shape),
            "cache_lengths": cache_lengths,
            "expected_cache_lengths": list(range(1, 9)),
            "max_abs": max_abs,
            "max_rel": max_rel,
            "error": None,
        }
    except Exception as exc:
        recurrence = {
            "pass": False,
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
        }
    return {"construction": construction, "recurrence": recurrence}


def selfcheck() -> None:
    assert transfer_decision([], [], [])["pass"] is True
    rejected = transfer_decision(["layer.bias"], [], [])
    assert rejected["pass"] is False and rejected["missing_count"] == 1
    print("liger_feasibility selfcheck: PASS")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, default=Path("/home/augus/src/Linearization"))
    parser.add_argument("--architecture", choices=["qwen3", "llama"], default="qwen3")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--selfcheck", action="store_true")
    args = parser.parse_args()
    if args.selfcheck:
        selfcheck()
        return 0
    if args.out is None:
        parser.error("--out is required unless --selfcheck is used")

    source_root = args.source_root.resolve()
    script_path = Path(__file__).resolve()
    result: dict[str, Any] = {
        "schema_version": SCHEMA,
        "runner_path": str(script_path),
        "runner_sha256": sha256_file(script_path),
        "python": sys.version,
        "executable": sys.executable,
        "versions": installed_versions(),
        "architecture": args.architecture,
        "provenance": provenance(source_root),
    }
    if result["provenance"]["pass"]:
        result["state_transfer"] = state_transfer_gate(source_root, args.architecture)
        static_pass = result["state_transfer"]["decision"]["pass"]
        if static_pass and args.architecture == "llama":
            tensor_gates = llama_tensor_gates(source_root)
            result["construction_gate"] = tensor_gates["construction"]
            result["recurrence_gate"] = tensor_gates["recurrence"]
            if not tensor_gates["construction"]["pass"]:
                result["status"] = "blocked_construction"
            elif not tensor_gates["recurrence"]["pass"]:
                result["status"] = "blocked_recurrence"
            else:
                result["status"] = "complete_pass"
        else:
            result["status"] = "static_compatibility_pass" if static_pass else "blocked_static_compatibility"
            result["construction_gate"] = "pending" if static_pass else "not_run_fail_closed"
            result["recurrence_gate"] = "pending" if static_pass else "not_run_fail_closed"
    else:
        result["status"] = "blocked_provenance"
        result["state_transfer"] = "not_run_fail_closed"
        result["construction_gate"] = "not_run_fail_closed"
        result["recurrence_gate"] = "not_run_fail_closed"

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "out": str(args.out)}, indent=2))
    return 0 if result["status"] in {"static_compatibility_pass", "complete_pass"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
