#!/usr/bin/env python3
"""Fail-closed provenance helpers for local-labs experiment receipts."""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import pathlib
import platform
import subprocess
import sys
import time
from typing import Iterable


ROOT = pathlib.Path(__file__).resolve().parents[2]


def sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _run_text(argv: list[str], cwd: pathlib.Path | None = None) -> dict:
    try:
        completed = subprocess.run(
            argv,
            cwd=str(cwd) if cwd else None,
            text=True,
            encoding="utf-8",
            errors="replace",
            capture_output=True,
            check=False,
            timeout=30,
        )
        return {
            "argv": argv,
            "returncode": completed.returncode,
            "stdout": completed.stdout.strip(),
            "stderr": completed.stderr.strip(),
        }
    except Exception as exc:
        return {"argv": argv, "returncode": None, "stdout": "", "stderr": repr(exc)}


def git_identity(repo: pathlib.Path = ROOT) -> dict:
    head = _run_text(["git", "rev-parse", "HEAD"], repo)
    status = _run_text(["git", "status", "--porcelain=v1", "--untracked-files=all"], repo)
    status_text = status["stdout"]
    return {
        "head": head["stdout"] if head["returncode"] == 0 else "UNKNOWN",
        "dirty": bool(status_text),
        "status_entry_count": len(status_text.splitlines()) if status_text else 0,
        "status_sha256": hashlib.sha256(status_text.encode("utf-8")).hexdigest(),
        "errors": [
            result["stderr"]
            for result in (head, status)
            if result["returncode"] != 0 and result["stderr"]
        ],
    }


def package_versions(names: Iterable[str]) -> dict[str, str]:
    versions: dict[str, str] = {}
    for name in names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "NOT_INSTALLED"
    return versions


def gpu_identity() -> dict:
    query = _run_text([
        "nvidia-smi",
        "--query-gpu=name,uuid,driver_version,memory.total",
        "--format=csv,noheader,nounits",
    ])
    return {
        "available": query["returncode"] == 0,
        "query": query["stdout"] if query["returncode"] == 0 else None,
        "error": query["stderr"] if query["returncode"] != 0 else None,
    }


def input_identities(paths: Iterable[pathlib.Path]) -> list[dict]:
    identities = []
    for raw_path in paths:
        path = raw_path.resolve()
        if not path.exists():
            identities.append({"path": str(path), "status": "MISSING"})
            continue
        if not path.is_file():
            identities.append({"path": str(path), "status": "NOT_A_FILE"})
            continue
        identities.append({
            "path": str(path),
            "status": "HASHED",
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        })
    return identities


def build_provenance(
    *,
    script_path: pathlib.Path,
    started_at_utc: str,
    started_monotonic: float,
    input_paths: Iterable[pathlib.Path] = (),
    packages: Iterable[str] = (),
    runtime: dict | None = None,
) -> dict:
    script_path = script_path.resolve()
    finished_at_utc = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    return {
        "schema": "local-labs-experiment-provenance-v1",
        "started_at_utc": started_at_utc,
        "finished_at_utc": finished_at_utc,
        "elapsed_seconds": round(time.monotonic() - started_monotonic, 6),
        "command": [sys.executable, *sys.argv],
        "cwd": os.getcwd(),
        "repository": git_identity(ROOT),
        "script": {
            "path": str(script_path),
            "sha256": sha256_file(script_path),
        },
        "inputs": input_identities(input_paths),
        "environment": {
            "python": sys.version,
            "executable": sys.executable,
            "platform": platform.platform(),
            "packages": package_versions(packages),
            "gpu": gpu_identity(),
        },
        "runtime": runtime or {},
    }


def provenance_complete(provenance: dict) -> tuple[bool, list[str]]:
    errors = []
    if provenance.get("schema") != "local-labs-experiment-provenance-v1":
        errors.append("invalid provenance schema")
    if not provenance.get("command"):
        errors.append("missing command")
    script = provenance.get("script", {})
    if len(script.get("sha256", "")) != 64:
        errors.append("missing script SHA-256")
    repository = provenance.get("repository", {})
    if len(repository.get("head", "")) != 40:
        errors.append("missing repository HEAD")
    for item in provenance.get("inputs", []):
        if item.get("status") != "HASHED":
            errors.append(f"unbound input: {item.get('path')}")
    return not errors, errors


def canonical_json_sha256(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
