from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from model_lifecycle.integrations.slop import (  # noqa: E402
    SlopManifestError,
    ingest_generate_manifest,
    sha256_file,
    validate_generate_manifest,
    validate_ingest_receipt,
)


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _manifest(tmp_path: Path) -> tuple[dict, Path]:
    model = tmp_path / "model.safetensors"
    config = tmp_path / "config.json"
    tokenizer = tmp_path / "tokenizer.json"
    model.write_bytes(b"real model bytes")
    config.write_bytes(b'{"model_type":"qwen2"}')
    tokenizer.write_bytes(b'{"version":"1.0"}')
    output = " Paris"
    receipt = {
        "version": 1,
        "execution_id": "generation_0",
        "layer_id": "qwen.decoder.0",
        "device_name": "cpu:f32",
        "substrate": "CpuNative",
        "state": "Exercised",
        "counters": {
            "flops": 42,
            "duration_nanos": 100,
            "peak_memory_bytes": 64,
            "allocated_bytes": 64,
        },
        "path_evidence": ["QwenModel::forward_incremental"],
        "output_sha256": _sha(b"hidden state"),
        "verified_invariants": ["Invariants::IncrementalKv"],
    }
    document = {
        "schema": "slop.rs/generate-run/1.0",
        "issued_at_unix_ms": 1,
        "substrate": "CPU_NATIVE",
        "artifacts": {
            "format": "SAFETENSORS",
            "model_path": str(model.resolve()),
            "config_path": str(config.resolve()),
            "tokenizer_path": str(tokenizer.resolve()),
            "identity": {
                "config_sha256": sha256_file(config),
                "weights_sha256": sha256_file(model),
                "tokenizer_sha256": sha256_file(tokenizer),
            },
        },
        "parameters": {
            "prompt_sha256": _sha(b"The capital of France is"),
            "prompt_tokens": 5,
            "max_new_tokens": 1,
            "context_capacity": 6,
            "temperature": 0.0,
            "top_p": 1.0,
            "seed": 7,
        },
        "result": {
            "generated_text": output,
            "continuation_token_ids": [9707],
            "generated_tokens": 1,
            "total_tokens": 6,
            "stop_reason": "MaxNewTokens",
            "total_flops": 42,
            "prompt_eval_nanos": 80,
            "time_to_first_token_nanos": 90,
            "decode_nanos": 20,
            "decode_tokens_per_second": 50_000_000.0,
            "duration_nanos": 100,
            "tokens_per_second": 10_000_000.0,
            "output_sha256": _sha(output.encode()),
        },
        "receipt_count": 1,
        "receipts": [receipt],
        "manifest_sha256": "",
    }
    body = json.dumps(document, ensure_ascii=False, separators=(",", ":")).encode()
    document["manifest_sha256"] = _sha(body)
    manifest_path = tmp_path / "run.json"
    manifest_path.write_text(json.dumps(document, indent=2), encoding="utf-8")
    return document, manifest_path


def test_external_contract_validates_artifact_bytes_and_retains_raw_input(tmp_path: Path):
    document, manifest_path = _manifest(tmp_path)
    assert validate_generate_manifest(document)["receipt_count"] == 1

    retained = tmp_path / "retained"
    ingest = ingest_generate_manifest(
        manifest_path,
        retained,
        ROOT,
        require_clean_consumer=False,
    )
    assert ingest["qualification_status"] == "UNASSESSED"
    assert ingest["authority"] == "NONE"
    assert ingest["artifact_bytes_rehashed"] is True
    validated = validate_ingest_receipt(ingest, require_clean_consumer=False)
    assert validated["source"]["manifest_sha256"] == document["manifest_sha256"]
    assert Path(validated["retained_evidence"]["path"]).read_bytes() == manifest_path.read_bytes()


def test_tampered_manifest_and_artifact_fail_closed(tmp_path: Path):
    document, _ = _manifest(tmp_path)
    tampered = copy.deepcopy(document)
    tampered["result"]["total_flops"] += 1
    with pytest.raises(SlopManifestError, match="self-digest"):
        validate_generate_manifest(tampered)

    model = Path(document["artifacts"]["model_path"])
    model.write_bytes(b"different model")
    with pytest.raises(SlopManifestError, match="artifact digest mismatch"):
        validate_generate_manifest(document)


def test_ingest_never_overwrites_retained_evidence(tmp_path: Path):
    _, manifest_path = _manifest(tmp_path)
    retained = tmp_path / "retained"
    ingest_generate_manifest(
        manifest_path,
        retained,
        ROOT,
        require_clean_consumer=False,
    )
    with pytest.raises(SlopManifestError, match="overwrite"):
        ingest_generate_manifest(
            manifest_path,
            retained,
            ROOT,
            require_clean_consumer=False,
        )


def test_readback_rejects_a_forged_qualification_claim(tmp_path: Path):
    _, manifest_path = _manifest(tmp_path)
    ingest = ingest_generate_manifest(
        manifest_path,
        tmp_path / "retained",
        ROOT,
        require_clean_consumer=False,
    )
    ingest["qualification_status"] = "QUALIFIED"
    with pytest.raises(SlopManifestError, match="grant qualification"):
        validate_ingest_receipt(ingest, require_clean_consumer=False)


def test_dirty_consumer_gate_precedes_retained_evidence_write(tmp_path: Path):
    _, manifest_path = _manifest(tmp_path)
    dirty_repo = tmp_path / "dirty-repo"
    dirty_repo.mkdir()
    subprocess.run(["git", "init", "-q", str(dirty_repo)], check=True)
    subprocess.run(["git", "-C", str(dirty_repo), "config", "user.email", "test@example.invalid"], check=True)
    subprocess.run(["git", "-C", str(dirty_repo), "config", "user.name", "Test"], check=True)
    tracked = dirty_repo / "tracked.txt"
    tracked.write_text("clean", encoding="utf-8")
    subprocess.run(["git", "-C", str(dirty_repo), "add", "tracked.txt"], check=True)
    subprocess.run(["git", "-C", str(dirty_repo), "commit", "-qm", "fixture"], check=True)
    tracked.write_text("dirty", encoding="utf-8")

    retained = tmp_path / "must-not-exist"
    with pytest.raises(SlopManifestError, match="consumer repository is dirty"):
        ingest_generate_manifest(manifest_path, retained, dirty_repo)
    assert not retained.exists()


def test_cli_runs_ingest_and_readback_in_separate_processes(tmp_path: Path):
    _, manifest_path = _manifest(tmp_path)
    receipt = tmp_path / "ingest-receipt.json"
    cli = ROOT / "tools" / "integrations" / "slop_manifest.py"
    ingest = subprocess.run(
        [
            sys.executable,
            str(cli),
            "ingest",
            "--manifest",
            str(manifest_path),
            "--retain-dir",
            str(tmp_path / "retained"),
            "--receipt",
            str(receipt),
            "--allow-dirty-consumer",
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )
    assert ingest.returncode == 0, ingest.stderr
    query = subprocess.run(
        [
            sys.executable,
            str(cli),
            "query",
            "--receipt",
            str(receipt),
            "--allow-dirty-consumer",
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )
    assert query.returncode == 0, query.stderr
    readback = json.loads(query.stdout)
    assert readback["receipt_sha256"] == json.loads(receipt.read_text())["receipt_sha256"]
