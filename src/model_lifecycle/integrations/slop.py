"""Strict consumer for ``slop.rs/generate-run/1.0`` evidence.

Local Labs owns the scientific decision. This module only proves that the
exact bytes emitted by slop.rs crossed the process/repository boundary intact,
that their internal identities agree, and that the raw input was retained.
It deliberately cannot mint a QualificationSnapshot or production authority.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import tempfile
import time
from typing import Any, Mapping


SOURCE_SCHEMA = "slop.rs/generate-run/1.0"
INGEST_SCHEMA = "tare.tools.local-labs/slop-ingest-receipt/1.0"
HEX = frozenset("0123456789abcdef")


class SlopManifestError(ValueError):
    """The supplied evidence cannot be trusted as a valid slop.rs run."""


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in HEX for character in value)
    )


def _is_git_object_id(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) in {40, 64}
        and all(character in HEX for character in value)
    )


def _object(value: object, name: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise SlopManifestError(f"{name} must be a JSON object")
    return value


def _positive_int(value: object, name: str, *, allow_zero: bool = False) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise SlopManifestError(f"{name} must be an integer")
    if value < 0 or (value == 0 and not allow_zero):
        qualifier = "non-negative" if allow_zero else "positive"
        raise SlopManifestError(f"{name} must be {qualifier}")
    return value


def _finite_number(value: object, name: str, *, allow_zero: bool = True) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SlopManifestError(f"{name} must be a number")
    number = float(value)
    if not math.isfinite(number) or number < 0 or (number == 0 and not allow_zero):
        qualifier = "non-negative" if allow_zero else "positive"
        raise SlopManifestError(f"{name} must be finite and {qualifier}")
    return number


def _rust_json_bytes(document: Mapping[str, Any]) -> bytes:
    """Match serde_json's compact struct serialization for the v1 schema."""
    return json.dumps(
        document,
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")


def _manifest_self_digest(document: Mapping[str, Any]) -> str:
    unhashed = copy.deepcopy(dict(document))
    unhashed["manifest_sha256"] = ""
    return sha256_bytes(_rust_json_bytes(unhashed))


def _canonical_receipt_bytes(receipt: Mapping[str, Any]) -> bytes:
    return json.dumps(
        receipt,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _ingest_self_digest(receipt: Mapping[str, Any]) -> str:
    unhashed = copy.deepcopy(dict(receipt))
    unhashed["receipt_sha256"] = ""
    return sha256_bytes(_canonical_receipt_bytes(unhashed))


def _require_keys(document: Mapping[str, Any], expected: set[str], name: str) -> None:
    actual = set(document)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise SlopManifestError(f"{name} keys differ; missing={missing}, extra={extra}")


def _validate_receipt(receipt: object, index: int) -> int:
    item = _object(receipt, f"receipts[{index}]")
    _require_keys(
        item,
        {
            "version",
            "execution_id",
            "layer_id",
            "device_name",
            "substrate",
            "state",
            "counters",
            "path_evidence",
            "output_sha256",
            "verified_invariants",
        },
        f"receipts[{index}]",
    )
    if item["version"] != 1 or item["substrate"] != "CpuNative":
        raise SlopManifestError(f"receipts[{index}] is not a v1 CPU-native receipt")
    if item["state"] != "Exercised":
        raise SlopManifestError(f"receipts[{index}] did not reach Exercised")
    for field in ("execution_id", "layer_id", "device_name"):
        if not isinstance(item[field], str) or not item[field]:
            raise SlopManifestError(f"receipts[{index}].{field} is empty")
    if not isinstance(item["path_evidence"], list) or not item["path_evidence"]:
        raise SlopManifestError(f"receipts[{index}] has no path evidence")
    if not all(isinstance(value, str) and value for value in item["path_evidence"]):
        raise SlopManifestError(f"receipts[{index}] has invalid path evidence")
    if not isinstance(item["verified_invariants"], list):
        raise SlopManifestError(f"receipts[{index}] has invalid invariants")
    if not _is_sha256(item["output_sha256"]):
        raise SlopManifestError(f"receipts[{index}] has invalid output SHA-256")
    counters = _object(item["counters"], f"receipts[{index}].counters")
    _require_keys(
        counters,
        {"flops", "duration_nanos", "peak_memory_bytes", "allocated_bytes"},
        f"receipts[{index}].counters",
    )
    flops = _positive_int(counters["flops"], f"receipts[{index}].counters.flops")
    _positive_int(counters["duration_nanos"], f"receipts[{index}].counters.duration_nanos")
    _positive_int(
        counters["peak_memory_bytes"],
        f"receipts[{index}].counters.peak_memory_bytes",
        allow_zero=True,
    )
    _positive_int(
        counters["allocated_bytes"],
        f"receipts[{index}].counters.allocated_bytes",
        allow_zero=True,
    )
    return flops


def validate_generate_manifest(
    document: object,
    *,
    verify_artifacts: bool = True,
) -> dict[str, Any]:
    """Validate the complete v1 contract and optionally rehash source artifacts."""
    manifest = _object(document, "manifest")
    _require_keys(
        manifest,
        {
            "schema",
            "issued_at_unix_ms",
            "substrate",
            "artifacts",
            "parameters",
            "result",
            "receipt_count",
            "receipts",
            "manifest_sha256",
        },
        "manifest",
    )
    if manifest["schema"] != SOURCE_SCHEMA or manifest["substrate"] != "CPU_NATIVE":
        raise SlopManifestError("unsupported source schema or execution substrate")
    _positive_int(manifest["issued_at_unix_ms"], "issued_at_unix_ms")
    if not _is_sha256(manifest["manifest_sha256"]):
        raise SlopManifestError("manifest_sha256 is not a lowercase SHA-256")
    if _manifest_self_digest(manifest) != manifest["manifest_sha256"]:
        raise SlopManifestError("manifest self-digest mismatch")

    artifacts = _object(manifest["artifacts"], "artifacts")
    _require_keys(
        artifacts,
        {"format", "model_path", "config_path", "tokenizer_path", "identity"},
        "artifacts",
    )
    if artifacts["format"] != "SAFETENSORS":
        raise SlopManifestError("only SAFETENSORS evidence is accepted")
    identity = _object(artifacts["identity"], "artifacts.identity")
    _require_keys(
        identity,
        {"config_sha256", "weights_sha256", "tokenizer_sha256"},
        "artifacts.identity",
    )
    for field in ("config_sha256", "weights_sha256", "tokenizer_sha256"):
        if not _is_sha256(identity[field]):
            raise SlopManifestError(f"artifacts.identity.{field} is invalid")
    for path_field, hash_field in (
        ("config_path", "config_sha256"),
        ("model_path", "weights_sha256"),
        ("tokenizer_path", "tokenizer_sha256"),
    ):
        raw_path = artifacts[path_field]
        if not isinstance(raw_path, str) or not raw_path:
            raise SlopManifestError(f"artifacts.{path_field} is empty")
        if verify_artifacts:
            path = Path(raw_path)
            if not path.is_file():
                raise SlopManifestError(f"artifact is missing: {path}")
            if sha256_file(path) != identity[hash_field]:
                raise SlopManifestError(f"artifact digest mismatch: {path}")

    parameters = _object(manifest["parameters"], "parameters")
    _require_keys(
        parameters,
        {
            "prompt_sha256",
            "prompt_tokens",
            "max_new_tokens",
            "context_capacity",
            "temperature",
            "top_p",
            "seed",
        },
        "parameters",
    )
    if not _is_sha256(parameters["prompt_sha256"]):
        raise SlopManifestError("prompt_sha256 is invalid")
    prompt_tokens = _positive_int(parameters["prompt_tokens"], "prompt_tokens")
    max_new_tokens = _positive_int(parameters["max_new_tokens"], "max_new_tokens")
    capacity = _positive_int(parameters["context_capacity"], "context_capacity")
    if prompt_tokens + max_new_tokens != capacity:
        raise SlopManifestError("context capacity does not bind prompt + requested continuation")
    _finite_number(parameters["temperature"], "temperature")
    top_p = _finite_number(parameters["top_p"], "top_p", allow_zero=False)
    if top_p > 1.0:
        raise SlopManifestError("top_p must not exceed 1.0")
    _positive_int(parameters["seed"], "seed", allow_zero=True)

    result = _object(manifest["result"], "result")
    _require_keys(
        result,
        {
            "generated_text",
            "continuation_token_ids",
            "generated_tokens",
            "total_tokens",
            "stop_reason",
            "total_flops",
            "prompt_eval_nanos",
            "time_to_first_token_nanos",
            "decode_nanos",
            "decode_tokens_per_second",
            "duration_nanos",
            "tokens_per_second",
            "output_sha256",
        },
        "result",
    )
    generated_text = result["generated_text"]
    if (
        not isinstance(generated_text, str)
        or not _is_sha256(result["output_sha256"])
        or sha256_bytes(generated_text.encode()) != result["output_sha256"]
    ):
        raise SlopManifestError("generated output digest mismatch")
    continuation = result["continuation_token_ids"]
    if not isinstance(continuation, list) or not all(
        isinstance(token, int) and not isinstance(token, bool) and 0 <= token <= 0xFFFFFFFF
        for token in continuation
    ):
        raise SlopManifestError("continuation_token_ids is invalid")
    generated_tokens = _positive_int(result["generated_tokens"], "generated_tokens", allow_zero=True)
    total_tokens = _positive_int(result["total_tokens"], "total_tokens")
    if generated_tokens != len(continuation) or prompt_tokens + generated_tokens != total_tokens:
        raise SlopManifestError("token accounting is inconsistent")
    if generated_tokens > max_new_tokens or total_tokens > capacity:
        raise SlopManifestError("generation exceeded its declared bounds")
    if result["stop_reason"] == "MaxNewTokens" and generated_tokens != max_new_tokens:
        raise SlopManifestError("MaxNewTokens stop reason contradicts token count")
    total_flops = _positive_int(result["total_flops"], "total_flops")
    _positive_int(result["prompt_eval_nanos"], "prompt_eval_nanos")
    _positive_int(result["decode_nanos"], "decode_nanos")
    _positive_int(result["duration_nanos"], "duration_nanos")
    _finite_number(result["decode_tokens_per_second"], "decode_tokens_per_second")
    _finite_number(result["tokens_per_second"], "tokens_per_second")
    first_token = result["time_to_first_token_nanos"]
    if generated_tokens == 0 and first_token is not None:
        raise SlopManifestError("zero-token run cannot have a first-token timestamp")
    if generated_tokens > 0:
        _positive_int(first_token, "time_to_first_token_nanos")

    receipts = manifest["receipts"]
    if not isinstance(receipts, list) or not receipts:
        raise SlopManifestError("receipts must be a non-empty array")
    receipt_count = _positive_int(manifest["receipt_count"], "receipt_count")
    if receipt_count != len(receipts):
        raise SlopManifestError("receipt_count does not match receipts")
    observed_flops = sum(_validate_receipt(receipt, index) for index, receipt in enumerate(receipts))
    if observed_flops != total_flops:
        raise SlopManifestError("receipt FLOPs do not match result.total_flops")

    return copy.deepcopy(dict(manifest))


def repository_identity(repository: Path) -> dict[str, Any]:
    root = repository.resolve()
    try:
        head = subprocess.run(
            ["git", "-C", os.fspath(root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "-C", os.fspath(root), "status", "--porcelain"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError) as error:
        raise SlopManifestError(f"cannot bind Local Labs repository identity: {error}") from error
    if not _is_git_object_id(head):
        raise SlopManifestError("Local Labs Git HEAD is not a full Git object identity")
    return {"path": os.fspath(root), "git_head": head, "dirty": bool(status.strip())}


def _write_new(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary = Path(stream.name)
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
    except FileExistsError as error:
        raise SlopManifestError(f"refusing to overwrite existing evidence: {path}") from error
    except OSError as error:
        raise SlopManifestError(f"cannot publish retained evidence {path}: {error}") from error
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def ingest_generate_manifest(
    manifest_path: Path,
    retain_dir: Path,
    consumer_repository: Path,
    *,
    verify_artifacts: bool = True,
    require_clean_consumer: bool = True,
) -> dict[str, Any]:
    """Validate, content-address and acknowledge one external slop.rs run."""
    source_path = manifest_path.resolve()
    try:
        source_bytes = source_path.read_bytes()
        document = json.loads(source_bytes)
    except (OSError, json.JSONDecodeError) as error:
        raise SlopManifestError(f"cannot read source manifest {source_path}: {error}") from error
    manifest = validate_generate_manifest(document, verify_artifacts=verify_artifacts)
    source_file_sha256 = sha256_bytes(source_bytes)

    # Check the consumer before the first persistent write. A failed clean-tree
    # gate must not leave evidence that looks successfully ingested.
    consumer = repository_identity(consumer_repository)
    if require_clean_consumer and consumer["dirty"]:
        raise SlopManifestError("Local Labs consumer repository is dirty")

    retained_path = retain_dir.resolve() / f"{source_file_sha256}.json"
    _write_new(retained_path, source_bytes)
    if sha256_file(retained_path) != source_file_sha256:
        raise SlopManifestError("retained evidence failed SHA-256 readback")

    receipt: dict[str, Any] = {
        "schema": INGEST_SCHEMA,
        "issued_at_unix_ms": time.time_ns() // 1_000_000,
        "source": {
            "schema": SOURCE_SCHEMA,
            "path": os.fspath(source_path),
            "file_sha256": source_file_sha256,
            "manifest_sha256": manifest["manifest_sha256"],
        },
        "subject": {
            "artifact_identity": manifest["artifacts"]["identity"],
            "substrate": manifest["substrate"],
            "receipt_count": manifest["receipt_count"],
            "total_flops": manifest["result"]["total_flops"],
            "generated_tokens": manifest["result"]["generated_tokens"],
        },
        "consumer": consumer,
        "retained_evidence": {
            "path": os.fspath(retained_path),
            "sha256": source_file_sha256,
            "size_bytes": len(source_bytes),
        },
        "artifact_bytes_rehashed": verify_artifacts,
        "qualification_status": "UNASSESSED",
        "authority": "NONE",
        "receipt_sha256": "",
    }
    receipt["receipt_sha256"] = _ingest_self_digest(receipt)
    return receipt


def validate_ingest_receipt(
    receipt: object,
    *,
    require_clean_consumer: bool = True,
) -> dict[str, Any]:
    item = _object(receipt, "ingest receipt")
    _require_keys(
        item,
        {
            "schema",
            "issued_at_unix_ms",
            "source",
            "subject",
            "consumer",
            "retained_evidence",
            "artifact_bytes_rehashed",
            "qualification_status",
            "authority",
            "receipt_sha256",
        },
        "ingest receipt",
    )
    if item["schema"] != INGEST_SCHEMA:
        raise SlopManifestError("unsupported ingest receipt schema")
    if item["qualification_status"] != "UNASSESSED" or item["authority"] != "NONE":
        raise SlopManifestError("ingest receipt attempted to grant qualification or authority")
    _positive_int(item["issued_at_unix_ms"], "issued_at_unix_ms")
    if not _is_sha256(item["receipt_sha256"]) or item["receipt_sha256"] != _ingest_self_digest(item):
        raise SlopManifestError("ingest receipt self-digest mismatch")
    retained = _object(item["retained_evidence"], "retained_evidence")
    _require_keys(retained, {"path", "sha256", "size_bytes"}, "retained_evidence")
    retained_path = Path(retained.get("path", ""))
    if not retained_path.is_file():
        raise SlopManifestError("retained evidence is missing")
    expected_sha = retained.get("sha256")
    if not _is_sha256(expected_sha) or sha256_file(retained_path) != expected_sha:
        raise SlopManifestError("retained evidence digest mismatch")
    if retained_path.stat().st_size != retained.get("size_bytes"):
        raise SlopManifestError("retained evidence size mismatch")
    source = _object(item["source"], "source")
    _require_keys(source, {"schema", "path", "file_sha256", "manifest_sha256"}, "source")
    if source["schema"] != SOURCE_SCHEMA or not _is_sha256(source["manifest_sha256"]):
        raise SlopManifestError("source identity is invalid")
    if source.get("file_sha256") != expected_sha:
        raise SlopManifestError("source and retained evidence identities differ")
    subject = _object(item["subject"], "subject")
    _require_keys(
        subject,
        {"artifact_identity", "substrate", "receipt_count", "total_flops", "generated_tokens"},
        "subject",
    )
    if subject["substrate"] != "CPU_NATIVE":
        raise SlopManifestError("ingested subject has an unsupported substrate")
    _positive_int(subject["receipt_count"], "subject.receipt_count")
    _positive_int(subject["total_flops"], "subject.total_flops")
    _positive_int(subject["generated_tokens"], "subject.generated_tokens", allow_zero=True)
    artifact_identity = _object(subject["artifact_identity"], "subject.artifact_identity")
    _require_keys(
        artifact_identity,
        {"config_sha256", "weights_sha256", "tokenizer_sha256"},
        "subject.artifact_identity",
    )
    if not all(_is_sha256(value) for value in artifact_identity.values()):
        raise SlopManifestError("subject artifact identity is invalid")
    consumer = _object(item["consumer"], "consumer")
    _require_keys(consumer, {"path", "git_head", "dirty"}, "consumer")
    if not _is_git_object_id(consumer["git_head"]):
        raise SlopManifestError("consumer Git identity is invalid")
    if require_clean_consumer and consumer.get("dirty") is not False:
        raise SlopManifestError("consumer identity is dirty")
    if not isinstance(item["artifact_bytes_rehashed"], bool):
        raise SlopManifestError("artifact_bytes_rehashed must be boolean")
    return copy.deepcopy(dict(item))
