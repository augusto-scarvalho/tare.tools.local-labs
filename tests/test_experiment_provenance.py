from __future__ import annotations

import pathlib
import time

from tools.analysis.experiment_provenance import (
    build_provenance,
    canonical_json_sha256,
    provenance_complete,
    sha256_file,
)


def test_sha256_file_and_canonical_json_are_deterministic(tmp_path: pathlib.Path):
    artifact = tmp_path / "artifact.bin"
    artifact.write_bytes(b"frozen-evidence")
    assert sha256_file(artifact) == sha256_file(artifact)
    assert canonical_json_sha256({"b": 2, "a": 1}) == canonical_json_sha256({"a": 1, "b": 2})


def test_build_provenance_binds_command_repo_script_and_input(tmp_path: pathlib.Path):
    artifact = tmp_path / "input.json"
    artifact.write_text("{}", encoding="utf-8")
    started = time.monotonic()
    provenance = build_provenance(
        script_path=pathlib.Path(__file__),
        started_at_utc="2026-08-25T00:00:00Z",
        started_monotonic=started,
        input_paths=[artifact],
    )
    complete, errors = provenance_complete(provenance)
    assert complete, errors
    assert provenance["inputs"][0]["status"] == "HASHED"
    assert isinstance(provenance["repository"]["dirty"], bool)


def test_missing_input_fails_closed(tmp_path: pathlib.Path):
    provenance = build_provenance(
        script_path=pathlib.Path(__file__),
        started_at_utc="2026-08-25T00:00:00Z",
        started_monotonic=time.monotonic(),
        input_paths=[tmp_path / "missing.bin"],
    )
    complete, errors = provenance_complete(provenance)
    assert not complete
    assert any("unbound input" in error for error in errors)
