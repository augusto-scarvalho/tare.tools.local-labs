"""Qualify preservation only; historical experiment claims are not reexecuted."""
import hashlib
import json
from pathlib import Path


ARCHIVE = (Path(__file__).resolve().parents[1] / "docs/history"
           / "2026-09-08-library-filtered-publications")
RECEIPT_SHA256 = "6b26266d9919bc69097a076f1db7c2c8f98240f6deafdc554aba7e3f19549960"


def test_remote_history_retains_exact_source_bytes_and_temporal_authority():
    raw = (ARCHIVE / "RECOVERY_RECEIPT.json").read_bytes()
    assert hashlib.sha256(raw).hexdigest() == RECEIPT_SHA256
    receipt = json.loads(raw)
    assert receipt["status"] == "HISTORICAL"
    assert receipt["indexing"] == "HISTORY_ONLY"
    assert receipt["authority"] == "NONE"
    assert receipt["current_paper"] is False
    assert receipt["executed_experiments"] is False
    assert receipt["file_count"] == len(receipt["files"]) == 25
    assert sum(row["bytes"] for row in receipt["files"]) == receipt["source_bytes"] == 108480
    assert len({row["git_blob"] for row in receipt["files"]}) == 25
    for row in receipt["files"]:
        path = (ARCHIVE / row["path"]).resolve()
        assert path.is_relative_to(ARCHIVE.resolve())
        data = path.read_bytes()
        assert len(data) == row["bytes"]
        assert hashlib.sha256(data).hexdigest() == row["sha256"]
        assert hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest() == row["git_blob"]
        assert row["authority"] == "NONE" and row["indexing"] == "HISTORY_ONLY"
        assert row["source_repository"] == "tare.tools.library"
        assert {"ref": row["source_ref"], "commit": row["source_commit"]} in row["all_observed_sources"]
        assert row["source_ref"].startswith("refs/remotes/origin/")


def test_distinct_historical_versions_do_not_overwrite_each_other():
    receipt = json.loads((ARCHIVE / "RECOVERY_RECEIPT.json").read_text(encoding="utf-8"))
    paths = [row["path"] for row in receipt["files"]]
    assert len(paths) == len(set(paths))
    versions = [row for row in receipt["files"] if row["source_path"] == "experiments/README.md"]
    assert len(versions) == 2
    assert len({row["sha256"] for row in versions}) == 2
    assert all(row["comparison"]["does_not_claim_no_semantic_overlap"] for row in receipt["files"])
