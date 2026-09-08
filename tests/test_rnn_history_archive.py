"""Archive integrity and temporal authority, without running an experiment."""
import hashlib
import json
from pathlib import Path


ARCHIVE = (Path(__file__).resolve().parents[1] / "docs/campaigns/rnn-mamba/history"
           / "2026-08-12-library-publication")


def test_recovered_rnn_sources_are_exact_and_have_no_current_authority():
    manifest = json.loads((ARCHIVE / "MANIFEST.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "HISTORY_ONLY"
    assert manifest["operational_authority"] == "NONE"
    assert manifest["scientific_qualification"] == "NOT_REEVALUATED"
    assert manifest["execution_performed"] is False
    assert manifest["backlog_changed"] is False
    assert manifest["source_commit"] == "9da8a843426c076895af609fe3c8c8433ac87d33"
    assert len(manifest["records"]) == 4
    assert len({row["source_path"] for row in manifest["records"]}) == 4
    for record in manifest["records"]:
        path = (ARCHIVE / record["archive_path"]).resolve()
        assert path.is_relative_to(ARCHIVE.resolve())
        raw = path.read_bytes()
        assert len(raw) == record["bytes"]
        assert hashlib.sha256(raw).hexdigest() == record["sha256"]
        git_header = f"blob {len(raw)}\0".encode()
        assert hashlib.sha1(git_header + raw).hexdigest() == record["git_blob"]


def test_original_false_green_and_unqualified_boundaries_are_retained():
    receipt = json.loads((ARCHIVE / "sources/archaeology/publication-receipts"
        / "2026-08-12-rnn-06t2-audit-update.json").read_text(encoding="utf-8"))
    assert receipt["canonical_promotion"] is False
    boundary = receipt["scientific_boundary"]
    assert boundary["end_to_end_recovery_utility_historical_mint"] == "RECONCILED_FALSE_GREEN"
    assert boundary["end_to_end_recovery_utility"] == "NOT_COMPARABLE"
    assert boundary["batch_shape_numerical_portability"] == "OUT_OF_SCOPE_NOT_QUALIFIED"
    assert boundary["qwen"] == "DEFER"
