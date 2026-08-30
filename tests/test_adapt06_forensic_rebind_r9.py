import json
from pathlib import Path

from tools.research.run_adapt06_forensic_rebind_r9 import verify_with_terminal_files


def test_r9_projects_sealed_terminal_files(monkeypatch, tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "run.terminal.json").write_text(json.dumps({"files": {"samples.jsonl": "abc"}}), encoding="utf-8")
    monkeypatch.setattr("tools.research.run_adapt06_forensic_rebind_r9.ORIGINAL_VERIFY", lambda _: {"valid": True})
    assert verify_with_terminal_files(raw)["manifest"] == {"samples.jsonl": "abc"}


def test_r9_binds_delegated_r8_and_its_own_wrapper():
    source = Path("tools/research/run_adapt06_forensic_rebind_r9.py").read_text(encoding="utf-8")
    assert "run_adapt06_forensic_rebind_r8.py" in source
    assert "r8.__file__ = __file__" in source
    assert "r8.verify_run = verify_with_terminal_files" in source
