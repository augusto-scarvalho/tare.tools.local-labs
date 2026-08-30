import subprocess
import sys
from pathlib import Path


SCRIPT = Path("tools/research/run_adapt06_forensic_rebind_r10.py")


def test_r10_is_directly_invocable_from_repository_root():
    completed = subprocess.run(
        [sys.executable, str(SCRIPT), "--help"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert "--outdir" in completed.stdout


def test_r10_changes_only_import_and_packet_binding():
    source = SCRIPT.read_text(encoding="utf-8")
    assert "sys.path.insert(0, str(ROOT))" in source
    assert "r8.verify_run = r9.verify_with_terminal_files" in source
    assert "r8.__file__ = __file__" in source
    assert "BACKLOG-ADAPT06-SLOP-LIVE-10" in source
