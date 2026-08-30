from __future__ import annotations

import pathlib
import subprocess
import sys


def test_r8_direct_file_import_bootstrap():
    root = pathlib.Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [sys.executable, str(root / "tools/research/run_slx08_relevance_prefill_r8.py"), "--help"],
        cwd=root,
        text=True,
        capture_output=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    assert "--outdir" in result.stdout
