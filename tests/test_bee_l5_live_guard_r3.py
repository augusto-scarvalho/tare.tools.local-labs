import pathlib
from tools.research import run_bee_l5_live_guard_r3 as target


def test_bootstrap_root_is_repo_root():
    assert (target.ROOT / "tools" / "analysis" / "reasoning_loop_guard.py").is_file()
    assert target.ROOT == pathlib.Path(target.__file__).resolve().parents[2]
