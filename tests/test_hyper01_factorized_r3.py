from pathlib import Path

from tools.research.hyper01_factorized_worker_r2 import factorized_parameter_count


def test_rank_135_is_exact_discrete_frontier_under_twenty_mib():
    output = 8 * 1024 + 3584 * 8
    assert factorized_parameter_count(64, 256, 512, 135, output) * 4 / (1024 ** 2) <= 20.0
    assert factorized_parameter_count(64, 256, 512, 136, output) * 4 / (1024 ** 2) > 20.0


def test_r3_is_harness_bound_and_terminal_by_design():
    source = Path("tools/research/run_hyper01_factorized_r3.py").read_text(encoding="utf-8")
    assert "with ExperimentRun(raw, TASK_ID, inputs) as run" in source
    assert "receipt = run.seal" in source
    assert '"--rank", "135"' in source
    assert '"--steps", "3000"' in source
    assert '"no_further_rank_or_step_successor": True' in source
