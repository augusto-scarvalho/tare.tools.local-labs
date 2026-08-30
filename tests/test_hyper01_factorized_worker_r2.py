from pathlib import Path

from tools.research.hyper01_factorized_worker_r2 import factorized_parameter_count, summarize_seed_cosines


def test_factorized_parameter_count_is_below_frozen_twenty_mb_gate():
    output = 8 * 1024 + 3584 * 8
    count = factorized_parameter_count(64, 256, 512, 32, output)
    assert count == 1_381_152
    assert count * 4 / (1024 ** 2) < 20.0


def test_seed_summary_uses_worst_complete_seed_not_only_best_seed():
    rows = [
        {"target_cosines": [0.99, 0.98, 0.97, 0.96]},
        {"target_cosines": [0.95, 0.95, 0.95, 0.95]},
    ]
    summary = summarize_seed_cosines(rows)
    assert summary["completed_seeds"] == 2
    assert summary["seed_mean_cosines"] == [0.975, 0.95]
    assert summary["worst_seed_mean_cosine"] == 0.95
    assert summary["mean_weight_delta_cosine"] == 0.9624999999999999


def test_worker_retains_states_generated_tensors_and_raw_timings():
    source = Path("tools/research/hyper01_factorized_worker_r2.py").read_text(encoding="utf-8")
    assert "save_file(state_tensors" in source
    assert "save_file(generated_tensors" in source
    assert '"latencies_ms": latencies' in source
    assert "target_A = torch.randn" not in source
    assert "target_B = torch.randn" not in source


def test_scorer_reopens_both_retained_files():
    source = Path("tools/research/hyper01_factorized_scorer_r2.py").read_text(encoding="utf-8")
    assert "args.generated" in source
    assert "args.states" in source
    assert "target_b @ target_a" in source


def test_runner_freezes_five_seeds_and_worst_seed_gate():
    source = Path("tools/research/run_hyper01_factorized_r2.py").read_text(encoding="utf-8")
    assert "SEEDS = [20260824, 20260825, 20260826, 20260827, 20260828]" in source
    assert '"worst_seed_fidelity": ("worst_seed_mean_cosine", "ge", 0.95)' in source
    assert '"worker.json"' in source
    assert '"generated_tensors.safetensors"' in source
