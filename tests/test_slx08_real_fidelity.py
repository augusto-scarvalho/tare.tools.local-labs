from pathlib import Path

from tools.research.slx08_real_fidelity_worker import aggregate


def test_aggregate_uses_all_real_cells_and_median():
    rows = [
        {
            "tensor_source": "frozen_qwen",
            "computed_indices_materially_used": True,
            "selected_block_context_cosine": 0.96 + index / 1000,
            "legacy_first_half_context_cosine": 0.70,
        }
        for index in range(12)
    ]
    scores = aggregate(rows)
    assert scores["actual_qkv_cells"] == 12
    assert scores["all_decisive_tensors_from_frozen_model"] is True
    assert scores["computed_top_block_indices_materially_used"] is True
    assert scores["median_selected_block_context_cosine"] > 0.95


def test_original_probe_computes_but_ignores_selected_indices():
    source = Path("tools/probes/slx08_speculative_prefill_oracle.py").read_text(encoding="utf-8")
    assert "selected_indices = torch.topk" in source
    assert "k_selected = k[:, :, : (top_k_blocks * block_size), :]" in source
    assert "v_selected = v[:, :, : (top_k_blocks * block_size), :]" in source
