from tools.research.negative_kv_real_worker import aggregate
from tools.research.run_negative_kv_real_screen import build_gates, candidate_decisions


def _rows():
    rows = []
    for index in range(12):
        common = {"context": 0, "layer": index // 2, "slice": index % 2,
                  "tensor_source": "frozen_qwen"}
        rows.append({"candidate": "RSH-01", **common, "fib_mse_ratio_vs_uniform": 0.6,
                     "fib_sqnr_gain_db": 3.0, "fib_cosine_similarity": 0.999})
        rows.append({"candidate": "RSH-03", **common, "rank4_mse_recovery": 0.6,
                     "rank4_output_cosine": 0.999, "rank4_parameter_overhead": 0.008})
    for context in range(3):
        for layer in (3, 7, 11, 15, 19, 23):
            common = {"context": context, "layer": layer, "tensor_source": "frozen_qwen"}
            rows.append({"candidate": "REP-03", **common, "hadamard_mse_reduction": 0.6,
                         "hadamard_attention_cosine": 0.999})
            rows.append({"candidate": "RSH-04", **common, "binary_top_block_recall": 0.95,
                         "retained_fraction": 0.25})
            rows.append({"candidate": "REP-06", **common, "average_bits_per_element": 6.0,
                         "dynamic_attention_cosine": 0.999, "dynamic_beats_static_int4": True})
    return rows


def test_aggregate_counts_real_cells_and_candidates():
    scores = aggregate(_rows())
    assert scores["actual_model_activation_cells"] == 18
    assert scores["actual_model_weight_matrices"] == 12
    assert scores["candidate_hypotheses_evaluated"] == 5
    assert scores["all_decisive_tensors_from_frozen_model"] is True


def test_candidate_decisions_apply_frozen_conjunctions():
    decisions = candidate_decisions(aggregate(_rows()))
    assert set(decisions) == {"RSH-01", "REP-03", "RSH-03", "RSH-04", "REP-06"}
    assert all(row["pass"] for row in decisions.values())


def test_gates_recompute_each_observation():
    gates = build_gates(aggregate(_rows()), recompute_match=True, service_restored=True)
    assert len(gates) == 19
    assert all(row["pass"] for row in gates.values())


def test_any_synthetic_source_fails_integrity_gate():
    rows = _rows()
    rows[0]["tensor_source"] = "synthetic"
    scores = aggregate(rows)
    gates = build_gates(scores, recompute_match=True, service_restored=True)
    assert gates["no_synthetic_decisive_tensors"]["pass"] is False
