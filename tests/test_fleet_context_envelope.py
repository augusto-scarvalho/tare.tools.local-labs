from tools.research import run_fleet_context_envelope as target


def test_matrix_has_seventy_two_cases_and_slot_corrected_moe_limit():
    total = sum(len(target.TARGETS[model]) * len(target.POSITIONS) * len(target.REPLICATES)
                for model in target.MODELS)
    assert total == 72
    assert target.SLOT_CONTEXT["qwen36-moe"] == 73728 // 4
    assert max(target.TARGETS["qwen36-moe"]) == 17000


def test_needles_are_unique_and_positioned():
    codes = {target.access_code(model, length, position, replicate)
             for model in target.MODELS for length in target.TARGETS[model]
             for position in target.POSITIONS for replicate in target.REPLICATES}
    assert len(codes) == 72
    start = target.make_prompt(10, "start", "NX-START")
    middle = target.make_prompt(10, "middle", "NX-MIDDLE")
    end = target.make_prompt(10, "end", "NX-END")
    assert start.index("NX-START") < start.index("Archive record")
    assert middle.index("Archive record 000004") < middle.index("NX-MIDDLE") < middle.index("Archive record 000006")
    assert end.index("NX-END") > end.index("Archive record 000009")


def test_exact_recall_allows_only_normalized_code():
    assert target.exact_recall("`NX-Q38-04000-S0`", "NX-Q38-04000-S0")
    assert not target.exact_recall("The code is NX-Q38-04000-S0", "NX-Q38-04000-S0")
    assert not target.exact_recall("NX-WRONG", "NX-Q38-04000-S0")
