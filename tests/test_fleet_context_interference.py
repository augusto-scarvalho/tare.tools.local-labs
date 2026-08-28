from tools.research import run_fleet_context_interference as target


def test_prompt_has_one_exact_target_and_thirty_one_decoys():
    prompt = target.interference_prompt(100, "middle", "NX-Q38-04000-M0")
    assert prompt.count("SECURE ACCESS RECORD") == 32
    assert prompt.count("SECURE ACCESS RECORD [ORION-DELTA]:") == 1
    assert sum(f"[ORION-DELTA-{index:02d}]" in prompt for index in range(1, 32)) == 31
    assert "NX-Q38-04000-M0" in prompt


def test_target_position_moves_without_changing_decoy_count():
    prompts = {position: target.interference_prompt(100, position, "NX-HAU-16000-X0")
               for position in target.core.POSITIONS}
    locations = {position: prompt.index("SECURE ACCESS RECORD [ORION-DELTA]:")
                 for position, prompt in prompts.items()}
    assert locations["start"] < locations["middle"] < locations["end"]
    assert all(prompt.count("ORION-DELTA-") == 32 for prompt in prompts.values())
