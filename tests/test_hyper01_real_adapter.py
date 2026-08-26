from pathlib import Path

from tools.research.hyper01_real_adapter_worker import TARGET_KEY_A, TARGET_KEY_B


def test_physical_target_keys_are_matched_lora_pair():
    assert TARGET_KEY_A.endswith("gate_proj.lora_A.weight")
    assert TARGET_KEY_B.endswith("gate_proj.lora_B.weight")


def test_original_hyper_probe_used_random_target_adapters():
    source = Path("tools/probes/hyper01_capsule_generator.py").read_text(encoding="utf-8")
    assert "target_A = torch.randn" in source
    assert "target_B = torch.randn" in source


def test_successor_has_no_random_target_substitution():
    source = Path("tools/research/hyper01_real_adapter_worker.py").read_text(encoding="utf-8")
    assert "safe_open" in source
    assert "target_A = torch.randn" not in source
    assert "target_B = torch.randn" not in source
