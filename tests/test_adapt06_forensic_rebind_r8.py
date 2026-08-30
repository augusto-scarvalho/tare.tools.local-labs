from tools.research.run_adapt06_forensic_rebind_r8 import route_lora, route_switches, text_sha256


def test_utf8_digest_is_over_exact_unicode_text_bytes():
    assert text_sha256("ação") == "0664077f33cc3ebbaa4bbdacac0eb70e740983080f01dce29929e73b7785a7ad"
    assert text_sha256("ação") != text_sha256("aÃ§Ã£o")


def test_route_controls_are_explicit_for_both_adapters():
    assert route_lora("base") == [{"id": 0, "scale": 0.0}, {"id": 1, "scale": 0.0}]
    assert route_lora("mlp") == [{"id": 0, "scale": 1.0}, {"id": 1, "scale": 0.0}]
    assert route_lora("attn") == [{"id": 0, "scale": 0.0}, {"id": 1, "scale": 1.0}]


def test_switch_counter_counts_transitions_not_rows():
    rows = [{"route": route} for route in ("base", "mlp", "base", "base", "attn")]
    assert route_switches(rows) == 3
