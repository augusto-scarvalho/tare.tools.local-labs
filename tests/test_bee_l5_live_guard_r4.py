from tools.research import run_bee_l5_live_guard_r4 as target


def test_tokenize_normalizes_string_and_byte_array(monkeypatch):
    monkeypatch.setattr(target.core,"post_json",lambda *_:{"tokens":[{"piece":" wait"},{"piece":[32,195]}]})
    assert target.tokenize("x")==[" wait"," �"]
