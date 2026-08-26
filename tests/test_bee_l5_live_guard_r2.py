from tools.research import run_bee_l5_live_guard_r2 as target


def test_tokenize_extracts_active_server_record_shape(monkeypatch):
    monkeypatch.setattr(target.core, "post_json", lambda *_: {"tokens": [{"id": 1, "piece": " wait"}, {"id": 2, "piece": " now"}]})
    assert target.tokenize("ignored") == [" wait", " now"]
