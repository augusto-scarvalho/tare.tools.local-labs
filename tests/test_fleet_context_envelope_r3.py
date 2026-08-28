from tools.research import run_fleet_context_envelope_r3 as target


def test_backend_token_count_uses_verified_backend(monkeypatch):
    called = {}
    monkeypatch.setattr(target.r2.fleet, "gateway_status", lambda: {
        "backend_port": 18080, "backend_healthy": True})
    def fake_http(url, payload, timeout):
        called.update({"url": url, "payload": payload, "timeout": timeout})
        return 200, {"tokens": [1, 2, 3]}
    monkeypatch.setattr(target.r2.fleet, "http_json", fake_http)
    assert target.backend_token_count("hello") == 3
    assert called["url"] == "http://127.0.0.1:18080/tokenize"
    assert called["payload"]["add_special"] is False


def test_backend_identity_is_fail_closed(monkeypatch):
    monkeypatch.setattr(target.r2.fleet, "gateway_status", lambda: {
        "backend_port": 9999, "backend_healthy": True})
    try:
        target.backend_token_count("hello")
    except RuntimeError as error:
        assert "unexpected tokenizer backend identity" in str(error)
    else:
        raise AssertionError("wrong backend port was accepted")
