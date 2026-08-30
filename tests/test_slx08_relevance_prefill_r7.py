from __future__ import annotations

from tools.research import run_slx08_relevance_prefill_r7 as runner


def test_freeze_fixtures_uses_live_qwen38_backend_and_restores_temp_base(monkeypatch):
    original = runner.r6.base.TEMP_BASE
    observed = []
    monkeypatch.setattr(runner.r6.base, "health", lambda _port: (200, {"current_model": "qwen38", "backend_healthy": True, "backend_port": 18080}))

    def fixture(case_id: int):
        observed.append(runner.r6.base.TEMP_BASE)
        return {"case_id": case_id, "tokens": [case_id] * 4096}

    monkeypatch.setattr(runner.r6, "build_fixture", fixture)
    fixtures = runner.freeze_fixtures()
    assert len(fixtures) == 64
    assert set(observed) == {"http://127.0.0.1:18080"}
    assert runner.r6.base.TEMP_BASE == original


def test_freeze_fixtures_rejects_wrong_resident_model(monkeypatch):
    monkeypatch.setattr(runner.r6.base, "health", lambda _port: (200, {"current_model": "fable-tc", "backend_healthy": True, "backend_port": 18080}))
    try:
        runner.freeze_fixtures()
    except RuntimeError as error:
        assert "qwen38" in str(error)
    else:
        raise AssertionError("wrong resident model was accepted")
