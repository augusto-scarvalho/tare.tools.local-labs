import importlib.util
from pathlib import Path
import pytest

spec = importlib.util.spec_from_file_location('decision_worker', Path(__file__).parents[1]/'tools/serving/decision_worker.py')
worker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(worker)


@pytest.fixture
def calls(monkeypatch):
    from types import ModuleType
    import sys
    # No model dependencies or downloads are needed for orchestration tests.
    package = ModuleType('compute_plane'); module = ModuleType('compute_plane.choice_assessment')
    module.validate_request = lambda value: None
    monkeypatch.setitem(sys.modules, 'compute_plane', package)
    monkeypatch.setitem(sys.modules, 'compute_plane.choice_assessment', module)
    calls = []
    def cpu(*_):
        calls.append('cpu')
        return {'choice': 'insufficient', 'model_identity': {}}
    monkeypatch.setattr(worker, 'cpu_assess', cpu)
    return calls


def test_busy_gpu_falls_back_once_without_model_switch(calls, monkeypatch):
    def gpu(*_): raise worker.Unavailable('GPU_LEASE_BUSY')
    monkeypatch.setattr(worker, 'gpu_assess', gpu)
    value = worker.evaluate({}, {})
    assert calls == ['cpu'] and value['model_identity']['fallback_reason'] == 'GPU_LEASE_BUSY'


def test_semantic_abstention_never_shops_for_another_answer(calls, monkeypatch):
    monkeypatch.setattr(worker, 'gpu_assess', lambda *_: {'choice': 'insufficient'})
    assert worker.evaluate({}, {})['choice'] == 'insufficient'
    assert not calls


def test_inference_failure_does_not_launch_second_evaluator(calls, monkeypatch):
    def gpu(*_): raise ValueError('ASSESSMENT_MISSING_CANDIDATE')
    monkeypatch.setattr(worker, 'gpu_assess', gpu)
    with pytest.raises(ValueError): worker.evaluate({}, {})
    assert not calls


def test_no_ram_fails_closed(calls, monkeypatch):
    def gpu(*_): raise worker.Unavailable('GPU_LEASE_BUSY')
    def cpu(*_): raise ValueError('ASSESSMENT_INSUFFICIENT_RAM')
    monkeypatch.setattr(worker, 'gpu_assess', gpu); monkeypatch.setattr(worker, 'cpu_assess', cpu)
    with pytest.raises(ValueError, match='INSUFFICIENT_RAM'): worker.evaluate({}, {})
