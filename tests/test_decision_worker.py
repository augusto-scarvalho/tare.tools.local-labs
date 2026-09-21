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


@pytest.mark.parametrize('write_fails', [False, True])
def test_hard_stop_writes_protocol_even_when_stdout_redirected(monkeypatch, write_fails):
    import json
    seen = []
    class Exit(BaseException): pass
    def write(fd, payload):
        assert fd == 1
        seen.append(json.loads(payload))
        if write_fails: raise OSError('closed protocol pipe')
    def exit_(code):
        assert code == 124
        raise Exit()
    monkeypatch.setattr(worker.os, 'write', write)
    monkeypatch.setattr(worker.os, '_exit', exit_)
    with pytest.raises(Exit): worker.exit_with_failure('ASSESSMENT_TIMEOUT', 'CPU_SCORING', 124)
    assert seen == [{'failure': 'ASSESSMENT_TIMEOUT', 'stage': 'CPU_SCORING'}]


def test_cpu_capacity_refusal_precedes_model_load(tmp_path, monkeypatch):
    import sys
    from types import SimpleNamespace
    module = SimpleNamespace()
    def reject(*args): raise ValueError('ASSESSMENT_CAPACITY_EXCEEDED')
    module.preflight_cpu_tokens = reject
    module.OpenJevChoiceModel = lambda *_: pytest.fail('model loaded despite capacity refusal')
    monkeypatch.setitem(sys.modules, 'compute_plane.openjev_choice_worker', module)
    monkeypatch.setitem(sys.modules, 'psutil', SimpleNamespace(
        virtual_memory=lambda: SimpleNamespace(available=40 * 1024**3)))
    monkeypatch.setitem(sys.modules, 'fcntl', SimpleNamespace(LOCK_EX=1, LOCK_NB=2, flock=lambda *_: None))
    handlers = {}
    alarms = []
    def signal_(number, handler):
        old = handlers.get(number, 'prior')
        handlers[number] = handler
        return old
    monkeypatch.setattr(worker.signal, 'SIGALRM', 14, raising=False)
    monkeypatch.setattr(worker.signal, 'signal', signal_)
    monkeypatch.setattr(worker.signal, 'alarm', alarms.append, raising=False)
    with pytest.raises(ValueError, match='CAPACITY_EXCEEDED'):
        worker.cpu_assess({}, {'cpu_lock': str(tmp_path / 'cpu.lock'), 'checkpoint': 'fixture',
                              'cpu_max_total_tokens': 2500})
    assert alarms == [52, 0]
    assert set(handlers.values()) == {'prior'}
