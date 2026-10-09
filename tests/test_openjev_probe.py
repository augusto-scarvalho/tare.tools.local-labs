import os
import sys
from types import ModuleType

import pytest

from tools.analysis import probe_openjev


@pytest.fixture(autouse=True)
def lease_module(monkeypatch):
    # inherited_lease lives in tare.tools.node, which tests that a flag alone is no authority; here only the
    # probe's use of it: no lease refuses, a refused lease stops the probe, both before inputs are read.
    def inherited_lease():
        if os.environ.get('TARE_GPU_TRAINING_LEASE') is None:
            return None
        raise ValueError('Inherited GPU lease is invalid or no longer held; refusing unprotected execution.')
    module = ModuleType('tare_node.gpu_lease_run')
    module.inherited_lease = inherited_lease
    monkeypatch.setitem(sys.modules, 'tare_node', ModuleType('tare_node'))
    monkeypatch.setitem(sys.modules, 'tare_node.gpu_lease_run', module)


def test_gpu_probe_requires_verified_parent_lease_before_reading_inputs(monkeypatch, tmp_path):
    monkeypatch.delenv('TARE_GPU_TRAINING_LEASE', raising=False)
    monkeypatch.setattr(sys, 'argv', ['probe', '--device', 'cuda', '--kernel-root', str(tmp_path),
        '--checkpoint', 'missing', '--protocol', 'missing', '--output', str(tmp_path/'out.json')])
    with pytest.raises(ValueError, match='GPU_PROBE_REQUIRES_SUPERVISING_LEASE'):
        probe_openjev.main()
    assert not (tmp_path/'out.json').exists()


def test_untrusted_inherited_lease_flag_is_not_gpu_authority(monkeypatch, tmp_path):
    monkeypatch.setenv('TARE_GPU_TRAINING_LEASE', '{}')
    monkeypatch.setattr(sys, 'argv', ['probe', '--device', 'hybrid', '--kernel-root', str(tmp_path),
        '--checkpoint', 'missing', '--protocol', 'missing', '--output', str(tmp_path/'out.json')])
    with pytest.raises(ValueError, match='Inherited GPU lease is invalid'):
        probe_openjev.main()
    assert not (tmp_path/'out.json').exists()
