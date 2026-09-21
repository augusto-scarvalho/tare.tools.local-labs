import sys

import pytest

from tools.analysis import probe_openjev


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
