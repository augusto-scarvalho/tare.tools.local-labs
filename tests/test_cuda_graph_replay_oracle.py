from __future__ import annotations

import math

from tools.probes.slx05_launch_oracle import (
    _percentile,
    _restore_cache_state,
    _snapshot_cache_state,
)


def test_percentile_is_deterministic_and_order_independent():
    values = [5.0, 1.0, 3.0, 2.0, 4.0]
    assert _percentile(values, 0.5) == 3.0
    assert _percentile(list(reversed(values)), 0.5) == 3.0
    assert _percentile(values, 0.95) == 5.0


def test_empty_percentile_fails_as_nan():
    assert math.isnan(_percentile([], 0.5))


class _FakeTensor:
    def __init__(self, value):
        self.value = value

    def detach(self):
        return self

    def clone(self):
        return _FakeTensor(self.value)

    def copy_(self, other):
        self.value = other.value


class _FakeLayer:
    def __init__(self):
        self.keys = _FakeTensor("prefill-keys")
        self.values = _FakeTensor("prefill-values")
        self.cumulative_length = _FakeTensor(128)
        self.conv_states = {0: _FakeTensor("prefill-conv")}
        self.recurrent_states = {0: _FakeTensor("prefill-recurrent")}


class _FakeCache:
    def __init__(self):
        self.layers = [_FakeLayer()]


def test_hybrid_cache_snapshot_restores_all_mutable_tensor_state():
    cache = _FakeCache()
    snapshot = _snapshot_cache_state(cache)
    layer = cache.layers[0]
    layer.keys.value = "decode-keys"
    layer.values.value = "decode-values"
    layer.cumulative_length.value = 129
    layer.conv_states[0].value = "decode-conv"
    layer.recurrent_states[0].value = "decode-recurrent"

    _restore_cache_state(cache, snapshot)

    assert layer.keys.value == "prefill-keys"
    assert layer.values.value == "prefill-values"
    assert layer.cumulative_length.value == 128
    assert layer.conv_states[0].value == "prefill-conv"
    assert layer.recurrent_states[0].value == "prefill-recurrent"


def test_hybrid_cache_restore_rejects_topology_drift():
    cache = _FakeCache()
    snapshot = _snapshot_cache_state(cache)
    cache.layers.append(_FakeLayer())

    try:
        _restore_cache_state(cache, snapshot)
    except ValueError as error:
        assert "topology" in str(error)
    else:
        raise AssertionError("topology drift must fail closed")
