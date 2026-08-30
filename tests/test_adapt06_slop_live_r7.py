import pytest

from tools.research.run_adapt06_slop_live_r7 import make_gateway_service


def test_gateway_compatibility_preserves_real_identity_and_exposes_binary():
    responses = iter([
        {"raw": {}, "values": {"ExecStart": "{ path=/usr/bin/python3 ; argv[]=/usr/bin/python3 qualified_model_gateway.py ; ignore_errors=no }", "MainPID": "10", "NRestarts": "0", "ActiveState": "active"}},
        {"raw": {}, "values": {"ExecStart": "{ path=/usr/bin/python3 ; argv[]=/usr/bin/python3 qualified_model_gateway.py ; ignore_errors=no }", "MainPID": "11", "NRestarts": "0", "ActiveState": "active"}},
    ])
    service, observations = make_gateway_service(lambda: next(responses), "/bin/llama-server")
    before, after = service(), service()
    assert "path=/bin/llama-server" in before["values"]["ExecStart"]
    assert "qualified_model_gateway.py" in before["values"]["GatewayExecStart"]
    assert after["gateway_identity"]["gateway_main_pid"] == "11"
    assert len(observations) == 2


def test_gateway_compatibility_rejects_command_drift():
    responses = iter([
        {"raw": {}, "values": {"ExecStart": "qualified_model_gateway.py a"}},
        {"raw": {}, "values": {"ExecStart": "qualified_model_gateway.py b"}},
    ])
    service, _ = make_gateway_service(lambda: next(responses), "/bin/llama-server")
    service()
    with pytest.raises(RuntimeError, match="drifted"):
        service()
