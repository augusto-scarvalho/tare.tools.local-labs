from tools.research import run_trace_distillation_confirmation_r6 as r6
from tools.research.run_trace_distillation_confirmation_r7 import TASK_ID, configure_r6


def test_r7_changes_only_task_identity_timeout_and_static_bindings():
    configure_r6()
    assert r6.TASK_ID == TASK_ID
    assert len(r6.EXPECTED_STATIC) == 11
    assert r6.r2.systemctl.__name__ == "long_systemctl"
