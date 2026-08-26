# BACKLOG-ADAPT-TRACE-DISTILL-02 aborted host orchestration

The preregistered GPU workers themselves did not fail. The host orchestration aborted after the fourth worker because `subprocess.run(text=True)` used the Windows cp1252 default decoder; a byte emitted by WSL stderr caused `UnicodeDecodeError`, after which `completed.stderr` was `None` and the log writer raised `TypeError`.

The persistent inference service was restored by `finally`; ports 8080 and 8081 returned healthy and `NRestarts` remained zero.

Four completed worker JSONs and their checkpoints are preserved under `raw/`:

| Worker | Math | QA | Worker JSON SHA-256 |
|---|---:|---:|---|
| seed 20260824 answer-only | 11/32 | 4/16 | `be30915b4b5c8b402a98953ecd0aba829afbe9fa58be5116616546614ccde79c` |
| seed 20260824 full-trace | 9/32 | 4/16 | `c389a8290effa80b45685516375499a0f0c41a79348b3490e489608d27eaa7db` |
| seed 20260825 answer-only | 9/32 | 4/16 | `a8fcd0a8bb782176840882513e60d9670550e13c1493f994b09f409c55a2ef36` |
| seed 20260825 full-trace | 12/32 | 2/16 | `0220a38e5ae1c74695ac91b25807c85a907fa8d0c949306c59e6be6e35f872f0` |

No receipt, result or claim was produced. A successor must freeze these exact outputs and continue the unchanged preregistered design with the two seed-20260826 workers. It must not change the hypothesis, thresholds or training recipe in response to the partial scores.
