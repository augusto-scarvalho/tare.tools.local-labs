# Amendment — vLLM WSL UVA recovery

Status: **FROZEN BEFORE RECOVERY GENERATION**  
Date: 2026-08-22

The first vLLM 0.27.1 startup stopped before model load with `RuntimeError: UVA is not available`.
The vLLM environment-variable reference documents `VLLM_WSL2_ENABLE_PIN_MEMORY=1` specifically for
WSL2 when pinned memory or UVA is required by the v2 model runner. One recovery attempt is admitted
with only that environment variable added. Model, dtype, engine version, memory fraction, workload,
and all decision gates remain unchanged.

The failed startup JSON and log are retained with the `.pre-uva-recovery` suffix. If this documented
recovery does not produce a valid server and ten valid probes, classify the vLLM arm
`BLOCKED_RUNTIME`; do not substitute another model, quantization, or unregistered engine version.

