# Amendment — cross-round prefix-cache control

Status: **FROZEN BEFORE QUALIFIED RERUN**  
Date: 2026-08-22

The preliminary llama.cpp and SGLang blocks reused byte-identical prompts. SGLang prefill throughput
rose from 18,366 to 106,750 tokens/s across the five rounds, proving that the intended fresh-prefill
measurement was confounded by cross-round prefix-cache reuse. Both preliminary JSON/log pairs are
retained with the `.pre-amendment` suffix and are invalid for engine comparison.

The only workload change is a deterministic, equal-form nonce at the **start** of each warm-up and
measured prompt (`Control nonce 0000` through `0005`). This defeats shared-prefix reuse while keeping
the prompts and token lengths matched across engines. All original gates, engine order, generation
settings, and five retained rounds remain unchanged. Decode and TTFT are rerun too; no preliminary
metric is carried into the decision.

The SGLang preflight also found an environment dependency mismatch before model load: Pydantic 2.10.6
with FastAPI 0.141.1 caused `typed_dict_schema(... cls_name)` to fail. The environment repair upgraded
only Pydantic/Pydantic Core to 2.13.4/2.46.4, after which the frozen SGLang 0.5.16 server loaded. The
failed preflight log is retained; the engine version and model/runtime flags did not change.

