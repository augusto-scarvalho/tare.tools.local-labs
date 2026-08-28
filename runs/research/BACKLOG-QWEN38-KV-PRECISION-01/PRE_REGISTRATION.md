# BACKLOG-QWEN38-KV-PRECISION-01 preregistration

Task: Qualify Qwen3.8 Q4_0 KV cache against F16 on physical serving
Evidence class: `serving_runtime`
Executor: Codex executor
Date: 2026-08-26

## Hypothesis

On the frozen Qwen3.8 runtime, replacing F16 K/V cache with physical Q4_0 K/V
cache will save at least 1,000 MiB of resident GPU memory while retaining at
least 90% extracted-answer parity, losing at most one of 32 GSM8K answers and
retaining at least 95% of F16 decode throughput. Any gate failure rejects the
bounded configuration claim.

## Frozen inputs

- Fleet registry: `config/qualified_model_fleet.json`, SHA-256 `042fedf5907f031fb9993c03058f3cc9c8fe2c8d75a3235ea4b5e11c7412cd82`.
- Math panel: `workloads/gsm8k.jsonl`, SHA-256 `68a72276898a45dacb893154477621ff3d05ae7e043a10820644b691d8b63d77`.
- Protected-panel registry: `runs/requalification/QWEN38-HAUHAUCS-NORMAL-QA-2026-08-23/tasks.jsonl`, SHA-256 `56434ebd53ff9f5adb477bd902566e28644fe9ea88619834f81417a06c66b84f`.
- Consolidated handoff: `docs/HANDOFF_2026-08-26_CONSOLIDATED_RESEARCH_BACKLOG.md`, SHA-256 `895fec3ac345bdf26350b4a97f513bf4f4b3bad9898d09701db07a985f8b7d55`.
- Binary: `/home/augus/opt/slop.cpp/b10165-71676e46c/bin/llama-server`, 17,920 bytes, SHA-256 `efb2f06c19d26605a1934c0a9ed5b65dd69034e8765f2d29d0426b7a011cfbe2`.
- Model: `/home/augus/models/qwen38-27b/unsloth/Qwen3.8-27B-UD-Q4_K_XL.gguf`, 17,923,394,624 bytes, SHA-256 `bee238bbeb3dc0a34bde4d0dedbaee1f98c009e8bb4226f03070054c12fb1372`.

## Command

```powershell
python tools/research/run_qwen38_kv_precision.py --outdir runs/research/BACKLOG-QWEN38-KV-PRECISION-01
```

## Factors

- Four fresh-process blocks in `F16, Q4_0, Q4_0, F16` order.
- Both K and V cache precision change together; binary, model, context, MTP,
  prompt order, seed and decode settings remain fixed.
- Each block has four discarded warmups and the same first 32 frozen GSM8K
  tasks, totaling 128 recorded responses and 64 paired cross-precision cases.
- Runtime: RTX 3090, context 32,768, FlashAttention, one slot, MTP depth 3,
  temperature 0, top-k 1, seed 20260826 and 256 output-token cap.
- Resident GPU memory is sampled after health and discarded warmups. The
  physical saving is the median F16 resident allocation minus median Q4_0
  resident allocation under the same background service state.
- Accuracy is rescored from frozen numeric answers; throughput uses server
  timing counters, not wall-clock model-load time.

## Acceptance gates

- `treatment_identity`: `explicit_cache_controls_verified eq True`
- `balanced_crossover`: `valid_abba_blocks eq 4`
- `request_integrity`: `successful_response_rate eq 1.0`
- `semantic_parity`: `extracted_answer_parity_rate ge 0.9`
- `quality_non_regression`: `q4_accuracy_regression le 0.03125`
- `physical_memory_saving`: `vram_saving_mib ge 1000`
- `throughput_non_regression`: `q4_vs_f16_throughput_ratio ge 0.95`
- `service_recovery`: `service_and_embedding_restored eq True`

## Abort conditions

- Any frozen source, binary or model identity differs.
- The inference gateway or embedding endpoint is unhealthy before maintenance.
- The persistent inference service cannot be stopped through systemd, port
  18080 remains occupied, or a reserved transient unit already exists.
- A block fails to load, reports the wrong cache-type argv, runs out of memory,
  or produces three consecutive request errors.
- Port 8081 becomes unhealthy at a block boundary.
- The original gateway service and initially resident model cannot be restored.

Wrong math answers and cross-arm semantic differences are evidence and do not
abort safe remaining blocks.

## Allowed claims

- `QWEN38_Q4_KV_PHYSICALLY_QUALIFIED_R1`
- `QWEN38_Q4_KV_PHYSICALLY_REJECTED_R1`

Claims outside these codes are forbidden even if a metric looks favorable.
The executor stops at `EXECUTED`; independent review is required for promotion.
