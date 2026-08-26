# BACKLOG-ADAPT-REQUAL-02 preregistration

Task: Requalify saved adapters with process-isolated clean base reloads  
Evidence class: `artifact_requalification`  
Executor: Codex executor  
Date: 2026-08-25

## Hypothesis

When every evaluation arm starts in a distinct WSL Python process that loads a new, unmodified `Qwen/Qwen3.5-0.8B-Base`, the outputs for `base`, `lokr_1ep`, and `target_mlp_only` will be byte-identical under forward and reverse smoke order. If that isolation canary passes, all 13 saved adapters can be requalified without cross-arm PEFT contamination on the frozen 32-item GSM8K and 16-item protected-QA panels.

This experiment tests artifact loading, isolated behavior and deterministic scoring only. It does not select or promote a finalist and does not establish training reproducibility.

## Frozen inputs

- Admission specification: `config/research_backlog_admissions/BACKLOG-ADAPT-REQUAL-02.json`, 2639 bytes, SHA-256 `789eaff0490d7d8f939d3af2445551400b22ae2432ea2a67de07abc616a64f8a`.
- Independent invalidation audit: `docs/AUDIT_2026-08-25_CODEX_INDEPENDENT_AGY_EXECUTION.md`, 16057 bytes, SHA-256 `e4364456156a3c2f015306d986192792fb1aa9ae9333b63a2237ec46e3ffc11f`.
- Frozen 13-adapter ledger: `runs/research/BACKLOG-ADAPT-REQUAL-01/raw/artifact_hashes.json`, 6007 bytes, SHA-256 `b19fa60e5d122219934a1563cdf231dac0a847393327d35b214763711582c5fc`. Every one of its 26 config/weight digests must be recomputed before GPU execution.
- Prior dataset ledger: `runs/research/BACKLOG-ADAPT-REQUAL-01/raw/dataset_hashes.json`, 442 bytes, SHA-256 `ec389b0a3eb63460edd92eef26ca3361966f0bd6b869cd1b47078b008ee2d652`.
- Math dataset: `workloads/gsm8k.jsonl`, 389701 bytes, SHA-256 `68a72276898a45dacb893154477621ff3d05ae7e043a10820644b691d8b63d77`.
- Protected QA dataset: `runs/requalification/QWEN38-HAUHAUCS-NORMAL-QA-2026-08-23/tasks.jsonl`, 11016 bytes, SHA-256 `56434ebd53ff9f5adb477bd902566e28644fe9ea88619834f81417a06c66b84f`.
- Base weights: `/home/augus/models/adapt00/qwen3.5-0.8b-base-dc7cdfe/model.safetensors-00001-of-00001.safetensors`, 1746942600 bytes, SHA-256 `c2b1e5a17d9c1e27685d92ed9b382911ebb99955ecd89052d1721241adfbab6c`.
- Base config: `/home/augus/models/adapt00/qwen3.5-0.8b-base-dc7cdfe/config.json`, 2907 bytes, SHA-256 `b90b86f35c8e6925ef74ee04d0e758f0a845c83a42089ad82bbaa948de9b4204`.
- Base tokenizer: `/home/augus/models/adapt00/qwen3.5-0.8b-base-dc7cdfe/tokenizer.json`, 12807196 bytes, SHA-256 `fe000e3ed39ed12b8d2481d527d44f93c65d37e87645d2dcc80d1bf9d50d2927`.
- Adapter identities and evaluation IDs are exactly those frozen by the admission specification and 13-adapter ledger; no substitution or fallback path is allowed.

## Command

```powershell
python tools/research/run_adapter_requalification_r2.py --outdir runs/research/BACKLOG-ADAPT-REQUAL-02
```

The host command launches each arm as a separate WSL process using `/home/augus/.venvs/adapt00-20260824/bin/python`. No worker may reuse an in-memory base model from another arm.

## Factors

- Smoke order A: `base`, `lokr_1ep`, `target_mlp_only`.
- Smoke order B: `target_mlp_only`, `lokr_1ep`, `base`.
- The smoke compares semantic fields byte for byte: task ID, output text, extracted answer, correctness, generated-token count and natural-EOS flag. Timing is excluded.
- Full run: 1 base plus 13 adapter arms. Outputs from smoke order A are reused for those three arms; the other 11 arms each receive a fresh worker process.
- Expected clean workers: 6 smoke workers plus 11 remaining full workers = 17.
- Math panel: the same 32 frozen GSM8K IDs as R1.
- Protected panel: the same 16 frozen QA IDs as R1.
- Decoding: greedy, `do_sample=False`, no temperature/top-p sampling, seed 20260824, 192 max new tokens for math and 128 for QA.
- Model precision: `torch.float16`, matching R1.
- Hardware: NVIDIA RTX 3090. `llm-inference.service` may be stopped through systemd only if required for VRAM; embedding port 8081 must remain healthy. The original service ExecStart/PID/restart baseline must be recorded and the service restored after execution.
- Scoring is recomputed on the host from `raw/samples.jsonl`; the worker's correctness flags cannot alone satisfy `independent_score`.

## Acceptance gates

- `artifact_identity`: `hashed_artifacts eq 13`.
- `clean_base_per_arm`: `workers_with_zero_preexisting_peft_modules eq 17`.
- `isolation_smoke`: `smoke_order_invariant eq True`.
- `frozen_math_panel`: `scored_math_samples_per_arm ge 32`.
- `frozen_qa_panel`: `scored_qa_samples_per_arm ge 16`.
- `base_control`: `base_control_present eq True`.
- `independent_score`: `independent_scorer_match eq True`.

No performance threshold or finalist-selection gate is part of this packet.

## Abort conditions

- Any frozen SHA-256, file size, adapter path, dataset ID or base-model identity mismatch.
- Any worker reports a PEFT/tuner module before adapter injection.
- Any smoke semantic field differs between order A and order B.
- Any worker exits nonzero, OOMs, produces fewer than 48 samples or writes a task ID outside the frozen panel.
- The embedding endpoint on port 8081 becomes unhealthy.
- The inference service cannot be restored to an active, healthy state after the run.
- Receipt provenance, raw samples, clean-worker receipts or WSL environment evidence is incomplete.

## Allowed claims

- `ARTIFACT_REQUALIFIED_R2`
- `ARTIFACT_REQUALIFICATION_R2_REJECTED`

Claims outside these codes are forbidden even if a metric looks favorable. In particular, this packet cannot promote an adapter finalist.
