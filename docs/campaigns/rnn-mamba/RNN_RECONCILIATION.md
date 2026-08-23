# RNN backlog ↔ current repo reconciliation (§3)

Packet: RNN Foundation R0/R1 · 2026-08-10 · HEAD `4f4e459`. Marks each RNN-backlog need against what
the repo already has. Rule (§3, §14): **reuse existing infrastructure; do not build parallel systems.**

**Note on §3's named files:** `CURRENT_STATE.md` and `FINDINGS.md` **do not exist** in this repo. The
actual equivalents that were read instead: `STATUS.md`, `EXPERIMENTS.md`, `MECHANISMS.md`,
`BACKLOG_V2_STATUS.md`, `GDN_KERNEL.md`/`GDN_NEXT_LEVERS.md`/`GDN_TF32_PLAN.md`, and the closed serving
campaign under `runs/serving/` + `.harness/handoff/`.

## ALREADY_DONE — reuse, do not rebuild
| Need | Existing asset | Reuse |
|---|---|---|
| Robust stats for small-n paired evals | `src/model_lifecycle/analysis/robust.py` (`bootstrap_ci`, `hodges_lehmann`, `sign_test_p`, `cliffs_delta`, `non_inferiority`, `trend_slope_ci`, `min_rounds_for`) | Any RNN quality/latency eval reuses this — no new stats code |
| Benchmark-instrument QA discipline (LAB-QA-001) | `tests/benchmark_harness/` (23 cases, green), `benchmark_harness_qa.py` (`run_identity`, `{stem}__identity.json`), `DATASET_IDENTITY.json` | **RNN-00B RULER smoke reused this philosophy** (tokenizer identity, known-good/bad, no silent truncation, reproducibility) |
| GDN recurrence math (kernel level) | `GDN_KERNEL.md`: chunk-parallel gated-delta kernel, **validated to ~1e-16** vs sequential | Cross-checked against HF `torch_recurrent_gated_delta_rule` — same rule; we did NOT re-derive the math |
| Model registry / serve profiles | `src/model_lifecycle/models.py`, `serve_profiles.py` | Registry pattern for any new RNN model artifact |
| Run/evidence storage | `runs/**` convention | New evidence under `runs/rnn/**` (no parallel store) |
| Deterministic GDN in PyTorch | transformers 5.12.1 `qwen3_5` / `qwen3_next` (installed) with pure-torch fallbacks | RNN-01/02/03 ran on it; **no custom kernels written** (§14) |

## PARTIAL — started this packet, bounded
| Need | State |
|---|---|
| GDN **state** archaeology (RNN-01) | DONE for shapes/dtypes/bytes/semantics via faithful surrogate; **real-weight magnitude capture NOT done** (optional) |
| Checkpoint/restore/branch semantics (RNN-02/03) | DONE, BIT_EXACT (weight-independent) — gate cleared |
| Research ledger (RNN-00) | DONE for 18 works + 3 repos (primary-source verified); not every peripheral repo cloned (§14) |
| Architecture matrix (RNN-00A) | DONE (7 rows, Qwen 0.8B/27B verified from config) |
| Eval instrument qual (RNN-00B) | RULER **harness discipline** qualified; **actual RULER-vs-model run deferred** (GPU); BABILong/NoLiMa recorded, not built (§8 allows RULER-only) |

## MISSING — not present, not started (by design)
- Long-context benchmark *integration* (RULER/BABILong/NoLiMa runners against a model) — needs GPU/serving.
- Any TTT / Titans / Memory-Caching / HAM / TPTT / LoLCATs / Liger implementation — none in repo; ledger
  records where each lives. Memory Caching has **OFFICIAL_CODE_NOT_FOUND**.
- `flash-linear-attention` / `causal-conv1d` not installed (pure-torch fallback used). Installing FLA is
  the natural first dependency **if** a GDN/linear-attn experiment is authorized (§14) — not done now.
- Custom memory eval suite (revision/interference/contamination, RNN-EVAL-*) — not started.

## BLOCKED — gated on an explicit decision (not hard-blocked)
- **RNN-04** (Memory Caching reference reproduction) — no official code; from-scratch pretraining at
  760M–1.3B. Cost estimate in the handoff. Gated on authorization; **PARK** direct repro.
- **RNN-08** (TPTT reference reproduction) — feasible on a 3090 (LoRA, ≤7B); cost estimate in handoff.
- Scale to Qwen3.6-27B (RNN-31/32) — explicitly gated behind small-model positive evidence (not near).

## Standing guardrails carried in (unchanged)
Serving campaign is **CLOSED** at `4f4e459`; do not reopen LAB-SERVE-001d / 002 / soak / energy. Do not
modify the deploy profile or `llama.cpp` fork. Do not integrate with tare.tools. Handoffs live untracked
under `.harness/`.
