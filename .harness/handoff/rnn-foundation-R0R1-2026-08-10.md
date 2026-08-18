# HANDOFF — RNN Foundation Packet R0/R1 (2026-08-10)

Opening packet of the recurrent/neural-memory research line. **Foundation only** — research ledger,
architecture matrix, Gated-DeltaNet state archaeology, checkpoint/restore, RULER harness qualification.
**No training, no llama.cpp changes, no deploy-profile changes, CPU-only.** Serving campaign stays
CLOSED. Not pushed.

## 1. Git identities
- Branch **master**. **Starting HEAD `4f4e459`** → **Ending HEAD `09a012e`**.
- Commit `09a012e` *research(rnn): RNN Foundation R0/R1 — ledger, arch matrix, GDN state archaeology*
  (14 files, +1571). Diffstat at the bottom of this file.
- Working tree after commit: **only `.harness/` untracked** (handoffs, incl. this file — kept untracked
  by lab convention). `trackedTreeClean=true`, `stagedTreeClean=true`, **not pushed** (no upstream).
- Excluded by design: model weights, venvs, caches, cloned upstream repos, the 6.8 MB state-tensor dump
  (its result — disk round-trip BIT_EXACT — is recorded in JSON instead).

## 2. CURRENT reconstruction (§3)
Full table in `RNN_RECONCILIATION.md`. Highlights:
- **Reused, not rebuilt:** `analysis/robust.py` (paired stats), the LAB-QA benchmark-identity discipline
  (`tests/benchmark_harness/`, `benchmark_harness_qa.py`), the `runs/**` evidence convention, and the
  lab's existing **GDN kernel math** (`GDN_KERNEL.md`, validated to 1e-16).
- **§3's named files `CURRENT_STATE.md`/`FINDINGS.md` do not exist** — read `STATUS.md`,
  `EXPERIMENTS.md`, `MECHANISMS.md`, `BACKLOG_V2_STATUS.md`, `GDN_*.md`, serving handoffs instead.
- **Prior GDN work was kernel-level only** (chunk-parallel prefill in the fork); GDN as a *research /
  mutable-memory* substrate was MISSING before this packet.

## 3. Research ledger (§4/§5) — `RNN_RESEARCH_LEDGER.md`
18 works + 3 repos, each verified from primary sources (arXiv + official GitHub; commit SHA + LICENSE
read this packet). Pinned commits incl. FLA `7843b32` (MIT), RULER `c3f5e3b` (Apache-2.0), Gated DeltaNet
`b53d6d3` (**NVIDIA NC**), TTT-pytorch `cd831db` (MIT), In-Place-TTT `be23248` (Apache-2.0), LaCT
`a648340` (MIT), MambaInLlama `b03f123` (Apache-2.0), RADLADS `1b362eb` (Apache-2.0), TPTT `242e214`
(Apache-2.0), LoLCATs `375df84` (Apache-2.0), Liger `0b364eb` (Apache-2.0).
- **OFFICIAL_CODE_NOT_FOUND: Memory Caching, Titans, ATLAS, MIRAS.** Memory Caching's only impl is the
  community `sypherin/growing-memory` (app-layer, MIT) → `COMMUNITY_IMPLEMENTATION`, not authoritative.
- Licensing watch-outs: Gated DeltaNet (NVlabs) is **non-commercial** — use the FLA (MIT) reimpl;
  Longhorn repo has **no LICENSE**.

## 4. Architecture matrix (§6/§7) — `RNN_ARCHITECTURE_MATRIX.{csv,json}`
Verified from official config.json (not memory):
- **Qwen3.5-0.8B** and **Qwen3.6-27B** are **dense-FFN Gated-DeltaNet + gated-attention hybrids**
  (`model_type: qwen3_5_text`, `full_attention_interval: 4` → 3:1 linear:full, `attn_output_gate: true`),
  **not MoE**. 0.8B = 24 layers (18 linear/6 full, hidden 1024, `linear_num_value_heads 16`); 27B = 64
  layers (48 linear/16 full, hidden 5120, `linear_num_value_heads 48`). **27B `text_config` == local
  `fp16/base`.** Smallest official Qwen GDN hybrid = **Qwen3.5-0.8B**. Older lineage: Qwen3-Next-80B
  (`qwen3_next`, MoE).

## 5. Qwen GDN state — source excerpts + shapes + bytes (§9) — `RNN_STATE_MODEL.md`
Source: `transformers/models/qwen3_5/modeling_qwen3_5.py` (installed, transformers 5.12.1):
- `class Qwen3_5GatedDeltaNet` **L371**; `conv_dim = key_dim*2 + value_dim` **L389**; depthwise
  `nn.Conv1d(kernel=4)` **L390-397**.
- recurrent state alloc `torch.zeros(batch, num_heads, k_head_dim, v_head_dim, dtype=value.dtype)`
  **L346-347**; gated-delta update **L359-363**; cache read **L457-458**; writes `update_conv_state`
  **L487** / `update_recurrent_state` **L550**; prefill(`chunk_gated_delta_rule`)/decode(`recurrent_
  gated_delta_rule`) select **L524-541**. `class Qwen3_5ForCausalLM` **L1613**. Cache = `DynamicCache`,
  linear layer object = `LinearAttentionLayer`.
- **Same recurrence as the lab's ggml kernel** (`GDN_KERNEL.md`) — independent cross-check.

Per-linear-layer state (OBSERVED, real 0.8B dims): `recurrent_states [1,16,128,128]` fp32 = **1.0 MiB**;
`conv_states [1,6144,4]`. **Per-request GDN memory is constant in sequence length:** **~18.84 MiB
(0.8B, 18 linear)**, **~147.75 MiB (27B, 48 linear)** — only the ¼ full-attention layers carry a growing
KV cache. (OBSERVED per-layer × COMPUTED layer counts.)

## 6. Checkpoint / restore / branch (§10/§11) — raw in `runs/rnn/RNN-01-gdn-state/`
Deterministic CPU fp32, greedy:

| test | max|Δ| | verdict |
|---|---:|---|
| checkpoint → destroy → restore → continue | 0.0 | **BIT_EXACT** |
| disk round-trip (`torch.save`/`load`) | 0.0 | **BIT_EXACT** |
| cached-incremental decode vs full recompute | 7.75e-6 | **NUMERICALLY_EQUIVALENT** (≤1e-4; kernel fp non-assoc.) |
| RNN-03 branch A vs B separation | 5.66 | branches independent |
| RNN-03 each branch restored & reproduced | 0.0 | **BIT_EXACT** |

⇒ The GDN recurrent state is a cleanly **checkpointable / restorable / serializable / forkable** object.
The RNN-02/03 state-semantics gate is **passed**. (Weight-independent ⇒ transfers to the real model.)

## 7. Benchmark qualification (§8) — `runs/rnn/RNN-00B-ruler/rnn00b_ruler_smoke.json`
RULER-style single-needle harness smoke, LAB-QA philosophy: tokenizer identity (fp16/base, vocab 248044,
files sha `66647ee6…`), exact 4096-token delivery (**no silent truncation**), needle survives window,
known-good PASS / known-bad FAIL, reproducible (same seed identical, diff seed differs) →
**`RULER_HARNESS_DISCIPLINE_QUALIFIED`**. Upstream RULER pinned (`c3f5e3b`, Apache-2.0); **full
benchmark-vs-model run DEFERRED** (needs GPU). BABILong/NoLiMa recorded, not built (§8 allows RULER-only).

## 8. Failures / negative evidence (honesty)
- `flash-linear-attention` + `causal-conv1d` **not installed** → transformers pure-torch GDN fallback
  (logged). Fine for correctness/determinism; a real GPU-perf study would install FLA.
- The official Qwen3.5-0.8B checkpoint is a **VLM wrapper** (`Qwen3_5ForConditionalGeneration`) — loading
  just its text stack is awkward; the faithful **surrogate** (real per-layer GDN dims) sidesteps this and
  is sufficient for shape/byte/semantics. Real-weight magnitude capture NOT done (optional).
- No negative *result* to report (all gates passed); no benchmark artifact, no contamination — none
  tested against a real model yet.

## 9. Exact commands (reproduce; CPU-only)
```
cd /mnt/c/projects/local-model-lifecycle
/home/augus/sglang-venv/bin/python ops/rnn_gdn_state_probe.py \
    --outdir runs/rnn/RNN-01-gdn-state --branching
/home/augus/sglang-venv/bin/python ops/rnn_ruler_smoke.py \
    --outdir runs/rnn/RNN-00B-ruler --target-tokens 4096
/home/augus/sglang-venv/bin/python ops/rnn_arch_matrix_gen.py
```

## 10. Next-experiment cost estimates (§13)
**RNN-04 — Memory Caching reference reproduction**
| field | estimate |
|---|---|
| official code | **none** (OFFICIAL_CODE_NOT_FOUND) → reimplement aggregation from paper (correctness risk) |
| model | small FLA recurrent LM (DeltaNet/GDN ~125–350M) — NOT Qwen |
| VRAM / RAM | fits 24 GB easily (≤4 GB train for ~160M); RAM modest |
| disk | FineWeb subset ~10–50 GB (if LM) — or ~0 for synthetic MQAR |
| GPU time | **mechanism-only on synthetic MQAR/small-RULER: single-digit GPU-hours**; a real from-scratch LM (paper is 760M@30B / 1.3B@100B tok) is multi-day → **out of scope** |
| training tokens | synthetic: ≪1B; LM signal: 1–5B (weak) |
| artifacts | quality vs cached-state-count curve (Pareto) |
| main risk | reimplementing the aggregation/selection correctly with no reference code |
→ **Scope RNN-04 to synthetic associative-recall, mechanism-only.** Full-LM repro is PARKED.

**RNN-08 — TPTT reference reproduction**
| field | estimate |
|---|---|
| official code | **yes** — `fabienfrfr/tptt` @ `242e214` (Apache-2.0) + PyPI `tptt` |
| model | smallest upstream example (Gemma3-270m or Qwen2.5-1.5B) |
| VRAM / RAM | LoRA on ≤1.5 B ~6–12 GB → **comfortable on the 3090** |
| disk | base model 0.5–3 GB + small alignment set |
| GPU time | LoRA alignment smoke **~1–4 GPU-hours** |
| training tokens | small (parameter-efficient); + mandatory BASE / LoRA-only / TPTT+LoRA arms (RNN-09) |
| artifacts | MMLU-subset: BASE vs LoRA-only vs TPTT+LoRA (isolates whether recurrence, not LoRA, helps) |
| main risk | **dependency compat** — `tptt`/FLA vs installed transformers 5.12.1 (may need an isolated venv / version pin) |
→ **Feasible on the 3090** with the mandatory LoRA control.

## 11. Exactly one recommended next packet
**RNN-08 TPTT reference smoke (≤1.5 B, LoRA) with the mandatory RNN-09 LoRA control — beginning with a
dependency-feasibility canary.** Rationale: it has official Apache-2.0 code, directly tests the lab's
core question (memory gain *without* full pretraining) on a pretrained Transformer, fits the 3090in a
few GPU-hours, and forces the BASE / LoRA-only / TPTT+LoRA discipline early so no gain is misattributed
to recurrence. First step is a cheap **dependency canary** (isolated venv: install `tptt`+FLA, load the
smallest example, confirm forward + a few LoRA steps, log VRAM/time) *before* committing GPU — matching
the lab's canary-first method, and directly de-risking the one identified failure mode (version compat).
*(Alternative if TPTT deps prove hostile: RNN-04 scoped to synthetic MQAR, mechanism-only.)*

Optional cheap addenda (not the packet): install FLA; capture real-weight Qwen3.5-0.8B state magnitudes
to append to RNN-01.

## 12. Guardrails (unchanged)
Do NOT reopen serving (LAB-SERVE-001d/002/soak/energy); do NOT modify the deploy profile or the
llama.cpp fork; do NOT integrate tare.tools; do NOT start meaningful training in this line without an
explicit next packet. Handoffs stay untracked under `.harness/`.

## Diffstat (commit 09a012e)
```
 RNN_ARCHITECTURE_MATRIX.csv                        |   8 +
 RNN_ARCHITECTURE_MATRIX.json                       | 157 ++++
 RNN_RECONCILIATION.md                              |  47 ++
 RNN_RESEARCH_LEDGER.md                             | 176 +++++
 RNN_STATE_MODEL.md                                 | 132 +++
 ops/rnn_arch_matrix_gen.py                         | 154 +++
 ops/rnn_gdn_state_probe.py                         | 333 ++++++
 ops/rnn_ruler_smoke.py                             | 129 ++
 runs/rnn/RNN-00B-ruler/rnn00b_ruler_smoke.json     |  42 ++
 runs/rnn/RNN-01-gdn-state/RNN_STATE_INVENTORY.json | 333 ++++++
 runs/rnn/RNN-01-gdn-state/RNN_STATE_SIZES.csv      |  17 ++
 runs/rnn/RNN-01-gdn-state/rnn01_inventory.txt      |  21 ++
 runs/rnn/RNN-01-gdn-state/rnn02_checkpoint_restore.json | 14 +
 runs/rnn/RNN-01-gdn-state/rnn03_branching.json     |   8 +
 14 files changed, 1571 insertions(+)
```
