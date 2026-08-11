# RNN-06-P0 — Frozen-Checkpoint BASE Regime Scout · PROTOCOL

**Status: EXPLORATORY pre-packet. NOT confirmatory. This protocol is written and committed BEFORE substantive GPU outcome work.**

Authored 2026-08-11. Sequence position (from the accepted RNN-06 research/design packet, HEAD `7f04f48`): **`P0 → 06A → 06B → 06C → 06D`**. This is P0.

## 0. What P0 is and is not

P0 answers exactly one bounded question:

> On a frozen real pretrained recurrent LM, can we cheaply find a plausible **non-ceiling, non-cliff** memory-pressure regime suitable for later confirmatory qualification?

P0 **MUST NOT**:
- fine-tune / train either backbone; modify weights;
- run or modify Qwen; modify llama.cpp/SGLang/vLLM;
- implement checkpoint/restore, historical-state substitution, linear probes/readers, Memory Caching, or any recovery machinery;
- mint `FIXED_BACKBONE_GRADED_REGION = QUALIFIED` (only RNN-06B may);
- call any P0 result confirmatory; push anything.

P0's only band statuses are **`P0_GRADED_BAND = PLAUSIBLE | NOT_FOUND_WITHIN_BUDGET | MODEL_NOT_RUNNABLE`**, plus candidate-level `MODEL_RUNNABLE ∈ {YES,NO}` and `TASK_COMPETENT ∈ {YES,NO,N/A}`.

`PLAUSIBLE` means only: *a pressure region appears suitable for a later independent confirmatory experiment.* It is **not** a qualification verdict.

## 1. Candidates (order)

- **A — PRIMARY (as designed):** `linear-moe-hub/Gated-Deltanet-1.3B` (real pretrained Gated DeltaNet via `flash-linear-attention`; the direct pretrained analog of the Qwen-GDN deployment mechanism).
- **B — anchor / fallback:** `AntonV/mamba2-1.3b-hf` (HF-format conversion of `state-spaces/mamba2-1.3b`; transformers-native `Mamba2ForCausalLM`). Independent scientific anchor for whether the construction can find a graded state-capacity regime in a real recurrent model. **Not** reshaped to force GDN to match it; if GDN is flat-high and Mamba shows the useful regime, that outcome is preserved.

Do not spend the whole budget fishing for a GDN band. Sub-GPU-hour scout per candidate.

### Candidate substrate resolution (append-only; recorded BEFORE the outcome-bearing sweep — runnability scouting is P0's explicit authority §2)
Determined by load-only smoke tests on the pinned stack (torch 2.6.0+cu124, fla 0.5.2, transformers 4.48.3, Triton 3.2.0):
- **`linear-moe-hub/Gated-Deltanet-1.3B` = MODEL_NOT_RUNNABLE on this pinned stack.** Its checkpoint uses a **fused gate+up MLP projection** (`mlp.gate_proj` = `[11264, 2048]` = 2×5632, with `mlp.down_proj` = `[2048, 5632]`), whereas fla 0.5.2's `GatedDeltaNetMLP` expects **separate** `gate_proj`/`up_proj` each `[5632, 2048]`. The config also under-specifies `intermediate_size` (`None`, `hidden_ratio=4` → fla recomputes 5632). Loading fails with a `state_dict` size mismatch. A fused→split remap is **not attempted in a scout** because the concat order (gate-first vs up-first) is unverifiable from the artifacts and a wrong split silently produces an incorrect model. **Deferred to RNN-06A** (pin the exact authoring fla revision or verify the remap). This is genuine negative tooling evidence, not a scientific result about GDN.
- **Runnable delta-rule–family arm (substituted PRIMARY):** **`fla-hub/delta_net-1.3B-100B`** — the official fla **DeltaNet** checkpoint (ungated delta rule), a grade-A candidate in the design packet ("ungated mechanism-proper arm", Phase-9 matrix). Loads cleanly (1.366 B params, `DeltaNetForCausalLM`, `LlamaTokenizerFast` vocab 32000, ~3.0 GB VRAM bf16). DeltaNet shares the delta-rule matrix-state mechanism with GDN minus the decay gate; it is a real pretrained recurrent LM with first-class recurrent state. **Qwen-transplant proximity is slightly lower than GDN** (Qwen uses *gated* DeltaNet); GDN-proper remains for RNN-06A once the fla revision is pinned. Per the packet, **the phenomenon (graded regime on a real recurrent LM) has priority over architectural proximity to Qwen**, so the substitution preserves the P0 objective.
- **Anchor unchanged:** `AntonV/mamba2-1.3b-hf` (Mamba-2, `gpt-neox`-style tokenizer).

### Value-vocabulary calibration (append-only)
The DeltaNet arm uses a Llama/Mistral SentencePiece tokenizer that **splits multi-digit numbers into single-digit tokens**, so a numeric-only value vocabulary is nearly empty for it. The scout therefore builds value tokens as: **prefer** 2–4-digit single-token numbers if ≥ `pool_size` exist (Mamba's gpt-neox tokenizer), **else** fall back to single-token **alphanumeric** tokens disjoint from the key pool (DeltaNet). Keys are always alphabetic single-token words disjoint from values. The chosen `value_kind` and pool counts are recorded per candidate in `MODEL_IDENTITY_*.json` / `P0_RESULTS_*.json`.

## 2. BASE task — MQAR-style associative recall (token-id level)

Memory-bound retrieval on the real pretrained LM, following Based/Zoology MQAR (the published graded knob: fix the model, sweep #KV-pairs → smooth recall decline).

**Construction (tokenizer-agnostic, deterministic, exact positions):**
- Each key and each value is exactly **one token id** (single-token pools built per tokenizer): keys = alphabetic word-tokens, values = 2–4 digit number-tokens. This makes every prompt at a given dose **identical length** (no padding) and puts the answer at a **known position** → exact constrained scoring.
- A prompt is a flat token-id sequence: for each pair `[key_id, "=", value_id, "\n"]`, then the query `[probe_key_id, "="]`. The gold answer is the probe key's value token.
- **Pressure knob = number of pairs `P`** (write-capacity / interference load).
- **Nested-monotonic superset (EXT2 §5 pattern):** each example has one ordered list of `P_max` pairs; dose `P` uses the first `P`. The probed pair sits at a fixed per-example index `< P_min`, so it lives in the shared prefix of **every** dose — its **write position is held constant** while trailing interference and the write→query gap grow. Higher pressure = same underlying challenge + additional associations.

**Why token-id level:** it removes tokenizer/format confounds (no `"=NNN"` BPE-merge ambiguity), guarantees single-token self-delimiting values, and holds sequence length fixed within a dose. The write→query **gap grows with P** (a length/position covariate) — this is recorded per dose (`seq_len_tokens`) and flagged as a confound for RNN-06B to control (length-matched packing / position-only control); P0 does not attempt to fully control it.

## 3. Evaluation semantics (deterministic, low-variance)

- **Primary metric — constrained accuracy:** one forward pass; at the single answer position take the logits, **argmax over the fixed value vocabulary**; correct iff `== gold`. No sampling, no free generation. Identical across all doses within a candidate.
- **Format adherence (denominator-exposing):** fraction of examples whose **unconstrained** global argmax token is inside the value vocabulary. Low format adherence at low pressure ⇒ interface/competence problem, not memory failure — recorded separately, never silently scored as a memory miss.
- **Unconstrained exact accuracy:** unconstrained global argmax `== gold` (a stricter, format-sensitive competence read).
- **Chance** `= 1/|value vocab|` recorded per dose.

All denominators (`n`, `n_constrained_correct`, `n_format_ok`, …) are exposed in `P0_RESULTS_*.json` / `P0_CURVES.csv`.

## 4. Low-pressure competence gate

Before interpreting any forgetting: at the **lowest** pressure dose the frozen model must reach `constrained_acc ≥ τ_hi` (competence). Otherwise the candidate is **`TASK_NOT_COMPETENT`**, classified separately from memory failure, and its band verdict is `NOT_FOUND_WITHIN_BUDGET` (a task/interface problem, not evidence about forgetting). P0 may adjust the **task surface** (candidate value vocab, separators, wrapper, pressure range) to establish competence — every adjustment recorded as an append-only amendment. **No fine-tuning** to establish competence.

## 5. Desired vs disqualifying shapes

- **Target (PLAUSIBLE):** competent-high at low pressure → progressive material-but-non-total loss → ≥1 interior dose with `τ_lo < acc < τ_hi` (a mid-band rung), not a single-cell cliff straight to the floor.
- **Flat-high** (e.g. 0.99→0.99→0.98): `NOT_FOUND_WITHIN_BUDGET`.
- **Immediate cliff** (0.95→0.92→0.15→0.03) with no interior rung: `NOT_FOUND_WITHIN_BUDGET` unless a bounded finer sweep reveals a reproducible transition band. A single unstable transition cell is **not** a graded regime.
- **Flat-low:** `TASK_NOT_COMPETENT` (interface not qualified), classified separately.

## 6. Thresholds (exploratory; may be amended append-only)

- `τ_hi = 0.75` (competence / high band), `τ_lo = 0.45` (material loss floor for "mid-band"). These mirror EXT2's `grade_hi/grade_lo` for continuity but carry **no** confirmatory authority here.
- Mid-band rung = a dose with `τ_lo < constrained_acc < τ_hi`.
- `PLAUSIBLE` requires competence at low pressure **and** ≥1 interior mid-band rung.

## 7. Pressure ladder (ORIGINAL — amendments append-only below)

**Original coarse ladder (both candidates):** `P ∈ {4, 8, 16, 32, 64, 128}` pairs.
- `P_min = 4` (competence rung), `P_max = 128`.
- `n_eval = 200` examples per dose (shared across doses via nesting), `master_seed = 20260811`, `pool_size = 256` single-token keys and values, `dtype = bf16`.
- Coarse-to-fine rule (§14 of the packet): run coarse; if a transition appears between two rungs, run **one** bounded finer sweep around it; then stop. Ladder extension (e.g. beyond 128 if flat-high) is permitted **only** as a recorded amendment within the hard sub-GPU-hour-per-candidate budget.

### Amendment log (append-only; original ladder above is never deleted)
- *(none yet — populated during the scout with: original ladder, observed inadequacy, new ladder, amendment order.)*

## 8. Identity, determinism, contamination boundary

- RNG: fixed integer seeds via numpy `PCG64`/`SeedSequence` (process-stable; **not** python `hash()`).
- `calibrationSetSha256` = sha256 of the **abstract** example spec (slot indices + probe index; tokenizer-independent). Per-model **materialized** token-id sweep is separately hashed (`materialized_sweep_sha256`).
- These are **P0 calibration examples**. The confirmatory RNN-06B `qualificationSetSha256` MUST be independent and disjoint; it is **not** created or consumed here.
- `MODEL_IDENTITY_*.json` freezes HF id + resolved revision sha + config/tokenizer file hashes + backend/version stack → establishes *same frozen checkpoint across all pressure conditions* (no per-condition model/seed variation; no seed screening).

## 9. No lifecycle claims

`STATE_OBSERVABLE ≠ STATE_SEMANTICALLY_CHECKPOINTABLE`. P0 does not observe, capture, restore, branch, or substitute recurrent state. Ordinary `model(input_ids)` forward passes only. Any obvious backend bug encountered during plain execution is recorded, not repaired.

## 10. Compute budget / futility

Target sub-GPU-hour per small candidate. Coarse ladder → optional single bounded finer sweep → stop. Reported: GPU wall time, total wall time, peak VRAM, model load time, throughput, #examples/queries evaluated. Futility = `NOT_FOUND_WITHIN_BUDGET`.

## 11. Required negative evidence

Preserved regardless of how it looks: load/dependency/backend failures, OOM, parser/format failure rates, low-pressure incompetence, ceiling regimes, cliff regimes, non-monotone curves, candidate differences, and **all** ladders actually tried. Not only the best-looking ladder.

## 12. Outputs

`P0_PROTOCOL.md` (this), `machine_config.json`, `MODEL_IDENTITY_GDN.json` / `MODEL_IDENTITY_MAMBA2.json` (per candidate that runs), `calibration_examples.json`, `P0_RESULTS_*.json`, `P0_CURVES.csv`, `P0_DECISION.md`, `HANDOFF.md`, plus the scout source `ops/rnn_06_p0_mqar.py`. Inapplicable artifacts (candidate never ran) are recorded as such, not fabricated.

## 13. Git ordering

reconstruct CURRENT → write+commit this protocol locally **before** substantive GPU outcome work → execute scout → write results → commit results locally. The pre-run protocol commit is **not** amended after seeing outcomes. Nothing pushed.
