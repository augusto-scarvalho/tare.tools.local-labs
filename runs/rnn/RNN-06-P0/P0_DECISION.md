# RNN-06-P0 — DECISION (EXPLORATORY scout; carries NO confirmatory authority)

**Packet:** RNN-06-P0 · Frozen-Checkpoint BASE Regime Scout
**Date:** 2026-08-11 · **Protocol commit:** `7d7feed` (pre-run) · **Nothing pushed.**
**Reminder:** P0 emits only `P0_GRADED_BAND ∈ {PLAUSIBLE, NOT_FOUND_WITHIN_BUDGET, MODEL_NOT_RUNNABLE}`. It does **NOT** mint `FIXED_BACKBONE_GRADED_REGION = QUALIFIED` (only RNN-06B may), does not qualify lifecycle/state semantics (RNN-06A), and implements no probe/reader/recovery/Memory-Caching machinery.

## Per-candidate result

### Designed PRIMARY — `linear-moe-hub/Gated-Deltanet-1.3B` (GDN)
- **MODEL_RUNNABLE = NO** (on the pinned stack: torch 2.6.0+cu124 / fla 0.5.2 / transformers 4.48.3 / Triton 3.2.0).
- **Reason:** checkpoint uses a **fused gate+up MLP** (`mlp.gate_proj` `[11264,2048]`=2×5632, `mlp.down_proj` `[2048,5632]`) while fla 0.5.2's `GatedDeltaNetMLP` expects **separate** `gate_proj`/`up_proj` (`[5632,2048]` each); config also under-specifies `intermediate_size` (`None`, `hidden_ratio=4`). `state_dict` size-mismatch on load. A fused→split remap was **not** attempted (unverifiable concat order → silent-correctness risk; inappropriate for a scout).
- **Disposition:** genuine negative **tooling** evidence, not a scientific result about GDN. **Deferred to RNN-06A** (pin the exact authoring fla revision, or verify a remap against a reference forward).
- `P0_GRADED_BAND = MODEL_NOT_RUNNABLE`.

### Runnable delta-family arm (substituted PRIMARY) — `fla-hub/delta_net-1.3B-100B` (DeltaNet)
- **MODEL_RUNNABLE = YES** · `DeltaNetForCausalLM`, 1.366 B params, `LlamaTokenizerFast` vocab 32000, fla/Triton, bf16, peak 8.78 GB, resolved rev `b4dcbbafd4fde802717bdec3008d4aba9cb3a1f8`.
- Value vocab: **alphanumeric-fallback** (the Llama tokenizer yields **0** multi-digit single-token numbers → values are single-token alphanumerics disjoint from keys), 256/256 pools, chance = 1/256 ≈ 0.0039.
- **Curve (constrained_acc, n=200/dose):** P4 0.730 · P8 0.700 · P16 0.760 · P32 0.755 · P64 0.680 · P128 0.530. Format-adherence falls 0.725→0.220.
- **TASK_COMPETENT = NO** — low-pressure (P=4) constrained accuracy **0.730 < τ_hi 0.75** (marginal), with weak format adherence even at low pressure. There is a **soft tail decline** (0.76→0.53) but from a non-high, format-degrading baseline, so it cannot be cleanly attributed to graded forgetting vs. a marginal interface.
- `P0_GRADED_BAND = NOT_FOUND_WITHIN_BUDGET` (competence not cleanly established; flat-medium body + soft tail, no clean competent→graded structure).

### Anchor — `AntonV/mamba2-1.3b-hf` (Mamba-2; HF conversion of `state-spaces/mamba2-1.3b`)
- **MODEL_RUNNABLE = YES** · `Mamba2ForCausalLM` (transformers-native), 1.3 B, gpt-neox-style tokenizer, bf16, peak 8.53 GB, resolved rev `703e19a43f397c70315244a3424d79456b54fb34`.
- Backend note: **no `mamba_ssm`/`causal_conv1d` kernels** → transformers-native **naive** path; `chunk_size` overridden **256→32** — a **mathematically-equivalent** chunk-tiling knob (sanity output identical: `apple=`→`273` at both), applied only to fit the naive path's O(chunk²) diagonal tensor in VRAM. Value vocab = **numeric** (1992 multi-digit single-token numbers available), 256/256 pools, chance ≈ 0.0039. `n_eval = 64` here (vs 200 for DeltaNet) because the kernel-free naive path is ~4 s/forward; the 64 examples are the **nested prefix** of the same deterministic generation (identical example `i` for all runs).
- **Curve (constrained_acc, n=64/dose):** P4 **0.953** · P8 0.922 · P16 0.781 · P32 0.719 · P64 0.516 · P128 0.234. Format-adherence falls 0.641→0.078.
- **TASK_COMPETENT = YES** (0.953 ≥ τ_hi). Curve is **monotone non-increasing** with **2 interior mid-band rungs** (0.719, 0.516 ∈ (0.45,0.75)) and a **material-but-non-total** floor (0.234 ≫ chance 0.0039).
- `P0_GRADED_BAND = PLAUSIBLE` — competent-high → progressive material loss → non-total floor, matching the Based/Zoology/Stuffed-Mamba state-capacity prediction (recall degrades as #KV-pairs rises against a fixed state).

## Summary table

| Candidate | RUNNABLE | COMPETENT (P=4) | shape | P0_GRADED_BAND |
|---|---|---|---|---|
| `linear-moe-hub/Gated-Deltanet-1.3B` | **NO** (fused-MLP vs fla 0.5.2) | — | — | `MODEL_NOT_RUNNABLE` |
| `fla-hub/delta_net-1.3B-100B` (DeltaNet) | YES | NO (0.730) | flat-medium + soft tail | `NOT_FOUND_WITHIN_BUDGET` |
| `AntonV/mamba2-1.3b-hf` (Mamba-2) | YES | YES (0.953) | **graded: 0.95→0.23, monotone** | **`PLAUSIBLE`** |

The outcome is exactly the contingency the design packet anticipated: *"if GDN is flat-high and Mamba shows the scientifically useful regime, preserve that result … the phenomenon has priority over architectural proximity to Qwen."* The experimental construction (token-id MQAR, constrained scoring, nested pressure ladder) **is capable of surfacing a graded state-capacity regime in a real pretrained recurrent LM** — it did so on Mamba-2.

## Caveats bounding the PLAUSIBLE verdict (exploratory only)
1. **Not confirmatory.** `n_eval=64` (SE≈0.06); single master seed; single prompt template family; `τ_hi/τ_lo` are P0 heuristics. RNN-06B must re-qualify on **independent, disjoint** deterministic examples with all preregistered seeds/templates and the EXT2 §7 all-seeds overlap gate.
2. **Length/gap confound not controlled.** Higher pressure also means longer sequence and a longer write→query gap (`seq_len` grows 18→514). P0 records this; 06B must add length-matched packing + a position-only control to isolate **state capacity** from length/position.
3. **Format-adherence declines with pressure** (Mamba 0.64→0.08). Constrained scoring still recovers a clean graded recall signal, but 06B should report the format/decoding channel explicitly (a competent model that merely stops emitting the value format is different from one that forgets).
4. **Backend caveats:** Mamba ran on the kernel-free naive path with a `chunk_size` tiling override (math-equivalent, sanity-verified) — 06A/06B should install `mamba_ssm`/`causal_conv1d` (fast path) or verify equivalence. `STATE_OBSERVABLE ≠ STATE_CHECKPOINTABLE`: no lifecycle claim is made.
5. **GDN-proper unrun.** The Qwen-closest arm (gated DeltaNet) did not load on the pinned stack; its band status is unknown, not flat.

## Exactly ONE next recommendation (NOT executed)
**Proceed to `RNN-06A` (state observability & full lifecycle qualification) on `AntonV/mamba2-1.3b-hf` (Mamba-2-1.3B),** targeting the P0-observed graded window **P ∈ [8, 128] pairs (mid-band ≈ P 32–64)** as the pressure range to carry into the confirmatory `qualificationSetSha256`. Rationale: Mamba-2 is the only candidate that cleared the competence gate **and** showed a plausible non-ceiling/non-cliff graded band; DeltaNet was marginal/flat; GDN-proper is blocked on a tooling fix. This honors the packet's "phenomenon > Qwen-proximity" rule. In parallel, RNN-06A should (a) resolve the GDN load (pin authoring fla revision) so the Qwen-closest arm can re-enter, and (b) install the Mamba-2 fast kernels. **Do NOT** treat this as `FIXED_BACKBONE_GRADED_REGION = QUALIFIED` (only 06B, on independent examples, may mint that). **QWEN_GDN_TRANSPLANT_GATE = DEFER.**

*(This recommendation is stated, not executed. STOP after P0.)*
