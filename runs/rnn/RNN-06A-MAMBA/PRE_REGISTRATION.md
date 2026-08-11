# RNN-06A-MAMBA — Pre-Registration (Frozen-Backbone Lifecycle Qualification)

**Status:** PRE-REGISTRATION. Committed BEFORE any outcome-bearing lifecycle run.
**Packet:** RNN-06A-MAMBA. Follows the CLOSED exploratory packet RNN-06-P0.
**Author:** Claude (Opus 4.8) via Claude Code. **Date:** 2026-08-11.

Nothing here is an outcome. This file fixes the subject, the state contract, the
claims, the equivalence classes, the numerical tolerances, and the single gate
decision **before** observing results. Do not amend this file after seeing
outcomes (append an `Execution deviations` section instead if reality forces a
change, mirroring the P0 protocol convention).

---

## 0. Single scientific objective

> Determine whether the exact frozen Mamba-2 candidate **`AntonV/mamba2-1.3b-hf @ 703e19a4…`** on the exact pinned **transformers 4.48.3 naive `torch_forward`** backend (torch 2.6.0+cu124, bf16, no `mamba_ssm`/`causal_conv1d`) exposes a **complete, request-isolated recurrent state** whose **checkpoint / restore / continuation / branch** semantics are **BIT_EXACT or bounded-numerically-equivalent** under this preregistered lifecycle contract.

This is a **lifecycle/semantics** qualification, **not** a memory-quality experiment.
No historical-state recovery, no Memory Caching, no RNN-06B confirmatory run, no GDN
repair, no Qwen, nothing pushed.

## 1. Subject (exact, non-generalizable)

- Repo+revision: `AntonV/mamba2-1.3b-hf` @ `703e19a43f397c70315244a3424d79456b54fb34` (== P0 subject).
- Class: `transformers.Mamba2ForCausalLM`; backend `Mamba2Mixer.torch_forward` (naive; `is_fast_path_available=False`).
- Versions: transformers `4.48.3`, torch `2.6.0+cu124`, CUDA 12.4; bf16; no quantization.
- Executed backend source pinned by sha256 (see `MODEL_IDENTITY.json`):
  - `modeling_mamba2.py` = `83685d78…04fdb`
  - `configuration_mamba2.py` = `f9a3c2f5…c4a1c`
- Full state contract: see `STATE_CONTRACT.json`. Complete sequence-owned state =
  `{conv_states [48,B,4352,4], ssm_states [48,B,64,64,128]}`, bf16, ≈ **49.59 MiB/sequence**.
  No `seqlen_offset`/position is stored in the cache (caller-managed `cache_position`).

## 2. Backend policy

Primary and **only** substrate for 06A = the P0 naive `torch_forward` path. No fast
kernels are installed to "go faster." If the naive path could not expose/restore the
complete state, that would be recorded as `NOT_TESTABLE_ON_PINNED_BACKEND` — it is not
a licence to swap backends silently. (Source inspection already shows the state IS
exposed via `Mamba2Cache`, so we expect it is testable.)

## 3. Executed-source identity (mandatory P0 repair)

Before every outcome-bearing run the runner deterministically self-records, INTO
`LIFECYCLE_RESULTS.json`: runner-file sha256, git blob sha (if tracked), git commit +
dirty flag, `modeling_mamba2.py`/`configuration_mamba2.py` sha256, transformers/torch
versions, model repo+revision, config hash, and a cheap model-weights fingerprint
(section 10). The source must not change between identity freeze and outcome collection
without a new run identity.

## 4. State inventory (full-module; no partial-state false green)

Enumerated from the pinned source (`STATE_CONTRACT.json`). The runner re-enumerates at
runtime and byte-accounts **both** components for **all 48 layers** before minting any
PASS. A matrix-only (ssm-only) proof is INSUFFICIENT; conv_states must round-trip too.
RNN-05A lesson: partial recurrent-state lifecycle ≠ full-module lifecycle.

## 5. Lifecycle claims (each tested and classified separately)

Let `prefill(X)` = one forward over sequence X with `cache_position[0]==0`; `step(t)` =
one single-token decode forward with `cache_position[0]=offset>0`. `logits(·)` compared
in fp32. Continuation is done token-by-token because the decode path is single-token
(STATE_CONTRACT `decode_path_constraint`).

- **A. Fresh-sequence determinism.** Two independent `prefill(X)` on a freshly-built
  cache. Compare full logits tensor. Also compare the two resulting `{conv,ssm}` states.
- **B. Full-sequence vs segmented continuation.** Reference `R` = `prefill(A+B)` then read
  per-position logits over B. Candidate `S` = `prefill(A)` then `step` each token of B,
  collecting each step's next-token logits. Compare the aligned continuation logits.
  Multiple split points |A| ∈ {2, 5, 8}. (Cross-path: chunked-prefill vs recurrent-decode.)
- **C. Serialize → destroy runtime → restore → continue.** `prefill(A)`; **serialize** the
  complete state to disk bytes; `del model`, `del cache`, `torch.cuda.empty_cache()`;
  **rebuild** a clean model+cache from the pinned revision; **deserialize** state; `step`
  through B. Compare to an in-memory reference that ran `prefill(A)`+`step(B)` without any
  destroy/serialize. This is the core checkpoint/restore claim.
- **D. Branch restore.** From one saved state@t produce two continuations B1, B2 (distinct
  token streams). Verify each branch == its own independent restore-and-continue reference;
  verify the stored parent snapshot is byte-unchanged after each branch; verify B1 cannot
  contaminate B2 (run B1 then B2 from the same snapshot and compare B2 to B2-alone).
- **E. Request isolation.** Two independent equal-length sequences P and Q. Run each ALONE
  (batch=1) and TOGETHER (batch=2, rows [P;Q]). Verify P-alone logits == P-in-batch and
  Q-alone == Q-in-batch (no cross-row leakage). Also verify per-row state slices match.
- **F. Reset / fresh state.** After an executed sequence, `cache.reset()` must yield a state
  byte-identical to a newly constructed fresh cache (all zeros), and a subsequent `prefill`
  on the reset cache must equal `prefill` on a fresh cache.
- **G. State round-trip.** Serialize→deserialize every component; verify component count,
  names, shapes, dtypes, devices, byte counts, and exact element equality.

## 6. Equivalence classes (predeclared)

Comparisons are on fp32-upcast logits unless stated; state-tensor comparisons are on the
stored (bf16) tensors.

| Class | Definition (predeclared) |
|---|---|
| `BIT_EXACT` | `torch.equal` holds (identical bits): max_abs_err == 0 **and** 100% argmax agreement. State tensors: exact element equality. |
| `NUMERICALLY_EQUIVALENT` | not bit-exact, but `max_abs_err ≤ 2e-2` **and** `mean_abs_err ≤ 2e-3` **and** 100% argmax agreement. |
| `BOUNDED_DIFFERENCE` | 100% argmax agreement but `max_abs_err ∈ (2e-2, 5e-1]`. |
| `NOT_EQUIVALENT` | any argmax disagreement, or `max_abs_err > 5e-1`. |
| `NOT_TESTABLE` | state not exposed / cannot serialize or restore / OOM / API absent. |

Reported metrics per comparison: `max_abs_err`, `mean_abs_err`, `rel_err` (where
meaningful), `argmax_agreement_frac`, and exact-state-equality booleans. Token match alone
is never accepted as proof of state equivalence (state tensors are compared directly).

### Predeclared expectation per claim (falsifiable)
- A: expect `BIT_EXACT`; acceptable down to `NUMERICALLY_EQUIVALENT` (cuBLAS run-to-run).
- B: expect `NUMERICALLY_EQUIVALENT` or `BOUNDED_DIFFERENCE` (cross-path); **required: argmax agreement == 100%** (i.e. not `NOT_EQUIVALENT`).
- C: expect `BIT_EXACT` (identical decode code path; lossless bf16 state round-trip).
- D: expect `BIT_EXACT`; parent-unchanged and no-cross-contamination are hard booleans.
- E: expect `BIT_EXACT` (Mamba has no cross-sequence interaction at equal length, no padding).
- F: expect `BIT_EXACT` (reset==fresh zeros).
- G: expect `BIT_EXACT` for every component.

## 7. Load-bearing set and the single gate

`FROZEN_BACKBONE_LIFECYCLE ∈ { QUALIFIED | NOT_QUALIFIED | NOT_TESTABLE_ON_PINNED_BACKEND }`.

**Load-bearing claims** = {A, B, C, D, E, F, G}. Gate rule (no averaging away failures):

- `QUALIFIED` iff **all** hold:
  - C, D, E, F, G each `BIT_EXACT` (state serialize/restore/branch/isolation/reset/round-trip are exact);
  - D parent-unchanged == true AND no-cross-branch-contamination == true;
  - E no-leakage == true;
  - A ∈ {`BIT_EXACT`, `NUMERICALLY_EQUIVALENT`};
  - B argmax agreement == 100% (class ∈ {`BIT_EXACT`,`NUMERICALLY_EQUIVALENT`,`BOUNDED_DIFFERENCE`}).
- `NOT_TESTABLE_ON_PINNED_BACKEND` iff the complete state cannot be serialized/restored on
  this backend (any of C/G structurally impossible), as predeclared.
- `NOT_QUALIFIED` otherwise — including: any `NOT_EQUIVALENT`; any request-isolation leak;
  any branch contamination or parent mutation; any partial-state-only success (e.g. ssm
  round-trips but conv does not).

If request isolation (E) fails, lifecycle fails. If full-module state cannot be
serialized/restored (C/G), lifecycle fails or is NOT_TESTABLE as above. Partial-state
success does NOT qualify.

## 8. Challenge inputs (deterministic; not a memory-quality probe)

Small fixed token sequences over the model vocab (id-space, process-stable), chosen to
exercise: short prefix; longer prefix; multiple split positions {2,5,8}; convolution
warmup boundary (conv_kernel=4, so splits at 2 and 5 straddle it); chunk boundary
(section 9); branch divergence (B1 vs B2 differ from token 0 of the continuation).
One P0-like retrieval sequence MAY be included **solely** as a lifecycle stress input; its
retrieval score carries **no** 06A scientific authority and mints nothing.

## 9. Chunk-size semantics

Primary subject uses the model-native `chunk_size = 256`. 06A prefixes are short
(≤ ~24 tokens ≪ 256) so **no chunk tiling occurs** on the primary runs. As a declared
sub-experiment, run one prefill with a prefix LONGER than a reduced `chunk_size` (set
`chunk_size=8`, prefix length ≥ 20 crossing multiple chunk boundaries) and compare logits
to `chunk_size=256` on the identical prefix. If they differ beyond `NUMERICALLY_EQUIVALENT`,
record `CHUNK_SIZE_IS_PART_OF_EXECUTION_IDENTITY = TRUE` and keep 06A pinned to one value
(256 for the lifecycle claims). This is a single controlled check, not a chunk-size sweep.

## 10. Weight immutability

Prove no training/weight mutation: assert no optimizer constructed, model in `eval()`,
`torch.no_grad()` around all forwards, no `.backward()`. Capture a cheap deterministic
weight fingerprint = sha256 over a fixed low-cost reduction (per-parameter
`(name, shape, dtype, float(sum), float(sum of squares))` folded into one hash) BEFORE and
AFTER all lifecycle work; require equality. Record HF revision unchanged. (Avoid hashing
full multi-GB tensors repeatedly.)

## 11. Performance evidence (secondary, recorded not optimized)

model load time; state bytes (measured); serialized snapshot size on disk; capture,
serialize, restore, continuation latencies; peak VRAM; total GPU runtime. Inputs to a
later SESOI/storage-economics packet; NOT optimized here.

## 12. Explicit non-goals / boundaries

No RNN-06B, no `qualificationSetSha256`, no `FIXED_BACKBONE_GRADED_REGION` mint, no
historical-state test, no Memory Caching / recovery, no GDN repair (`GDN_COMPATIBILITY_GAP`
stays OPEN), no Qwen, no serving-infra change, no push. P0 statuses are inherited unchanged;
06A only adds a lifecycle verdict. `P0_GRADED_BAND` remains `PLAUSIBLE_EXPLORATORY`.

## 13. Evidence ordering

CURRENT reconstruction → API/source discovery (done, DISCOVERY_ONLY) → **this
pre-registration + protocol commit** → identity freeze → lifecycle qualification → results
→ decision → audit bundle. The protocol commit is not amended after outcomes.
