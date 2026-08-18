# HANDOFF — RNN-05B: Memory Caching on actual DeltaNet / Gated-DeltaNet (2026-08-11)

First reproduction of Memory Caching on **real delta / gated-delta** recurrences (not the RNN-04/05A Linear
Attention), with a **complete-state** lifecycle (recurrent matrix **+ conv boundary**), a **causal 2×2**
co-adaptation control, and matched LA/DN/GDN family. **Not pushed. No Qwen weights used. No llama.cpp / deploy
/ serving touched.**

## Git
- Branch **master**. **Start HEAD `40c5e7c`** (`40c5e7c813749ba10d8c8b543c0ee40e9195faf2`) → end HEAD
  **`25c28f0`** (see git_evidence). New code `ops/rnn_delta_substrate.py`, `ops/rnn_mc_05b.py`; evidence
  `runs/rnn/RNN-05B-delta-gdn/`. **No weights in Git**: the 3 frozen+reader checkpoints live in durable
  non-Git `.harness/artifacts/` and in the bundle `external_artifacts/`.
- Untracked-by-policy: `.harness/`, `git_evidence*.txt`, RNN-08 adapter dirs.

## Central question and hypotheses
Does Memory Caching behave differently when the substrate is a real DeltaNet/GDN whose historical states do
**not** collapse to the additive final-state sufficient statistic (as LA does, RNN-05A)?
- **H1** MC only helps with backbone–memory **co-adaptation**.
- **H2** RNN-05A failed specifically because LA collapses history into the final state.
- **H3** a real delta/gated-delta update stores useful history not recoverable from the final state.

**Answer: H1 is decisive. The non-collapsing DN/GDN state did NOT change the frozen-backbone verdict.**
Co-adaptation (2×2 interaction) is large and consistent; frozen-backbone MC still does not exceed base for
DN or GDN. H3 shows only a weak, sub-threshold directional signal (DN/GDN caches carry slightly
non-redundant info); the task ceiling limits its test.

## Pinned upstream sources (equation authority) + provenance
| source | role | provenance / license |
|---|---|---|
| `GDN_KERNEL.md` (repo) | ggml/llama.cpp GDN recurrence `out=Sᵀx`, `d_t=exp(g_t)`, chunk==seq to ~1e-16 | llama.cpp (MIT); ggml op `gated_delta_net` |
| `scratchpad/modeling_qwen3_next.py` | `torch_recurrent_gated_delta_rule` (ground-truth scan) + `torch_chunk_gated_delta_rule` (chunk-parallel) | HF Transformers Qwen3-Next (Apache-2.0) |
| `runs/rnn/RNN-01-gdn-state/RNN_STATE_INVENTORY.json` | real Qwen cache = `{recurrent_states, conv_states}` | this lab (RNN-01) |
FLA is **not installed** in the venv → parity oracle = our own sequential scan vs our own chunk-parallel path
(both ported from the two Qwen functions above; no shared code). Full equation→source→function map in
`runs/rnn/RNN-05B-delta-gdn/RNN05B_DELTA_SEMANTICS.md`.

## LA vs DN vs GDN semantic matrix (one recurrence, two switches)
Read (all): `o_t = Sᵀ q_t` after the write; `q,k` L2-normalized, `q` scaled `1/√d_k`; `g_t≤0` per-head scalar.

| substrate | decay `d_t` | write `u_t` | additive? | `ADDITIVE_COLLAPSE` (measured) |
|---|---|---|---|---|
| LA  | `1`        | `v_t`                | yes `S=Σ k_i⊗v_i` | **YES** (sum rel-err 0.0) |
| DN  | `1`        | `β_t(v_t−Sᵀk_t)`     | no | **NO** (best linear combo 0.41 err) |
| GDN | `exp(g_t)` | `β_t(v_t−Sᵀk_t)`     | no (+decayed history) | **NO** (final≈recent, decay) |

`delta_scan` (ground truth) ← `torch_recurrent_gated_delta_rule`; `delta_chunked`/`la_chunked` (fast) ←
`torch_chunk_gated_delta_rule`.

## Gates (packet §5–8) — all PASS (STOP-conditions cleared)
| substrate | REFERENCE_PARITY (scan↔chunked, ≤1e-4) | FULL_MODULE_CHECKPOINT_RESTORE | REQUEST_ISOLATION / BRANCH_RESTORE |
|---|---|---|---|
| LA  | PASS (≤7e-7) | NUMERICALLY_EQUIVALENT | PASS / PASS |
| DN  | PASS (≤1e-7) | NUMERICALLY_EQUIVALENT | PASS / PASS |
| GDN | PASS (≤1e-7) | NUMERICALLY_EQUIVALENT | PASS / PASS |
Parity sub-tests: single step, small seq, chunked seq, incremental decode (decode exactly 0.0). The
standalone substrate self-test (`substrate_selftest.json`, d_k=24) reports **BIT_EXACT**; the harness gate
(d_k=64, longer seq on GPU) reports NUMERICALLY_EQUIVALENT (≤1e-4, GPU reduction-order) — both qualify.

## Complete sequence-owned state (packet §6-7) — closes the RNN-05A gap
`RNN05B_STATE_INVENTORY.json` / `RNN05B_STATE_MODEL.md`. Live state per request = **18688 B**:
`recurrent_state_S [64,64]=16384` + `conv_state [192,3]=2304`. Maps 1:1 to Qwen `{recurrent_states,
conv_states}` (RNN-01). The full-module checkpoint serializes **both**, destroys runtime, restores, and feeds
**only** continuation tokens (conv rebuilt from the restored boundary; recurrence from restored S) — the exact
protocol RNN-05A's matrix-only proof could not satisfy. `FULL_MODULE_CHECKPOINT_RESTORE` qualified for real
DeltaNet and GDN.

## Model / training config (matched family, packet §10)
`d_model=128`, single head `d_k=d_v=64`, one depthwise causal conv (kernel 4, SiLU) over `[q;k;v]`,
`w_g` bias −4.5 / weight ×0.3 (decay≈1 init; **required** or GDN catastrophically forgets), `w_b=sigmoid`.
MQAR (blake2b process-stable seeds): `L=256`, seg=32, num_keys=128, num_vals=64, num_queries=8, density=0.3.
`AdamW lr=3e-3, wd=0.01, cosine, batch=128, clip=1`; backbone/reader **3000** steps; calib 1800.
**Unavoidable, recorded (§10):** single lr=3e-3 for all (the delta family needs ≥2e-3; LA fine at any); the
delta rule is rank≈`d_k` with a **sharp capacity cliff** (stable+learnable at ≤40 pairs, chaotic above) → D
selected **36** (base ~0.94–0.99, all three learn stably; ceiling limits MC upside — see caveats).

## Experiments
### 2×2 co-adaptation (packet §11-12). GDN=3 seeds [42,43,44]; LA/DN=1 seed [42]
cells: A train-single/infer-single · B train-single/infer-MC(param-free) · C train-MC/infer-single ·
D train-MC/infer-MC. Interaction = (D−C)−(B−A).

| substrate | A(base) | B | C | D | post-hoc B−A | joint D−A | D−C | **interaction** |
|---|---|---|---|---|---|---|---|---|
| LA  | 0.997 | 0.852 | 0.995 | 0.995 | −0.146 | −0.002 | −0.000 | **+0.145** |
| DN  | 0.971 | 0.838 | 0.829 | 0.977 | −0.132 | +0.007 | +0.148 | **+0.280** |
| GDN | 0.974 | 0.85  | 0.79  | 0.979 | −0.124 | +0.006 | +0.188 | **+0.312** |
GDN per-seed interaction: 0.303 / 0.343 / 0.290 (tight). **MC helps only when training and inference are
matched** — both mismatched cells (B, C) fall well below base; D returns to base. `JOINT_TRAINED_MC_EFFECT =
NO_EFFECT` (D≈A, at ceiling). `COADAPTATION_INTERACTION = SUPPORTED` — the causal 2×2 RNN-04/05A could only
"support, not isolate."

### Frozen transfer (packet §11B, the RNN-05A analog on real DN/GDN)
Single-state backbone trained once, **frozen** (`FROZEN_BACKBONE_VALIDITY=PASS`, mutation 0 all modes), then:

| substrate | base | param-free MC | trained reader (w_u only) | FROZEN_POST | FROZEN_READER |
|---|---|---|---|---|---|
| LA  | 0.997 | 0.852 (−0.146) | 0.976 (−0.021) | NEGATIVE | NO_EFFECT |
| DN  | 0.971 | 0.838 (−0.132) | 0.975 (+0.004) | NEGATIVE | NO_EFFECT |
| GDN | 0.982 | 0.861 (−0.124) | 0.984 (+0.002) | NEGATIVE | NO_EFFECT |
Frozen-backbone MC **does not exceed base** for DN or GDN — same qualitative verdict as RNN-05A's LA negative.
Trained reader (3 durable checkpoints, SHA below) only **recovers** the param-free damage back to base.

### Pure cache-count sweep (packet §13) — identical weights, fixed checkpoint positions, retain most-recent K
`PURE_CACHE_COUNT_CURVE = QUALIFIED` (weights constant, verified). **Substrate difference:**
- LA acc **decreases** with K: 0.997→0.997→0.989→0.976 (K=1,2,4,8) — additive caches are redundant + add noise.
- DN acc **increases**: 0.971→0.972→0.974→0.975. GDN: 0.981→0.982→0.985→0.984 — non-collapsing caches carry
  slightly complementary info. Direction consistent with H3; magnitude sub-margin.

### Historical-state novelty (packet §14) — recall of early-written associations, base vs MC
early-half write-segment recall gain (MC − base): LA **−0.001** (NOT_DETECTED); DN **+0.011**, GDN **+0.014**
(INCONCLUSIVE, sub-margin). Same directional H3 signal, ceiling-limited. `HISTORICAL_STATE_NOVELTY`: LA
NOT_DETECTED, DN/GDN INCONCLUSIVE.

### DeltaNet collapsibility (packet §15, diagnostic)
rel-err of composing INDEPENDENT segment states vs the true final state: LA sum **0.0** / weighted **0.0**
(additive); DN sum 0.89 / weighted-lstsq **0.41**; GDN dominated by recent (decay). LA collapses; DN/GDN do
not.

## Cost (direct-measured) + memory (packet §19-20)
| substrate | proj+conv | recurrent | ckpt copy | read | gate | **total (grm,cont)** | live state B | hist cache B (full) | reader params |
|---|---|---|---|---|---|---|---|---|---|
| LA  | 0.60 | 1.14  | 0.055 | 0.57 | 0.65 | **4.20 ms** | 18688 | 16384×7 | 64×128 |
| DN  | 0.60 | 11.36 | 0.082 | 0.68 | 0.72 | **16.46 ms** | 18688 | 16384×7 | 64×128 |
| GDN | 0.60 | 10.89 | 0.055 | 0.57 | 0.64 | **13.33 ms** | 18688 | 16384×7 | 64×128 |
Matrix 16384 B + conv 2304 B = 18688 B live/req (all substrates); 8 checkpoints/seq (7 historical × 16384 B
matrix). DN/GDN recurrence ~10× LA (delta forward-substitution + inter-chunk carry; pure-PyTorch, no kernel).
Cost components directly timed; total is end-to-end MC forward.

## Outcome matrix (packet §24)
| | LA | DN | GDN |
|---|---|---|---|
| REFERENCE_PARITY | PASS | PASS | PASS |
| FULL_MODULE_LIFECYCLE | NUM_EQUIV | NUM_EQUIV | NUM_EQUIV |
| REQUEST_ISOLATION | PASS | PASS | PASS |
| ADDITIVE_COLLAPSE | YES | NO | NO |
| JOINT_TRAINED_MC_EFFECT | NO_EFFECT | NO_EFFECT | NO_EFFECT |
| FROZEN_POST_MC_EFFECT | NEGATIVE | NEGATIVE | NEGATIVE |
| FROZEN_READER_MC_EFFECT | NO_EFFECT | NO_EFFECT | NO_EFFECT |
| PURE_CACHE_COUNT_CURVE | QUALIFIED (↓K) | QUALIFIED (↑K) | QUALIFIED (↑K) |
| HISTORICAL_STATE_NOVELTY | NOT_DETECTED | INCONCLUSIVE | INCONCLUSIVE |
| STATE_BYTES | 18688 | 18688 | 18688 |
| COMPUTE_TOTAL_MS | 4.20 | 16.46 | 13.33 |

**Classifications:** `DELTA_MC_SIGNAL = GDN_MC_SIGNAL = LA_MC_SIGNAL = NO_EFFECT_NAIVE_MC_NEGATIVE` (best/
trained MC arm neutral; only the naive param-free arm hurts). `COADAPTATION_INTERACTION = SUPPORTED`.

## Failures / negatives / honesty
- **NEGATIVE core**: frozen-backbone Memory Caching does not help real DeltaNet or GDN (same as RNN-05A/LA).
- **Ceiling caveat**: the delta rule's sharp capacity cliff forced a base ~0.94–0.99 (no stable lower base) →
  limited upside headroom; the H3 novelty test is INCONCLUSIVE (not disproven). The NEGATIVE param-free and
  NO_EFFECT trained/joint verdicts are robust (arms never exceed base across seeds); the weak H3 direction
  (cache-count ↑K, novelty +0.01 for DN/GDN vs LA) is sub-margin.
- **Implementation canaries (§22)**: decay must init ≈1 (else GDN forgets); delta family needs lr≥2e-3 and
  d_k≥ pairs; DN/GDN training is unstable near capacity (single-seed basin lottery, e.g. calib D=40 DN=0.03).
- `TRAINING_REPLICATION_COUNT`: GDN=3 seeds (tight); LA/DN=1 seed (predeclared).

## Artifacts / hashes
Durable non-Git `.harness/artifacts/` + bundle `external_artifacts/` (SHA-256):
- `rnn05b_la_frozen_reader.pt`  375679 B  `f2f3849c3b5014ad…`  reader `e0d81e2a0c130297…`
- `rnn05b_dn_frozen_reader.pt`  375679 B  `30bd4a144303bfd2…`  reader `d13b3844e6833ad8…`
- `rnn05b_gdn_frozen_reader.pt` 375696 B  `9096c19e10d0dabc…`  reader `28c60d5856d9650c…`
Reader **durably saved this time** (fixes RNN-05A's non-durable-reader blocker).

## QWEN_GDN_TRANSPLANT_GATE = **CONDITIONAL / DEFER**
Parity + **full-module lifecycle** + request-isolation are now QUALIFIED for real GDN (preconditions RNN-05A
could not meet — real progress toward §25 (1)(2)(3)(6)). But §25 (4) is not met: frozen/co-adapted MC shows
**no positive** state-behavior signal, and **co-adaptation (not state non-collapsibility) is the decisive
factor**. Do **not** auto-transplant into Qwen.

## Exact commands
```
V=/home/augus/tptt-venv/bin/python ; PP=/mnt/c/projects/local-model-lifecycle/ops
PYTHONPATH=$PP $V $PP/rnn_delta_substrate.py --selftest --out .../substrate_selftest.json --chunk 32
PYTHONPATH=$PP CUDA_VISIBLE_DEVICES= $V $PP/rnn_delta_substrate.py --inventory .../RNN05B_STATE_INVENTORY.json --dk 64
PYTHONPATH=$PP $V $PP/rnn_mc_05b.py --outdir runs/rnn/RNN-05B-delta-gdn --artifacts .harness/artifacts
```

## Recommended next packet (exactly one)
**RNN-05B-EXT — break the ceiling to test H3 directly.** Same qualified substrate; make the task memory-bound
so BASE fails on early/decayed associations (e.g. fixed pairs but longer L / higher distractor density /
value-recall over long distance, keeping pairs < delta capacity to stay stable). Then the frozen trained
reader has room to exceed base **iff** DN/GDN historical caches carry recoverable information the decayed
final state lost. If MC then helps DN/GDN (and not LA), that is the first positive isolation of H3; if not,
the frozen-backbone negative is confirmed beyond the ceiling regime. Keep the 2×2, add ≥3 seeds for DN too.
Do **not** touch Qwen.
