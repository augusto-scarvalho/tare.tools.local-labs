# RNN-06 — Research & Experimental-Design Packet
## Real Recurrent LM: Long-Horizon State Loss & Historical-State Recovery

**Status: RESEARCH + DESIGN ONLY. No implementation. No GPU runs. Nothing modified. Nothing pushed.**
Authored 2026-08-11 · **Revision: reconciliation-1 (2026-08-11).** This packet designs RNN-06; it does not execute it. Memory Caching is **not** assumed to be the intervention.

### Reconciliation-1 changes (2026-08-11)
Applied after an external research/design audit (the audit's requirements were supplied inline in the reconciliation request; the referenced `AUDIT_RECONCILIATION_RNN-06_RESEARCH_DESIGN_2026-08-11.md` file was not present in the working tree, so its points are transcribed here from that request):
1. **CSV repaired** — `RNN-06_candidate_matrix.csv` is now regenerated from the JSON by a proper CSV writer (comma-containing fields quoted) and **round-trip validated** record-by-record (**PASS**; 14 candidates, 22 cols, 308 cells). Until now the **JSON is the machine-readable authority**.
2. **Stage model preserved; exploratory scout separated** — added **RNN-06-P0 (Frozen-Checkpoint BASE Regime Scout)** as an EXPLORATORY pre-packet; RNN-06A/B/C/D keep their meanings. Only **06B** may mint `FIXED_BACKBONE_GRADED_REGION = QUALIFIED`.
3. **Lifecycle risk upgraded** — *state observable* is now separated from *state semantically checkpointable/restorable*; BIT_EXACT is **not** inferred from the existence of a Cache API. For GDN-1.3B (and all candidates) lifecycle semantics are **NOT_QUALIFIED** until 06A proves them on the exact pinned backend/version/checkpoint.
4. **Mamba wording narrowed** — "strongest directly-relevant state-capacity/long-context evidence among runnable fallbacks"; a smooth common graded band on our `mamba2-1.3b` checkpoint stays **NOT_QUALIFIED** until measured.
5. **Qwen backend wording narrowed** — Transformers/FLA is the *currently identified* clean state-introspection path; equivalent access via vLLM/SGLang/llama.cpp is **NOT_QUALIFIED**, not proven impossible.
6. **Calibration/qualification separated** — added `calibrationSetSha256` / `qualificationSetSha256` / `stressGridSha256` with a contamination boundary (P0 tunes the pressure range on calibration examples; 06B qualifies on independent deterministic examples).
Candidate decision is **unchanged** (PRIMARY Gated-DeltaNet-1.3B, FALLBACK Mamba-2-1.3B).

### Provenance & evidence caveat (read first)
- Phase 0 git facts are **verified against live `git`**, not assumed.
- Phase 1 candidate facts came from three parallel primary-source research sweeps. **Several key papers are dated after the assistant's Jan-2026 knowledge cutoff** (DART 2608, Sparse Delta Memory 2607, HOLA 2607, GDN-2 2605, FG2-GDN 2604, "Memory Caching" 2602, StateX 2509-v3). These are reported *as retrieved from fetched primary sources* with the researcher's evidence grade and an explicit **[verify]** flag; they are **not** treated as settled. The **runnable-substrate enablers** (fla/`mamba_ssm` expose recurrent+conv state; Mamba-2, RWKV-7, RecurrentGemma, DeltaNet/GDN/GLA fla checkpoints) are **pre-cutoff, grade A**.
- Evidence grades: **A** = primary source verified + reproducible/live artifact · **B** = paper+repo, numbers to re-verify · **C** = community · **D** = our inference. Fact labels: **[CIF]** current implementation fact · **[PC]** paper claim · **[CO]** community obs · **[OI]** our inference.

---

## Phase 0 — CURRENT reconstruction (verified)

- **CURRENT HEAD** = `03a863f8b27736b62885916ba98bada26d720fd3` · branch `master` · **no upstream, NOT pushed**. Matches the packet's stated closure HEAD (confirmed, not assumed).
- **EXT2 chain intact & ordered:** `5abeab4` pre-registration → `12157b4` results (Case A BLOCKED) → `2b5e946` git-evidence → `03a863f` audit reconciliation (append-only). Pre-registration precedes results.
- **Closure reconciliation exists & committed:** `runs/rnn/RNN-05B-EXT2/AUDIT_RECONCILIATION.md` at `03a863f`. Both LOW audit findings reconciled with **no** original artifact modified.
- **Prior evidence immutable:** no tracked file under `runs/rnn/RNN-05B-EXT2/**`, `RNN-05B-EXT/**`, `RNN-05B-delta-gdn/**` is modified in the working tree (only pre-existing *untracked* `git_evidence.txt`/`stdout.log` helpers). RNN-04 / 05A / 05B / EXT / EXT2 are byte-identical.
- **This packet modifies nothing** — RNN-06 is new scope under `runs/rnn/RNN-06/`.

### Canonical prior artifacts RNN-06 must respect
`ops/rnn_05b_ext2.py`; `runs/rnn/RNN-05B-EXT2/{PRE_REGISTRATION.md, machine_config.json, BASE_QUALIFICATION.json, HANDOFF.md, rnn05bext2_results.json, rnn05bext2_outcomes.json, AUDIT_RECONCILIATION.md}`; substrate `ops/{rnn_delta_substrate,rnn_mc_substrate,rnn_mc_bench,rnn_05b_ext}.py`. `challengeGridSha256=66ff2476…a9e5`; SESOI(DELTA_AURC)=0.05; 3% margin = OPERATOR_HEURISTIC.

### Exact lessons that constrain RNN-06
1. **Two independent synthetic H3 attempts already failed to yield a testable graded regime.** EXT failed via an **unstable base** (train-per-condition seed cliff → TRAIN_PER_CONDITION_STABILITY FAILED). EXT2 removed that confound (train once/freeze/vary inference pressure) but the mixture-trained fixed backbone stayed **flat-high (≥0.98 recall through 250 gap distractors)** → no graded region, MC correctly never ran. Synthetic dense MC is **PARKED**.
2. **Ceiling and cliff are both useless.** The scientific prerequisite is a **graded** region: competent → progressive **material-but-non-total** loss → measurable transition. This is now the **first objective** of RNN-06, ahead of any recovery mechanism.
3. **Mixture-training over the whole stress ladder buys robustness**, and training on the stress ladder then calling it an inference stress test is circular (the EXT2 confound). RNN-06 must separate *training* / *qualification* / *stress-extrapolation* distributions without inventing arbitrary adversarial OOD.
4. **Control-flow invariant works — keep it, and keep it ordered.** BASE qualification persisted + hash-verified STRICTLY before any recovery mechanism; absent/mismatch/unqualified = STOP. EXT2 honored it (no exploratory MC); EXT violated *ordering* (post-block exploratory MC → EXPLORATORY_ONLY).
5. **Seed screening is prohibited for qualification.** Every preregistered seed/checkpoint/prompt-set counts.
6. **The toy single-state DeltaNet may be structurally too clean to forget gracefully** (RNN-04/05A/05B: one large state either solves the toy or fails at a cliff). This is the core reason to move to a **real pretrained recurrent LM** with a fixed, non-tunable state budget and realistic interference.
7. **The intervention is not pre-decided.** EXT2 parks *synthetic dense MC specifically*; it does not prove MC is useless on a real model. Establish the **phenomenon** first; MC is one Phase-06D candidate.
8. **Structural-analogy caveat carries forward.** Even a real GDN state is only STRUCTURALLY_ANALOGOUS to the Qwen deployment cache; representational equivalence NOT_PROVEN. `QWEN_GDN_TRANSPLANT_GATE` stays **DEFER** until a real-model phenomenon + recovery is shown.

---

## Phase 1 — candidate landscape (primary-source synthesis)

### 1A. The enabler is real and mature (grade A, pre-cutoff)
- **[CIF, A]** `flash-linear-attention` (**fla**, MIT, ~5.5k★) implements DeltaNet, **Gated DeltaNet**, GDN-2, RWKV-7, GLA, Mamba-2/3, KDA and ~30 others, wired into a HF-Transformers `Cache` that **stores and returns the per-layer recurrent state matrix `S` and the short-conv state** — programmatically readable, clonable, and **re-injectable** (`forward(..., past_key_values: Cache, use_cache=True)` → updated `Cache`). `fused_recurrent` = per-token granularity; `chunk` = per-chunk-boundary. Repo: github.com/fla-org/flash-linear-attention.
- **[CIF, A]** `mamba_ssm` (state-spaces/mamba) exposes `InferenceParams.key_value_memory_dict[layer] = (conv_state, ssm_state)` as ordinary tensors; HF `transformers` Mamba2/RecurrentGemma/Zamba2/FalconH1/Qwen3Next each ship a `*Cache` with `ssm_states`+`conv_states`. **Observable and capturable.**
- **[CIF/OI, B] Observable ≠ semantically checkpointable.** That an observation API exists does **NOT** establish that a captured state can be restored/branched with BIT_EXACT-or-bounded fidelity. There is public evidence of **GDN segmented / manual-cache inference discrepancies**, and the HF Mamba2 doc notes `mamba_ssm` batched-vs-cached "slight discrepancies"; fla Triton `chunk` vs `fused_recurrent` also differ. **Do not infer BIT_EXACT from the Cache API.** Therefore, for every candidate, `state_semantically_checkpointable_restorable = NOT_QUALIFIED` and lifecycle status `NOT_QUALIFIED` until **RNN-06A** re-earns the lifecycle proof (as RNN-05B did for the toy) on the **exact pinned backend/version/checkpoint**. The candidate matrix now carries separate `base_inference_complexity`, `state_introspection_complexity`, and `lifecycle_qualification_risk` columns to keep these distinct.

### 1B. Runnable pretrained checkpoints that fit a 3090 (grade A, pre-cutoff)
| Family | Checkpoints (HF id) | Params | State form | License |
|---|---|---|---|---|
| **Gated DeltaNet** | `linear-moe-hub/Gated-Deltanet-{340M,1.3B}`, `Idiap/gated-deltanet-attn-1.4B-30B` | 0.34–1.4B | matrix `S[H,d_k,d_v]` + conv | Apache-2.0 |
| **DeltaNet** | `fla-hub/delta_net-{1.3B,2.7B}-100B` | 1.3/2.7B | matrix `S[H,d_k,d_v]` + conv | Apache-class |
| **GLA** (LA control) | `fla-hub/gla-1.3B-100B` | 1.3B | gated linear-attn state | Apache-class |
| **RWKV-7 "Goose"** | `fla-hub/rwkv7-{0.1,0.4,1.5,2.9,7.2}B`, `BlinkDL/rwkv-7-world` | 0.1–7.2B | diag+rank-1 wkv state (head 64) | Apache-2.0 |
| **Mamba-2** | `state-spaces/mamba2-{130m,370m,780m,1.3b,2.7b}` | 0.13–2.7B | matrix `ssm_state (nheads,headdim,d_state≈128)` | Apache-2.0 |
| **RecurrentGemma / Griffin** | `google/recurrentgemma-2b(-it)` | 2B | RG-LRU recurrence + local-attn KV | Gemma (gated) |
| **Falcon-Mamba** | `tiiuae/falcon-mamba-7b` | 7B | Mamba-1 `d_state=16` (vector-ish) | TII Falcon-Mamba 1.0 |
| **Codestral-Mamba** | `mistralai/Mamba-Codestral-7B-v0.1` | 7B | Mamba-2 matrix state | Apache-2.0 |
| **Nemotron-Nano-2** | `nvidia/NVIDIA-Nemotron-Nano-9B-v2` | 9B | Mamba-2/attn hybrid | NVIDIA (verify) |

All of the above (≤~7–9B) fit a 3090 in bf16 or 4-bit. Mamba-2 130M–2.7B gives the **biggest seed/size budget** (a built-in state-capacity scaling axis); the fla GDN/DeltaNet/GLA line is the **direct pretrained analog of the RNN-05B ports** (same library, first-class state).

### 1C. The Qwen deployment target (grade A/B)
- **[CIF, A]** **Qwen3.6-35B-A3B** (2026-04-14, **Apache-2.0**, 35B/3B-active, 40 layers = 30 GDN + 10 gated-attn) runs on a 3090 only at **4-bit/AWQ + offload** (~18–21 GB). **For RNN-06, the transformers/FLA path (4-bit load) is the currently identified clean state-introspection route; equivalent experimental recurrent-state access through vLLM, SGLang, llama.cpp or other serving backends is `NOT_QUALIFIED` (not established impossible, just not yet identified/verified).** **[OI, D]** est. total recurrent state ~16 MB bf16 (constant in context) — **[verify config.json head counts].**
- **[CIF]** **Qwen3-Next-80B**, **Qwen3.5 (~397B community-grade)**, **Jamba-1.5 (52B)** — **too big** for state-introspection on a 3090 → PARK (architecture reference only).

### 1D. On-thesis "escape-hatch" architectures — most relevant, least runnable (post-cutoff, [verify])
These explicitly architect a path to recover what the single compressed final state overwrites — i.e. **our exact research question stated as their thesis**:
- **[PC, B, verify]** **DART** — "Decoded Attention over Recurrent States," arXiv **2608.02032**. Mamba-2 + State-Memory-Attention that decodes token-conditioned keys/values from *retained per-chunk states* and attends over them. **No public code/weights found.**
- **[PC, B+, verify]** **HOLA** ("A Hippocampus for Linear Attention: an exact memory for what the recurrent state forgets"), arXiv **2607.02303**. Compressive delta-rule state + bounded **exact KV side-cache**, writes gated by prediction-residual `β·‖e‖`. Most on-point framing; code/weights unconfirmed.
- **[PC/CIF, A/B]** **Sparse Delta Memory (SDM)**, arXiv **2607.07386**, Meta FAIR — GDN with an explicit N×d sparse memory table (product-key addressing). **Code + Triton kernels + configs released** (github.com/facebookresearch/sparse-delta-memory, CC-BY-**NC**), **no weights** (train-from-scratch; 1.4B is the 3090 ceiling).
- **[PC/CIF, A]** **Gated DeltaNet-2 (GDN-2)**, arXiv **2605.22791**, NVlabs — splits GDN's scalar gate into channel-wise **erase (key-side)** + **write (value-side)** gates; **official repo + in fla** (`GDN-2`); **NC license**, downloadable weights **[verify]**.
- **[PC, B, verify]** **FG²-GDN** (2604.19021, Meituan) — channel-wise β (erase/write decoupling); no code/weights.
- **[PC, B, verify]** **"Memory Caching: RNNs with Growing Memory"**, arXiv **2602.24281** (Google, Behrouz et al.) — the **official write-up of the Memory-Caching line** the RNN-04/05 campaign reverse-engineered ("no official code" note now has a paper; code still unconfirmed). Directly relevant as a Phase-06D candidate and a validity check on the campaign's earlier reconstruction.
- **[PC/CIF]** **StateX** (2509.22630, THUNLP, **code released**, CC-BY-SA) — post-training **state expansion** on public GLA/Mamba2-1.3B checkpoints; "recall ∝ recurrent-state size." A tool to *manipulate* state capacity on existing checkpoints, not a new backbone.
- **[CO/CIF]** **ReplaySSM** (Tri Dao blog) is a **decode-speed kernel** (ring-buffer input replay, math-equivalent) — **not** a memory mechanism; the architecture-level "replay" is **SMR** (arXiv 2405.17534). Relevant only as proof that *linear-recurrence state is exactly reconstructible from cached inputs* (supports exact branch replay in Phase 4).

### 1E. Literature that already measures the graded regime (grade A, mostly pre-cutoff)
- **[PC, A]** **Based** (arXiv **2402.18668**) + **Zoology/MQAR** (arXiv **2312.04927**), Arora et al. — canonical **MQAR** harness; a **fundamental recurrent-state-size ↔ recall tradeoff**: fix the model, sweep #KV-pairs → **smooth graded** accuracy decline. **This is the graded knob.** Repos: HazyResearch/{zoology,based}.
- **[PC, A]** **Stuffed Mamba** (arXiv **2410.07145**, COLM 2025) — state-size sets a **contextual-memory capacity bound**; "state collapse" when trained on too-short contexts; Mamba-2 near-perfect ≤8K, **degrades past 16K**. Graded capacity curve + a distinct collapse regime.
- **[PC, A]** **Effective State-Size (ESS)** (arXiv **2504.19561**, ICML 2025) — control-theoretic *measured* state utilization → a preregisterable manipulation-check / dependent variable, not inferred from cache bytes.
- **[PC, A]** **MAD** (arXiv **2403.17844**) — synthetic unit tests (MQAR, selective copy, compression, fuzzy recall) with graded difficulty.
- **[PC, A]** **ROME-for-Mamba** (arXiv **2404.03646**) + IOI-in-Mamba (2407.14008) + selective-memory autoencoders (2512.15653, [verify]) — precedent that **linear probing / causal tracing / editing the SSM hidden state** is a published, legitimate method → supports the Phase-4 probe.
- **[PC, A] (cliff controls, use as negative contrast):** **Repeat-After-Me** (2402.01032, copy-length cliff) and **Illusion of State** (2404.08819, TC⁰ state-tracking wall). Design implication: **MQAR-load & context-vs-capacity = graded; copy/passkey-length & formal state-tracking = cliff.** Preregister the graded knob; keep a cliff task only as a contrast.

---

## State-semantics analysis
- **DeltaNet/GDN/GDN-2 (and Qwen GDN):** matrix `S∈ℝ[H,d_k,d_v]` + short-conv buffers; delta rule `S_t=S_{t-1}(I−β_t k_tk_tᵀ)+β_t v_tk_tᵀ`, GDN adds decay `α_t`, GDN-2 splits erase/write. Substituting a *compatible* historical `S` needs **no retraining**; changing state *shape/semantics* does. Gating "bakes in" decay history → historical-state substitution is semantically trickier for GDN than for ungated DeltaNet (an argument for including both).
- **Mamba-2:** matrix `ssm_state (nheads,headdim,d_state≈128)` + conv — the cleanest small pretrained matrix state; closest ecological analog to the campaign's matrix-state work.
- **RWKV-7:** diagonal-plus-rank-1 wkv state (head 64) — *related but not identical* to GDN (vector gates, in-context learning-rate); a good extra arm, **not** a GDN drop-in.
- **RecurrentGemma (Griffin):** genuine RG-LRU linear recurrence **plus local sliding-window attention** — gives a **built-in attention control inside one model** (isolate recurrent vs attention pathway).
- **Observation vs exploitation:** information *present* in an earlier state (probe/replay recoverable) is a **different claim** from a mechanism being able to *use* it. RNN-06 keeps these separate (Phases 4 vs 5-D).

## Failure-regime analysis (what "graded" must thread)
| Regime | Example evidence | Verdict for H3 |
|---|---|---|
| **Ceiling everywhere** | EXT2 fixed GDN (≥0.98 through 250 distractors); Mamba-2 ≤8K | DISQUALIFY (no signal) |
| **Seed/train cliff** | EXT train-per-condition GDN/DN | DISQUALIFY (instability, not forgetting) |
| **Capacity cliff** | copy-length (Repeat-After-Me), TC⁰ wall (Illusion of State) | DISQUALIFY as primary; keep as contrast |
| **Graded capacity/interference** | Based/Zoology MQAR #pairs↑; Stuffed Mamba length↑ toward collapse | **TARGET** — preregister here |

---

## Phase 2 — define the phenomenon before the intervention
**BASE target (no recovery mechanism):** on **one frozen pretrained recurrent LM**, information written early and queried late degrades **progressively** as controlled pressure rises (competent low-pressure → material-but-non-total high-pressure), *and* that information is separately shown to persist in **earlier** recurrent states. Ceiling-everywhere and cliff/collapse both DISQUALIFY.

**Probe shortlist (down-select in 06B against the chosen model):**
| Probe | Stresses | Graded prospect | State-specificity | Confound risk |
|---|---|---|---|---|
| **MQAR, #KV-pairs vs fixed state** | write-capacity | **HIGH (published)** | HIGH | low (synthetic); realism low |
| Delayed KV retrieval (early write / late query) | retention | HIGH | HIGH | moderate |
| Selective retrieval under interference | overwrite | HIGH | HIGH | moderate |
| Continual entity/state tracking (NL) | running state | HIGH & realistic | HIGH | LM-competence/tokenizer |
| Passkey / needle (recurrent-adapted) | single-fact retention | MED (often cliff) | MED | length/position |
| Copy / reconstruction | bulk fidelity | MED (cliff) | MED | length |
| State-decoding probe | *is info in the state* | Phase-4 tool | HIGHEST | probe strength |

**Preferred BASE:** MQAR-style memory-bound retrieval on a real pretrained model, **#tracked-items/interference swept at fixed length & positions** to sit the model in the graded band (Based/Zoology knob), with a **paired natural-language** continual-entity-tracking variant on the same pressure axis. Reuse the EXT2 nested-monotonic superset ladder (preserves snapshot-position identity).

## Phase 3 — distinguish causes of degradation (falsifiers, not just a declining score)
| Competing cause | Control / falsification handle |
|---|---|
| **State capacity pressure (target)** | vary #items/interference at **fixed** length & positions → isolates state load from length |
| Sequence-length shift | length-matched conditions; pack so length constant while pressure varies |
| Positional / generic LM decay | fixed positions across doses (EXT2); position-only control (same length, no added items) |
| Attention-window effects (hybrids) | probe a **recurrence-only** layer; or use RecurrentGemma's local-attn split as the control |
| Tokenizer / input difficulty | fixed surface form; synthetic control with fixed token inventory |
| Prompt sensitivity / "confusion" | multiple templates; report variance |
| Training-distribution mismatch | keep task in-domain; report OOD explicitly |
| Quantization degradation | BASE at full precision first; quant is a **separate declared axis**, never confounded |
| Convolution-state corruption | checkpoint/restore conv too; conv-only vs matrix-only restore (05B pattern) |
| **Recurrent-state overwrite (target mechanism)** | Phase-4 historical-state probe + target-proximal ablation |
| Evaluator/probe weakness | eager/reference decode oracle; probe reads a *fresh* state sanity check |

**"Final state lost historical info" is earned only when:** score declines with pressure **AND** length/position/quant controls stay flat **AND** the info is shown present in an earlier state (Phase 4) **AND** the loss localizes to the recurrent pathway.

## Phase 4 — historical-state information probe (presence ≠ exploitability)
Two distinct claims, never conflated: **(I)** an earlier state contains info the final state no longer exposes; **(II)** a recovery mechanism can use it. Phase 4 establishes (I) only, preferring **minimal/no backbone modification**:
| Method | Backbone change | Strength | Main risk / mitigation |
|---|---|---|---|
| **Frozen linear probe** (predict queried value from captured state) | none | standard, low-confound | probe capacity → cap size; report probe@historical − probe@final **gap** |
| Shallow nonlinear probe | none | catches nonlinear encoding | over-power → matched-capacity control |
| **Exact branch replay** (restore state@t, continue, read answer) | none | causal, no learned probe | needs deterministic restore (06A) |
| Counterfactual state substitution (historical→final) | none | causal | interpretation care |
| State→token decoder | small head | expressive | training confound |
| Representational similarity / decodable-MI proxy | none | cheap screen | correlational only |

**Primary instrument:** frozen linear probe (presence, cheap) **corroborated by** exact branch replay (causal; uses the deterministic checkpoint/restore proven in 06A). **Information-presence gap** = accuracy from state@early − state@final at matched pressure; a positive gap = info was present and lost from the final state — the bridge EXT/EXT2 never reached (they blocked at BASE).

## Phase 5 — experimental causal hierarchy (gated stages)
- **RNN-06-P0 — Frozen-Checkpoint BASE Regime Scout (EXPLORATORY pre-packet, NOT a confirmatory stage).** Purposes, and only these: (1) prove the candidate checkpoint executes on this workstation; (2) freeze exact checkpoint/revision/config identity (`calibrationSetSha256`, model revision hash); (3) run a cheap **inference-only** calibration sweep; (4) judge whether a plausible **non-ceiling, non-cliff graded region** is reachable within a bounded budget; (5) choose/falsify the candidate model + pressure axis **before** lifecycle engineering. P0 is **not** confirmatory H3 evidence, does **not** qualify historical-state semantics, and **MUST NOT** emit `FIXED_BACKBONE_GRADED_REGION = QUALIFIED`. Its only statuses are **`P0_GRADED_BAND = PLAUSIBLE | NOT_FOUND_WITHIN_BUDGET | MODEL_NOT_RUNNABLE`**. P0 may tune the pressure range on its calibration examples; those examples are then quarantined from 06B.
- **RNN-06A — State observability & full lifecycle qualification.** On the real model: full state inventory (every recurrent matrix + conv + SSM hidden, per layer/head); per-request isolation (no cross-request leakage); checkpoint/restore **BIT_EXACT-or-bounded**; branch restore; weights provably immutable. Mints `FROZEN_BACKBONE_LIFECYCLE = QUALIFIED | NOT_QUALIFIED`. **Gate:** lifecycle must qualify (re-earn the 05A→05B gap on the real model, on the pinned backend — not inferred from the Cache API) before 06B.
- **RNN-06B — Confirmatory BASE forgetting qualification.** The **only** stage that may mint `FIXED_BACKBONE_GRADED_REGION = QUALIFIED | BLOCKED`. Preregistered challenge on **independent deterministic qualification examples** (disjoint from P0 calibration) → **stable graded degradation across ALL seeds/prompt-sets** (common overlapping graded region, EXT2 §7 gate reused). **No recovery mechanism.** No region → `H3_TESTABILITY = BLOCKED_ON_<MODEL>`, STOP (Case A). *The make-or-break gate.*
- **RNN-06C — Historical-state information (Phase 4).** Only if 06B qualifies. Establish info-presence gap > 0 (frozen probe + branch replay). **Gate:** gap positive & robust, else `HISTORICAL_INFO_ABSENT`, STOP.
- **RNN-06D — Recovery intervention.** Only if B **and** C qualify. Candidate arms: final-state BASE; **historical state via minimal frozen reader** (MC-style, now justified); **matched-capacity reader WITHOUT history** (the control EXT2 built but never ran); retrieval over external hidden-state snapshots; co-trained vs frozen; parameter-matched controls. MC is one arm, not the assumed winner.

Order: **P0 (exploratory scout) → 06A (lifecycle) → 06B (confirmatory BASE) → 06C (info) → 06D (recovery)**. Each confirmatory arrow is a hard gate with a **persisted + hash-verified** qualification artifact (EXT2 control-flow invariant), **ordered** (fixes the EXT ordering defect): 06B/06C/06D entrypoints LOAD+VERIFY the prior artifact; absent/mismatch/unqualified = STOP. **P0 outputs carry no confirmatory authority** and cannot substitute for 06A or 06B.

## Phase 6 — prevent the previous confounds (explicit)
- **EXT confound (fresh model per condition):** RNN-06 uses **ONE frozen pretrained checkpoint per experimental identity**; pressure varies at inference only; no per-condition training.
- **EXT2 confound (train on the exact stress ladder, then call it a stress test):** the model is **already pretrained** — its training distribution is fixed and is **not** our stress ladder, structurally breaking the circularity. Declared distributions: *pretraining* (given) / *qualification* (in-domain competence) / *stress-extrapolation* (increasing pressure), chosen to probe **state capacity**, not to manufacture arbitrary adversarial OOD.
- **Seed screening:** forbidden for qualification; every preregistered seed/prompt-set/checkpoint counts.
- **Outcome-before-gate:** forbidden; BASE (06B) + info (06C) artifacts persisted+verified before any 06D path can run; ordered so no exploratory recovery runs post-block.

## Phase 7 — candidate statistical design
- **Unit:** (frozen checkpoint × prompt/example) cluster; targets nested in sequence in checkpoint. Real pretrained ⇒ few checkpoints → shift inferential weight to **example/template-level** clustering + use multiple pretrained **sizes** (Mamba-2 130M–2.7B; GDN 340M/1.3B) as the cross-checkpoint/state-capacity axis.
- **Paired** across methods on identical examples (A/B/C/D share examples), EXT2 §10.
- **Nested monotonic** pressure axis (superset per dose), EXT2 §5 — preserves snapshot-position identity.
- **SESOI:** **re-derive** on the real model's cost model (historical-snapshot bytes × interval + reader latency); do **not** inherit 0.05 blindly. 3% stays OPERATOR_HEURISTIC.
- **CIs/bootstrap:** cluster-aware (example/template-level) bootstrap; hierarchical across sizes. State honestly whether N checkpoints supports population inference or only **direction/stability** (EXT2 was explicit: 3 backbones = direction only).
- **Dose-response:** monotone fit; **AURC** primary, or **D50 + transition width** if a single sharp-but-graded transition; isotonic-residual monotonicity diagnostic; **ESS** (2504.19561) as a manipulation check that state utilization actually moves.
- **Recovery/harm:** exposed denominators (n_base_wrong, n_recovered, RECOVERY_RATE; n_base_correct, n_harmed, HARM_RATE; NET), pooled query rates (not per-seed averaged), EXT2 §12.
- **Multiple comparisons:** control across methods × doses (hierarchical shrinkage or explicit FWER note); pre-declared primary contrast = **DELTA_AURC(reader vs BASE)** and **(reader vs matched-capacity no-history control)**.
- **Calibration:** report probe/reader accuracy-vs-confidence calibration (a reader that helps only by better calibration is not "recovering historical info").
- **Stopping/futility:** 06B graded-region gate is the futility rule; 06C info-gap ≤ 0 → stop; pre-declare a compute cap with "no band within budget → BLOCKED".
- **Exploratory vs confirmatory (contamination boundary):** three distinct frozen identities — **`calibrationSetSha256`** (examples P0 is permitted to tune the pressure range against — exploratory), **`qualificationSetSha256`** (independent deterministic examples generated **after** P0 freezes the pressure range — confirmatory, used by 06B), **`stressGridSha256`** (the frozen nested-monotonic pressure ladder). Rules: calibration and qualification **example sets MUST be disjoint**; template **families** preferably disjoint (if shared, declared + justified); RNG = fixed integer seeds (process-stable, **not** `hash()`-based) from a recorded master seed, generation version pinned; P0 may tune **only** the pressure range/ladder + probe/eval calibration; the qualification example set, stress grid, and metrics are **frozen before any 06B outcome-bearing run**. 06B must **not** silently validate on P0's tuned examples and call it confirmatory.

## Phase 8 — compute feasibility (RTX 3090 24 GB / 64 GB RAM)
Estimates use bf16; **[verify config.json]** for exact head counts/state dims.
| Item | GDN-1.3B (fla) | Mamba-2-1.3B | Qwen3.6-35B-A3B |
|---|---|---|---|
| Weights VRAM | ~2.6 GB (bf16) | ~2.6 GB | ~18–21 GB (4-bit/AWQ + offload) |
| Recurrent-state bytes / snapshot | `≈ Σ_layers H·d_k·d_v·2` ≈ **O(1–10 MB)** [verify] | `≈ Σ_layers nheads·headdim·d_state·2` ≈ **O(1–10 MB)** [verify] | ~16 MB [verify] |
| Historical-state store, seq 2K, snapshot/64 tok (≈32 snaps) | ~0.3–0.4 GB | ~0.3–0.4 GB | ~0.5 GB |
| KV/cache (hybrid attn layers) | n/a (pure) | n/a (pure) | bounded via SWA; declare window |
| BASE sweep (06B): sizes × doses × N examples, **inference-only** | minutes–1 GPU-hr | minutes–1 GPU-hr | hours (offload-bound) |
| Probe/reader training (06C/06D): tiny head, frozen backbone | minutes | minutes | tens of min |
| **P0 scout (exploratory band reachability)** | **< 1 GPU-hr** | **< 1 GPU-hr** | ~2–4 GPU-hr |
Disk: historical snapshots for a full sweep ≪ 50 GB. **Design scouts cheaply before scaling:** the exploratory **RNN-06-P0** go/no-go (is a plausible non-ceiling/non-cliff band reachable on a frozen pretrained model?) is a **sub-GPU-hour inference sweep** on GDN-1.3B or Mamba-2-1.3B — no training, no big model. P0 emits only `P0_GRADED_BAND = PLAUSIBLE | NOT_FOUND_WITHIN_BUDGET | MODEL_NOT_RUNNABLE`; the **confirmatory** `FIXED_BACKBONE_GRADED_REGION` verdict is minted only by 06B on independent qualification examples.

## Phase 9 — decision matrix & recommendation
Compact view below; **the full machine-readable matrix (`RNN-06_candidate_matrix.{json,csv}`, JSON authoritative) now carries separate `state_externally_accessible`, `state_semantically_checkpointable_restorable`, `base_inference_complexity`, `state_introspection_complexity`, `lifecycle_qualification_risk`, and `lifecycle_status` columns.** Compact columns: SciRel = scientific relevance; StateObs = observation API exists; **Ckpt? = semantically checkpointable/restorable (all `NOT_QUAL` until 06A)**; HistCap = historical-state capturable; 3090 = feasibility; LcRisk = lifecycle-qualification risk; BASEfg = BASE-forgetting-test feasibility; GPU$ = expected cost; Grade = evidence.

| Candidate | SciRel | StateObs | Ckpt? | HistCap | 3090 | LcRisk | BASEfg | GPU$ | Key blockers | Grade |
|---|---|---|---|---|---|---|---|---|---|---|
| **GDN-1.3B (fla / linear-moe-hub)** ⭐PRIMARY | **High** (Qwen-GDN transplant target) | Yes (fla Cache) | **NOT_QUAL** | Yes | **Fits easily** | **HIGH** (GDN segmented/manual-cache discrepancy) | Med-High | Low | robust-within-load risk (EXT2); Triton/WSL friction; lifecycle unproven | **A** |
| **Mamba-2 1.3B/2.7B (state-spaces)** ⭐FALLBACK | High (strongest *directly-relevant* state-capacity lit among runnable fallbacks; SSM≠GDN) | Yes (`mamba_ssm`/HF) | **NOT_QUAL** | Yes | **Fits easily** | MED (batched-vs-cached) | Med-High (graded band on *our* ckpt NOT_QUAL till measured) | Low | not delta-rule GDN; BIT_EXACT unproven | **A** |
| **RecurrentGemma-2B (Griffin)** | High (real recurrence + built-in attn control) | Yes (HF cache) | NOT_QUAL | Yes | Fits easily | MED | Med | Low | Gemma gated license; local-attn confound to isolate | **A** |
| **DeltaNet-1.3B (fla)** | High (ungated mechanism-proper arm) | Yes (fla) | NOT_QUAL | Yes | Fits easily | MED-HIGH | Med | Low | weak research-scale LM; lifecycle unproven | **A** |
| **GLA-1.3B (fla)** LA control | Control (campaign continuity) | Yes (fla) | NOT_QUAL | Yes | Fits easily | MED | Low-Med | Low | control-only; lifecycle unproven | **A** |
| **RWKV-7 0.1–2.9B (fla/BlinkDL)** | Med-High (mature state-first; related mech) | Yes (fla) | NOT_QUAL | Yes | Fits (all) | MED | Med | Low | not a GDN drop-in | **A** |
| **Nemotron-Nano-9B-v2** | Med-High (strong long-ctx hybrid) | Yes (HF) | NOT_QUAL | Yes | **Tight ~22 GiB** | MED-HIGH | Med-High | Med | tight VRAM; hybrid confound | **A/B** |
| **Falcon-Mamba-7B / Codestral-Mamba-7B** | Med (single-arch attn-free 7B) | Yes | NOT_QUAL | Yes | Fits (bf16/4-bit) | MED | Med | Low-Med | Falcon-Mamba d_state=16 less analogous; license | **A** |
| **GatedDeltaNet-2 (NVlabs)** | High (erase/write control) | Yes (in fla) | NOT_QUAL | Yes | Fits | HIGH | Med | Low | **NC license**; downloadable weights [verify]; post-cutoff | **B** |
| **Sparse Delta Memory (Meta)** | High (sparse slots resist overwrite) | Yes (code) | N/A (no weights) | Yes | 1.4B ceiling | HIGH | Med | **High** | **no weights**, NC, ~1T-tok train; post-cutoff | **A/B** |
| **DART / HOLA** | **Highest (their thesis = our question)** | [verify] | N/A (no code) | Yes (by design) | ≤~800M fits | UNKNOWN | ? | High | **no code/weights** found; post-cutoff | **B** |
| **StateX (THUNLP)** | Med (state-expansion *tool*) | Yes | NOT_QUAL | Yes | Fits | MED | tool | Low-Med | a method, not a backbone; post-cutoff | **A-/B** |
| **Qwen3.6-35B-A3B** | **Highest (deployment target)** | Transformers/FLA path (4-bit); serving backends NOT_QUAL | NOT_QUAL | Yes | **4-bit+offload only** | HIGH | Med | Med-High | offload-bound; serving-backend state access NOT_QUALIFIED | **A/B** |
| Qwen3-Next-80B / Qwen3.5-397B / Jamba-52B | Ref only | — | — | — | **Too big** | n/a | — | — | do not fit for state work | A |

**Recommendation:**
- **PRIMARY: `Gated-DeltaNet-1.3B` (fla / `linear-moe-hub`).** Maximizes *transplant relevance* (real pretrained GDN = the Qwen deployment mechanism), *state observability* (the same fla `Cache` the RNN-05B ports used — note: **observability, not proven checkpointability**), Apache-2.0, and *continuity* with the campaign — while fitting a 3090 with headroom. **Two explicit risks:** (a) a 100B-token pretrained GDN may, like EXT2's mixture-trained backbone, be **robust within its trained load** → 06B must push #KV-pairs/interference **past training-typical load**, and the BASE gate may still BLOCK; (b) **lifecycle semantics are `NOT_QUALIFIED`** — GDN segmented/manual-cache discrepancies mean 06A must *prove* BIT_EXACT-or-bounded restore on the pinned backend before 06C relies on it. These risks are why the go/no-go is a cheap exploratory **P0** scout ahead of any lifecycle build.
- **FALLBACK / graded-regime anchor: `Mamba-2` (state-spaces, 130M–2.7B).** Cleanest *observable* matrix state, the **strongest *directly relevant* published state-capacity / long-context evidence among the readily runnable fallback candidates** (Based/Zoology recall-vs-state; Stuffed-Mamba length-vs-capacity), a **built-in size-scaling axis** (five sizes), and ROME-for-Mamba probing precedent. **The actual existence of a smooth common graded band on our frozen `mamba2-1.3b` checkpoint remains `NOT_QUALIFIED` until measured.** Use if the GDN checkpoint is flat-high, or run **in parallel** as the anchor. Add **GLA-1.3B** (LA control) and **RecurrentGemma-2B** ("real recurrence + built-in attention control") as cross-checks.
- **WATCH / research (do not build):** DART, HOLA, FG²-GDN, "Memory Caching" paper (2602.24281), SDM, GDN-2, StateX — highly on-thesis but **no runnable weights** (or NC + train-from-scratch); revisit only if a **newly verified source materially changes the candidate decision**.
- **DEFER: Qwen3.6-35B-A3B** (`DEFER_VALIDATION_TARGET`) — only after a positive small-model result unlocks `QWEN_GDN_TRANSPLANT_GATE`; do not start on it.
- **Evidence still missing before implementation:** (1) **P0** — confirm the GDN checkpoint runs on this box and freeze its checkpoint/revision/config identity; (2) **P0** — cheap inference-only sweep → `P0_GRADED_BAND` status; (3) **06A** — prove lifecycle (BIT_EXACT-or-bounded checkpoint/restore/branch) on the pinned backend/version/checkpoint (**do not infer from the Cache API**); (4) **06B** — confirmatory graded-region qualification on independent deterministic examples; (5) exact `config.json` state dims for SESOI/storage; (6) weight/license status of GDN-2/SDM/DART/HOLA.

## Phase 10 — preregistration skeleton (DRAFT — DO NOT EXECUTE)
- **Hypotheses:** **H_BASE** (a common graded forgetting region exists on the frozen real model over the pressure axis); **H_INFO** (early-state info-presence gap > 0 at matched pressure); **H_REC** (a historical-state reader beats final-state BASE by ≥ SESOI **and** beats a matched-capacity no-history reader).
- **Falsifiers:** flat-high or cliff BASE → BLOCKED; info-gap ≤ 0 → HISTORICAL_INFO_ABSENT; recovery ≤ matched-capacity control → READER_CAPACITY_ARTIFACT; loss explained by length/position/quant controls → not a state phenomenon.
- **Stage identities:** **`calibrationSetSha256`** (P0-tunable, exploratory) · **`qualificationSetSha256`** (06B confirmatory, independent + disjoint from calibration) · **`stressGridSha256`** (frozen nested pressure ladder). Fixed integer seeds (process-stable, not `hash()`-based); generation version pinned; subprocess self-check; recorded identically in prereg / machine_config / BASE_QUALIFICATION / results / outcomes. **Only 06B mints `FIXED_BACKBONE_GRADED_REGION`; P0 emits only `P0_GRADED_BAND`.**
- **Challenge identity:** `challengeGridSha256` (≡ `stressGridSha256` at 06B) over frozen task config + pressure ladder + example-id hashes.
- **Model/checkpoint identity:** HF id + revision hash + file SHA-256; exact config; quantization state declared.
- **State identity:** full state-inventory manifest (per-layer recurrent matrix + conv + SSM hidden; shapes/bytes); snapshot schedule; snapshot hashes.
- **Pressure axis:** nested monotonic, inference-only, fixed length & positions (EXT2 pattern).
- **Competence criterion:** max BASE ≥ τ_hi at low pressure (the model can actually do the task).
- **Graded-region criterion:** min BASE ≤ τ_lo **AND** ≥ k mid-band doses **AND** mid-band **OVERLAPS across ALL** seeds/prompt-sets (EXT2 §7).
- **All-seeds/prompts policy:** every preregistered seed/prompt-set counts; **no screening**.
- **Stopping rule:** P0 scout → 06A lifecycle → 06B BASE gate → 06C info gate → 06D; compute cap; futility = P0 `NOT_FOUND_WITHIN_BUDGET` or no band in 06B budget.
- **Failure classes:** P0_MODEL_NOT_RUNNABLE / P0_GRADED_BAND_NOT_FOUND_WITHIN_BUDGET / FROZEN_BACKBONE_LIFECYCLE_NOT_QUALIFIED / BLOCKED_ON_<MODEL> / HISTORICAL_INFO_ABSENT / NOT_DETECTED_IN_QUALIFIED_REGIME / READER_CAPACITY_ARTIFACT / POSITIVE_CANDIDATE.
- **Evidence ordering:** prereg commit → **P0 (exploratory, no confirmatory authority)** → 06A → 06B → 06C → 06D, each confirmatory stage **persisted + hash-verified before the next**.
- **Decision Cases:** **P0** exploratory only (`PLAUSIBLE | NOT_FOUND_WITHIN_BUDGET | MODEL_NOT_RUNNABLE`; never a QUALIFIED verdict). **A** no graded region at 06B → BLOCKED, stop. **B** region but no info-gap or no recovery → PARK. **C** graded + info-gap + recovery > matched control + ablation-supported + LA/attention control negative + path counters prove historical-state use → **POSITIVE_CANDIDATE** → authorizes *design* of a separate Qwen qualification packet only (no automatic Qwen run).
- **Gate:** 06C/06D code LOADS+VERIFIES the persisted BASE (and info) qualification artifact; absent/mismatch/unqualified = STOP (no exploratory recovery).

---

## Sources (primary; [verify] = post-cutoff, confirm before relying)
**Enabler/substrate (grade A):** fla github.com/fla-org/flash-linear-attention · `mamba_ssm` github.com/state-spaces/mamba · HF Mamba2 doc · checkpoints: `fla-hub/delta_net-{1.3B,2.7B}-100B`, `linear-moe-hub/Gated-Deltanet-{340M,1.3B}`, `fla-hub/gla-1.3B-100B`, `fla-hub/rwkv7-*`, `state-spaces/mamba2-{130m…2.7b}`, `google/recurrentgemma-2b`, `tiiuae/falcon-mamba-7b`, `mistralai/Mamba-Codestral-7B-v0.1`, `nvidia/NVIDIA-Nemotron-Nano-9B-v2`.
**GDN family:** GDN arXiv:2412.06464 + NVlabs/GatedDeltaNet · RWKV-7 arXiv:2503.14456 · Qwen3.6-35B-A3B (HF, Apache-2.0) · Qwen3-Next vllm blog 2025-09-11.
**Graded-forgetting literature (grade A):** Based 2402.18668 · Zoology/MQAR 2312.04927 · Stuffed Mamba 2410.07145 · Effective State-Size 2504.19561 · MAD 2403.17844 · ROME-for-Mamba 2404.03646 · IOI-in-Mamba 2407.14008 · Repeat-After-Me 2402.01032 (cliff) · Illusion of State 2404.08819 (cliff).
**On-thesis escape-hatch [verify, post-cutoff]:** DART 2608.02032 · HOLA 2607.02303 · Sparse Delta Memory 2607.07386 + facebookresearch/sparse-delta-memory (NC) · GDN-2 2605.22791 + NVlabs/GatedDeltaNet-2 (NC) · FG²-GDN 2604.19021 · "Memory Caching: RNNs with Growing Memory" 2602.24281 · StateX 2509.22630 + THUNLP/StateX · SMR 2405.17534 · selective-memory autoencoders 2512.15653.

## Exactly one next recommendation (NOT executed)
**Open a NEW implementation session for `RNN-06-P0 — Frozen-Checkpoint BASE Regime Scout` (NOT RNN-06A).** Its only deliverables: (1) prove the frozen pretrained `Gated-DeltaNet-1.3B` (fla) checkpoint runs on this workstation and freeze its checkpoint/revision/config identity (`calibrationSetSha256`); (2) run a cheap **inference-only** MQAR #KV-pair calibration sweep (with `Mamba-2-1.3B` as a parallel anchor); (3) report **`P0_GRADED_BAND = PLAUSIBLE | NOT_FOUND_WITHIN_BUDGET | MODEL_NOT_RUNNABLE`** and a candidate pressure range — **without** minting any confirmatory `FIXED_BACKBONE_GRADED_REGION` verdict (only RNN-06B may) and **before** any lifecycle, probe, or recovery machinery. P0 is exploratory; if it returns `PLAUSIBLE`, the confirmatory sequence RNN-06A (lifecycle) → 06B (BASE) → 06C (info) → 06D (recovery) follows under its own pre-registration.

**STOP.** (No implementation, no GPU run, no Qwen/serving changes, nothing pushed.)
