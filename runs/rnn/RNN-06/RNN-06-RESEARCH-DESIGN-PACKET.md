# RNN-06 — Research & Experimental-Design Packet
## Real Recurrent LM: Long-Horizon State Loss & Historical-State Recovery

**Status: RESEARCH + DESIGN ONLY. No implementation. No GPU runs. Nothing modified. Nothing pushed.**
Authored 2026-08-11. This packet designs RNN-06; it does not execute it. Memory Caching is **not** assumed to be the intervention.

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
- **[CIF, A]** `mamba_ssm` (state-spaces/mamba) exposes `InferenceParams.key_value_memory_dict[layer] = (conv_state, ssm_state)` as ordinary tensors; HF `transformers` Mamba2/RecurrentGemma/Zamba2/FalconH1/Qwen3Next each ship a `*Cache` with `ssm_states`+`conv_states`. **Observable, capturable, restorable.**
- **[CIF, B]** BIT_EXACT restore is backend-dependent (fla Triton `chunk` vs `fused_recurrent`; `mamba_ssm` batched-vs-cached "slight discrepancies" noted in the HF Mamba2 doc). → RNN-06A must re-earn the lifecycle proof (as RNN-05B did for the toy) per chosen backend.

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
- **[CIF, A]** **Qwen3.6-35B-A3B** (2026-04-14, **Apache-2.0**, 35B/3B-active, 40 layers = 30 GDN + 10 gated-attn) runs on a 3090 only at **4-bit/AWQ + offload** (~18–21 GB). State is cleanly introspectable **only via the transformers/fla path** (needs 4-bit load to fit), **not** via vLLM/SGLang/llama.cpp. **[OI, D]** est. total recurrent state ~16 MB bf16 (constant in context) — **[verify config.json head counts].**
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
- **RNN-06A — State observability & lifecycle qualification.** On the real model: full state inventory (every recurrent matrix + conv + SSM hidden, per layer/head); per-request isolation (no cross-request leakage); checkpoint/restore BIT_EXACT-or-bounded; branch restore; weights provably immutable. **Gate:** lifecycle must qualify (re-earn the 05A→05B gap on the real model) before 06B.
- **RNN-06B — BASE forgetting qualification.** Preregistered challenge → **stable graded degradation across ALL seeds/prompt-sets** (common overlapping graded region, EXT2 §7 gate reused). **No recovery mechanism.** No region → `H3_TESTABILITY = BLOCKED_ON_<MODEL>`, STOP (Case A). *The make-or-break gate.*
- **RNN-06C — Historical-state information (Phase 4).** Only if 06B qualifies. Establish info-presence gap > 0 (frozen probe + branch replay). **Gate:** gap positive & robust, else `HISTORICAL_INFO_ABSENT`, STOP.
- **RNN-06D — Recovery intervention.** Only if B **and** C qualify. Candidate arms: final-state BASE; **historical state via minimal frozen reader** (MC-style, now justified); **matched-capacity reader WITHOUT history** (the control EXT2 built but never ran); retrieval over external hidden-state snapshots; co-trained vs frozen; parameter-matched controls. MC is one arm, not the assumed winner.

Each arrow is a hard gate with a **persisted + hash-verified** qualification artifact (EXT2 control-flow invariant), **ordered** (fixes the EXT ordering defect): 06C/06D entrypoints LOAD+VERIFY the prior artifact; absent/mismatch/unqualified = STOP.

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
- **Exploratory vs confirmatory:** ladder/probe-capacity calibration = EXPLORATORY (tune before freeze); everything after the `challengeGridSha256` freeze = CONFIRMATORY.

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
| **Cheap falsifier (BASE graded-region check alone)** | **< 1 GPU-hr** | **< 1 GPU-hr** | ~2–4 GPU-hr |
Disk: historical snapshots for a full sweep ≪ 50 GB. **Design falsifies cheaply before scaling:** the entire go/no-go (does a graded band exist on a frozen pretrained model?) is a **sub-GPU-hour inference sweep** on GDN-1.3B or Mamba-2-1.3B — no training, no big model.

## Phase 9 — decision matrix & recommendation
Compact matrix (full machine-readable versions: `RNN-06_candidate_matrix.{json,csv}`). Columns: SciRel = scientific relevance; RealState / StateObs / HistCap = real recurrent state / externally accessible / historical-state capturable; 3090 = feasibility; Mat = maturity; Quant; BASEfg = BASE-forgetting-test feasibility; CausInt = causal interpretability; Cx = implementation complexity; GPU$ = expected cost; Grade = evidence.

| Candidate | SciRel | RealState | StateObs | HistCap | 3090 | Mat | Quant | BASEfg | CausInt | Cx | GPU$ | Key blockers | Grade |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **GDN-1.3B (fla / linear-moe-hub)** ⭐PRIMARY | **High** (Qwen-GDN transplant target) | Yes matrix+conv | **Yes (fla Cache)** | Yes | **Fits easily** | High | bf16/bnb | Med-High | High | Low | Low | may be robust (EXT2 risk); Triton/WSL friction | **A** |
| **Mamba-2 1.3B/2.7B (state-spaces)** ⭐FALLBACK | High (best graded lit; SSM≠GDN) | Yes matrix+conv | **Yes (`mamba_ssm`/HF)** | Yes | **Fits easily** | High | bf16/4-bit | **High (Based/Stuffed-Mamba)** | High (ROME-Mamba) | Low | Low | not delta-rule GDN; BIT_EXACT [verify] | **A** |
| **RecurrentGemma-2B (Griffin)** | High (real recurrence + built-in attn control) | Yes RG-LRU | Yes (HF cache) | Yes | Fits easily | High | bf16 | Med | Med | Low | Low | Gemma gated license; local-attn confound to isolate | **A** |
| **DeltaNet-1.3B (fla)** | High (ungated mechanism-proper arm) | Yes matrix+conv | Yes (fla) | Yes | Fits easily | High | bf16 | Med | High | Low | Low | weak research-scale LM | **A** |
| **GLA-1.3B (fla)** LA control | Control (campaign continuity) | Yes (LA state) | Yes (fla) | Yes | Fits easily | High | bf16 | Low-Med | High | Low | Low | control-only | **A** |
| **RWKV-7 0.1–2.9B (fla/BlinkDL)** | Med-High (mature state-first; related mech) | Yes diag+rank1 | Yes (fla) | Yes | Fits (all) | High | GGUF+ | Med | Med | Low | Low | not a GDN drop-in | **A** |
| **Nemotron-Nano-9B-v2** | Med-High (strong long-ctx hybrid) | Yes Mamba-2+attn | Yes (HF) | Yes | **Tight ~22 GiB** | High | 4-bit | Med-High | Med | Med | Med | tight VRAM; hybrid confound | **A/B** |
| **Falcon-Mamba-7B / Codestral-Mamba-7B** | Med (single-arch attn-free 7B) | Yes (M1 vec / M2 matrix) | Yes | Yes | Fits (bf16/4-bit) | High | GGUF/4-bit | Med | Med | Low | Low-Med | Falcon-Mamba d_state=16 less analogous; license | **A** |
| **GatedDeltaNet-2 (NVlabs)** | High (erase/write control) | Yes matrix | Yes (in fla) | Yes | Fits | New | bf16 | Med | High | Med | Low | **NC license**; downloadable weights [verify]; post-cutoff | **B** |
| **Sparse Delta Memory (Meta)** | High (sparse slots resist overwrite) | Yes N×d table | Yes (code) | Yes | 1.4B ceiling | New | — | Med | High | **High (train-from-scratch)** | **High** | **no weights**, NC, ~1T-tok train; post-cutoff | **A/B** |
| **DART / HOLA** | **Highest (their thesis = our question)** | Yes | [verify] | Yes (by design) | ≤~800M fits | Paper | — | ? | High | High | High | **no code/weights** found; post-cutoff | **B** |
| **StateX (THUNLP)** | Med (state-expansion *tool*) | operates on GLA/M2-1.3B | Yes | Yes | Fits | Released | — | tool | Med | Med | Low-Med | a method, not a backbone; post-cutoff | **A-/B** |
| **Qwen3.6-35B-A3B** | **Highest (deployment target)** | Yes GDN+attn | Yes **only via fla path (4-bit)** | Yes | **4-bit+offload only** | High | AWQ/GGUF/FP8 | Med | Med | High | Med-High | offload-bound; state introspection needs 4-bit fla load | **A/B** |
| Qwen3-Next-80B / Qwen3.5-397B / Jamba-52B | Ref only | Yes | — | — | **Too big** | High | — | — | — | — | — | do not fit for state work | A |

**Recommendation:**
- **PRIMARY: `Gated-DeltaNet-1.3B` (fla / `linear-moe-hub`).** Maximizes *transplant relevance* (real pretrained GDN = the Qwen deployment mechanism), *first-class state introspection* (the same fla `Cache` the RNN-05B ports used), Apache-2.0, and *continuity* with the campaign — while fitting a 3090 with headroom. **Explicit risk:** a 100B-token pretrained GDN may, like EXT2's mixture-trained backbone, be **robust within its trained load** → 06B must push #KV-pairs/interference **past training-typical load** to find the graded band, and the BASE gate may still BLOCK. That risk is exactly why the go/no-go is designed as a sub-GPU-hour falsifier.
- **FALLBACK: `Mamba-2` (state-spaces, 130M–2.7B).** Cleanest observable matrix state, the **strongest published graded-forgetting evidence** (Based/Zoology recall-vs-state; Stuffed-Mamba length-vs-capacity), a **built-in size-scaling axis** (five sizes) as the state-capacity knob, and ROME-for-Mamba probing precedent. Use if the GDN checkpoint is flat-high, or run **in parallel** as the graded-regime anchor. Add **GLA-1.3B** as the linear-attention control and **RecurrentGemma-2B** as the "real recurrence + built-in attention control" cross-check.
- **PARK (watch, do not build):** DART, HOLA, FG²-GDN, "Memory Caching" paper, SDM — highly on-thesis but **no runnable weights** (or NC + train-from-scratch); revisit if checkpoints appear. GDN-2 optional advanced arm pending weight/license check.
- **DEFER: Qwen3.6-35B-A3B** as an eventual *validation* target (only after a positive small-model result unlocks `QWEN_GDN_TRANSPLANT_GATE`); do not start on it.
- **Evidence still missing before implementation:** (1) confirm the **fla `recurrent_state` is exposed and BIT_EXACT-restorable** for the specific GDN checkpoint on this box (06A smoke); (2) confirm a **graded MQAR band exists** on the frozen pretrained GDN (and/or Mamba-2) — the sub-GPU-hour falsifier; (3) exact `config.json` state dims for SESOI/storage; (4) weight/license status of GDN-2/SDM/DART/HOLA before considering them.

## Phase 10 — preregistration skeleton (DRAFT — DO NOT EXECUTE)
- **Hypotheses:** **H_BASE** (a common graded forgetting region exists on the frozen real model over the pressure axis); **H_INFO** (early-state info-presence gap > 0 at matched pressure); **H_REC** (a historical-state reader beats final-state BASE by ≥ SESOI **and** beats a matched-capacity no-history reader).
- **Falsifiers:** flat-high or cliff BASE → BLOCKED; info-gap ≤ 0 → HISTORICAL_INFO_ABSENT; recovery ≤ matched-capacity control → READER_CAPACITY_ARTIFACT; loss explained by length/position/quant controls → not a state phenomenon.
- **Challenge identity:** `challengeGridSha256` over frozen task config + pressure ladder + example-id hashes; subprocess self-check; recorded identically in prereg / machine_config / BASE_QUALIFICATION / results / outcomes.
- **Model/checkpoint identity:** HF id + revision hash + file SHA-256; exact config; quantization state declared.
- **State identity:** full state-inventory manifest (per-layer recurrent matrix + conv + SSM hidden; shapes/bytes); snapshot schedule; snapshot hashes.
- **Pressure axis:** nested monotonic, inference-only, fixed length & positions (EXT2 pattern).
- **Competence criterion:** max BASE ≥ τ_hi at low pressure (the model can actually do the task).
- **Graded-region criterion:** min BASE ≤ τ_lo **AND** ≥ k mid-band doses **AND** mid-band **OVERLAPS across ALL** seeds/prompt-sets (EXT2 §7).
- **All-seeds/prompts policy:** every preregistered seed/prompt-set counts; **no screening**.
- **Stopping rule:** 06A lifecycle → 06B BASE gate → 06C info gate → 06D; compute cap; futility = no band in budget.
- **Failure classes:** BLOCKED_ON_<MODEL> / HISTORICAL_INFO_ABSENT / NOT_DETECTED_IN_QUALIFIED_REGIME / READER_CAPACITY_ARTIFACT / POSITIVE_CANDIDATE.
- **Evidence ordering:** prereg commit → 06A → 06B → 06C → 06D, each **persisted + hash-verified before the next**.
- **Decision Cases:** **A** no graded region → BLOCKED, stop. **B** region but no info-gap or no recovery → PARK. **C** graded + info-gap + recovery > matched control + ablation-supported + LA/attention control negative + path counters prove historical-state use → **POSITIVE_CANDIDATE** → authorizes *design* of a separate Qwen qualification packet only (no automatic Qwen run).
- **Gate:** 06C/06D code LOADS+VERIFIES the persisted BASE (and info) qualification artifact; absent/mismatch/unqualified = STOP (no exploratory recovery).

---

## Sources (primary; [verify] = post-cutoff, confirm before relying)
**Enabler/substrate (grade A):** fla github.com/fla-org/flash-linear-attention · `mamba_ssm` github.com/state-spaces/mamba · HF Mamba2 doc · checkpoints: `fla-hub/delta_net-{1.3B,2.7B}-100B`, `linear-moe-hub/Gated-Deltanet-{340M,1.3B}`, `fla-hub/gla-1.3B-100B`, `fla-hub/rwkv7-*`, `state-spaces/mamba2-{130m…2.7b}`, `google/recurrentgemma-2b`, `tiiuae/falcon-mamba-7b`, `mistralai/Mamba-Codestral-7B-v0.1`, `nvidia/NVIDIA-Nemotron-Nano-9B-v2`.
**GDN family:** GDN arXiv:2412.06464 + NVlabs/GatedDeltaNet · RWKV-7 arXiv:2503.14456 · Qwen3.6-35B-A3B (HF, Apache-2.0) · Qwen3-Next vllm blog 2025-09-11.
**Graded-forgetting literature (grade A):** Based 2402.18668 · Zoology/MQAR 2312.04927 · Stuffed Mamba 2410.07145 · Effective State-Size 2504.19561 · MAD 2403.17844 · ROME-for-Mamba 2404.03646 · IOI-in-Mamba 2407.14008 · Repeat-After-Me 2402.01032 (cliff) · Illusion of State 2404.08819 (cliff).
**On-thesis escape-hatch [verify, post-cutoff]:** DART 2608.02032 · HOLA 2607.02303 · Sparse Delta Memory 2607.07386 + facebookresearch/sparse-delta-memory (NC) · GDN-2 2605.22791 + NVlabs/GatedDeltaNet-2 (NC) · FG²-GDN 2604.19021 · "Memory Caching: RNNs with Growing Memory" 2602.24281 · StateX 2509.22630 + THUNLP/StateX · SMR 2405.17534 · selective-memory autoencoders 2512.15653.

## Exactly one next recommendation (NOT executed)
**Open RNN-06A as a separate implementation packet in a new session whose first and only gated deliverable is the sub-GPU-hour BASE falsifier: on the frozen pretrained `Gated-DeltaNet-1.3B` (fla) — with `Mamba-2-1.3B` as the parallel graded-regime anchor — run the inference-only MQAR #KV-pair sweep and decide `FIXED_BACKBONE_GRADED_REGION = QUALIFIED | BLOCKED` under the Phase-10 gate, before building any lifecycle, probe, or recovery machinery.**

**STOP.** (No implementation, no GPU run, no Qwen/serving changes, nothing pushed.)
