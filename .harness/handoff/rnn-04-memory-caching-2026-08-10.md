# HANDOFF — RNN-04: Memory Caching Mechanism Reproduction (2026-08-10)

Mechanism-level reproduction of **Memory Caching** (Behrouz et al., arXiv 2602.24281) on a small,
transparent pure-PyTorch linear-memory substrate + synthetic MQAR. Answers the packet's primary question —
*can caching multiple historical recurrent states improve associative retrieval vs a single state, under
measured memory/compute budgets?* — and gates a future Qwen GDN transplant. No Qwen/llama.cpp/deploy/serving
changes; TPTT PARKED; no push.

## 1. Git
- Branch **master**. **Start HEAD `4f5b139`** → **End HEAD `40ced93`**. Three small commits (§25):
  `743b799` research(spec), `8110b97` test(benchmark), `40ced93` experiment.
- Untracked (out of Git per §25): `.harness/`, old RNN-08/08b adapter dirs. No model weights, venv, upstream
  clone, or cache committed. **Not pushed.**

## 2. Primary question — ANSWERED
**Yes, with an important efficiency caveat.** Caching multiple historical states (GRM) improves MQAR recall
over a single equal-*sized* state by **+0.1145** (main) and **+0.1267** (independent replication), and the
gain grows as the fixed state saturates. BUT spending the same *bytes* on one bigger single state reaches
0.997 → **MC_OUTCOME = MC_ONLY_HELPS_WITH_MORE_MEMORY** (§21). MC's value is adding memory to a *fixed*
recurrent model, not beating a freely-enlargeable state at toy scale.

## 3. Official / community code disposition (§4)
**OFFICIAL_CODE_NOT_FOUND** (re-checked PDF/arXiv/alphaXiv/HF/author site). Community torch effort mentioned
on social media, unverified → COMMUNITY_IMPLEMENTATION, did NOT define the algorithm. Full record:
`runs/rnn/RNN-04-memory-caching/official_code_disposition.json`. Paper mechanism = PUBLISHED; our impl =
REPRODUCED/ADAPTED (toy substrate, **no paper-parity claim**).

## 4. Paper equations implemented + equation→source→code mapping (§3)
Authoritative spec: `RNN_MEMORY_CACHING_SPEC.md` (verbatim Eq. 4–17 + binding table). Implemented:
| paper | quantity | code |
|---|---|---|
| Eq. 2 | linear-attention base memory `S=Σ kᵢvᵢᵀ`, read `Sᵀq` | `DeltaMemory._seg_linear` |
| Eq. 4 | fixed segmentation; cache final segment state | `forward` segment loop → `cached_states` |
| Eq. 7 | Residual read (collapses for linear) | `agg_residual` (arm B0) |
| Eq. 9–10 | GRM: `γ=softmax_i⟨u_t,meanpool(Sⁱ)⟩`, `u=xWu` | `agg_grm` + gate logits (arm B) |
| Eq. 14–15 | Memory Soup (param-average; ≡GRM for linear) | `soup_states` (unit-tested, deferred as arm) |
| Eq. 16–17 | SSC Top-k router (+random control) | `ssc_gates` (arms SSC / D) |
| Sec. 4.3 | POST_TRAINING_MC param-free moving average | `agg_moving_average` (arm Post) |
Adapted decisions (paper silent): trained arms backprop end-to-end through cached states (detach starved
recall → chance); `Wu ~ N(0,0.02²)`; substrate is plain linear attention (a simplification of Qwen's gated
delta rule).

## 5. Dependency identity (§7)
Reused isolated venv `/home/augus/tptt-venv`: **torch 2.6.0+cu124, numpy 2.5.2, python 3.12.3**, CUDA on
RTX 3090. No CUDA/Triton kernels; no FLA dependency needed (pure-PyTorch reference, §7 fallback). Does not
touch sglang/evalplus/serving.

## 6. Benchmark + reproducibility (§5/§9/§10) — QUALIFIED
`ops/rnn_mc_bench.py` (spread-write MQAR). **Seeds = hashlib.blake2b(canonical_spec+idx) → int** (fixes the
RNN-08b `hash()` defect). `RNN-04-benchmark-selftest.json`:
- **SYNTHETIC_DATASET_REPRODUCIBILITY = QUALIFIED** — two fresh subprocesses produce identical example IDs
  (`proc_a_eq_proc_b` & `inproc_eq_proc` both true).
- **BENCHMARK_SELFTEST = PASS** — oracle 1.0; random 0.0117 (~chance 0.0156); exact seq length; kv survives;
  scorer exact/perturbed 1.0/0.0; no answer leak; no lexical shortcut (all 64 keys → >1 value).
- Independent axes: seq_len, num_pairs, num_queries, distractor_density; per-pair write→query distance
  recorded. Every sample has a stable canonical ID.

## 7. Model / training identity (§8)
`MQARModel`: Embedding → `DeltaMemory` (single head, d_model 128, d_k=d_v 24, depthwise causal conv k=3) →
LayerNorm(x+y) → head. AdamW lr 1e-3, cosine, wd 0.01, batch 128, float32. MQAR num_keys 128 / num_vals 64.
Tiny; **whole experiment ~13 GPU-min, well under the 1-hr budget** (no large-LM training).

## 8. Aggregation + checkpoint qualification (§13/§19)
`substrate_unittest.json`: **AGGREGATION_UNIT_TEST = PASS** (residual/grm/soup/moving-avg all match hand
computation to float noise; soup≡grm for linear; SSC selects exactly k+online; residual-collapse-for-linear
= 0.0). **CHECKPOINT_RESTORE = BIT_EXACT** (serialize→reload identical reads; additive continuation matches
full-sequence run, NUMERICALLY_EQUIVALENT). State identity: `[d_k,d_v]` fp32 = **2304 bytes/request**.

## 9. All experimental arms (§11–§12, §17–§18) — MC_TASK D=72, seg=32, N=8
| arm | agg | acc | note |
|---|---|---:|---|
| A BASE_RNN | single | 0.6960 | matched-size fixed state |
| B0 Residual (train-free on A) | residual | **0.6960** | == base → **linear collapse confirmed** (Eq. 7) |
| B GRM | grm | **0.8105** | **+0.1145 vs A**; gate signal TRUE |
| C equal-memory control | single, d=68 | **0.9973** | ~same bytes (18496) as B cache → dominates |
| SSC learned k=2 | ssc | 0.6562 | Top-k=2 discards most cache |
| D SSC random k=2 | ssc-rand | 0.2876 | learned ≫ random (**+0.3686** → policy learns) |
| Post moving-avg (frozen A) | moving_average | 0.3611 | **< base** → honest negative (§ below) |
Replication (fresh seeds, §24): base 0.6919 / GRM 0.8186 / **Δ +0.1267** → benefit replicates.

## 10. Memory-budget curves + storage/compute separation (§14–§16) — `pareto.csv`
| N | acc | cache_bytes | infer_ms | agg_read_ms | update_scan_ms |
|--:|--:|--:|--:|--:|--:|
| 1 | 0.6838 | 0 | 1.13 | ~0 | ~1.1 |
| 2 | 0.7036 | 4608 | 1.29 | 0.15 | ~1.1 |
| 4 | 0.7439 | 9216 | 1.55 | 0.40 | ~1.1 |
| 8 | 0.7859 | 18432 | 3.02 | 1.80 | ~1.1 |
| 16 | 0.8684 | 36864 | 6.32 | 5.08 | ~1.1 |
Accuracy rises monotonically with cached states; **cost is in aggregation/read (0→5 ms), state-update is
flat** (§16 — footprint ≠ compute, RNN-08b lesson upheld). acc-vs-seq_len (D=72): GRM beats base at all L
(256/384/512), and is more robust as L grows (base 0.688→0.648→0.600 vs GRM 0.792→0.804→0.739).
acc-vs-distance curves per arm in `rnn04_results.json` (`dist_curve`).

## 11. Negative / honest evidence (§21)
- **Equal-byte single state (0.997) ≫ GRM (0.81)** → at toy scale the gain is bytes, not structure →
  MC_ONLY_HELPS_WITH_MORE_MEMORY.
- **POST_TRAINING_MC failed** (0.361 < base 0.696): naive param-free averaging of segment states on a model
  trained as ONE continuous state degrades recall. (Paper's post-training claim = length-extrapolation, a
  different setup; not contradicted, but the naive transplant analog is negative here.)
- **SSC k=2 (0.656) < GRM/base**: selecting only 2 of 7 cached states loses information; its *value* is the
  learned-vs-random gap, not absolute accuracy.
- Effect sizes are small and toy-scale; substrate is linear (not gated-delta) memory.

## 12. QWEN_GDN_TRANSPLANT_GATE (§22) = **PASS (caveated)**
All 6 conditions met: benchmark qualified; aggregation qualified; **basic MC shows repeatable benefit**
(replicated +0.127); state-memory cost known (2304 B/req; cache N·2304); compute cost known (agg 0→5 ms);
state maps to Qwen GDN shape `[d_k,d_v]`. **Caveat (load-bearing):** PASS means *mechanism reproduced &
plausibly mappable*, NOT *will improve Qwen*. A transplant must use the **trained-with-MC** path (GRM), NOT
the naive post-training moving-average (which failed here), and its value is specifically that Qwen's GDN
state cannot be freely enlarged without retraining (so the equal-memory control that beat MC here is not
available there). **Do NOT touch Qwen GDN in this packet** (per §22) — this is a recommendation flag only.

## 13. Exact reproduction commands
```
cd /mnt/c/projects/local-model-lifecycle           # WSL Ubuntu-24.04
V=/home/augus/tptt-venv/bin/python
PYTHONHASHSEED=0 $V ops/rnn_mc_bench.py --selftest --out runs/rnn/RNN-04-memory-caching/RNN-04-benchmark-selftest.json
$V ops/rnn_mc_substrate.py --unittest --out runs/rnn/RNN-04-memory-caching/substrate_unittest.json
PYTHONPATH=ops PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True $V ops/rnn_mc_experiment.py --outdir runs/rnn/RNN-04-memory-caching
$V ops/rnn_mc_analyze.py --dir runs/rnn/RNN-04-memory-caching
```
Deterministic (fixed seeds); arms reproduce bit-identically across runs. Raw outputs: `rnn04_results.json`,
`rnn04_analysis.json`, `pareto.csv`, `run.log`.

## 14. Exactly one recommended next packet
**RNN-05 — Memory Caching on a FIXED pretrained recurrent backbone with a small TRAINED aggregation head,
plus deep (2-layer MLP) memory.** Rationale: the two open results from RNN-04 both point here — (a) the
equal-memory control means MC's real use case is a state that *can't* be freely enlarged (a frozen Qwen-like
backbone), and (b) for *linear* memory Residual/Soup collapse and only GRM's gate helps, whereas the paper's
larger gains are on *deep* memory (L_M≥2) where Soup/Residual do not collapse. RNN-05 should: freeze a
small pretrained recurrent LM, train ONLY a GRM/SSC aggregation head over cached segment states (the true
transplant analog), and add a 2-layer-MLP memory variant to test the non-collapsing regime. Keep it
small-GPU, synthetic + one small real recall task, no deploy-model training. This is the correct precursor
to any Qwen GDN transplant (still gated, still not automatic).

## 15. Guardrails (unchanged)
No Qwen GDN transplant in this packet · no llama.cpp/serving/deploy change · TPTT PARKED · no model/family
scaling · no kernel work · no tare.tools · no push. Serving CLOSED.
