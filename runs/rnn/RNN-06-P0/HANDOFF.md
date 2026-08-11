# RNN-06-P0 — Audit Handoff

**Packet:** RNN-06-P0 — Frozen-Checkpoint BASE Regime Scout (EXPLORATORY)
**Date:** 2026-08-11 · **Author:** Claude (Opus 4.8) via Claude Code · **Nothing pushed.**

## 1. Before / after HEAD, commits, tree state
- **BEFORE (session start) HEAD:** `7f04f48fa2bf266527d54f25ea00444da3255dde` (`research(rnn): RNN-06 reconciliation-1`).
- **Packaging claim resolved:** the previous session's manifest claim `HEAD=7f04f48` was classified `PACKAGING_GIT_HEAD_CLAIM = UNVERIFIED` and then **VERIFIED against live git** — `7f04f48` exists, is current HEAD at session start, and descends from the RNN-06 packet commit `3216229` (`git merge-base --is-ancestor 3216229 7f04f48` → YES).
- **Commit 1 (pre-run protocol, before any outcome-bearing GPU work):** `7d7feed` — `research(rnn): RNN-06-P0 pre-run protocol + scout (EXPLORATORY, no outcomes yet)`. Files: `ops/rnn_06_p0_mqar.py`, `ops/rnn_06_p0_curves.py`, `runs/rnn/RNN-06-P0/{P0_PROTOCOL.md,machine_config.json}`.
- **Commit 2 (results):** `46098e5` — adds identities, results JSON, curves, decision, this handoff, sweep logs, bundle script, append-only protocol execution-deviation note, and the `--impl`/`--autobatch`/OOM-resilience scout changes made during the run. (This handoff's own final commit updates this hash + the ZIP SHA below.)
- **Branch:** `master`, **no upstream, NOT pushed.** `git stash` empty.
- **Historical evidence immutable:** no tracked file under `runs/rnn/RNN-04-memory-caching`, `RNN-05A-fixed-backbone`, `RNN-05B-delta-gdn`, `RNN-05B-EXT`, `RNN-05B-EXT2` is modified (verified: `git status --porcelain <dirs> | grep -v '^??'` → empty). The pre-existing untracked `git_evidence.txt`/`stdout.log`/adapter-dir helpers are unchanged.

## 2. Exact model/checkpoint identities
| Candidate | HF id | resolved revision | class | tokenizer | params | config.json sha256 |
|---|---|---|---|---|---|---|
| **GDN (designed primary)** | `linear-moe-hub/Gated-Deltanet-1.3B` | repo `871079371ada2f8beba1e05e7d6b5cc7c9a1d5f6` | — (did not load) | — | — | — |
| **DeltaNet (runnable primary)** | `fla-hub/delta_net-1.3B-100B` | `b4dcbbafd4fde802717bdec3008d4aba9cb3a1f8` | `DeltaNetForCausalLM` | `LlamaTokenizerFast` v32000 | 1,365,677,056 | `506e45ed5e44aad13566ddd797dc16367bd00b30342697748e62e0fc6f57cff6` |
| **Mamba-2 (anchor)** | `AntonV/mamba2-1.3b-hf` | `703e19a43f397c70315244a3424d79456b54fb34` | `Mamba2ForCausalLM` (transformers-native) | gpt-neox-style | ~1.3 B | `1b7bd1a3f505e01ecaa071cd72509061901ef90347ac8bf0c11be85e3d944c50` |

`AntonV/mamba2-1.3b-hf` is the HF-format conversion of `state-spaces/mamba2-1.3b` (repo sha `c5b59d00ec85`, raw `mamba_ssm` format not loadable by `AutoModelForCausalLM`). Full per-file hashes + public config in `MODEL_IDENTITY_{DELTANET,MAMBA2}.json`. Frozen-checkpoint assertion recorded in each identity: *same frozen checkpoint across ALL pressure conditions; no per-condition model/seed variation; no seed screening.* Weights bf16, **quantization NONE**.

## 3. Dependency / backend versions (pinned)
python 3.12.3 · torch 2.6.0+cu124 · CUDA 12.4 · transformers 4.48.3 · flash-linear-attention 0.5.2 · Triton 3.2.0 (**below fla-recommended 3.3.0 — recorded risk axis**) · numpy 2.5.2 · huggingface_hub 0.36.2 · accelerate 1.14.0 · protobuf 7.35.1 · tokenizers 0.21.4 · safetensors 0.8.0 · **mamba_ssm = None, causal_conv1d = None** (Mamba-2 ran the transformers-native naive path). GPU: RTX 3090 24 GB, driver 591.86, CC 8.6. Substrate: WSL2 Ubuntu-24.04, 43 GB RAM visible. venv `~/rnn06_env` and HF cache `~/.cache/huggingface` are on the Linux FS, **outside the repo, NOT committed**. Full stack in `machine_config.json` + each identity's `backend_env`.

## 4. Commands executed (key)
```
# env (venv on Linux FS, not committed)
python3 -m venv ~/rnn06_env
pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install numpy einops safetensors huggingface_hub transformers==4.48.3 accelerate \
            sentencepiece flash-linear-attention protobuf
# scout — pre-run protocol committed BEFORE these outcome-bearing runs
python ops/rnn_06_p0_mqar.py --model-id fla-hub/delta_net-1.3B-100B --tag DELTANET \
       --outdir runs/rnn/RNN-06-P0 --doses 4 8 16 32 64 128 --n-eval 200 --batch-size 50
python ops/rnn_06_p0_mqar.py --model-id AntonV/mamba2-1.3b-hf --tag MAMBA2 \
       --impl transformers --config-overrides '{"chunk_size":32}' \
       --outdir runs/rnn/RNN-06-P0 --doses 4 8 16 32 64 128 --n-eval 64 \
       --batch-size 8 --autobatch-budget 1536
python ops/rnn_06_p0_curves.py runs/rnn/RNN-06-P0          # -> P0_CURVES.csv
python ops/rnn_06_p0_bundle.py                             # -> ZIP + SHA256SUMS
```
GDN load attempts (`AutoModelForCausalLM.from_pretrained("linear-moe-hub/Gated-Deltanet-1.3B", …)`, and again with `config.intermediate_size=11264`) both failed with `state_dict` size-mismatch — evidence in §7.

## 5. Source excerpts (paths + context)
- **Task construction** — `ops/rnn_06_p0_mqar.py::build_calibration_spec` (nested-monotonic superset; probe index `< p_min` so the probed pair lives in every dose's shared prefix) and `::materialize_dose` (flat token-id prompt `[key,'=',value,'\n']*P + [probe_key,'=']`, gold = probe's value token). Every key/value is a single token → equal-length prompts, exact answer position.
- **Deterministic scoring** — `::eval_dose`: one forward/prompt; `constrained_acc` = argmax over the value vocabulary at the last position == gold; `format_adherence` = unconstrained top-1 ∈ value vocab; `unconstrained_exact_acc`; per-batch `torch.cuda.OutOfMemoryError` caught → `n_oom_skipped` (denominators exposed, never silently scored as memory misses).
- **Determinism** — `::rng_for` uses numpy `PCG64`/`SeedSequence` on integer seeds (process-stable; **not** python `hash()`).
- **Band classifier** — `::classify_band`: `PLAUSIBLE` requires competence at low pressure AND ≥1 interior dose in `(τ_lo,τ_hi)` (guards flat-high and single-cell cliff); else `NOT_FOUND_WITHIN_BUDGET`.
- **Backend selection** — `::load_model`: `import fla` only for `auto/fla` impl (registers `gated_deltanet`/`delta_net`); `impl=="transformers"` uses the concrete `Mamba2ForCausalLM` to dodge fla's OOM `mamba2.torch_forward`. Config overrides (`intermediate_size`, `chunk_size`) recorded in `config_overrides_applied`.

## 6. Results — curves, denominators, parser failures
`chance = 1/256 ≈ 0.003906` (argmax over the 256-token value vocabulary) for all rows. Full machine-readable curves in `P0_CURVES.csv` / `P0_RESULTS_*.json`.

**DeltaNet** (`fla-hub/delta_net-1.3B-100B`, n=200/dose, value_kind = alphanumeric-fallback, peak 8.78 GB, 1200 forwards, 82 s):
| P | seq_len | constrained_acc | format_adherence | unconstrained_exact |
|--:|--:|--:|--:|--:|
| 4 | 18 | 0.730 | 0.725 | 0.580 |
| 8 | 34 | 0.700 | 0.690 | 0.540 |
| 16 | 66 | 0.760 | 0.550 | 0.485 |
| 32 | 130 | 0.755 | 0.465 | 0.430 |
| 64 | 258 | 0.680 | 0.365 | 0.355 |
| 128 | 514 | 0.530 | 0.220 | 0.200 |
→ TASK_COMPETENT=NO (0.730<0.75); `P0_GRADED_BAND = NOT_FOUND_WITHIN_BUDGET`.

**Mamba-2** (`AntonV/mamba2-1.3b-hf`, n=64/dose, value_kind = numeric, chunk_size 256→32, peak 8.53 GB, 384 forwards, 90 s):
| P | seq_len | batch_used | constrained_acc | format_adherence | unconstrained_exact |
|--:|--:|--:|--:|--:|--:|
| 4 | 18 | 8 | **0.953** | 0.641 | 0.609 |
| 8 | 34 | 8 | 0.922 | 0.578 | 0.562 |
| 16 | 66 | 8 | 0.781 | 0.453 | 0.422 |
| 32 | 130 | 8 | 0.719 | 0.297 | 0.281 |
| 64 | 258 | 5 | 0.516 | 0.125 | 0.125 |
| 128 | 514 | 2 | 0.234 | 0.078 | 0.062 |
→ TASK_COMPETENT=YES (0.953); monotone; 2 interior mid-band rungs (0.719, 0.516); `P0_GRADED_BAND = PLAUSIBLE`.

**Parser/format failures are not hidden:** `format_adherence` is the fraction whose *unconstrained* top-1 was inside the value vocabulary; it falls with pressure for both models. Constrained scoring still recovers a clean signal, so low format adherence at high pressure is reported as its own channel, not counted as a memory miss. No prompts were dropped except any OOM batch (Mamba: `n_oom_skipped=0` — the chunk/autobatch settings held; all n reported are actual denominators).

## 7. Negative evidence (preserved)
- **GDN MODEL_NOT_RUNNABLE** — `RuntimeError: Error(s) in loading state_dict for GatedDeltaNetForCausalLM: size mismatch for model.layers.*.mlp.gate_proj.weight: checkpoint [11264,2048] vs model [5632,2048]`. With `intermediate_size=11264` forced, the error moves to `mlp.down_proj: checkpoint [2048,5632] vs model [2048,11264]` → the checkpoint's `gate_proj` is a **fused gate+up** (2×5632) while fla 0.5.2 uses separate projections. Deferred to RNN-06A (no fused→split remap attempted — unverifiable concat order).
- **Mamba-2 OOM chain (backend, resolved):** fla's `mamba2.torch_forward` tried to allocate **100 GiB** at batch 50 (`G_intermediate = C[...] * B[...]`, O(batch·L²)); transformers-native naive path used **22.8 GB at batch 2, L=514**; the `(b,c,chunk,chunk,h,n)` diagonal tensor OOM'd at chunk 256 → resolved with the math-equivalent `chunk_size=32` tiling + seq-length-adaptive batch. All OOMs preserved in `stdout_sweep*.log` / task logs.
- **Triton 3.2.0 < fla-recommended 3.3.0** — warned on every fla load; recorded, not silently ignored.
- **DeltaNet flat/marginal** and **format-adherence decline** are reported, not smoothed.

## 8. P0 statuses
```
linear-moe-hub/Gated-Deltanet-1.3B : MODEL_RUNNABLE=NO   P0_GRADED_BAND=MODEL_NOT_RUNNABLE
fla-hub/delta_net-1.3B-100B (DN)   : MODEL_RUNNABLE=YES  TASK_COMPETENT=NO   P0_GRADED_BAND=NOT_FOUND_WITHIN_BUDGET
AntonV/mamba2-1.3b-hf (Mamba-2)    : MODEL_RUNNABLE=YES  TASK_COMPETENT=YES  P0_GRADED_BAND=PLAUSIBLE
```

## 9. Identity / determinism hashes
- `calibrationSetSha256` (abstract example spec, tokenizer-independent): DeltaNet (n=200) `25ef820e5a8085228a238e44526e72317ee85c2960f45d9557071d535399e46f`; Mamba (n=64, nested prefix) `779fb37af14eea0e36b25b0407f5aa32fc23dc84b8add51a2df4ee0ad88c45f3`. Saved `calibration_examples.json` = the 200-example superset.
- `materialized_sweep_sha256` (per-model token-id sweep): DeltaNet `59944f0f…`, Mamba `b55c83d8…`.
- The **confirmatory** `qualificationSetSha256` (RNN-06B) is **independent + disjoint** and was **not** created or consumed here.

## 10. Compute / runtime evidence
DeltaNet: load 2.4 s, 6-dose sweep 82 s, peak 8.78 GB, 1200 forwards. Mamba-2: load 1.7 s, sweep 90 s, peak 8.53 GB, 384 forwards. Both **well under the sub-GPU-hour-per-candidate** target. Multiple exploratory backend probes (GDN load attempts, Mamba OOM/chunk/batch calibration) were additional but bounded. Nothing beyond inference-only forward passes.

## 11. Staged vs working-tree / push state
Results commit stages: `runs/rnn/RNN-06-P0/{MODEL_IDENTITY_*,P0_RESULTS_*,P0_CURVES.csv,P0_DECISION.md,HANDOFF.md,calibration_examples.json,git_evidence.txt,stdout_sweep*.log,P0_PROTOCOL.md(append-only edit)}`, `ops/rnn_06_p0_{mqar,curves,bundle}.py`. The ZIP `RNN-06-P0-audit-bundle.zip` is a deliverable (may be left untracked or committed per convention). **Confirmation: nothing pushed** (no upstream on `master`; no `git push` executed).

## 12. Exactly one next recommendation (NOT executed)
**Proceed to RNN-06A (lifecycle qualification) on `AntonV/mamba2-1.3b-hf` (Mamba-2-1.3B)** over the P0-observed graded window **P ∈ [8,128] (mid-band ≈ 32–64)**; in parallel resolve the GDN load (pin authoring fla revision) and install Mamba-2 fast kernels. **Not executed.** P0 mints no `FIXED_BACKBONE_GRADED_REGION`; `QWEN_GDN_TRANSPLANT_GATE = DEFER`.

**STOP** after P0 (no 06A/06B, no historical-state machinery, no Qwen, no push).
