# GDN M4 — session checkpoint & resume (2026-08-03; CLOSED OUT 2026-08-04)

Durable state so nothing is lost across the **GPU-lost reboot** (see incident at bottom). Read this
first on resume, then `GDN_KERNEL.md` / `GDN_NEXT_LEVERS.md`. All file edits below are **on disk**.

## ⟢ FINAL STATUS 2026-08-04 — M4b COMPLETE + fork CONSOLIDATED. Nothing owed; GDN thread is closed.
Post-reboot the whole M4b plan ran clean (GPU stable, clock-lock working). All 4 experiments DONE, all
committed. **The chunked GDN kernel is a CLOSED lever** — correct + quality-neutral but wins nowhere but
the starved H=32 deploy shape (~3% @8k), so it ships OFF-by-default opt-in. Then per user request the fork
was CONSOLIDATED (details in each section below; also in the project memory + FORK.md):
- **Occupancy sweep:** more heads → chunked STRICTLY WORSE at B=1 (§ "Occupancy sweep" below).
- **Dense-27B H=48 A/B:** −2–4% E2E (§ "Dense-27B" below).
- **Concurrent-serving A/B:** real null R(8)=1.008, engagement proven (§ "Concurrent-serving" below).
- **Quality bless:** pass@1 34/40 base, 33/40 plus — identical off/on (§ task-2 below).
- **Fork consolidation (user opt-B):** every campaign line now a local branch in `llama.cpp-master`
  (`lifecycle` deploy `068764d92` + `turbo-stack` `cca05a3ac` [rescued from a fragile detached HEAD] +
  `prefetch-skip-pinned` + `fable5-prefetch-experts`); prefetch-skip-pinned FOLDED onto lifecycle
  (CLI flags preserve the env path; inert on default). **Re-blessed 3/3, test-backend-ops 13349/13349,
  GDN 46/46.** Nothing deleted. Commits: project `41a9fa7`+`754f762`, fork `c8761b40c`→`068764d92`.

**COMMITS THIS SESSION are all in git; tree is clean.** Runner scripts were moved out of scratchpad into
the repo (commit `83e46e1`): `gdn_conc_arm.sh` at root; `enum_builds.sh` / `check_recoverable.sh` /
`consolidate_audit.sh` / `preserve_branches.sh` under `ops/fork-consolidation/` (+ README).

## TL;DR of where we WERE (2026-08-03, pre-close — kept for provenance)
- **Lever B (bf16 W-readout): DONE — negative, REVERTED to TF32.** bf16 = flat vs TF32 (deploy 2048:
  2868.8 vs 2878.7 µs; worse at 4096/8192) → kernel is **latency/occupancy-bound, not throughput-bound**.
  Precision levers exhausted. Correct 46/46 @ 2e-7 confirmed after revert (before the GPU died).
- **M4 (llama-bench real prefill A/B): DONE.** The isolated ~5% kernel win does NOT surface E2E at B=1:
  ncmoe=8 → +1.0%/+0.5% (noise); ncmoe=0 → −0.4%/−0.5% (noise). Full numbers in `GDN_KERNEL.md` M4.
- **The 3 user-requested tasks + occupancy sweep — ALL DONE 2026-08-04** (see FINAL STATUS above).

## Uncommitted edits currently on disk (do NOT lose; nothing committed — commit only when asked)
1. `ggml/src/ggml-cuda/gated_delta_net.cu` — **reverted to TF32** (shipped state; bf16 fully backed out,
   comments + `GGML_CUDA_GDN_TC` doc restored). This is the correct/clean kernel.
2. `tests/test-backend-ops.cpp` —
   - tolerance restored to `2e-7` (bf16's 1e-4 backed out), line ~4111.
   - **ADDED (keep): head-count occupancy perf cases** after the PP-8192 row (~line 10038): H=48 @
     {512,1024,2048,4096} and H=64 @ {512,1024,2048}, all d=128 seq=1. H=48 == the on-disk dense-27B
     GDN shape. Binary rebuilt WITH these (verified: `grep -c head_count=48` on perf output = present).
3. `quality_bench.py` (repo root, runs on Windows) — **ADDED (keep): GDN env passthrough.** `import os` +
   a loop copying `GGML_CUDA_GDN_CHUNKED` / `GGML_CUDA_GDN_CHUNKED_MIN_TOKENS` / `GGML_CUDA_GDN_TC` from
   `os.environ` into the server `env` dict (the LlamaCppAdapter uses `env K=V` with ONLY its dict, so
   shell exports don't reach the WSL server otherwise). This is what lets the harness bless the kernel.
4. `gdn_conc_bench.py` (repo root) — **NEW: the concurrent-serving load driver** (stdlib, ready to run).

## Shape investigation (answers "are the results model-dependent?") — the free contrasts
The GDN perf verdict is **shape-bound (H, d, tokens), not weight-dependent** — verified by the real
model matching the synthetic sweep. On-disk model GDN shapes (from gguf metadata, `ssm.inner_size /
ssm.state_size` = GDN head count):
- **MoE-35B-A3B** (arch `qwen35moe`): state_size=128, inner_size=4096 → **H=32 GDN heads**, d=128. 40 layers.
- **Dense-27B** (arch `qwen35`): state_size=128, inner_size=6144 → **H=48 GDN heads**, d=128. 65 layers.
  → **We ALREADY have a "more heads" model on-disk (H=48).** B=1 grid = 48·4 = 192 blocks vs MoE's 128.
- Build supports `LLM_ARCH_QWEN3NEXT` — so a real Qwen3-Next GGUF WOULD load. BUT the marquee candidate
  (Qwen3-Next-80B-A3B) uses GDN with **num_v_heads=32, head_dim=128 = the SAME shape as our MoE** → a
  45GB download adds no occupancy contrast. **Recommendation: skip the download**; use MoE(H=32) vs
  dense-27B(H=48) + the synthetic H=48/64 sweep. (User was offered this; do the download only if they
  still want a different-family datapoint despite same shape.)

## Occupancy sweep (free) — the "does shape change the story" experiment — **DONE 2026-08-03 (post-reboot)**
Full sweep re-ran clean (2× reproducible, GPU healthy after). Result file:
`runs/gdn/occupancy_sweep_H32-48-64_20260803.txt`. **VERDICT: more heads makes chunked STRICTLY WORSE at
B=1 — the occupancy hypothesis is refuted.** tf32/seq ratio (lower = chunked faster) by head count:
| H (heads) | grid = H×n_seqs | 1024 tok | 2048 (deploy ub) | 4096 | 8192 |
|---|---|---|---|---|---|
| **32** (deploy MoE) | 32 | 1.03 | **1.00 parity** | 0.98 | 0.97 |
| **48** (dense-27B) | 48 | 1.17 | 1.14 | 1.13 | — |
| **64** (synthetic) | 64 | 1.18 | 1.16 | — | — |
- fp32-chunked is 1.2–1.45× everywhere → **TF32 remains the only viable chunked variant** (tf32-vs-fp32
  ~0.81–0.84, the LDS-relief win holds).
- **n_seqs co-batching does NOT rescue it either:** H=32/512 at n_seqs {1,2,4,8} = all parity
  (tf32/seq 1.00–1.01); H=32/2048 n_seqs=2 = 1.15 (slower). → previews a **null for the concurrent-serving
  A/B** (more sequences ≠ a chunked win).
- **Mechanism:** the sequential kernel is occupancy-starved ONLY at H=32; at H≥48 the extra heads already
  give the sequential scan enough resident blocks, so chunked's parallelism edge vanishes while its
  extra-GEMM overhead stays → net loss. The marginal H=32 win (parity@2048, ~3% @8192) was the *best case*,
  not a floor. **This closes the chunked kernel as a general B=1 prefill lever — it wins nowhere but the
  one starved shape, and even there only ~3% at 8k.**

## Concurrent-serving A/B (M4b, Arm B) — **DONE 2026-08-03 (post-reboot). VERDICT: NO-GO / real null.**
ncmoe=0, `-b 8192 -ub 2048 -ctxcp 0`, prompts=1024 tok, REPS=7 (5 measured), driver `gdn_conc_bench.py`,
runner `scratchpad/gdn_conc_arm.sh`, JSON in `runs/gdn/armB__*__ncmoe0__k*.json`. Aggregate prefill t/s:
| k | OFF (seq) | ON (chunked, MINTOK=128) | R=ON/OFF |
|---|---|---|---|
| 1 | 1486.9 | 1687.5 | 1.135 (noise; OFF IQR 1430–1902) |
| 2 | 2488.3 | 2542.2 | 1.022 |
| 4 | 2902.4 | 2771.5 | 0.955 |
| 8 | 2997.8 | 3021.7 | **1.008** |
R(k) is **non-monotonic, flat around 1.0** → NO-GO (threshold was R(8)≥1.03 CI-excluding-1; got 1.008).
**Engagement PROVEN, so this is a real null not a setup error:** detector arm (chunked set but
`MINTOK=1000000` → provably never engages) at k=8 = **2957.0**, statistically identical to ON 3021.7 and
OFF 2997.8 (all IQRs overlap 2704–3069). ON≈detect≈OFF ⇒ the chunked kernel engaging vs not makes **zero**
measurable difference; gate-firing was source-verified (n_seqs=8, tokens/seq=256≥128). **Why null:** 8-way
co-batching saturates the GPU on the dominant attention/FFN GEMMs (OFF alone scales 1487→2998 t/s, ~2×,
hitting a ~3000 t/s ceiling any arm); GDN is a small slice and is only ~parity isolated at H=32 anyway.
**ncmoe=8 SKIPPED as subsumed:** adding CPU-MoE offload only shrinks the GDN slice further (more
transfer-bound) → strictly more null than this cleaner all-on-GPU test. **The B=1 verdict now extends to
the concurrent regime: the chunked GDN kernel is not a serving lever at any concurrency on this HW.**

## THE THREE TASKS TO RESUME (user asked for all three) — in priority order

### 1. Concurrent-serving A/B (M4b) — Fable methodology (full, since it only lived in the transcript)
**KEY STRUCTURAL FINDING (from reading the server source):** at deploy config there is a hard ceiling
**`n_seqs ≤ n_ubatch / MIN_TOKENS`.** So with `-ub 2048` + default gate `MIN_TOKENS=1024`, the chunked
kernel **can never see n_seqs > 2**, and with ≥2048-token prompts + default `-b 2048` there is **no
cross-seq co-batching at all**. i.e. the 8-slot concurrency does NOT feed the kernel unless we deviate
config. Mechanics (all source-verified):
- Server (`tools/server/server-context.cpp` ~3045-3060): cont-batching fills ONE `llama_batch`,
  greedily appending each slot's prompt tokens until `batch.size() >= n_batch`. k slots co-batch only
  if `n_batch >= k × prompt_len_remaining`.
- Ubatch split (`llama-batch.cpp` ~510-679, hybrid uses `split_equal`): `tokens_per_seq =
  floor(n_ubatch / n_seqs)`.
- Gate (`gated_delta_net.cu` ~768): tests n_tokens = **per-sequence** tokens in the ubatch.
- **Context-checkpoint trap:** default `n_ctx_checkpoints=32` breaks prompt filling **4 tokens before
  prompt end** → a 1024-token prompt yields only 1020 < 1024 gate → chunked silently OFF. Use `-ctxcp 0`.
- `kv_unified=false` → `split_equal(sequential)` needs **consecutive slot ids** → fire all k requests
  simultaneously at an idle server, equal prompt lengths.

**Three arms** (server launch lines below; driver = `gdn_conc_bench.py`; ncmoe=8 AND ncmoe=0 each):
- **Arm A (deploy-faithful):** `-b 2048 -ub 2048`, default gate, checkpoints on. Expected FLAT by
  construction (n_seqs≤2, and none for long prompts) — run to confirm the null; it is itself a finding.
- **Arm B (occupancy probe — the real experiment):** `-b 8192 -ub 2048 -ctxcp 0`, prompts = 1024 tok,
  ON arm `GGML_CUDA_GDN_CHUNKED_MIN_TOKENS=128`. At concurrency k: GDN op gets n_seqs=k,
  tokens_per_seq=2048/k → constant total work, grid y-dim 1→8. Sweep **k ∈ {1,2,4,8}**.
- **Arm C (gate-respecting max occupancy):** `-b 8192 -ub 8192 -ctxcp 0`, prompts 1024, DEFAULT gate,
  k=8 → single ubatch of 8×1024, chunked engages under shipped gate with n_seqs=8. Preflight VRAM at
  `-ub 8192`; if OOM at ncmoe=0 fall back to `-ub 4096` + `MIN_TOKENS=512` (8×512).

**Engagement detector (no profiler):** an "ON + `GGML_CUDA_GDN_CHUNKED_MIN_TOKENS=1000000`" cell (chunked
provably never engages). If ON ≈ ON+BIG in a cell, the kernel wasn't engaging there → diagnosed null.

**Metrics:** per `/completion` response `timings`: `prompt_n` (uncached prompt tokens; assert=full len),
`cache_n` (**must be 0**), `prompt_ms`/`prompt_per_second`. **Primary = aggregate prefill tok/s =
Σ prompt_n / burst_wall_time** (n_predict=1). Secondary = per-req `prompt_ms` p50/p95 (TTFT proxy).
Warmups: 2 discarded bursts/launch. Reps: ~10 measured, median + IQR. Log
`nvidia-smi --query-gpu=clocks.sm,temperature.gpu,power.draw --format=csv -l 1` during runs; reject
cells with >5% SM-clock spread (WSL can't lock clocks).

**Go/no-go** (Arm B, ncmoe=0 first): R(k) = agg_tps(ON)/agg_tps(OFF). **GO** if R(k) rises monotonically
and R(8) ≥ 1.03 with 95% CI excluding 1.00 (expected 5-15% since GDN=22.5% of GPU prefill × batch-driven
op speedup). **NO-GO** if flat ±1% for all k WHILE the engagement detector confirms the kernel ran →
B=1 verdict stands, lever closed. **Diagnosed null** if flat AND ON≈ON+BIG → setup/gate error, fix it.
Arm A expected flat by construction → report as "deploy config cannot feed the kernel n_seqs>2" (a
primary finding on its own; if we want the win at deploy we'd ship Arm-C `-ub 8192` or a
lower `MIN_TOKENS` when n_seqs>1).

**Server launch lines (HEADLESS via background tool; literal paths — WSL `$VAR` expands empty):**
```
# Arm B OFF (sequential), ncmoe=8:
env -u GGML_CUDA_GDN_CHUNKED /home/augus/src/llama.cpp-master/build/bin/llama-server \
  -m /home/augus/models/qwen36-35b-a3b/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf \
  --n-cpu-moe 8 -c 131072 --parallel 8 -b 8192 -ub 2048 -ctxcp 0 --host 127.0.0.1 --port 8080
# Arm B ON: prefix  GGML_CUDA_GDN_CHUNKED=1 GGML_CUDA_GDN_CHUNKED_MIN_TOKENS=128  (drop `env -u`)
# Engagement detector: ON but MIN_TOKENS=1000000
# Arm C: -ub 8192 and drop MIN_TOKENS on ON.  Arm A: -b 2048 -ub 2048, no -ctxcp 0.
# ncmoe=0 variants: --n-cpu-moe 0 -c 32768 (fits 24GB; verify compute-buffer line in server log)
# NO --spec-type anywhere (MTP is a decode lever; disable to isolate prefill).
```
Driver per cell (Windows, server up): `python gdn_conc_bench.py <k> 1024 > runs/gdn/armB__on__k<k>.json`.

### 2. Quality bless (TF32 chunked = quality-neutral) — **DONE 2026-08-04. BLESSED: quality-neutral.**
Model key = **`qwen36-35b-q4`** (the Q4_K_M MoE; ncmoe defaulted to 24, ctx 8192). subset 40, HumanEval+,
`MIN_TOKENS=2` forces chunked prefill (decode stays n=1 → sequential, as deploy). Scored with
`score_subset.py` in the evalplus venv (pads to 164, runs official evalplus, reports pass@1 over the 40):
| tag | base (HumanEval) | plus (HumanEval+) |
|---|---|---|
| gdn-off (sequential) | 34/40 (0.850) | 33/40 (0.825) |
| gdn-on (chunked TF32) | 34/40 (0.850) | 33/40 (0.825) |
→ **pass@1 IDENTICAL** — matches project history (flat 33-34/40). **Engagement proven for free:** the
*sets* of failed problems differ (off: 134,93 unique; on: 118,8 unique; 5 shared) — bit-identical outputs
would give identical fail sets, so the chunked path demonstrably engaged and altered numerics. Same
signature as MTP: **quality-neutral but NOT bit-identical** (2e-7 NMSE flips greedy ties on borderline
problems; net pass@1 unchanged). Files: `runs/quality/gdn-{off,on}__qwen36-35b-q4*.json`. **The kernel is
now blessed on all three axes: correctness (46/46 @ 2e-7), no-regression perf (T≥1024 gate + OFF default),
and quality-neutrality.**

### 3. Different-shape model — RECOMMEND SKIP the download (redundant; see Shape investigation above).
The free contrasts (MoE H=32 vs dense-27B H=48 real A/B + synthetic H=48/64 sweep) answer the occupancy
question. If user still wants a download, the only genuinely-more-heads option would need checking HF
for a larger `qwen35`/`qwen35moe` GGUF (custom arch names — likely same source as the on-disk ones).

**Dense-27B (H=48) real-model A/B — DONE 2026-08-03 (post-reboot).** `ngl 99` (all on GPU), ub=2048, r=5
(the r=3 combined run gave a spurious ±778 first-rep outlier on pp2048; re-run isolated, r=5, clean):
| test | OFF (seq) | ON (chunked TF32) | ratio |
|---|---|---|---|
| pp2048 | 1378.5 ± 14.7 | 1325.6 ± 10.3 | **0.962 (−3.8%)** |
| pp4096 | 1371.1 ± 9.2 | 1340.9 ± 3.9 | **0.978 (−2.2%)** |
→ **Chunked is 2–4% SLOWER E2E at H=48** — the sweep's isolated −14% dilutes to −2–4% (GDN ≈15% of dense
prefill). Contrast MoE H=32 M4 (ncmoe=0: −0.4%/−0.5%, noise). **The head-count verdict holds at real-model
E2E: the chunked kernel should stay OFF by default (it already is) and must NEVER auto-engage on H≥48
models.** Confirms the T≥1024 + env-gate design is correct: no-regression precisely because it's opt-in.

## GPU-LOST INCIDENT (why we're rebooting)
Mid-way through the occupancy sweep, `nvidia-smi` (even on the Windows host) returned **"GPU is lost.
Reboot the system to recover this GPU"** — the RTX 3090 fell off the PCIe bus. `wsl --shutdown` did NOT
recover it (host-level, not WSL passthrough). **Only a reboot recovers.** The
GPU was healthy through all of Lever B + revert + M4; it died only in the last sweep. **Lesson for
resume: avoid `ncu`; keep GPU benchmarks from overlapping heavy interactive PC use.**
→ **Root cause now confirmed from the event log — see the post-mortem section below.**

## GPU STABILITY — event-log post-mortem & installed protections (2026-08-03)
**Forensics (Windows System log):** the crash = **nvlddmkm Event ID 153** firing 7× in 14s
(02/08 19:23:44→19:23:58) — a GPU engine-reset/TDR cascade. Corroborating evidence:
- **No WHEA-Logger fatal** (only 1 corrected *Information* record, unrelated) → **NOT** a physical PCIe
  link fault.
- **No TDR 4101/4102 recovery** event and **no BugCheck / minidump / LiveKernelReport** → the driver's
  resets FAILED and the OS stayed up (GPU-only death) — the "fell off the bus" (Xid-79 class) signature.
- **Reboot came ~16h later** (Kernel-Power 41 @ 03/08 11:37) → box ran on with a dead GPU until a manual
  reboot. Not a power-loss event.
- **Recurring, not a fluke:** 153 clusters on 07-05, 07-18 (×16), 07-19, 07-20, 08-02 05h, 08-02 19h —
  always under heavy GPU work → workload-triggered instability at high boost, not degrading hardware.
Root cause: sustained heavy compute drove the 3090 into its top boost/voltage bins where it
transient-hangs; the WDDM 2 s TDR + display contention (display is on the 3090) finished it off.
Baseline after reboot (gpt-oss-20b Q4_K_M, ngl99): pp512=**5688 t/s**, tg128=**197 t/s**, peak **304 W @
1905 MHz**; idle healthy at P8 / ~27 W.

**Protections installed** (`ops/gpu-stability/setup_gpu_protections.ps1`, must run **as admin**):
1. **Core-clock cap** — scheduled task **`RTX3090-ClockLock`** (SYSTEM, at-startup +30 s, retries + logs
   to `C:\ProgramData\gpu-tools\gpu_clocklock.log`) running **`nvidia-smi -lgc 210,1800`**. Range form:
   caps boost at 1800 MHz (below the 1905 that ran fine) but **still idles to P8 (~27 W)**. Decode is
   bandwidth-bound → ~0 t/s cost while removing the high-voltage region that hangs. Applies immediately
   too, no reboot needed. **1800 chosen & installed** (see sweep below).
2. **TDR delay** — `TdrDelay = TdrDdiDelay = 10 s` under
   `HKLM\SYSTEM\CurrentControlSet\Control\GraphicsDrivers` (`TdrLevel` left = 3/recover, safety net ON).
   **Needs a reboot to activate.** Complementary to #1 (attacks the 2 s-timeout side; #1 attacks the
   instability side).

**Clock-lock sweep (2026-08-03, `ops/gpu-stability/uv_sweep.ps1`, gpt-oss-20b ngl99):** ALL clocks
stable (New153=0, RC=0 including stock). Decode (tg512) FLAT ~190-205 t/s across the whole range
(confirms bandwidth-bound); prefill (pp2048) scales ~linear; peak power monotonic.
| lock | pp2048 | vs stock | tg512 | peak W |
|---|---|---|---|---|
| stock(1905) | 5865 | - | 205 | 341 |
| 1860 | 5742 | -2.1% | 198 | 329 |
| 1815 | 5658 | -3.5% | 194 | 329 |
| **1800 (chosen)** | ~5615 | ~-4% | flat | ~325 |
| 1770 | 5576 | -4.9% | 196 | 320 |
| 1725 | 5433 | -7.4% | (270 noise) | 314 |
| 1680 | 5353 | -8.7% | 190 | 306 |
→ **1800 kept**: ~-4% prefill, ~0% decode, 105 MHz / ~16 W below the 1905 crash point. To change, edit
`$CoreClockMax` in `setup_gpu_protections.ps1` and re-run it (admin). Caveat: the bench is ~1 min/step so
it measures the perf/watt curve, not a stability threshold (all clocks ran); the cap's stability benefit
is inferred from lower voltage, hence the conservative pick since decode is free.

**Revert everything:** `Unregister-ScheduledTask RTX3090-ClockLock -Confirm:$false; nvidia-smi -rgc;
Remove-ItemProperty 'HKLM:\SYSTEM\CurrentControlSet\Control\GraphicsDrivers' -Name TdrDelay,TdrDdiDelay`
(then reboot to clear TDR). Optional extra margin against transient spikes: add `nvidia-smi -pl 380`.

## RESUME ORDER after "gpu voltou"
1. `nvidia-smi` sanity (GPU back, 0 MiB used).
2. Re-verify kernel: `GGML_CUDA_GDN_CHUNKED=1 GGML_CUDA_GDN_CHUNKED_MIN_TOKENS=2 test-backend-ops test -o GATED_DELTA_NET -b CUDA0` → expect 46/46 @ 2e-7 (confirms TF32 revert intact).
3. Occupancy sweep: `python3 ~/gdn_perf_sweep.py 3` (now includes H=48/64) → map crossover vs H.
4. Dense-27B (H=48) real A/B (commands above).
5. Concurrent-serving A/B (Arm B first, ncmoe=0, k=1,2,4,8) — the headline experiment.
6. Quality bless (find --model key first).
7. (Optional) download decision.
Everything HEADLESS (no visible windows); servers via background tool, killed after.
