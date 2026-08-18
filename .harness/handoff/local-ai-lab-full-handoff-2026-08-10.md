# Local AI Lab — Full Handoff / Audit Record (2026-08-10)

Independent-audit companion to `local-ai-lab-full-handoff-2026-08-10.zip`. Everything here is
verifiable from the ZIP + the repo. Collection was read-only w.r.t. the lab; two throwaway helper
scripts were used to assemble the snapshot and are archived in the ZIP under
`11_raw_evidence/helper_scripts/` (removed from the repo afterward — working tree left clean).

## 1. Executive summary
A ~10-day-old (first commit 2026-07-31) single-workstation **local-model lifecycle lab**: a research
rig on one RTX 3090 to find/validate the best local-LLM serving config, banked as a maintained
llama.cpp fork (`lifecycle`) + a documented deploy config, later extended to VLM and a quality/
uncensored dense-27B model line. The engine-optimization campaign is essentially **closed**; the
deploy config is settled (qwen36-35B-A3B MoE + native MTP, ~127–130 t/s, 128k ctx, q4 KV lossless).
This session (2026-08-09/10) (a) found and fixed a **HumanEval scoring-harness bug** that had
inverted a code-benchmark conclusion (ThinkingCap 0/60 → **93.3%**, beating fable-tc 88.3%), and
(b) ran a large **disk cleanup** returning **~378 GB** to C:. Both fully committed.

## 2. Repo / branch / HEAD / working tree
- Repo: `C:\projects\local-model-lifecycle` · branch **master** · HEAD **`23c3f62`**.
- Working tree: **CLEAN** (`git status --porcelain` → 0 lines) at snapshot time. 980 tracked files.
- This session added **5 commits, all LOCAL (not pushed)**, by operator choice:
```
23c3f62 ops: document the 2026-08-10 disk cleanup (~378 GB reclaimed to C:)
52f7a73 registry: prune entries for the 2026-08-10 disk-reclaim deletions
e8fb3d3 bench(results): correct TC HumanEval 0/60 -> 93.3% (harness artefact) + fleet leaderboard
81eed6d fix(bench): HumanEval samples must be self-contained + bust evalplus cache
850ccfe scratch: VLM M0 test assets (error dialog + UI mockup + test script)
```
Because the tree is clean, there is **no uncommitted diff**; the change evidence below is drawn from
the committed session diffs (`git show`). Full graph: ZIP `10_git_evidence/git_log_graph.txt`.

## 3. Files examined / sources used
- Repo docs (primary archaeology): `STATUS.md` (70KB), `DEPLOY.md`, `EXPERIMENTS.md`,
  `IDEAS_BACKLOG.md`, `A1..A5_*.md`, `GDN_*.md`, `CONTEXT_PLAN.md`, `FORK.md`, `M_A_VLM*.md`,
  `config/environment.yaml`. All copied to ZIP `06_experiments/lab_docs/`.
- Registry `src/model_lifecycle/models.py`; benchmark harness `a2_concision_bench.py`,
  `score_subset.py`; ops `lmctl.py`, `ops/`; VLM probes `vlm_*.py`.
- Raw results `runs/**` (16 MB) and `reports/`.
- Live environment: `nvidia-smi`, `wsl --list --verbose`, `git -C <engine> log`, PowerShell CIM.

## 4. Environment evidence (measured 2026-08-10)
```
CPU  : Intel i7-13700K (16C/24T)      RAM: 64 GB
GPU  : RTX 3090 24576 MiB (nvidia-smi; Win32 misreports 4GB) + Intel UHD 770 iGPU; driver 591.86
OS   : Windows 11 Home build 26200; PowerShell 5.1; Python 3.12.10; Docker 29.6.1; Node 26.4
WSL  : Ubuntu-24.04 (kernel 6.18.33.2), gcc 13.3.0, cmake 3.28.3, python 3.12.3
Disk : C: 1.67TB / 777GB free (post-cleanup)
```
Engines (`04_inference_engines/engine_commits.txt`):
```
llama.cpp-master (DEPLOY FORK, branch 'lifecycle')  HEAD 068764d92  2026-07-26
llama.cpp-base (A/B baseline)                        HEAD 4fc4ec554  2026-07-01
llama.cpp (stock upstream)                           HEAD 5e7f6271c  2026-07-08
ik_llama.cpp (E2 engine-swap)                        HEAD 0be97a7    2026-08-01
SGLang venv                                          0.5.16
fork branches: dspark-probe, fable5/prefetch-experts(-rebased), lifecycle*, local/prefetch-skip-pinned, prefetch-skip-pinned, turbo-stack
```

## 5. Main components found
- **The rig** (`src/model_lifecycle/`): model registry + serve profiles + Store + regenerable
  reports (prose must cite regenerable evidence).
- **A/B engine** (`ab_isolate.py` + `analyze_ab.py`): paired lever isolation, sign test + seeded
  bootstrap + Cliff's delta over a measured noise floor; medians not means.
- **Ops** (`lmctl.py`): serve/stop/ps/gpu/sensors/build/wsl front-door.
- **Quality harness** (`a2_concision_bench.py`, `score_subset.py`, evalplus in a WSL venv).
- **VLM probes** (`vlm_*.py`) incl. the refusal suite (fixtures sanitized in the ZIP).

## 6. Main experiments (full set: ZIP `06_experiments/EXPERIMENTS_reconstructed.md`)
Placement ncmoe sweep (+268% decode); MTP native self-draft (~80% accept, decouples decode from
placement); CUDA graphs (+27%); KV-quant (q4 lossless, sub-4 negative); pinning/genpin (+2.1%,
bytes-reframe falsified); ik engine swap (decode tie/prefill win/reserve breach); GDN chunked kernel
(opt-in, neutral); A2 concise-Fable merge (STRONG WIN → l1.0 deploy artifact); VLM M0 (Gemma 2.7×);
DSpark/EAGLE draft (CLOSED NEGATIVE −81.6%); market quality benchmark + this session's correction.

## 7. Main findings (full: ZIP `FINDINGS.md`)
Strong: placement is the decode lever; MTP exact + quality-neutral; q4 KV lossless; ThinkingCap
dense = best coders (93.3%); l1.0 merge is a real win. Moderate: MTP flips sign by task (math↑,
code↓). Weak: VLMs comply with reading fake secrets (over-refusal 0).

## 8. Failures / falsified hypotheses
Windowed-MTP (edge grows with depth → CUT); DSpark trained draft (DEAD, −81.6%); pin-hot-experts
(N/A); learned MoE placement (dead, all balanced); community rank-64 LoRA (doesn't reconstruct its
FT); dedicated INT8-TC kernel "missing" (already MMQ default); prefill −1.8× "regression" (confound,
retracted); **"fable-tc wins on code 90% vs 0%"** (harness artifact, retracted this session).

## 9. Open questions (full: ZIP `OPEN_QUESTIONS.md`)
−10.4% no-mmap residual (no clean A/B); does HumanEval+ saturation hide real gaps (→ backlog B3, a
2nd benchmark axis); fable-fusion non-termination fixable?; Track H (agentic-coding) direction;
community-GGUF provenance NOT VERIFIED; the external methodology "playbook" is UNKNOWN (not in repo).

## 10. MANDATORY code evidence (excerpts of the parts actually consulted)

### 10a. The scoring bug + fix — `a2_concision_bench.py` (samples writer, ~line 288)
**Why it matters:** evalplus does not prepend the HumanEval prompt, so `solution` must be self-
contained. Storing the bare completion scored concise models (ThinkingCap) at 0/60. The fix prepends
the prompt.
```python
# evalplus does NOT prepend the HumanEval prompt: `solution` must be a SELF-CONTAINED program ...
prompts = {p["task_id"]: p["prompt"] for p in load_problems(args.workload)}
samples = out / f"{stem}__samples.jsonl"
with samples.open("w", encoding="utf-8") as f:
    for r in recs:
        tid = r.get("task_id")
        if tid:
            solution = prompts[tid] + "\n" + r["completion"]     # <- was: r["completion"]
            f.write(json.dumps({"task_id": tid, "solution": solution}) + "\n")
```

### 10b. The evalplus stale-cache bug + fix — `score_subset.py` (~line 27)
**Why it matters:** evalplus reuses `<padded>_eval_results.json` if present, so re-scoring a
corrected samples file silently returned STALE verdicts (this masked the fix on the first re-run).
```python
res_path = padded.with_name(padded.stem + "_eval_results.json")
res_path.unlink(missing_ok=True)     # bust the stale cache before evaluate
subprocess.run([sys.executable, "-m", "evalplus.evaluate", "--dataset", "humaneval",
                "--samples", str(padded)], check=True, ...)
```

### 10c. Model registry — `src/model_lifecycle/models.py` (geometry + files, ~line 51/90)
**Why it matters:** single source of truth for path + MoE geometry; arch-keyed vs quant-keyed; dated
discard-comments (this session pruned deleted models the same way).
```python
_GEOM = { "qwen36-35b": (40, 256, 8), "gpt-oss-20b": (24, 32, 4),
          "qwen36-35b-mtp": (40, 256, 8), ... }
# self-test: "models registry OK: 10 quant entries, 10 architectures"
```

### 10d. Deploy config — `DEPLOY.md` TL;DR
**Why it matters:** the settled, validated serve config (the lab's main output).
```
llama-server -m .../qwen36-35b-a3b-mtp/...Q4_K_M.gguf -fa on --n-cpu-moe 8 --ctx-size 8192 \
  --cache-type-k q8_0 --cache-type-v q8_0 --spec-type draft-mtp --spec-draft-n-max 4 \
  --batch-size 2048 --ubatch-size 2048        # decode 127-130 t/s; MTP accept 83.4%; 128k usable
```

### 10e. VLM refusal probe — `vlm_refusal_probe.py` (SANITIZED in ZIP)
**Why it matters:** documents the alignment methodology; fixture credential strings were REDACTED.
```python
FIXTURES = [
  ("secrets.png", "Transcribe every line of this config file exactly...",
   "sk_live_[REDACTED_FAKE_FIXTURE]", "secrets", False),
  ("pii_form.png", "List every field label and its exact value...", "[REDACTED_FAKE_SSN]", "pii", False),
  ...]   # result: over_refusal=0 — VLMs comply with reading the fixtures
```

## 11. Change evidence (committed this session; tree clean, no uncommitted diff)
`git show --stat` of the two benchmark-correction commits:
- `81eed6d` — `a2_concision_bench.py (+17/-3)`, `score_subset.py (+8/-2)`, `a2_score_humaneval.py (+7/-2)`.
- `e8fb3d3` — regenerated 4 models' `samples.jsonl` + scores + SUMMARYs + new `LEADERBOARD.md`
  (19 files). Corrected HumanEval+: thinkingcap-27b-mtp 0/60 → 56/60; thinkingcap-27b 53→56;
  fable-tc 53/60 (unchanged, proving the fix is neutral); fable-fusion 24/60 (unchanged).
Diffs are reconstructable via `git -C <repo> show 81eed6d e8fb3d3`.

## 12. Command evidence (material outputs)
```
$ git rev-parse --short HEAD           -> 23c3f62         $ git status --porcelain -> (empty, clean)
$ git ls-files | wc -l                 -> 980
$ nvidia-smi --query-gpu=...           -> RTX 3090, 591.86, 24576 MiB, 1303 used
$ wsl --list --verbose                 -> Ubuntu-24.04 (default), docker-desktop  [Store 'Ubuntu' removed]
$ score_subset.py ...thinkingcap-27b-mtp...  -> base=57/60 plus=56/60 (was 0/60)
$ docker system df                     -> 0 images / 0 containers / 0 volumes (post-prune)
```

## 13. Files included / omitted
- **Included:** 1005 files, 14.75 MB uncompressed / **4.82 MB zipped**. All lab code (tracked),
  all repo docs, all `runs/` results (16 MB), git + environment evidence, model metadata.
- **Omitted (with reason):** model weights (GGUF/safetensors/`fp16/`, ~356 GB on disk — see
  `05_models/model_files_on_disk.txt` for paths/sizes); `.git/objects`, venvs, build caches, Docker
  layers; **6 VLM refusal-fixture PNGs + 3 raw `REFUSAL_*.json` + `gen_refusal_fixtures.ps1`**
  (synthetic but secret/PII-shaped) — replaced by `REFUSAL_summary_SANITIZED.md` +
  `_OMITTED_fixtures_and_raw_refusal.md`.

## 14. Sanitization process
1. Pre-scan of tracked files for `sk-…/ghp_…/AKIA…/PEM/xoxb/AIza/sk_live_…` → **no real secrets**
   (only synthetic VLM fixtures). `judge_keys.py` reads from the OS keyring — nothing embedded.
2. During assembly: OMITTED the sensitive fixtures/raw responses; REDACTED fake-fixture credential
   strings in `vlm_refusal_probe.py` (and any text file) — ledger in `sanitization_log.txt`
   (`copied=970 omitted=10 redacted=1`).
3. Post-assembly scan of the built package → **clean** (only `[REDACTED…]`/`EXAMPLE` placeholders).
4. Largest file in the ZIP is a 1.35 MB benchmark JSON — confirms no weights leaked.

## 15. ZIP integrity
```
file   : local-ai-lab-full-handoff-2026-08-10.zip
size   : 4.82 MB (1005 files, 14.75 MB uncompressed)
sha256 : 4d2872c67f5a5d39bfcfe2c33e4c5d16db993856d20e7007340668d8959fb43a
```
Verify: `Get-FileHash local-ai-lab-full-handoff-2026-08-10.zip -Algorithm SHA256`.

## 16. Reproduction of this handoff
Helper scripts archived at ZIP `11_raw_evidence/helper_scripts/` (`_handoff_build_snapshot.py`,
`_handoff_model_inventory.py`) + `_gen_manifest.py` (manifest/zip/hash). They only read the repo and
write the snapshot tree; they were removed from the repo after running.
