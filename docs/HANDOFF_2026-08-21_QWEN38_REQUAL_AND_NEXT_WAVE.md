# Local inference lab handoff — 2026-08-21

**Authority:** current dated handoff for the Qwen3.8 requalification, broad portfolio, artifact assessment,
and next controlled-maintenance wave.

**Repository:** `C:\projects\local-model-lifecycle`

**Handoff parent HEAD:** `294990b` (`docs(research): make qualification explicitly supersedable`)

**Branch:** `dev/local-labs-relay-q0`, 20 commits ahead of `local-labs/dev/local-labs-relay-q0` before
this handoff commit. Nothing was pushed.

**Incoming-agent rule:** read this document before the older `docs/HANDOFF.md`. The older handoff remains
valuable historical evidence, but several of its deployment statements and closed/dogmatic labels have been
superseded or reopened by the 2026-08-20/21 work.

## 1. User intent and operating posture

The user wants the local inference lab pursued autonomously and broadly. Do not reduce the work to one live
endpoint smoke test. Requalify fragile experiments, use the measurement cookbook and the wider
`tare.tools.library` theory, and exercise the `slop.cpp` fork as an experimental system.

Specific instructions and preferences established in this continuation:

1. A controlled server maintenance window is acceptable and now desirable; the previous wave was too
   constrained by keeping the live endpoint up.
2. Do **not** restart or extend the 24/48/72-hour soak unless the user explicitly changes direction.
3. Be open to new requants, imatrices, fine-tunes, methods, and fork levers.
4. Hashes preserve attribution, not operational primacy. Models, experiments, and rules may be superseded by
   stronger evidence.
5. Preserve old receipts and label displaced conclusions `SUPERSEDED`; do not silently erase history.
6. Prefer a broad but dependency-gated portfolio over spending the whole session on one expensive benchmark.
7. No remote push has been authorized. Keep commits local unless the user explicitly asks otherwise.

## 2. Current live state captured at handoff

Captured on 2026-08-21 in the `America/Sao_Paulo` timezone.

### 2.1 Text inference endpoint

- health: `http://127.0.0.1:8080/health` returned `{"status":"ok"}`;
- service: `llm-inference.service`, active/running;
- service user: `augus`;
- restart policy: `Restart=always`;
- Linux PID at capture: `135546`;
- unit file: `/etc/systemd/system/llm-inference.service`;
- binary: `/home/augus/src/llama.cpp/build/bin/llama-server`;
- model: `/home/augus/models/qwen38-27b/unsloth/Qwen3.8-27B-IQ4_XS.gguf`;
- model alias: `qwen38-27b`;
- port: `8080`;
- context: `32768`;
- slots: server default/current unit behavior; the qualification responses reported four slots;
- FlashAttention: on;
- KV: `q8_0/q8_0`;
- GPU layers: all;
- speculative decoding: off;
- template: `/home/augus/models/templates/qwen-sharp.jinja`;
- metrics and Jinja: enabled.

Exact observed argv:

```text
/home/augus/src/llama.cpp/build/bin/llama-server
  -m /home/augus/models/qwen38-27b/unsloth/Qwen3.8-27B-IQ4_XS.gguf
  --alias qwen38-27b --host 0.0.0.0 --port 8080 --ctx-size 32768
  --flash-attn on --cache-type-k q8_0 --cache-type-v q8_0
  --gpu-layers all --metrics --jinja
  --chat-template-file /home/augus/models/templates/qwen-sharp.jinja
```

This is **not** the consolidated `lifecycle` fork binary. Recent endpoint qualification establishes the
stock-tree substrate above, not the authorial fork levers.

### 2.2 Embedding endpoint

An independent embedding server was also running:

- Linux PID at capture: `34282`;
- binary: `/home/augus/src/llama.cpp/build/bin/llama-server`;
- model: `/home/augus/models/embedding/nomic-embed-text-v1.5.Q8_0.gguf`;
- port: `8081`;
- embedding mode, context 32768, eight slots.

Do not stop or kill this process as collateral damage when taking the text endpoint into LAB mode.

### 2.3 Resources at capture

- GPU: RTX 3090, 24,576 MiB total;
- GPU memory: 18,776 MiB used, 5,547 MiB free;
- GPU temperature: 33 C;
- GPU power: 38.48 W;
- WSL memory limit: 43 GiB;
- WSL available memory: approximately 39 GiB;
- WSL swap: 16 GiB total, 9.5 GiB used;
- `/dev/sdd`: 1,007 GiB total, 433 GiB available.

The 43 GiB WSL memory ceiling is a real constraint for custom BF16-to-GGUF quantization of a roughly
55 GB source. Do not assume host-installed 64 GB is fully available inside WSL.

## 3. Repository and fork state

### 3.1 Local-labs repository

At handoff parent HEAD:

```text
## dev/local-labs-relay-q0...local-labs/dev/local-labs-relay-q0 [ahead 20]
?? runs/reliability/LAB-REL-001-24h-2026-08-21/
```

The untracked reliability directory is intentional partial evidence, not garbage. Do not add, delete, or
classify it accidentally.

### 3.2 `slop.cpp` / llama.cpp fork tree

Path: `/home/augus/src/llama.cpp-master`

Captured state:

- checked-out branch: `main`;
- HEAD: `87a416bd75d5a64e66e55846b779c0a54eca21bd`;
- untracked file: `a4_spec_metrics_probe.py`;
- preserved `lifecycle` branch: `068764d92`;
- other preserved lines include `prefetch-skip-pinned`, `fable5-prefetch-experts`, `dspark-probe`, and
  `turbo-stack`.

Do not reset, clean, or checkout over the dirty tree. Use an existing separate build tree/worktree or create
a new explicitly named worktree after verifying paths. The authorial fork levers are documented in
[`research/FORK.md`](research/FORK.md).

## 4. What was completed in this continuation

### 4.1 Qwen3.8 IQ4_XS evidence requalification

Frozen substrate:

- model SHA-256: `9fd40d7036f5e0918e20aaeebf11468fafd06bb53d4d980eef6bb7e4e4ace666`;
- model bytes: `15,705,861,088`;
- llama.cpp: `5e7f6271c06b9104862ab799278a1b7f1323a449` (`b9863`);
- context: 32,768;
- KV: Q8_0;
- greedy/deterministic response path;
- prompt cache disabled for the three requalification matrices.

Qualified results:

| Campaign | Result | Interpretation |
|---|---:|---|
| MQAR exact recall | 240/240 | Exact associative retrieval through 28,876 rendered tokens |
| NIAH positive retrieval | 60/60 | Exact random-code retrieval from roughly 4k to 30k tokens |
| NIAH negative controls | 10/10 | Correctly returned `NOT_PRESENT`; no fabrication |
| GSM8K strict | 95/100 | Four reasoning errors and one correct-but-truncated response |

The invalid MQAR 2,048-pair terminal dose exceeded the live context and was not pooled. The prior nominal
24k NIAH failure did not reproduce under calibrated prompts and controls.

Primary report: [`../runs/requalification/QWEN38-2026-08-20/REQUALIFICATION_REPORT.md`](../runs/requalification/QWEN38-2026-08-20/REQUALIFICATION_REPORT.md).

### 4.2 Liger feasibility and replication

Three fail-closed campaigns were completed:

1. **Qwen3 Liger extension:** blocked at state transfer. It adds Q/K/V biases and drops trained Q/K norms.
2. **Original Llama path on Transformers 4.52.4:** weights transfer exactly, but construction fails because
   caller/callee disagree on a two-value versus three-value attention return contract.
3. **Historical paper stack on Transformers 4.47.1:** construction and finite forward/backward pass, but
   cached recurrence fails. Cache length advances by head count instead of token count and the local-attention
   branch is not persisted.

No checkpoint download or fine-tuning followed the failed gates. Untouched upstream Liger is not a qualified
cached-inference substrate.

Reports:

- [`../runs/requalification/LIGER-FEASIBILITY-2026-08-20/RESULT.md`](../runs/requalification/LIGER-FEASIBILITY-2026-08-20/RESULT.md)
- [`../runs/requalification/LIGER-LLAMA-FEASIBILITY-2026-08-20/RESULT.md`](../runs/requalification/LIGER-LLAMA-FEASIBILITY-2026-08-20/RESULT.md)
- [`../runs/requalification/LIGER-PAPER-SNAPSHOT-2026-08-20/RESULT.md`](../runs/requalification/LIGER-PAPER-SNAPSHOT-2026-08-20/RESULT.md)

### 4.3 Broad live-safe portfolio

| Campaign | Result | Important boundary |
|---|---:|---|
| Agent/tool suite | 8/8 | Functional core only; perturbation and scale remain |
| Cache/cancel live slice | 4/4 | Cold/warm byte equality and oracle correctness; disk restore blocked |
| MBPP+ | base 326/378; Plus 284/378 | `Mbpp/260` non-terminates at both 768 and 2,048 tokens |
| Context paired slice | retrieval/multikey/multihop 18/18 | local RULER-inspired, not official RULER |
| Context aggregation | 10/10, 9/10, 10/10 at 8k/16k/28k | one reproducible positional error, no monotonic collapse |
| Energy instrument | qualified | 0.206/0.262 J per prompt token; 8.80/9.52 J per generated token |
| Provenance/QA | 23/23 self-tests | source revision of community requant remains unknown |
| RetNet mechanism | 7/7 gates | mechanism-only synthetic reproduction, not a real checkpoint |

Portfolio report: [`../runs/requalification/PORTFOLIO-2026-08-21/EXPERIMENT_PORTFOLIO_REPORT.md`](../runs/requalification/PORTFOLIO-2026-08-21/EXPERIMENT_PORTFOLIO_REPORT.md).

### 4.4 Fragile attempts that were corrected, not hidden

- Cache r1 exceeded context because a high-entropy nonce was repeated in every filler line.
- Cache r2 exposed an inadequate 16-token response budget; corrected r3 passed.
- Energy r1 failed its own boundary/interpolation self-test and used biased fixed ordering; r2 alternated order
  and interpolated phase boundaries.
- Context r1 changed random facts with context length; the valid paired run held facts constant.
- EvalPlus direct n=50 MBPP scoring was invalid for the required 378 IDs; the scorer now pads, invalidates stale
  cache, and preserves the intended denominator.

These are evidence about the harness. Do not pool superseded attempts with qualified runs.

## 5. Reliability soak disposition

The 24-hour run was stopped at the user's request and must not be resumed automatically.

Directory: `runs/reliability/LAB-REL-001-24h-2026-08-21/` (untracked)

Partial receipt:

- status: `CANCELLED_BY_USER`;
- started: `2026-08-21T03:01:35.305872Z`;
- stopped: `2026-08-21T09:10:21.477087Z`;
- iterations: 369;
- operation pass/fail: 369/0;
- health failures: 0;
- temperature peak: 53 C;
- VRAM peak: 18,840 MB;
- power peak: 353.61 W;
- records: 369 JSONL lines, 635,915 bytes;
- `soak.pid` contains stale historical PID `8124`; that process is stopped.

This is a clean partial observation, **not** a 24-hour PASS. Preserve it without pass/fail classification.

## 6. New external candidates and artifact drift

Assessment: [`research/QWEN38_IMATRIX_COLD_FUSION_ASSESSMENT_2026-08-21.md`](research/QWEN38_IMATRIX_COLD_FUSION_ASSESSMENT_2026-08-21.md).

### 6.1 Unsloth imatrix

Frozen at assessment time:

- repository revision: `4ca720788d1e01f1bff70c033e0d0028fd02e502`;
- `imatrix_unsloth.gguf` SHA-256:
  `0ee5b10bd0c2fa2127c6f4b43dbfe1efd71e383b63217af9dade1de36599f1c1`;
- bytes: 13,642,656;
- type: GGUF v3 imatrix;
- 1,251 chunks × 8,192 declared tokens;
- 992 importance records;
- no `nextn`/MTP importance record found.

An ephemeral inspection copy exists at `/tmp/qwen38-imatrix-inspect/imatrix_unsloth.gguf`. It is not an
admitted model artifact and may disappear with `/tmp` cleanup.

### 6.2 Unsloth current-revision drift

The current Hub files differ from the locally qualified generation:

| Candidate | Current Hub bytes | Current Hub SHA-256 |
|---|---:|---|
| `UD-IQ4_XS` | 14,252,845,984 | `40fac4050e940397dbf13087afd50f4734a11805bf9d65ef8ddd7483470e6199` |
| `UD-Q2_K_XL` | 9,828,981,664 | `fd4730dd8aad070517978752b63d530aeb1740d2283cab9fa24f1e404032ddb0` |

The Hub now publishes an MTP head separately and removed an earlier IQ2_M artifact. Do not overwrite local
files with same-label downloads. Admit into revision-qualified paths and allow the new requants to supersede
the historical frontier if they win.

### 6.3 Cold Fusion

Frozen at assessment time:

- GGUF repository revision: `21dd13a4a43d7570a9496948f6265310681fa9f4`;
- BF16 source revision: `9c44193f07782c85c0f437a5d8466ba5c95c95fe`;
- recommended first artifact:
  `Qwen3.8-27B-Cold-Fusion-GAIN-V1.1-NM-DAU-NEO-MAX-NEO-MTP-IQ4_XS.gguf`;
- bytes: 17,033,680,384;
- SHA-256: `523bf4fbe2a2e0ce7aa54f812d85746294483b579443dd6e50e8ab684d7852f9`.

Cold Fusion is a new fine-tuned model, not a quant arm of base Qwen. Its reduced-reasoning and benchmark
claims are unqualified locally. The model is interesting because it targets the exact base-model failure we
measured: verbose thinking, truncation, and worse coding accuracy at high reasoning effort.

No current-revision Unsloth quant or Cold Fusion model was admitted to `/home/augus/models` during this
continuation.

## 7. Epistemic rule: supersession

This rule is binding for the next agent:

- freeze identity to make results attributable;
- do not freeze authority;
- a newer requant/model/method may replace the incumbent after equal or stronger qualification;
- mark displaced conclusions `SUPERSEDED` and retain their receipts;
- do not use previous promotion as a tie-breaker;
- reopen even a previously closed hypothesis if a material substrate change attacks the failure mechanism.

The current IQ4_XS is the only Qwen3.8 artifact qualified by the new broad packet. That is a temporary status,
not a presumption against current Unsloth requants or Cold Fusion.

## 8. Recommended next wave

The user explicitly identified the prior wave as too constrained by keeping the server alive. The next wave
should use controlled maintenance and exercise quant/model/fork axes rather than merely repeating endpoint
tests.

### Phase 0 — controlled LAB entry

1. Re-read this handoff and the external-candidate assessment.
2. Capture `git status`, live server argv, unit state, health, GPU state, and exact baseline model/template
   identities again; PIDs are not stable.
3. Stop `llm-inference.service` through `systemctl`, not by killing its PID. `Restart=always` will resurrect a
   manually killed child. The current user previously lacked non-interactive authorization; if that remains
   true, report the exact authorization blocker rather than fighting systemd.
4. Leave the embedding service on port 8081 untouched.
5. Start experimental servers on an explicit port/profile and record every argv/env vector.
6. After each maintenance tranche, stop the experimental server, restart the baseline unit, verify `/health`,
   and compare its argv with the captured baseline.

### Phase 1 — current Unsloth revision screen

Admit into new revision-qualified paths:

- current Unsloth `UD-IQ4_XS`;
- current Unsloth `UD-Q2_K_XL`.

Compare against their historical local counterparts using the compact discriminating packet:

- agent suite plus perturbation/scale;
- cache/cancel/reuse;
- paired context at 8k/16k/32k/64k;
- MBPP+ failure-focused subset including `Mbpp/260`;
- strict replay of the five GSM8K failures;
- MTP inventory and a throughput/acceptance sentinel where applicable.

If the current Q2_K_XL is non-inferior or better, promote it to the full packet and allow it to supersede the
historical Pareto candidate.

### Phase 2 — Cold Fusion practical A/B

Download only the revision-pinned Cold Fusion `NEO-MTP-IQ4_XS` first. Use the same file with speculation off
and on so head presence does not confound the MTP factor.

Design:

| Artifact | MTP off | MTP on |
|---|---:|---:|
| qualified base IQ4_XS | yes | `n-max=3` |
| Cold Fusion MTP IQ4_XS | yes | `n-max=2`, then `n-max=3` |

Primary claim test: instruct/off, low, medium, and xhigh with adequate budgets. Measure strict correctness,
reasoning tokens, truncation, non-termination, task success per 1,000 tokens, time/joules per correct answer,
tool validity, and irreversible-action safety.

The first comparison answers `which final artifact is better?`, not `did the fine-tune alone cause the
gain?`. Only perform matched BF16 requantization later if causal attribution is worth its cost.

### Phase 3 — fork lever isolation

Fix the winning weight artifact, then compare the stock baseline and `lifecycle` default path before toggling
one lever at a time.

High-value arms:

1. MTP off/n2/n3/n4 × concurrency 1/2/4/8;
2. symmetric KV `f16/f16`, `q8_0/q8_0`, `q4_0/q4_0` at increasing context;
3. context checkpoints/prefix reuse under multi-turn, cancellation, and session switching;
4. mmap on/off with load time, cold/warm prefill, RAM, page faults, energy, and decode;
5. ubatch 512/1024/2048;
6. KV host pinning with `--no-kv-offload` at long context;
7. Qwen3.6-35B-A3B MoE placement × expert prefetch × pinning in normal and heavy-offload regimes.

Use the dense Qwen3.8 for KV/checkpoint/MTP questions and the Qwen3.6 MoE for expert-transfer levers. Dense
Qwen cannot establish the prefetch-experts claim.

### Phase 4 — full promotion packet and restoration

Run full MBPP+ or other expensive axes only on finalists. Freeze a decision packet containing artifact SHA,
engine commit, template SHA, runtime vector, receipts, invalid attempts, and supersession status. Restore the
original unit and verify health before ending the turn.

## 9. What not to spend the next wave on

- Do not run another soak.
- Do not repeat the full seven-quant HumanEval+/MATH screen before the compact new-revision gate.
- Do not treat IQ2_M's historical failure as eternal dogma; use it as a negative control if a materially new
  quant recipe is available.
- Do not continue untouched-upstream Liger inference without a declared cache patch.
- Do not open the high-cost three-kernel GDN rewrite on the 3090 before an occupancy profile justifies it.
- Do not revive TurboQuant merely because the branch exists.
- Do not promote expert caching on a load-balanced MoE without a routing-concentration screen.
- Do not pool deterministic and model-recommended sampling tracks.
- Do not infer MTP success from acceptance alone; require task-correct latency/energy benefit.

## 10. Measurement discipline

Standing harness self-test: 23/23. Run it before a new benchmark becomes decision-bearing.

Rules:

1. preregister the question, factor, invariant controls, invalidation conditions, and decision rule;
2. change one factor at a time unless the interaction is the explicit question;
3. alternate arm order and preserve cold/warm state intentionally;
4. retain raw responses and incremental receipts;
5. record artifact bytes/full SHA, source revision, quantizer/imatrix, engine commit, template SHA, and runtime
   levers;
6. mark truncation/non-termination separately from ordinary wrong answers;
7. never pool an instrumentation failure with a corrected run;
8. compare task-correct outcomes, not throughput alone;
9. restore the service baseline after maintenance.

Useful commands at the start of the next turn:

```powershell
git status -sb
git log --oneline -20
Invoke-RestMethod http://127.0.0.1:8080/health
wsl -d Ubuntu-24.04 -- pgrep -a llama-server
wsl -d Ubuntu-24.04 -- systemctl show llm-inference.service `
  -p ActiveState -p SubState -p ExecStart -p Restart -p User --no-pager
python tests/benchmark_harness/benchmark_harness_selftest.py
```

## 11. Commit map for this continuation

Qwen requalification:

- `a4554f9` preregistration;
- `153150a` valid MQAR terminal-dose amendment;
- `378c954` evidence and report.

Liger feasibility/replication:

- `a7e1090`, `0b6629a`, `85c2f27` — Qwen path;
- `2c22529`, `7b9f89c`, `3b250b0`, `a5d9020` — Llama/current environment;
- `6a2d8f9`, `e41efb9`, `e460540` — paper snapshot and recurrence verdict.

Broad portfolio and reliability disposition:

- `7390d05` broad qualification portfolio;
- `15830a2` soak cancellation bookkeeping.

External candidates and supersession posture:

- `ee5214f` Unsloth imatrix / Cold Fusion assessment;
- `294990b` explicit supersession doctrine.

This handoff is the next local commit after the parent above.

## 12. Primary reading order

1. **This file** — current live state and next execution order.
2. [`research/QWEN38_IMATRIX_COLD_FUSION_ASSESSMENT_2026-08-21.md`](research/QWEN38_IMATRIX_COLD_FUSION_ASSESSMENT_2026-08-21.md)
3. [`../runs/requalification/PORTFOLIO-2026-08-21/EXPERIMENT_PORTFOLIO_REPORT.md`](../runs/requalification/PORTFOLIO-2026-08-21/EXPERIMENT_PORTFOLIO_REPORT.md)
4. [`../runs/requalification/QWEN38-2026-08-20/REQUALIFICATION_REPORT.md`](../runs/requalification/QWEN38-2026-08-20/REQUALIFICATION_REPORT.md)
5. [`research/BACKLOG_V2_STATUS.md`](research/BACKLOG_V2_STATUS.md)
6. [`research/FORK.md`](research/FORK.md)
7. [`HANDOFF.md`](HANDOFF.md) — historical master handoff, useful but no longer authoritative alone.

## 13. Clean stopping condition for the next agent

A good next handoff should leave:

- the exact service baseline restored and healthy;
- all admitted artifacts revision-pinned and hashed;
- raw and summarized receipts for every valid arm;
- invalid/superseded attempts preserved and labeled;
- one explicit Pareto decision or a precise blocker;
- old conclusions marked `SUPERSEDED` where evidence warrants it;
- no unreported background process;
- no remote push unless explicitly authorized.
