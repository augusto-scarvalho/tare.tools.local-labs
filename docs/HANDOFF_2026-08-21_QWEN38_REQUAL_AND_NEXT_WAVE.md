# Local inference lab handoff — 2026-08-21

**Authority:** current dated handoff for the Qwen3.8 requalification, broad portfolio, artifact assessment,
and next controlled-maintenance wave.

**Repository:** `C:\projects\tare.tools.local-labs`

**Handoff parent HEAD:** `294990b` (`docs(research): make qualification explicitly supersedable`)

**Branch:** `dev/local-labs-relay-q0`, 20 commits ahead of `local-labs/dev/local-labs-relay-q0` before
this handoff commit. Nothing was pushed.

**Incoming-agent rule:** read this document before the older `docs/HANDOFF.md`. The older handoff remains
valuable historical evidence, but several of its deployment statements and closed/dogmatic labels have been
superseded or reopened by the 2026-08-20/21 work.

> **Operational amendment, 2026-08-21:** the repository and fork worktrees were migrated to their canonical
> ecosystem names after this state capture. The text service was intentionally stopped for that maintenance
> window and was later restored by the 2026-08-22 continuation below; the embedding endpoint on port 8081
> remained healthy throughout. Read
> [`PATH_CONTRACT.md`](PATH_CONTRACT.md) for the canonical paths, compatibility aliases, and rollback evidence.

> **Execution continuation, 2026-08-22:** the prioritized Muse, current-Unsloth, and Cold Fusion packets are
> now closed. Muse is `HOLD` with DFlash rejected; neither current Unsloth quant superseded its historical
> peer; Cold Fusion failed its compact base-role gate and did not receive broad promotion stages. A later
> explicitly authorized descriptive nine-cell MTP arm also closed as `MTP_REJECTED`.
> The exact service baseline is restored and healthy: historical Q4_K_XL on 8080, build `b9863-5e7f6271c`,
> 131,072 context, one slot, q4_0/q4_0 KV, MTP n3, 32 checkpoints; embedding 8081 is healthy. Read the three
> result files under `runs/requalification/{MUSE-GLIMMER-2026-08-21,QWEN38-UNSLOTH-REVISION-2026-08-21,COLD-FUSION-2026-08-22}/` before opening another model-candidate wave.

> **Repository-agent and context-policy continuation, 2026-08-22:** `LAB-CODE-003` is no longer
> infrastructure-blocked. Its official gold gate passed and the frozen Qwen3.8/mini-SWE-agent pilot resolved
> 5/10: all five submitted patches passed, while five cases exhausted the fixed 40-call budget with empty
> patches. Docker's temporary loopback API was closed and Desktop stopped afterward. The context-policy
> backlog is also closed: retain 131,072 for exclusive SERVE, with 81,920 as the named 4 GiB-reserve profile.
> See `runs/code/LAB-CODE-003-SWEBENCH-VERIFIED-2026-08-22/RESULT.md` and
> `runs/ops/CANONICAL-CONTEXT-POLICY-2026-08-22/DECISION.md`.

> **Provenance continuation, 2026-08-22:** LAB-PROV-001 now inventories all 32 GGUFs as 11 fully
> pinned, 20 exact upstream receipts pending local hash, and one authorial derivation. The local merge's
> original cache supplied exact receipts for all 31 parent shards; its preserved quantizer embeds commit
> `068764d92` and is binary-hashed. Third-party quantizer builds remain `UNKNOWN` because their cards do
> not disclose exact revisions. See `runs/provenance/LAB-PROV-001-FLEET-2026-08-22/RESULT.md`.

> **Cache continuation, 2026-08-22:** `LAB-CACHE-001` is now closed for explicit no-spec slot persistence
> and blocked for MTP persistent state. The original MTP cache run failed its long-context oracle and its
> first save/restore run persisted the same `!`-only output; four fresh cache replicas and one slot replica
> passed, but correctness gates do not average away a known failure. Paired chat remained 16/16 per arm and
> byte-identical through 32k. See `runs/cache/LAB-CACHE-001-MTP-2026-08-22/RESULT.md`. The canonical service
> baseline was restored afterward; AGENT-002 perturbation robustness is the next unblocked P0 slice.

> **Agent robustness continuation, 2026-08-22:** AGENT-002 is complete and failed 39/40. Rephrasing,
> function renaming, and irrelevant-tool arms passed, but reversing the tool list made irreversible recovery
> emit safe prose instead of calling `check_transfer_status`. Seed-fixed paired evidence was canonical 5/5
> versus reordered 0/5; tool-list-only 0/3 versus schema-order-only 3/3. Preserve canonical ordering and
> enforce unknown-outcome recovery in application control logic. AGENT-003 stress/scale is next.

> **Agent scale continuation, 2026-08-22:** AGENT-003 passed its corrected 16/16 bounded matrix through
> 32 tools, 12 parallel calls, sequential depth 8, and 16 history turns. The first 15/16 artifact is retained
> but explicitly invalid/superseded because its depth-0 fixture required a starting token absent from the
> prompt; the full corrected rerun passed. BigCodeBench Tier-1 is now the next dependency-gated code slice.

> **Practical-code continuation, 2026-08-22:** official BigCodeBench-Hard Instruct is complete at 48/148
> pass@1 = 32.43% (48/147 = 32.65% excluding the sole failed ground truth `/590`, reproducible Wikibooks
> HTTP 403). All 148 generations were nonempty, compilable, syntax-valid, and non-truncated. `/1042`'s
> memory failure reproduced in isolation and is a real model error. See
> `runs/code/LAB-CODE-002-BCB-HARD-2026-08-22/RESULT.md`. Docker Desktop was stopped; 8080/8081 are healthy.

> **Long-context continuation, 2026-08-22:** official RULERv1 task generation/scoring is complete as a
> bounded local endpoint run. The 13-task n=1 pilot scored 82.82% at 64k (preregistered FAIL) and 100% at
> 128k (bounded PASS). Triggered VT/CWE/FWE replication produced 66.7/40.0/88.9% at 64k versus
> 100/100/100% at 128k; four 64k responses exhausted their official output budgets, while all 19 128k
> receipts ended with `stop`. Do not infer that 128k is inherently easier because official generation is
> length-conditioned. See `runs/context/LAB-CTX-002-RULER-V1-2026-08-22/RESULT.md`. Repo-context is next.

> **Repo-context continuation, 2026-08-22:** full LongBench RepoBench-P is complete. Stock raw completion
> was stopped at n=20 after scoring 1.90 with two empty EOS results; a frozen model-native chat amendment
> improved the same sample to 24.55 and triggered the full run. The 500-example result is 39.56 official code
> similarity, below the 55.0 gate, with 109/500 exact first lines. Scores degrade 45.21 (<4k) → 36.78 (4–8k)
> → 14.10 (8k+); Python 49.83 versus Java 30.38. All prompts fit and all 500 chat outputs were nonempty.
> See `runs/context/LAB-CTX-003-REPOBENCH-P-2026-08-22/RESULT.md`.

> **Energy continuation, 2026-08-22:** LAB-ENERGY-002 completed all 24 counterbalanced cells at
> 420/378/336/294 W with the stock voltage curve (no undervolt). At long context, 378 W retained 99.31%
> prefill and 95.66% decode throughput but slightly increased both gross energy metrics; 336/294 W saved
> 3–7% energy while losing 7–18% throughput. No reduced limit met the frozen 95% rule, so 420 W remains
> recommended and deployment defaults were not mutated. The harness restored and verified 420 W; 8080/8081
> remain healthy. See `runs/energy/LAB-ENERGY-002-POWER-CURVE-2026-08-22/RESULT.md`.

> **mmap close-out, 2026-08-22:** the historical −10.4% no-mmap decode penalty is now formally
> `CONFOUNDED`. Six alternating Qwen3.6 MoE/ncmoe=6 pairs gave a warm-cache no-mmap median delta of
> +0.18% with a bootstrap interval including zero. In fresh processes, no-mmap nevertheless reduced total
> elapsed time by median 10.87% (95% CI 3.98–28.29) and avoided the severe initial mmap page-in, so it is
> recommended for that exact MoE profile. This does not transfer automatically to the dense Qwen3.8
> incumbent; no current service default was changed. The canonical service was restored and 8080/8081 are
> healthy. See `runs/close-outs/LAB-CLOSE-001-MMAP-2026-08-22/RESULT.md`.

> **Fable termination close-out, 2026-08-22:** the 32-cell budget/EOS/stop/sampling matrix confirms
> instruct-only bounded safety (8/8 natural stops) but disqualifies thinking-enabled agent use (6/16 natural
> stops, two prompts exhausted both 512 and 2,048). Explicit `</think>` stops produced no final content in
> 4/4, so `finish_reason=stop` alone would have masked the failure. Ignore-EOS instruct exhausted 512 in
> 4/4, confirming the safe instruct results are genuine EOS termination. The experimental service was
> removed and canonical 8080/8081 health restored. See
> `runs/close-outs/LAB-CLOSE-002-FABLE-TERMINATION-2026-08-22/RESULT.md`.

> **Operating-mode continuation, 2026-08-22:** LAB-OPS-001 adds a qualified fail-closed SERVE/LAB lock
> to `tools/benchmarks/lmctl.py`. Ten deterministic tests and live negative/transition checks passed:
> experimental launch in SERVE, LAB transition over active 8080, and 8080 launch in LAB were all refused
> before mutation. Embedding 8081 is explicitly auxiliary in both modes. The persistent machine state is
> now coherent `SERVE`; canonical 8080 and embedding 8081 are healthy. See
> `runs/ops/LAB-OPS-001-MODE-LOCK-2026-08-22/RESULT.md`.

> **Interference continuation, 2026-08-22:** LAB-OPS-002 completed 15 counterbalanced cells against the
> canonical endpoint. Bounded CPU/RAM/disk contenders shifted short-workload prefill by only 4.3–5.1%.
> Same-GPU FP16 matmul reduced decode 7.14% and increased gross prefill energy/token 54.0%, so GPU
> colocation is material even when a throughput-only 10% alarm would not fire. All contenders were cleaned
> up; SERVE mode is coherent and 8080/8081 remain healthy. See
> `runs/ops/LAB-OPS-002-INTERFERENCE-2026-08-22/RESULT.md`.

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
- binary: `/home/augus/src/slop.cpp/build/bin/llama-server`;
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
/home/augus/src/slop.cpp/build/bin/llama-server
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
- binary: `/home/augus/src/slop.cpp/build/bin/llama-server`;
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

Path: `/home/augus/src/slop.cpp-main`

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

The 2026-08-22 official-checkpoint follow-up is `BLOCKED_UPSTREAM`: neither Microsoft RetNet nor
TorchScale publishes a pretrained RetNet checkpoint, and the upstream checkpoint request remains open.
No community checkpoint was relabeled as official. The recurrent-architecture lane continues with the
official `RWKV/RWKV7-1.5B-20260805` checkpoint, which is sized for the RTX 3090.

That RWKV7 follow-up is now complete: `QUALIFIED_MECHANISM / RESEARCH-LOCAL`. The BF16 model used
3.06 GB allocated VRAM, left ~19.7 GiB free, held recurrent state at exactly 12,779,524 bytes from
32 to 1,024 tokens, and matched full vs cached continuation logits exactly. It is not serving-promoted:
first-use compilation/prefill was slow, and the publisher does not assert a license for the weights.
See `runs/requalification/RWKV7-1.5B-20260805-2026-08-22/RESULT.md`.

LAB-VLM-001 is also closed. Four deterministic coding screenshots (stack trace, UI overflow, visual
diff and pytest failure) were added; the resident Gemma-4-12B Vision profile passed 4/4 cases and
20/20 frozen clauses in 1.9–2.3 seconds per request. The old Qwen3-VL-8B profile is still registered,
but its local weight/projector files are absent. See
`runs/vlm/LAB-VLM-001-2026-08-22/RESULT.md`.

The next open-weight breadth arm, official Falcon-H1R-7B Q8 (hybrid Transformer+Mamba2), is
`HOLD_ROLE`: it fit with 14,275 MiB free and passed smoke 4/4, agent/tool 8/8 and the five-case GSM
replay 4/5, but `Mbpp/260` returned empty content at both 2,048 and diagnostic 4,096 token budgets.
Per the frozen dependency gate, context expansion was not opened. The artifact and new
`falcon-h1r-7b-q8` `lmctl` profile remain available for research. See
`runs/requalification/FALCON-H1R-7B-2026-08-22/RESULT.md`.

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

### Priority insertion — Muse Glimmer open-weight breadth scout

**Amendment, 2026-08-21 — executed:** `LAB-MUSE-000` through the compact `LAB-MUSE-002/004` gates ran before the
expensive full-packet and fork-lever phases below. The objective is to add an independent, multimodal,
open-weight option for the RTX 3090 rather than concentrating the fleet entirely in Qwen lineages.

Use the official revision-pinned Muse Glimmer 17 GB quant first. Build llama.cpp `b10353+` in an isolated
experimental worktree because neither the deployed `b9863` lane nor the current fork-main binary registers
the architecture. Test text, vision, and the official target-matched DFlash drafter as additive arms. Do not
mutate the deployed fork or service unit, do not start with the 19.7 GB 32-GB-targeted quant, and preserve the
embedding endpoint on port 8081.

Decision: `HOLD` base candidate, `DRAFT_REJECTED`, no deployed-role promotion. Text/vision fit separately and
the compact context curve passed through 120k, but agent 7/8, `Mbpp/260` non-termination, cache transcript drift,
DFlash output drift, and the combined stack's VRAM-reserve failure invoked the early stop. Packet:
[`research/MUSE_GLIMMER_3090_EXPERIMENT_PACKET_2026-08-21.md`](research/MUSE_GLIMMER_3090_EXPERIMENT_PACKET_2026-08-21.md).
Evidence: [`../runs/requalification/MUSE-GLIMMER-2026-08-21/RESULT.md`](../runs/requalification/MUSE-GLIMMER-2026-08-21/RESULT.md).

### Phase 1 — current Unsloth revision screen — EXECUTED / NO SUPERSESSION

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

### Phase 2 — Cold Fusion practical A/B — BASE REJECTED / DESCRIPTIVE MTP REJECTED

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

## 14. 2026-08-22 autonomous backlog continuation

- LAB-CODE-003 remains 5/10. Prompt-only and corrected duplicate-command middleware follow-ups were
  gated off: even after 29 executions were blocked, the model repeated the same request through 40 calls.
- LAB-AGENT-004 promoted a bounded application policy: unknown irreversible outcome routes immediately
  to an available idempotent status checker, never to a retry or permission question. Evidence was 5/5
  targeted and 16/16 full canonical/reversed.
- Five open-weight candidates were full-hashed and role-screened. Mistral Small 24B Heretic, Gemma 4
  26B Heretic, GPT-OSS 20B, official Gemma 4 26B and newly downloaded Ornith 1.5 35B-A3B all remain
  HOLD on fit, agent or cache gates. Ornith passed fit and agents 8/8 but cache 3/4; no GSM/MBPP
  expansion was spent after a failed gate.
- Fleet provenance is now 31 fully pinned, one content-pinned local derivation, and one explicitly
  isolated ThinkingCap MTP local/upstream digest mismatch.
- Canonical Qwen3.8 SERVE was restored at the end of the 2026-08-22 wave. On 2026-08-23 the user then
  explicitly authorized stopping the idle text service and excluded all soaks. Final verified state:
  `lmctl` is coherent `LAB`, `llm-inference.service` is inactive/dead, port 8080 is unavailable, the
  embedding server alone remains healthy on 8081, and no soak or experimental text endpoint is active.
