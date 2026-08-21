# Local inference experiment portfolio — 2026-08-20/21

This packet resumes the lab broadly after the earlier Qwen3.8/Liger-only requalification. It records
completed evidence, invalid/superseded attempts, current blockers, and the one long-running campaign.
No remote push or deployment-default change was performed.

## Frozen live substrate

- Qwen3.8-27B IQ4_XS, GGUF SHA-256
  `9fd40d7036f5e0918e20aaeebf11468fafd06bb53d4d980eef6bb7e4e4ace666`
- llama.cpp `5e7f6271c06b9104862ab799278a1b7f1323a449` (`b9863`)
- 32,768 context, four slots, Q8_0 K/V, greedy, no speculative decoder
- systemd unit `llm-inference.service`, `Restart=always`; server healthy after all completed work

## Completed campaigns

| Campaign | Result | Evidence / interpretation |
|---|---|---|
| LAB-AGENT-001-v2 | **8/8** | Selection, nested args, abstention, parallel, sequential, multi-turn, error recovery and irreversible-no-blind-retry all dispatch correctly. `runs/agent/LAB-AGENT-001-v2/`. |
| LAB-CACHE-001 live slice | **4/4** | Cold and cached answers byte-identical and oracle-correct for divergent suffix, partial removal, cancel/reuse and long context. Warm `cache_n`: 11,039; 12,113; 13,915; 24,552. `runs/cache/LAB-CACHE-001-v2/*r3.json`. |
| LAB-ENERGY-001 | **instrument qualified** | Alternating-order medians: prefill 0.206 J/token (~2.7k) and 0.262 J/token (~13.2k); decode 8.80 and 9.52 J/token; 42.1 and 39.5 t/s; peak 385.3 W, 72 C. `runs/energy/LAB-ENERGY-001/*r2.json`. |
| LAB-CODE-001 MBPP+ | **full 378** | EvalPlus base 326/378 = 86.24% (Wilson 82.40–89.35); Plus 284/378 = 75.13% (70.54–79.22). All 378 answered/fenced. `runs/code/LAB-CODE-001-MBPP-plus/qwen38-iq4xs-full378/`. |
| LAB-CTX-001 local slice | **paired, bounded** | Retrieval/multikey/multihop 18/18 at 8k/16k/28k. Aggregation expanded to n=10 per length: 10/10, 9/10, 10/10. One seed repeats the same 217→209 error at 16k but passes at 8k/28k: positional sensitivity, not monotonic collapse. `runs/context/LAB-CTX-001-v2/`. |
| LAB-PROV-001 core | **QA qualified, fleet partial** | Identity schema now records full artifact hash/bytes/source/revision/quantizer/imatrix/class; invalid lineage class fails closed. Harness QA 23/23. Current IQ4_XS fully pinned; upstream source revision remains UNKNOWN. |
| RNN-09 RetNet mechanism | **7/7 gates** | Parallel/recurrent max error 4.27e-14; chunkwise, save/reload and isolation bit-exact; gradient max 3.56e-15; fp32 finite at T=4096; explicit stale-state leak detected (47.08) and reset recovered exactly. Mechanism only, not checkpoint reproduction. `runs/rnn/RNN-09-retnet-retention/`. |

## Fragility findings and superseded attempts

1. Cache long-context attempt 1 exceeded n_ctx because a high-entropy nonce was repeated in every filler
   line. Attempt 2 exposed a 16-token answer-budget truncation. Both are preserved; corrected r3 passes 4/4.
2. Energy attempt 1 used nearest-sample power at phase boundaries and fixed short→long order. Its self-test
   failed. Linear boundary interpolation and alternating order were frozen before the qualified r2 rerun.
3. The first context matrix changed random facts with context length, confounding length and problem. It is
   exploratory only. The paired r3 repeats identical facts across lengths; the aggregation n=10 expansion
   was fixed after that task became discriminating.
4. `Mbpp/260` generated runaway analysis inside its code docstring and hit both 768 and 2,048 tokens.
   It is a confirmed termination failure, not a small-budget artifact.
5. EvalPlus rejected direct n=50 scoring because MBPP+ requires all 378 IDs. The qualified scorer pads
   missing IDs with guaranteed failures, busts stale result caches, then reports only the selected denominator.

## Running and blocked

- **LAB-REL-001 24h is RUNNING** from `2026-08-21T03:01:35Z` (00:01:35 BRT), PID recorded in
  `runs/reliability/LAB-REL-001-24h-2026-08-21/soak.pid`. It rotates the eight agentic cases and a periodic
  long known-answer control every 60 seconds, with health/RAM/VRAM/power/temp receipts. Do not classify
  before the full duration ends.
- Explicit slot file save/restore is implemented, but the live unit cannot be stopped by the current user:
  `systemctl stop llm-inference.service` requires interactive authentication and direct PID termination is
  immediately undone by `Restart=always`. The baseline was left healthy. The same blocker prevents clean
  no-mmap and engine-swap launches in this session.
- Speculative/MTP rollback is not covered because the frozen server reports `speculative=false`.

## Next autonomous queue

1. Close LAB-REL-001 only after 24 elapsed hours; if clean, define the envelope before any 48/72h run.
2. With authorized service control, run slot save/erase/restore, speculative rollback, and paired mmap on/off;
   restore the exact baseline argv afterward.
3. Add agent perturbation robustness and stress scale, then BigCodeBench Tier-1 rather than another easy code
   smoke.
4. Run official NVIDIA RULER plus repo-context under a 64k/128k launch; keep local RULER-inspired results
   explicitly non-comparable to upstream scores.
5. RetNet next step is an official small checkpoint reproduction. TPTT RNN-08b remains parked because its
   corrected, valid experiment already showed quality regression and ~9.6x training cost.
