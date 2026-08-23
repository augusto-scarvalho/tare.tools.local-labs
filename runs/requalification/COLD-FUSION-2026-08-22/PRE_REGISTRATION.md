# Cold Fusion Qwen3.8 practical candidate packet

**Status:** `PREREGISTERED`  
**Date:** 2026-08-22  
**Hardware:** one RTX 3090 24 GB, WSL2  
**Question:** does the revision-pinned Cold Fusion IQ4_XS artifact provide a better task-correct local option
than the qualified base Qwen3.8 artifacts, and does its embedded MTP head accelerate greedy decoding without
changing answers?

## Frozen artifact

Repository: `DavidAU/Qwen3.8-27B-Cold-Fusion-GAIN-V1.1-NM-DAU-NEO-MAX-MTP-GGUF`  
Revision: `27a5cb2cce434341c2a8a4a50130268e0eccae34`  
File: `Qwen3.8-27B-Cold-Fusion-GAIN-V1.1-NM-DAU-NEO-MAX-NEO-MTP-IQ4_XS.gguf`  
Bytes: `17,033,680,384`  
Expected SHA-256: `523bf4fbe2a2e0ce7aa54f812d85746294483b579443dd6e50e8ab684d7852f9`

Repository drift after the 2026-08-21 assessment is acceptable only because the target blob identity did not
change. Store it in a revision-qualified directory and never overwrite a historical Qwen artifact.

## Controls

- Runtime: deployed llama.cpp `b9863-5e7f6271c`.
- External template: `/home/augus/models/templates/qwen-sharp.jinja` for base-role comparability.
- Full GPU offload, FlashAttention on, q4_0/q4_0 KV, one slot, 65,536 context tokens.
- Deterministic qualification: temperature 0, top-k 1, adequate per-task token budgets.
- Base-role screen uses the candidate with speculation disabled and no vision projector.
- The embedding service on 8081 remains untouched.
- The deployed model and unit file remain unchanged; restore the exact baseline after each maintenance tranche.

## Stage A — admission and compact base-role gate

1. Verify bytes and full SHA-256, then inventory GGUF architecture, quant types, tokenizer/template metadata,
   output tensor, and embedded `nextn`/MTP tensors.
2. Load at 65k and require at least 4,096 MiB free VRAM.
3. Run deterministic smoke, agent 8-case suite, and cache/cancel/reuse 4-case suite.
4. Run retrieval, multikey, multihop, and aggregation at 8k/32k/64k, one replicate per cell.
5. Run `Mbpp/260` with 2,048 tokens and the five historical GSM8K failures with 512 tokens.
6. Run a fixed known-answer reasoning control in instruct/off, low, medium, and xhigh modes, recording
   correctness, reasoning tokens, total tokens, termination, and latency.

The candidate is ineligible if it violates the 4 GB reserve, fails any critical agent or cache oracle,
introduces a repeatable context cliff, blindly retries an irreversible action, or shows non-termination on
the compact discriminators. GSM failure replay must score at least 3/5 to substantiate a useful reasoning
shift before the broad packet.

## Stage B — broad correctness

Only a Stage-A survivor receives:

1. all 378 MBPP+ tasks at the historical 768-token cap with EvalPlus 0.2.0;
2. the frozen 100-item GSM8K manifest at 512 tokens;
3. three replicates of all four context families at 8k/32k/64k if the compact matrix contains any miss.

For a base-role promotion, agent and cache must remain perfect; MBPP Base must be at least 324/378 and Plus
at least 281/378; GSM8K must be at least 94/100; and no repeatable new context failure may appear. These are
the same small non-inferiority tolerances used for the current-Unsloth IQ4 screen. A claimed reasoning-token
reduction is beneficial only when strict correctness and termination remain non-inferior.

## Stage C — same-file MTP A/B

Only a Stage-B survivor may run the exact same candidate file with:

- speculation off;
- embedded MTP at `n-max=2`;
- embedded MTP at `n-max=3`.

Use fixed prompts, greedy decoding, alternating arm order, at least three measured replicates, and equal
output budgets. Record task-correct latency, decode throughput, draft acceptance, output hashes, and VRAM.
An MTP arm can promote only if answers remain byte-equivalent across arms, correct within every arm,
deterministic within arm, and materially faster in median task-correct latency or throughput. Acceptance
alone is not a win.

## Stop and restoration rules

- Stop later stages immediately when their eligibility prerequisite fails.
- Preserve invalid attempts and distinguish harness/runtime faults from model failures.
- Do not download additional Cold Fusion variants or a vision projector during this packet.
- Do not modify the service unit, deployed artifact path, or fork defaults.
- End with port 8092 closed, port 8080 restored to its exact historical argv, and port 8081 healthy.
