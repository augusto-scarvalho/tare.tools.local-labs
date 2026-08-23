# LAB-COLD-FUSION-002 — descriptive embedded-MTP A/B

Status: **FROZEN BEFORE GENERATION**  
Date: 2026-08-22

The user explicitly reopened the remaining experiment backlog. This packet therefore runs the formerly
dependency-gated mechanism A/B as descriptive evidence. It does not waive the Cold Fusion base-role failure
and cannot promote the candidate or deployment default.

## Frozen controls

- Candidate: `Qwen3.8-27B-Cold-Fusion-GAIN-V1.1-NM-DAU-NEO-MAX-NEO-MTP-IQ4_XS.gguf`.
- Revision: `27a5cb2cce434341c2a8a4a50130268e0eccae34`.
- SHA-256: `523bf4fbe2a2e0ce7aa54f812d85746294483b579443dd6e50e8ab684d7852f9`.
- Runtime: the same llama.cpp `b9863-5e7f6271c` lane used by Stage A.
- Common configuration: full GPU offload, Flash Attention, 32,768 context, one slot, q4_0 KV,
  external `qwen-sharp.jinja`; embedding 8081 remains resident.
- Arms differ only by embedded speculation: off, `draft-mtp n-max=2`, `draft-mtp n-max=3`.
- Greedy raw completion, top-k 1, seed 42, cache disabled, fixed 384-token maximum.

## Workload and design

Three fixed tasks cover exact arithmetic, bounded Python code, and red-black-tree prose. Each task has an
explicit correctness oracle. Run three counterbalanced server-level macro blocks, each containing all arms,
for three measured replicates per task and arm. Preserve startup argv/logs, load time, residency, content,
hash, timings, acceptance and correctness.

An MTP arm is mechanism-qualified only if every task is correct, deterministic within arm, byte-identical to
off for every matched task/replicate, and median task-correct decode throughput improves by at least 10%.
Otherwise report `MTP_REJECTED` or `MTP_UNRESOLVED`. The base candidate remains `REJECT_BASE_ROLE` regardless.

