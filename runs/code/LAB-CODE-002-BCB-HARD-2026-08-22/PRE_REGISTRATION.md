# LAB-CODE-002 BigCodeBench-Hard Tier-1

**Status:** `PREREGISTERED`  
**Date:** 2026-08-22  
**Model:** historical Qwen3.8-27B Q4_K_XL on canonical slop.cpp service `b9863-5e7f6271c`  
**Official package commit:** `09dd993f46c3fbf3a799465bb96d524edcb0b199`  
**Dataset:** BigCodeBench-Hard v0.1.4, 148 tasks, 1,343,732 bytes, SHA-256
`cee31f14f29927ca276744b15da05e80ea4d06f0724e6053e3aa6ce17c5b6e7c`

## Protocol

Use the official `instruct` prompts and sanitizer. Generate one greedy sample per task, temperature 0,
top-p 0.95, seed 0, 1,280-token maximum, thinking disabled according to the qualified deployment policy,
and batch size one. Preserve raw and sanitized solutions plus per-task hashes, usage, timings, and finish
reason. This is local model evaluation; no leaderboard submission is authorized.

Execute generated code only inside the official BigCodeBench evaluation container. First verify official
ground truths, then score the samples. Do not execute model-generated samples directly on Windows or the
host WSL environment.

## Dependency-gated stages

1. **Pilot generation:** 12 deterministic spread indices
   `0,13,26,39,52,65,78,91,104,117,130,147`.
2. **Pilot infrastructure gate:** official syncheck/scorer must accept the artifact, ground truths must pass,
   and at least 6/12 samples must be syntactically valid. Functional score is descriptive at this stage.
3. **Full Hard:** if infrastructure is sound, resume the same file to all 148 tasks and run official local
   sandbox evaluation for pass@1.

Any dataset/ground-truth/scorer mismatch blocks quality interpretation. Report answered, fenced/sanitized,
syntax-valid, timeout/non-termination, and functional pass counts separately. The active result must bind
package commit, dataset hash, model identity, and exact sample file hash.
