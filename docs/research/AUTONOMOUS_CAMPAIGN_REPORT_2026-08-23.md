# Autonomous experiment campaign report — 2026-08-21 to 2026-08-23

## Executive outcome

The authorized non-soak backlog was executed through its dependency gates. The campaign expanded the
RTX 3090 evidence base across open-weight model candidates, multimodal/image generation, agent behavior,
coding, long context, serving engines, energy, provenance, harness-product primitives, and mechanistic
ablation. It did not produce a new general-purpose replacement for the incumbent Qwen3.8 deployment,
but it removed substantial uncertainty and produced several bounded operational improvements.

There are no remaining dependency-free non-soak experiments in the reconciled queue. Remaining items
require upstream artifacts, human labels, a new falsifiable cache hypothesis, publisher licensing, or a
new explicit instruction to re-open soaks. The canonical queue is
[`REMAINING_EXPERIMENTS_2026-08-22.md`](REMAINING_EXPERIMENTS_2026-08-22.md).

## What advanced and why it matters

| Track | Result | Practical value |
|---|---|---|
| Current Qwen3.8 Unsloth revision | Current IQ4_XS and Q2_K_XL rejected for supersession | Prevented an unearned upgrade based on freshness alone; historical incumbent evidence remains stronger |
| Muse Glimmer | Overall `HOLD`; 107/150 MMStar and 5/5 bounded multimodal-safety cases; DFlash rejected | Added the strongest measured VQA specialist in the local panel without misclassifying it as a safe general agent or full-stack deploy |
| Cold Fusion | Base role rejected; later nine-cell MTP arm rejected | Showed that large task-dependent speedups do not compensate for output drift, tiny-answer slowdown, truncation, and base-quality failures |
| Resident open-weight breadth | Mistral, two Gemma variants, GPT-OSS, and Ornith all `HOLD` | Replaced model-card speculation with fit, agent, cache, reasoning, and coding evidence on the actual 3090 |
| Recurrent/hybrid models | RWKV7 mechanism qualified but deploy/license blocked; Falcon-H1R `HOLD_ROLE`; RetNet upstream-blocked | Established constant-state viability while preserving the distinction between mechanism evidence and deploy eligibility |
| Image generation | Qwen-Image 10/13 and deterministic replay, `QUALITY HOLD`; SDXL 3/13, `QUALITY REJECT` | Identified Qwen-Image as the research candidate and quantified the speed-versus-semantic-quality trade-off |
| Visual coding | Gemma-4-12B Vision passed 4/4 fixtures and 20/20 clauses | Validated a bounded screenshot-analysis role beyond basic OCR |
| Agent robustness | Corrected stress/scale 16/16; irreversible recovery policy 16/16; positional tool-order weakness isolated | Converted a fragile behavior into an explicit safe policy without hiding the underlying model sensitivity |
| Agent harness product | 83.85% token reduction, 5/5 required-file recall, 6/7 mutations killed, critic 8/8 | Demonstrated that digest-bound contracts, structural evidence, independent tests, and criticism can reduce context while retaining fail-closed checks |
| Code evaluation | BigCodeBench-Hard baseline 48/147 adjusted; mini SWE-bench pilot 5/10; repeat-command guards did not improve empty-patch cases | Added harder, repository-oriented evidence and rejected a plausible but ineffective middleware fix |
| Long context | RULER bounded 128k pass; RepoBench-P 39.56 quality fail with sharp 8k+ degradation | Separated retrieval capacity from useful repository completion quality |
| Cache correctness | No-spec persistence passed; intermittent MTP restore failure retained | Prevented ephemeral speculative-decoding speed from being generalized to persistent-state correctness |
| Serving engines | SGLang/vLLM led fresh prefill by about 18%; llama.cpp used less VRAM and started faster; decode unresolved | Established regime-specific trade-offs instead of declaring a universal engine winner |
| Serving isolation | Closed-loop TPOT run stopped on reproducible CUDA illegal-memory-access paths | Preserved a runtime crash as a blocker rather than promoting incomplete throughput data |
| Energy and interference | Retain 420 W; CPU/RAM/disk below the frozen threshold; GPU colocation materially increased energy | Avoided a slower power cap that missed the Pareto gate and justified exclusive same-GPU operation |
| Context policy | Retain 131,072 for exclusive SERVE; name 81,920 as the 4 GiB-reserve profile | Reconciled measured 128k utility with an explicit reserve-preserving alternative |
| Artifact provenance | 33 GGUFs inventoried; authorial lineage closed; one ThinkingCap MTP mismatch isolated | Makes promotion claims traceable and prevents local/upstream identity from being inferred |
| Authorial requant | Source/build lineage closed; parity rejected despite small speed gains | Showed that reproducibility of construction does not imply behavioral or operational parity |
| A2 Stage-2 refusal direction | 0/44 eligible directions; `G0 KILL`; no weight edit | Saved all downstream conversion, merge, and evaluation cost after a decisive causal gate failure |
| Close-outs | no-mmap decode residual confounded; Fable thinking termination disqualified | Closed two longstanding questions with bounded recommendations instead of carrying ambiguous backlog |

## Decisions that changed operations

- Keep the incumbent Qwen3.8 deployment and current canonical 131,072 context profile unchanged.
- Use the 81,920 profile when a task explicitly requires at least 4 GiB of free VRAM.
- Keep same-GPU experimental/judge workloads isolated from canonical serving; embedding port 8081 remains
  an allowed independent auxiliary.
- Retain 420 W as the board-power recommendation under the frozen throughput guardrail.
- Treat MTP as qualified for ephemeral decode only; persistent slot/cache restoration remains blocked.
- Apply the idempotent status-check policy immediately after an unknown irreversible action outcome and
  never blindly retry the irreversible action.
- Keep Qwen-Image as a research candidate, not a promoted production image model.

## Negative results that saved future work

The campaign's most useful outcomes were often early stops. Five compact open-weight text candidates did
not clear the full role gates; Muse's DFlash speedup did not preserve output equivalence or full-stack
reserve; Cold Fusion's embedded MTP was not uniformly task-correct; SDXL missed the semantic gate; A2 had
no eligible causal direction; repeated-command middleware did not fix mini-SWE empty patches; reduced
power caps missed the throughput guardrail; and the authorial requant failed parity. These results prevent
larger context, quality, conversion, or deployment campaigns from being spent on candidates that already
failed a cheaper discriminating gate.

## Evidence and repository policy

- Human-readable decisions live in each run's `RESULT.md`; raw JSON/JSONL receipts remain alongside them.
- Superseded and invalid attempts are retained and explicitly labeled rather than deleted.
- Runtime logs, PID files, model weights, adapters, and PyTorch activation tensors are not repository
  artifacts. The A2 tensors remain local and are represented by
  [`LOCAL_ARTIFACTS.sha256`](../../runs/a2/stage2-2026-08-22/LOCAL_ARTIFACTS.sha256), preserving their
  byte counts and SHA-256 identities without adding 813.78 MiB to Git.
- The cancelled reliability observations remain incomplete and are not classified as soak passes.

## Remaining experiments

No non-soak item is ready for unconditional execution. The residual queue is:

1. Third-party quantizer provenance, blocked on publisher receipts.
2. ThinkingCap MTP identity, blocked on a receipt matching the local digest.
3. Human-judge calibration, blocked on 50–100 frozen blind human labels.
4. MTP persistence root cause, blocked on a new falsifiable cache-lifecycle hypothesis.
5. RetNet, blocked on an official pretrained checkpoint.
6. RWKV7 deployment, blocked on an asserted weight license and a serving-quality packet.
7. ThinkingCap Qwen3.8, blocked on an official release and 3090-fit artifact.
8. Reliability soaks, explicitly excluded and cancelled; re-open only on a new explicit instruction.

Product/cloud/cluster builds and open-ended training remain parked because they are not local experiments
with a current decision gate.

## Verification boundary

Repository CI is deterministic and CPU-only: Python compilation, unit tests, and the LAB-QA-001 harness
self-test. GPU/model experiments are intentionally represented by immutable run receipts rather than
re-executed in every CI run.
