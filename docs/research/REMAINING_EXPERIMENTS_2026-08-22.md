# Remaining experiment register — 2026-08-22

This is the reconciled queue after the autonomous 2026-08-21/22 wave. It distinguishes work that is
ready for a design packet from work that must not be launched without a dependency or trigger.

## Ready for execution

None. All dependency-free non-soak experiments have run or closed at their frozen gate. The user
clarified on 2026-08-23 that soak experiments are excluded; do not launch 24/48/72-hour campaigns.

## Blocked on a concrete dependency

| Item | Blocker | Unblock trigger |
|---|---|---|
| LAB-PROV-001 third-party quantizer builds | The retained model cards and metadata do not disclose exact llama.cpp/quantizer commits for 31 upstream artifacts | Publisher build receipt; keep `UNKNOWN` and hash only newly admitted or promotion-relevant files |
| LAB-PROV-001 ThinkingCap MTP identity | The local 17,221,641,152-byte file has SHA `b0987c4e...`, while revision `f015d8b` identifies a 16,810,713,408-byte file with SHA `0ba445d2...` | A publisher/download receipt that identifies the local content; do not infer or overwrite |
| LAB-JUDGE-001 human calibration | Requires 50–100 blind human preference labels; model-generated labels are not human calibration | Human rater availability and frozen blind packet |
| LAB-CACHE-001 MTP persistence | Intermittent restored-state correctness failure remains reproducible enough to block promotion | Root-cause hypothesis and a new falsifiable cache-lifecycle packet |
| RetNet official checkpoint | Microsoft/TorchScale publish implementation but no pretrained official checkpoint | Official upstream checkpoint release |
| RWKV7 deployment | Tested release does not assert a license for the weights | Publisher-asserted weight license plus serving-quality packet |
| ThinkingCap Qwen3.8 arrival watch | BottleCapAI still exposes ThinkingCap on Qwen3.6, not an official Qwen3.8 release | Official Qwen3.8 ThinkingCap weights plus a 3090-fit GGUF |
| LAB-REL-002 48/72 h soak | Explicitly excluded by the user; both the historical partial and mistaken 2026-08-23 partial are cancelled and not PASS | A new explicit direction that includes soak experiments |

## Parked or deliberately not auto-launched

- LAB-SERVE-001d closed-loop TPOT isolation: reopened and stopped `BLOCKED_RUNTIME_CRASH` after the
  MTP-off N=4 workload and a no-CUDA-graphs recovery both hit CUDA illegal memory access.
- LAB-ENGINE-001/002 closed `COMPARABLE_COMPLETE / VLLM_COMPLETE`: official Qwen3-4B revision
  `1cfa9a7...` ran in BF16 across llama.cpp, SGLang and vLLM. SGLang/vLLM led fresh prefill by about
  18%; decode was unresolved; llama.cpp used much less VRAM and started much faster.
- Full Muse Glimmer VQA/injection expansion closed after explicit authorization: 107/150 MMStar and 5/5
  bounded safety cases. DFlash remains rejected on the already-measured output-equivalence and full-stack
  reserve gates; overall Muse disposition remains HOLD.
- Cold Fusion embedded MTP A/B closed descriptively after explicit authorization: all nine runtime cells
  passed, but output equivalence and uniform task-correct speed failed; decision `MTP_REJECTED`.
- Full training/distillation, custom CUDA kernels without a measured bottleneck, distributed serving,
  sub-4-bit KV, Kubernetes and product integration remain parked by the backlog policy.
- B5 task-oriented Q3/mixed quantization is superseded by the completed seven-quant Qwen3.8 frontier:
  code and hard math remained flat through Q2_K_XL, while long-context retrieval identified IQ2_M as
  the actual cliff. A new layer-sensitivity build has no unresolved decision to answer.

## Newly closed in this continuation

- LAB-IMG-001 Qwen-Image: official pinned NF4/BF16-offload pipeline fit on the 3090, replayed
  byte-identically, and scored 10/13 frozen semantic clauses. Typography was exact; dashboard spelling
  and 3D-shape semantics held quality. Verdict `FIT PASS / MECHANISM PASS / QUALITY HOLD`.
- LAB-IMG-002 SDXL: matched official FP16 baseline was about 9.5x faster and used about 0.7 GiB less
  peak inference VRAM, but scored only 3/13 semantic clauses. Verdict `QUALITY REJECT`; retain Qwen-Image
  as the research candidate, promote neither.
- LAB-HARNESS-001..003: TaskContract/evidence-pack/baseline gate, independent mutation-test writer,
  deterministic maintainability gate, and independent critic all completed. Evidence packs reduced
  tokenizer-measured context 83.85%; mutation testing killed 6/7 seeded defects; critic classified
  8/8 frozen patches with zero unsafe accepts. Bounded primitives pass; no product integration implied.
- A2 Stage-2 E1/E2: explicit authorization opened the optional purism leg. Corrected answer-channel
  baseline was 25/32 harmful and 0/32 harmless refusals, but 0/44 candidate directions passed the
  induction and KL gates (best KL 0.566 vs <0.1). Verdict `G0 KILL`; no weight edit or merge was built.

- LAB-PROV-002: official Qwen3.8-27B was pinned and verified 18/18, converted to BF16, and requantized
  with pinned llama.cpp plus the exact Unsloth imatrix. Provenance is closed, but parity was rejected:
  the authorial IQ4_XS was 5.82% larger, used 680 MiB more VRAM, truncated `Mbpp/260`, and changed
  deterministic generations despite small prefill/decode gains. Retain Unsloth UD-IQ4_XS.
- RetNet official checkpoint lookup: `BLOCKED_UPSTREAM`, without relabeling community weights.
- RWKV7 1.5B official release: `QUALIFIED_MECHANISM / RESEARCH-LOCAL`, deployment license blocked.
- LAB-VLM-001: PASS, 4/4 screenshot cases and 20/20 frozen clauses.
- Falcon-H1R-7B official Q8: `HOLD_ROLE`; fit/tools/GSM passed, coding termination failed at 2k and 4k.
- LAB-OPT-001: qualified six-cell Optuna screen; `n4/ub1024` screen winner, but deploy decision withheld after exact-control reconciliation.
- LAB-OPT-001b: aborted by the frozen 4 GiB gate because the exact 131k control left 2,782 MiB free; no challenger/default change.
- LAB-OPS-003: qualified context/VRAM ladder; 81,920 is the largest tested point that preserves 4 GiB, while the unchanged 131k service leaves ≈2.8 GiB.
- LAB-PROV-001 fleet pass: all 33 GGUFs inventoried; 31 fully pinned, one authorial derivation content-pinned, and one ThinkingCap MTP local/upstream digest mismatch explicitly isolated. The authorial merge has exact 31-shard parent receipts plus quantizer commit/binary hash. Third-party quantizer builds remain upstream-undisclosed.
- LAB-SERVE-002: promotion packet frozen and decided `HOLD_MODEL_DRIFT`; historical Qwen3.6 MoE load guidance was not transferred to the live Qwen3.8 service, and no default changed.
- LAB-CODE-003: the official gold gate passed and the frozen Qwen3.8/mini-SWE-agent pilot resolved 5/10. All five submitted patches resolved; the other five exhausted 40 calls with empty patches. Infrastructure was restored and Docker stopped afterward.
- LAB-CODE-004/005/005B: prompt-only and corrected duplicate-command middleware failed their three-case gate; the middleware blocked 29 repeat executions but did not change the empty-patch outcome.
- LAB-AGENT-004: promoted the bounded irreversible-recovery policy after 5/5 targeted and 16/16 full canonical/reversed passes with zero blind retry.
- Resident open-weight breadth: Mistral Small 24B Heretic, Gemma 4 26B Heretic, GPT-OSS 20B, official Gemma 4 26B and newly admitted Ornith 1.5 35B-A3B were full-hashed and screened; all five remain HOLD on fit, agent or cache gates. Ornith was the strongest agent result at 8/8 but stopped at cache 3/4.
- Canonical context policy: retain 131,072 for exclusive SERVE because it has direct bounded 128k quality evidence; keep 81,920 as the 4 GiB-reserve profile. No live mutation was required.
