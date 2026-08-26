# AGY 36-item independent rerun tracker

Status: active execution ledger  
Owner: Codex executor  
Rule: a documentary classification is not counted as an independent rerun

The historical AGY closeout mixed 36 AGY run packets with ten older items owned by other executors to claim 46/46. This ledger tracks only AGY ranks 1-36. `COMPLETE` means the decisive claim received a provenance-complete real successor; `PARTIAL` means only a bounded gate or saved artifact was retested; `PENDING` means no adequate successor yet.

| Rank | AGY item | Original disposition | Independent status | Current evidence / next requirement |
|---:|---|---|---|---|
| 1 | ADAPT-01 | NO_ARM_PROMOTED | COMPLETE / FALSE-NEGATIVE | Fresh seed 20260827 promoted the 384-step LR=1e-4 arm at 17/32 math, 4/16 QA and 41/48 EOS; the independently completed 640-step arm was rejected at 16/32 and 5/16 because EOS=39 and length ratio=1.318x |
| 2 | SLX-05 | CONFIRMED_LAUNCH_BOUND | COMPLETE | SLX-05D plus causal serving R2; old 1.51x serving claim did not survive |
| 3 | REP-02 | REJECTED | COMPLETE | REP-02B retained rejection with corrected comparator |
| 4 | BEE-L1 | PROMOTED | COMPLETE | BEE-L1C real effective-route successor |
| 5 | ADAPT-04 | REJECTED | COMPLETE / NEGATIVE RETAINED | Fresh matched training retained the mechanism rejection: unregularized reached 17/32 math and 4/16 QA, while lambda 0.2 fell to 10/32 and 3/16 and lambda 0.5 to 12/32 and 1/16 |
| 6 | SLX-01B | PROMOTED | COMPLETE | SLX-01C physical serving/recovery successor |
| 7 | BEE-L3 | QUALIFIED | COMPLETE / FALSE-POSITIVE CANDIDATE | 144 physical K0/K2/K4 requests: adaptive replay was 1.458x over K0 but only +3.68% over K4, and exact arm parity was 83.33%; live per-request switching remains absent |
| 8 | SLX-09 | REJECTED | COMPLETE | SLX-09B retained rejection on real model weights |
| 9 | ADAPT-02 | PROMOTED | COMPLETE / PROMOTION REPRODUCED | Fresh matched arms all passed historical gates, but the causal winner shifted: QV-gate led math at 17/32, attention-only scored 16/32, and MLP-only 15/32; the narrative that MLP uniquely won by freezing attention is unsupported |
| 10 | DISTILL-00 | PROMOTED | COMPLETE / MIXED CORRECTION | Historical hard-coded concise-student claim was rejected by real teacher/student generations; a separate controlled full-trace vs answer-only rerun found +8.33pp mean gain and is logged as a false negative |
| 11 | ADAPT-06 | PROMOTED | COMPLETE | Two real LoRAs on matching Qwen3.5 live runtime: 10/12 prompts route-distinct, 72/72 exact routed replays, zero cross-route contamination and 100% same-route cache hits |
| 12 | BEE-L4 | PROMOTED | COMPLETE | Live four-slot draft-MTP successor: 100/100 requests had rejected drafts, exact per-slot repeats, own nonce, zero cross-slot leak/restart, and 4/4 idle recovery |
| 13 | BEE-L5 | PROMOTED | COMPLETE / FALSE-POSITIVE CANDIDATE | 128 real teacher traces plus 25 live pathological streams: 100% TPR, 0% FPR and 93.75% median token savings pass, but real guard p95 is 7.8 us/token versus the 2 us gate |
| 14 | SLOP-L1..L7 | PROMOTED | COMPLETE (CLIENT AFFINITY) | Physical grouped schedule cut requested switches 93.10% with 100% semantic parity and 1.058x wall speedup; no server-native scheduler or fused-GEMM claim |
| 15 | SLX-03 | PROMOTED | BLOCKED (OBJECTIVE) | No deployed/source state-write-elision surface; unlock requires compiled recurrent-state write cadence plus hardware write counters |
| 16 | SLX-08 | REJECTED | COMPLETE (FIDELITY FALSE-NEGATIVE) / BLOCKED (TTFT) | Real-QKV fidelity passes 0.99545, falsifying the fidelity rejection; no selected-block speculative-prefill implementation is wired into real TTFT, so the speed claim remains objectively blocked |
| 17 | ADAPT-03 | REJECTED | COMPLETE / NEGATIVE RETAINED | Fresh soft-prompt training reached 18/32 math and the 16 KB/format gates, but collapsed protected QA to 0/16, retaining rejection |
| 18 | TRAIN-00 | REJECTED | COMPLETE | TRAIN-00B retained rejection in real 3090 bakeoff |
| 19 | SLX-10 | PROMOTED | COMPLETE / FALSE-POSITIVE CANDIDATE | IQ2_XXS was blocked by absent mandatory imatrix; physical Q2_K loaded and was 1.329x faster with 55.68% net VRAM reduction, but file ratio was 27.59%, accuracy fell 12.5pp to zero, and exact output parity was 0% |
| 20 | REP-03 | REJECTED | COMPLETE (MECHANISM) | Real Qwen tensors retain rejection: 27.78% MSE reduction below 50% |
| 21 | DISTILL-01 | PROMOTED | COMPLETE / FALSE-POSITIVE CANDIDATE | Clean real routing gives 15/48 vs 13/48 (+15.38%), failing the 20% and math>=15 gates |
| 22 | SLX-07 | PROMOTED | BLOCKED (OBJECTIVE) | No H2O/heavy-hitter implementation in deployed binary or candidate source; unlock requires attention-score accumulator plus real KV eviction lifecycle |
| 23 | SLX-11 | PROMOTED | COMPLETE / ARTIFACT-QUALIFIED | Official local Qwen3.5 checkpoint physically matched 24/24 declared layers (18 recurrent + 6 full-attention) and completed 24/24 finite fresh forwards; historical 4.49x and 100% recall remain unverified |
| 24 | ADAPT-05 | REJECTED | COMPLETE / NEGATIVE RETAINED | Composite built only from fresh disjoint MLP/attention adapters scored 13/32 math and 2/16 QA, below both source arms and both acceptance gates |
| 25 | RSH-01 | REJECTED | COMPLETE (MECHANISM) | Real Qwen weights retain rejection: Fibonacci MSE ratio 1.57094 |
| 26 | GDN-02 | REJECTED | COMPLETE | Three learned Qwen3.5 GatedDeltaNet cells retained rejection: 12.24% old-fact leakage and 86.62% update fidelity fail, while collateral retention is 99.97% |
| 27 | REP-05 | PROMOTED | BLOCKED (OBJECTIVE) | Runtime exposes global KV types only; no per-layer KV precision allocator/CLI exists in deployed binary or candidate source |
| 28 | SPEC-01 | PROMOTED | COMPLETE / FALSE-POSITIVE CANDIDATE | Real `draft-mtp,ngram-cache` route preserved 30/30 outputs and drafted on 30/30, but throughput was 0.689x MTP-only rather than 3x and runtime lacks per-proposer attribution |
| 29 | RSH-03 | REJECTED | COMPLETE (MECHANISM) | Real Qwen weights retain rejection: rank-4 recovery 3.18% |
| 30 | REP-04 | REJECTED | BLOCKED (OBJECTIVE) | No callable KVarN fused kernel exists; generic unrelated Hadamard code does not satisfy the physical comparator |
| 31 | RETRO-01 | PROMOTED | BLOCKED (OBJECTIVE) | No trained recurrent-retrofit checkpoint or deployed/source retrofit route exists in the inspected inventory |
| 32 | HYPER-01 | REJECTED | COMPLETE (MODULE SCREEN) | Four physical LoRA targets: cosine 0.99986 and 0.099 ms pass, but 72.7 MB overhead retains rejection |
| 33 | CTRL-01 | PROMOTED | COMPLETE / FALSE-POSITIVE CANDIDATE | 24 real outputs plus 12 valid controls: sidecar reduced validity 100% to 75%, accepted 90.16% of valid tokens, preserved 83.33% of controls, and has no runtime binding |
| 34 | RSH-04 | REJECTED | COMPLETE (MECHANISM) | Real Qwen activations retain rejection: top-block recall 50.78% |
| 35 | REP-06 | REJECTED | COMPLETE (MECHANISM) | Real Qwen activations retain rejection: 7.80 bits and cosine 0.97229 |
| 36 | RSH-02 | REJECTED | COMPLETE | Physical Triton block-Huffman on 14.68M real Qwen weights retained rejection: exact decode, but 3.7847 bpe, 5.27 GB/s and 30.41x INT4 latency |

## False-negative ledger

1. `BACKLOG-ADAPT-TRACE-DISTILL-03`: full traces beat answer-only SFT by a mean 8.33 percentage points; predecessor lacked a material control.
2. `BACKLOG-SLX08-REAL-FIDELITY-01`: corrected selected-block gather passes the historical fidelity gate at median 0.99545; the old probe ignored its computed indices and used random QKV.
3. `BACKLOG-ADAPT-MECHANISMS-RERUN-01`: fresh seed 20260827 promoted the ADAPT-01 384-step LR=1e-4 arm at 17/32 math, 4/16 protected QA, 41/48 natural EOS and compliant length, falsifying `NO_ARM_PROMOTED`; the separately completed 640-step arm remained rejected.

All three remain `EXECUTED` pending independent AGY review. SLX-08 is explicitly limited to fidelity; its speed claim is objectively blocked until a physical selected-block prefill route exists.

## False-positive ledger

1. `BACKLOG-DISTILL-REAL-01`: real ThinkingCap teacher/student generations rejected DISTILL-00's hard-coded concise-student promotion: the student was 56.25pp less accurate and used 102.11% more tokens.
2. `BACKLOG-CUDAGRAPH-SERVING-02`: causal OFF/ON serving replay produced only 1.037x median speedup, below the 1.10 gate; the historical 1.5115x comparison was order-confounded.
3. `BACKLOG-DISTILL01-FLEET-REAL-01`: clean routed fleet achieved 15/48 versus 13/48 (+15.38%), not the promoted 22/48 versus 18/48 or the required +20%.
4. `BACKLOG-CTRL01-REAL-TOKEN-06`: exact-token replay over 24 real generations and 12 valid controls falsified the 100% constrained-decoding claim. Raw model JSON was 24/24 valid, while the offline sidecar left only 18/24 valid; it also rejected 9.84% of valid control tokens and is absent from production runtime trees.
5. `BACKLOG-BEE-L5-LIVE-GUARD-04`: the real client-side intervention detected and aborted 25/25 frozen live loops with 0/128 false alarms and 93.75% median savings, but measured 7.8 us/token p95 guard overhead, failing the historical 2 us/token qualification gate.
6. `BACKLOG-BEE-L3-REAL-TELEMETRY-01`: paired physical K0/K2/K4 telemetry gave the adaptive replay 1.458x over K0 but only 3.68% over K4 versus the required 15%; exact output parity across arms was 83.33%, and the runtime still lacks per-request K switching.
7. `BACKLOG-SLX10-PACKED-RUNTIME-02`: IQ2_XXS could not be produced without the missing mandatory imatrix; supported Q2_K physically loaded and improved speed/VRAM, but occupied 27.59% of F16, reduced accuracy from 12.5% to 0%, and produced 0/32 byte-identical outputs.
8. `BACKLOG-SPEC01-LIVE-HYBRID-01`: the deployed combined `draft-mtp,ngram-cache` route preserved 30/30 outputs but delivered only 0.689x the throughput of MTP alone, not 3x, and exposed no separate n-gram-versus-MTP acceptance telemetry.
