# tare.tools.local-labs — Master Research Catalog & Empirical Index 🔬📚

> **Epistemic Standard**: All findings in this catalog are backed by committed empirical evidence in `runs/`, verified against the hardware noise floor (~2.3% paired scatter on host `aaaaa`), and evaluated using distribution-free robust statistics (exact sign tests, percentile bootstrap CIs, and Cliff's $\delta$).

---

## 🏛️ The 7 Research Pillars

```mermaid
graph TD
    LAB[Local AI Lab / Inference Test Bench] --> P1[1. Quantization Frontier & Reasoning Budget]
    LAB --> P2[2. Linear Recurrence & Neural State Models]
    LAB --> P3[3. Speculative Decoding & MTP]
    LAB --> P4[4. Layer Ablation, Merging & Concision]
    LAB --> P5[5. Context Scaling & KV Cache Compression]
    LAB --> P6[6. High-Concurrency Serving & Slot Topology]
    LAB --> P7[7. Vision-Language Models & Multimodal]

    P1 --> OUT1[Deploy: Q2_K_XL / Instruct Mode]
    P2 --> OUT2[Deterministic State Replay 40/40]
    P3 --> OUT3[MTP 2.1x Speedup / n-max 3]
    P4 --> OUT4[Fable-TC 1.0 Conciseness Gate Pass]
    P5 --> OUT5[Lossless Q4_0 Symmetric KV @ 262k]
    P6 --> OUT6[2x Prefill Speedup via ub2048]
    P7 --> OUT7[Gemma-4 VLM Zero-Refusal UI Stack]
```

---

## 📊 1. Master Research & Campaign Index

| Pillar / Campaign | ID | Core Question / Hypothesis | Methodology & Workload | Epistemic Status | Key Findings & Metric Deltas | Primary Artifacts |
|---|---|---|---|---|---|---|
| **Quantization Frontier** | `EXP-QWEN38-QUANT` | Does aggressive sub-4-bit quantization degrade coding, competition math, or long context? | 7-quant ladder (`Q4_K_XL` 16.7G $\to$ `IQ2_M` 9.6G) on HumanEval+ (n=164), MATH-500 L5 (n=50), and single-needle NIAH to 131k. | **OPERATIONALIZED** | **`Q2_K_XL` (9.9GB) is the Pareto sweet spot**: 0.896 on HumanEval+, 90% on MATH-500 L5, 100% deep retrieval at 65k+. `IQ2_M` suffers deep long-context retrieval failure at $\ge 32k$. Frees ~7GB VRAM. | [`QUANT_FRONTIER_CAMPAIGN.md`](file:///C:/projects/local-model-lifecycle/ops/qwen38-bringup/QUANT_FRONTIER_CAMPAIGN.md), [`quant-frontier.html`](file:///C:/projects/local-model-lifecycle/ops/qwen38-bringup/quant-frontier.html) |
| **Reasoning Budget Curve** | `EXP-QWEN38-BUDGET` | Does enabling thinking mode / extending reasoning token budget improve code generation accuracy? | Thinking budgets (512, 1024, 2048, 8192) vs. raw `instruct` on HumanEval+ ($n=60$, fail-fast pilot promoted). | **FALSIFIED / CUT** | **Instruct (95.0%) strictly beats thinking mode (86.7% @ 8192)**. Thinking mode introduces rambling, non-termination, and truncation without accuracy gain. Deploy with `enable_thinking: false`. | [`BUDGET_CURVE_CLOSURE.md`](file:///C:/projects/local-model-lifecycle/ops/qwen38-bringup/BUDGET_CURVE_CLOSURE.md) |
| **MTP Speculative Decoding** | `EXP-MTP-SPEED` | How much end-to-end throughput gain does Multi-Token Prediction yield on coding workloads? | Parallel draft acceptance benchmark on Qwen 3.8 / 3.6 with draft depth $n \in [1, 5]$. | **OPERATIONALIZED** | **~2.1x speedup** on code generation with $n_{max}=3$ (acceptance rate 83.4%). Draft KV overhead is negligible across native context lengths. | [`A1_WINDOWED_MTP.md`](file:///C:/projects/local-model-lifecycle/docs/campaigns/a1-mtp/A1_WINDOWED_MTP.md), `tools/gates/verify_mtp.py` |
| **Windowed MTP Draft** | `EXP-A1-MTP` | Does restricting draft attention to a sliding window reduce KV overhead in long context? | Context depth sweep from 8k to 262k on GDN hybrid architectures. | **CLOSED / NEGATIVE** | Windowing provides **no benefit** within reachable context sizes ($\le 262k$). MTP decode advantage grows with depth up to native ceiling (+176%). Draft KV tax only dominates near ~1M tokens. | [`A1_WINDOWED_MTP.md`](file:///C:/projects/local-model-lifecycle/docs/campaigns/a1-mtp/A1_WINDOWED_MTP.md) |
| **Linear State Caching** | `EXP-RNN-04` / `06` | Can recurrent state snapshots be serialized and reloaded deterministically to eliminate prefill? | Mamba-2 1.3B / GDN state capture across sequence lengths; state reload perturbation test. | **CONFIRMED** | **Bit-exact state reproducibility (40/40)** on official fast-path kernels. Validates O(1) inference state resumption for linear models. | [`RNN_RESEARCH_LEDGER.md`](file:///C:/projects/local-model-lifecycle/docs/campaigns/rnn-mamba/RNN_RESEARCH_LEDGER.md), [`RNN_MEMORY_CACHING_SPEC.md`](file:///C:/projects/local-model-lifecycle/docs/campaigns/rnn-mamba/RNN_MEMORY_CACHING_SPEC.md) |
| **In-Run State Recovery** | `EXP-RNN-07A-R1` | Does intermediate recurrent state capture allow recovering information lost in final generation? | True single-trajectory in-run capture on NoLiMa semi-synthetic bridge ($N=64$). | **NO_SIGNAL / FALSIFIED** | `HISTORICAL_INFORMATION_PRESENCE_R1`: **NOT_DETECTED**. In-run state capture showed net $\Delta \approx 0$ over final output. Historical prefix-reprefill claims were confounded. | [`HANDOFF-rnn-07a-bridge-r1-inrun-recovery.md`](file:///C:/projects/local-model-lifecycle/.harness/handoff/HANDOFF-rnn-07a-bridge-r1-inrun-recovery.md) |
| **Model Concision & Refusal** | `EXP-A2-FABLE` | Can refusal-ablated and merged models eliminate reasoning verbosity without prose quality degradation? | Multi-judge blind pairwise quorum (Claude Opus, GLM-5.2, MiniMax-M3) across 18 creative/coding briefs. | **PROMOTED** | `fable-tc-l1.0` passed the writing quality gate with **-55% reasoning tokens** and **-23% creative output length** at quality parity with full-length verbose references. | [`A2_GATE3_RESULT.md`](file:///C:/projects/local-model-lifecycle/docs/campaigns/a2-ablation-merging/A2_GATE3_RESULT.md), [`A2_STAGE1_CONCISE_FABLE.md`](file:///C:/projects/local-model-lifecycle/docs/campaigns/a2-ablation-merging/A2_STAGE1_CONCISE_FABLE.md) |
| **KV Cache Quantization** | `EXP-A3-KV` | Can asymmetric KV cache (`q8_0` K / `q4_0` V) or sub-4-bit KV improve context capacity on GPU? | Paired isolated runs (6 reps) on MoE architecture at 8k context with FlashAttention. | **CLOSED / OPTIMAL** | Symmetric **`q4_0 / q4_0` is the unbeatable sweet spot** (88.55 t/s). Asymmetric `q8_0 / q4_0` causes **-57% throughput penalty** due to GPU kernel fallback to CPU. Symmetric `q4_0` reaches 262k context losslessly. | [`A3_KV_QUANT.md`](file:///C:/projects/local-model-lifecycle/docs/campaigns/a3-kv-quant/A3_KV_QUANT.md) |
| **Prefill & UBatch Tuning** | `EXP-SERVE-UB` | What micro-batching parameters maximize prompt ingestion throughput without VRAM blowout? | Context sweeps to 128k with ubatch sizes 512, 1024, 2048 on PCIe Gen4 x16. | **OPERATIONALIZED** | **`ubatch=2048` cuts prefill latency by 50.7%** (137.8s $\to$ 67.9s at 128k context). Leaves ~1.6GB VRAM headroom on 24GB host. | [`DEPLOY.md`](file:///C:/projects/local-model-lifecycle/docs/campaigns/serving/DEPLOY.md), [`SERVING.md`](file:///C:/projects/local-model-lifecycle/docs/campaigns/serving/SERVING.md) |
| **VLM Evaluation** | `EXP-VLM-01` | How do multimodal vision models perform under agent UI and screenshot comprehension workloads? | Gemma-4 12B/26B vision models evaluated on OCR, UI hierarchy, and prompt refusal rates. | **CONFIRMED** | Zero-refusal pipeline established for desktop GUI automation and visual telemetry interpretation. | [`M_A_VLM.md`](file:///C:/projects/local-model-lifecycle/docs/campaigns/vlm/M_A_VLM.md), [`M_A_VLM_PERF.md`](file:///C:/projects/local-model-lifecycle/docs/campaigns/vlm/M_A_VLM_PERF.md) |

---

## 🧬 2. Model Lineage & Provenance Registry

To maintain rigorous scientific attribution across the lab's documentation and benchmarks:

* **`Qwen3.6-Fable-Heretic` (Hugging Face upstream fine-tune)**:
  * *Origin*: Community fine-tune from Hugging Face based on Qwen 3.6 27B, trained to remove standard alignment refusals and unlock unrestricted creative/system task execution.
* **`Qwen3.6-ThinkingCap` (BottleCapAI fine-tune)**:
  * *Origin*: Token-efficiency fine-tune by BottleCapAI (`bottlecapai`) on Hugging Face that achieves near-identical reasoning accuracy while halving generated reasoning tokens.
* **`Qwen3.6-Fable-TC` / `fable-tc-l1.0` (Author-Created Original Merge by Augusto Carvalho)**:
  * *Origin*: Proprietary model merge designed and built in this lab by Augusto Carvalho using full-rank task arithmetic:
    $$W = W_{\text{Fable}} + \lambda (W_{\text{ThinkingCap}} - W_{\text{Base}})$$
  * *Methodology*: Empirical sweep across $\lambda \in \{0.4, 0.7, 1.0\}$. The $\lambda=1.0$ candidate proved dominant, achieving **98.3% GSM8K accuracy**, reducing reasoning tokens by **-54.8%**, eliminating generation starvation (0 timeouts vs 12 on base), and passing the Gate 3 multi-judge writing parity quorum.
* **`Qwen3.6-Fable-Fusion-711` (DavidAU Community Merge)**:
  * *Origin*: External community merge by DavidAU (`Qwen3.6-27B-Fable-Fus-711-UnHeretic-NM-DAU-NEO-MAX-NEO-MTP-Q4_K_M.gguf`), evaluated strictly as an external baseline (identified in `ops/close-outs/fable_termination.sh` as non-terminating in thinking mode).

---

## ⚔️ 3. Cross-Model Comparative Shootout & Deep Narratives

### Benchmark Shootout Matrix ($n=60$ HumanEval+ Market-R0 Subset)

All models evaluated on the identical 60-problem deterministic benchmark fixture (`market-r0`), scored with `evalplus` without imatrix distortion:

| Model Candidate | Lineage / Creator | Setting / Mode | Pass@1 (HumanEval+) | Truncation / Starvation Rate | Effective Reasoning Tokens | Production Role & Verdict |
|---|---|---|---:|---:|---:|---|
| **Qwen3.8-27B Base** | Qwen Official / Unsloth UD | **Instruct (`enable_thinking: false`)** | **95.0%** | **0 / 60 (0%)** | 0 | 🏆 **Current Production Champion** |
| Qwen3.8-27B Base | Qwen Official | Reasoning Low (`effort=low`) | 93.3% | 0 / 60 (0%) | ~350 / prompt | High-difficulty reasoning fallback |
| ThinkingCap-3.6 | BottleCapAI HF Fine-tune | Native Thinking | 93.3% | 0 / 60 (0%) | ~520 / prompt | Qwen 3.6 Efficiency Reference |
| **Fable-TC l1.0** | **Augusto Carvalho Merge** | **Instruct / Concise** | **88.3%** | **0 / 60 (0%)** | ~480 / prompt | 🛡️ **Uncensored Agent Champion (3.6)** |
| Qwen3.8-27B Base | Qwen Official | Reasoning Medium (`effort=med`) | 88.3% | 0 / 60 (0%) | ~1,420 / prompt | Sub-optimal for code |
| Qwen3.8-27B Base | Qwen Official | Reasoning XHigh (Default) | 45.0% ⚠️ | 31 / 60 (51.7%) | >6,144 (cap hit) | ❌ Broken / Truncation Trap |
| Fable-Fusion-711 | DavidAU Community Merge | Thinking / NEO | 40.0% ⚠️ | 36 / 60 (60.0%) | Non-terminating | ❌ Parked / Runaway Loops |

---

### 🧠 Key Comparative Narratives & Architectural Insights

#### A. The Generational Transition: Qwen 3.6 vs. Qwen 3.8
In the **Qwen 3.6 generation**, base models suffered from severe verbosity, alignment hesitation, and budget exhaustion (starving 12/60 prompts). To solve this, the lab created **`fable-tc-l1.0`** via task arithmetic, blending ThinkingCap’s concision vector into the Fable base. This cured generation starvation and elevated GSM8K to 98.3% while maintaining zero-refusal capability.

With **Qwen 3.8-27B**, the upstream base weights natively integrated agentic tool-use, code instruction tuning, and compact reasoning primitives. In pure **instruct mode**, Qwen 3.8 achieves **95.0% pass@1 out-of-the-box**, surpassing both ThinkingCap-3.6 (93.3%) and Fable-TC (88.3%) without needing custom task-vector surgery.

#### B. The "Reasoning Trap" in Software Engineering
A central empirical finding of the lab is the **Reasoning Inversion Paradox**: for bounded, deterministic programming tasks (HumanEval-class problems), extending reasoning token budgets correlates *negatively* with output accuracy ($95.0\% \to 93.3\% \to 88.3\% \to 45.0\%$).
- **The Failure Mode**: Long reasoning chains introduce recursive self-doubt, where the model second-guesses correct syntax, enters circular internal monologues, and exhausts the response context window (`finish_reason="length"`).
- **Production Guideline**: For software engineering swarms, enforce `enable_thinking: false` (or `reasoning_effort: "low"` for complex algorithmic steps).

#### C. The Multi-Dimensional Quantization Knee
While traditional 1D scalar benchmarks (HumanEval+, GSM8K, MATH-500) show flat performance from 16.7GB down to 9.6GB (~2.4-bit), **2D Long-Context Needle-in-a-Haystack (NIAH)** sweeps expose the hidden degradation boundary:
- **`IQ2_M` (9.6GB)**: Degrades severely at context depths $\ge 32k$, systematically dropping retrieval on deep needles ($d \ge 0.75$).
- **`Q2_K_XL` (9.9GB)**: Maintains **100% retrieval fidelity across all depths up to 65k+**, identical to full 16-bit and Q4 baselines.
- **The Verdict**: `Q2_K_XL` represents the true mathematical Pareto floor, freeing ~7GB of VRAM with zero context or code penalties.

#### D. Linear Recurrence & Hybrid State Efficiency
- **Hybrid Transformers (GDN / Qwen 3.5/3.6 MoE)**: Only 10 of 40 layers maintain a KV cache (`full_attention_interval=4`), slashing memory pressure by 75% and enabling native 262k context within a single 24GB GPU.
- **Speculative Drafting**: Multi-Token Prediction (MTP) adds ~2.1x decode acceleration without draft-KV penalty, as draft heads scale linearly across context depth.

---

## 🎯 4. The Production Pareto Frontier & Golden Config

### Consolidated Best-in-Class Deployment Configuration
For standard agentic coding and reasoning tasks on 24GB GPUs (RTX 4090 / ADA Class):

```bash
/home/augus/src/llama.cpp-master/build/bin/llama-server \
  -m /home/augus/models/qwen38-27b/Qwen3.8-27B-UD-Q2_K_XL.gguf \
  -fa on \
  --ctx-size 65536 \
  --cache-type-k q4_0 --cache-type-v q4_0 \
  --spec-type draft-mtp --spec-draft-n-max 3 \
  --batch-size 2048 --ubatch-size 2048 \
  --host 0.0.0.0 --port 8080
```

### Measured Performance Summary:
- **Model Footprint**: 9.9 GB VRAM (frees >14GB for KV cache and context buffers).
- **Coding Accuracy (HumanEval+)**: 95.0% pass rate (pure instruct).
- **Math Accuracy (MATH-500 Level 5)**: 90.0% accuracy.
- **Speculative Acceleration**: ~2.1x decode speedup via MTP heads.
- **Max Context**: 65k+ single-needle NIAH with 100% retrieval fidelity.

---

## 🔬 5. Formal Falsifications & Closed Hypotheses

A key achievement of the lab is establishing clear negative boundaries to prevent speculative churn:

1. ❌ **Hypothesis: Thinking/Reasoning mode improves code generation.**
   - *Falsification*: Measured on HumanEval+ ($n=60$). Instruct achieved 95.0% vs Thinking 86.7%. Reasoning models suffered from prompt self-doubting and token budget truncation.
2. ❌ **Hypothesis: Asymmetric KV cache (`q8_0` keys / `q4_0` values) saves memory with minimal cost.**
   - *Falsification*: llama.cpp fused FlashAttention lacks asymmetric CUDA offloading, causing execution fallback to host CPU (-57% decode throughput).
3. ❌ **Hypothesis: Windowed MTP is required to prevent draft KV explosion in long context.**
   - *Falsification*: Up to the model's native 262k token limit, MTP decode throughput advantage increased monotonically (+176% at max context).
4. ❌ **Hypothesis: In-run recurrent state capture recovers lost signal in linear RNNs.**
   - *Falsification*: Strict in-run single-trajectory capture on NoLiMa benchmark ($N=64$) showed $\Delta \approx 0$ between intermediate state snap and final output.

---

## 📂 6. Research Navigation & Directory Mapping

- **Campaign Deep-Dives**:
  - [`docs/campaigns/a1-mtp/`](file:///C:/projects/local-model-lifecycle/docs/campaigns/a1-mtp/) — Multi-Token Prediction and speculation limits.
  - [`docs/campaigns/a2-ablation-merging/`](file:///C:/projects/local-model-lifecycle/docs/campaigns/a2-ablation-merging/) — Layer ablation, concision tuning, Gate 3 judge quorums.
  - [`docs/campaigns/a3-kv-quant/`](file:///C:/projects/local-model-lifecycle/docs/campaigns/a3-kv-quant/) — KV cache quantization and memory geometry.
  - [`docs/campaigns/a4-instrumentation/`](file:///C:/projects/local-model-lifecycle/docs/campaigns/a4-instrumentation/) — TTFT and latency instrumentation.
  - [`docs/campaigns/gdn-kernel/`](file:///C:/projects/local-model-lifecycle/docs/campaigns/gdn-kernel/) — Gated Delta Net CUDA kernel profiling & TF32 optimization.
  - [`docs/campaigns/rnn-mamba/`](file:///C:/projects/local-model-lifecycle/docs/campaigns/rnn-mamba/) — Recurrent state models, Mamba-2, and TPTT.
  - [`docs/campaigns/serving/`](file:///C:/projects/local-model-lifecycle/docs/campaigns/serving/) — Production serving topologies and micro-batching.
  - [`docs/campaigns/vlm/`](file:///C:/projects/local-model-lifecycle/docs/campaigns/vlm/) — Vision-Language Model latency and GUI comprehension.
- **Ops & Active Bring-ups**:
  - [`ops/qwen38-bringup/`](file:///C:/projects/local-model-lifecycle/ops/qwen38-bringup/) — Active 27B quant-frontier, budget curves, and test harnesses.
  - [`ops/wsl/`](file:///C:/projects/local-model-lifecycle/ops/wsl/) — Detached background runner (`wslx.sh`).
- **Strategic Synthesis**:
  - [`docs/research/STATUS.md`](file:///C:/projects/local-model-lifecycle/docs/research/STATUS.md) — Comprehensive empirical register.
  - [`docs/research/EXPERIMENTS.md`](file:///C:/projects/local-model-lifecycle/docs/research/EXPERIMENTS.md) — Historical chronological ledger.
  - [`docs/research/IDEAS_BACKLOG.md`](file:///C:/projects/local-model-lifecycle/docs/research/IDEAS_BACKLOG.md) — Strategic research backlog.
