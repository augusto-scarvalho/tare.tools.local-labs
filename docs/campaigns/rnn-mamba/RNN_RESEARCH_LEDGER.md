# RNN_RESEARCH_LEDGER — Recurrent / Neural-Memory Research (Local AI Lab)

Packet: **RNN Foundation R0/R1** · Date 2026-08-10 · Starting HEAD `4f4e459`.

**Method.** Every row was verified from **primary sources** — arXiv abstract/PDF pages and official
GitHub repos (commit SHA + LICENSE read from the repo/API) — during this packet, not from memory or
secondary blogs. Fields that could not be verified from a primary source are marked `UNKNOWN` rather
than guessed. Several works postdate the assistant knowledge cutoff (Jan 2026); those rows rest
entirely on live fetches performed in this packet.

**Epistemic key** (mandatory, §2): `PUBLISHED` = demonstrated by the cited work · `REPRODUCED` =
reproduced in this lab · `ADAPTED` = published mechanism moved to a new model · `PROPOSED` = our
hypothesis · `OBSERVED` = limited local evidence · `FALSIFIED` = contradicted locally.
Classification (§27): `ADOPT` / `ADAPT` / `REPRODUCE` / `INSPIRE` / `RESEARCH` / `PARK`.

> **Load-bearing determination (§5).** **Memory Caching has OFFICIAL_CODE_NOT_FOUND.** The only
> implementation located is `github.com/sypherin/growing-memory` — an app-layer, self-described
> non-official reproduction (MIT, 0 stars, created 2026-06-08). It is classified
> `COMMUNITY_IMPLEMENTATION` and must **not** be treated as upstream-authoritative. Titans, ATLAS,
> and MIRAS (all Google/Behrouz) also have OFFICIAL_CODE_NOT_FOUND.

---

## Master table

| # | Work | arXiv | v1 date | Pub | Official code | Commit pinned | License | Class |
|--:|---|---|---|---|---|---|---|---|
| 1 | Transformer-XL | 1901.02860 | 2019-01-09 | ACL 2019 | kimiyoung/transformer-xl | `44781ed` | Apache-2.0 | INSPIRE |
| 2 | Retentive Network (RetNet) | 2307.08621 | 2023-07-17 | preprint | microsoft/unilm /retnet | `833df7e`* | MIT | REPRODUCE |
| 3 | Longhorn | 2407.14207 | 2024-07-19 | NeurIPS'24 ENLSP wksp | Cranial-XIX/longhorn | `4ea1745` | **UNKNOWN (no LICENSE)** | RESEARCH |
| 4 | Gated DeltaNet | 2412.06464 | 2024-12-09 | ICLR 2025 | NVlabs/GatedDeltaNet | `b53d6d3` | **NVIDIA NC** | ADAPT |
| 5 | TTT (TTT-Linear/MLP) | 2407.04620 | 2024-07-05 | preprint (NeurIPS'24 claimed) | test-time-training/ttt-lm-{pytorch,jax,kernels} | `cd831db` (pt) | MIT (pt); UNKNOWN (jax,kernels) | REPRODUCE |
| 6 | Titans | 2501.00663 | 2024-12-31 | preprint | **OFFICIAL_CODE_NOT_FOUND** | — | — | INSPIRE |
| 7 | ATLAS | 2505.23735 | 2025-05-29 | preprint | **OFFICIAL_CODE_NOT_FOUND** | — | — | INSPIRE |
| 8 | It's All Connected / MIRAS | 2504.13173 | 2025-04-17 | preprint | **OFFICIAL_CODE_NOT_FOUND** | — | — | INSPIRE |
| 9 | In-Place TTT | 2604.06169 | 2026-04-07 | ICLR'26 Oral (claimed) | ByteDance-Seed/In-Place-TTT | `be23248` | Apache-2.0 | ADAPT |
| 10 | LaCT (TTT Done Right) | 2505.23884 | 2025-05-29 | preprint | a1600012888/LaCT | `a648340` | MIT | REPRODUCE |
| 11 | Memory Caching (Growing Memory) | 2602.24281 | 2026-02-27 | preprint | **OFFICIAL_CODE_NOT_FOUND** (community: sypherin/growing-memory) | — | community MIT | INSPIRE / PARK |
| 12 | Hybrid Associative Memories (HAM) | 2603.22325 | 2026-03-20 | preprint | **OFFICIAL_CODE_NOT_FOUND** | — | — | INSPIRE / RESEARCH |
| 13 | Mela | 2605.10537 | 2026-05-11 | preprint | **OFFICIAL_CODE_NOT_FOUND** | — | — | INSPIRE / RESEARCH |
| 14 | The Mamba in the Llama | 2408.15237 | 2024-08 | NeurIPS 2024 | jxiw/MambaInLlama | `b03f123` | Apache-2.0 | ADAPT |
| 15 | RADLADS | 2505.03005 | 2025-05 | preprint | recursal/RADLADS-paper | `1b362eb` | Apache-2.0 | ADAPT |
| 16 | TPTT | 2506.17671 | 2025-06-21 | preprint | fabienfrfr/tptt | `242e214` | Apache-2.0 | INSPIRE / ADAPT |
| 17 | LoLCATs | 2410.10254 | 2024-10-14 | ICLR 2025 | HazyResearch/lolcats | `375df84` | Apache-2.0 | ADOPT / REPRODUCE |
| 18 | Liger | 2503.01496 | 2025-03-03 | ICML 2025 | OpenSparseLLMs/Linearization | `0b364eb` | Apache-2.0 | ADOPT / REPRODUCE |
| 19 | Gated DeltaNet-2 (GDN-2) | 2605.08988 | 2026-05-14 | preprint | NVlabs/GatedDeltaNet-2 | `9f2a81c` | NVIDIA NC | ADAPT |
| 20 | RWKV-7 ("Goose") | — | 2026-01 | Open Source | BlinkDL/RWKV-LM | `c481b7e` | Apache-2.0 | INSPIRE / ADAPT |
| 21 | YOCO (You Only Cache Once) | 2405.05254 | 2024-05-08 | preprint | microsoft/unilm | `91ea83f` | MIT | INSPIRE / ADAPT |

\* RetNet SHA is the `microsoft/unilm` monorepo head, not a retnet-subfolder-specific commit.

---

## §5 High-priority upstream code (inspected)

| Repo | Commit pinned | License | Implements GDN/DeltaNet/linear-attn? | Reuse for |
|---|---|---|---|---|
| fla-org/flash-linear-attention | `7843b32` (2026-08-09) | MIT | **YES** — Gated DeltaNet, GDN-2, DeltaNet, GLA, RetNet, RWKV7, Mamba3, NSA/MoBA | Triton kernels + training-ready layers for any GDN/linear-attn retrofit; **Liger depends on it** |
| NVIDIA/RULER | `c3f5e3b` (2026-07-22) | Apache-2.0 | No (benchmark harness) | Synthetic long-context eval of effective context |
| test-time-training/ttt-lm-pytorch | `cd831db` (2024-07-14) | MIT | TTT-Linear / TTT-MLP layers (not DeltaNet) | Reference TTT layers (authors call it naive/tutorial) |

**Not cloned this packet** (frugality, §14): only metadata/SHAs were read via GitHub. No external repo
was vendored into the project. FLA is the single most reusable substrate and is the natural dependency
if/when a GDN/linear-attn experiment is authorized — it also removes any need to write custom GDN
kernels locally (§14).

---

## Per-work notes (verified detail; CLAIM = paper's claim, not lab-verified)

### 1 Transformer-XL — INSPIRE
Segment-level recurrence (stop-grad hidden-state memory from the previous segment) + relative
positional encoding. From-scratch architecture (requires retraining). Foundational/historical;
mechanism now absorbed into modern long-context stacks. State = fixed-length cached hidden buffer.

### 2 RetNet — REPRODUCE
"Retention" with three equivalent forms: parallel (train), **recurrent (O(1)-state inference, CLAIM)**,
chunkwise (long-seq). New arch, trained from scratch; no widely adopted large checkpoints → reproduce/
experiment rather than deploy. Reference kernels also in microsoft/torchscale.

**Local mechanism qualification (RNN-09, 2026-08-21): COMPLETE.** The frozen CPU microbenchmark
qualified parallel↔recurrent parity (max abs `4.27e-14` through T=513), bit-exact chunkwise carry,
save/reload, batch isolation and explicit reset, gradient parity (`3.56e-15` max), and finite fp32
recurrence through T=4096. Reusing one sequence's state in another produced a large detectable delta
(`47.08`), confirming that state ownership/reset is load-bearing. This verifies the algebra and lifecycle,
not an official RetNet checkpoint or model-quality claim; official-checkpoint reproduction remains open.

### 3 Longhorn — RESEARCH
SSM whose state-update is the closed-form solution to an online associative-recall objective. CLAIMS
~1.8× sample efficiency vs Mamba and 16× context extrapolation. **No LICENSE file in the repo** →
treat as all-rights-reserved; do not vendor. Small SSM research code, tractable on one 3090.

### 4 Gated DeltaNet — ADAPT
Gating (fast erasure) + delta rule (targeted KV memory update), improving Mamba2; chunkwise-parallel
training. **This is the exact recurrence the deploy Qwen3.6 hybrids use** (see RNN_STATE_MODEL.md).
Kernels upstreamed into FLA. **Caveat: official NVlabs code is non-commercial (NC)** — research use
only; the FLA reimplementation (MIT) is the usable path.

### 5 TTT (Learning to Learn at Test Time) — REPRODUCE
Hidden state = weights of an inner model (TTT-Linear / TTT-MLP) updated by self-supervised gradient
steps per step; linear complexity (CLAIM). 125M–1.3B evaluated. pytorch repo is a tutorial impl
("do not recommend training with it"); jax repo is the training path. Trainable at small scale on a 3090.

### 6 Titans — INSPIRE (no code)
Neural long-term memory (fast weights w/ momentum + weight decay) + attention as short-term memory;
three variants (memory as context/gate/layer); CLAIMS >2M context. **OFFICIAL_CODE_NOT_FOUND** (Google
blog + author page link PDFs only; no google-research/titans repo). From-scratch training → conceptual only.

### 7 ATLAS — INSPIRE (no code)
Follow-up to Titans; high-capacity memory optimized over a context window (not purely online);
"DeepTransformers". CLAIMS +80% at 10M-context BABILong. **OFFICIAL_CODE_NOT_FOUND.** 10M-context is far
beyond a single 3090.

### 8 MIRAS / It's All Connected — INSPIRE (no code)
Unifying framework (associative-memory arch × attentional-bias objective × retention gate × learning
algorithm) → Moneta/Yaad/Memora. **Highest conceptual value of the no-code set** — the lens that ties
attention, fast-weight memory, and online optimization into one family. **OFFICIAL_CODE_NOT_FOUND.**

### 9 In-Place TTT — ADAPT
Drop-in TTT for existing Transformers: updates MLP down/final-proj weights as fast weights at inference,
NTP-aligned objective, chunk-wise, context-parallel. README references Qwen3-8B / LLaMA-3.1-8B, 4B @
128k. Apache-2.0. **Strongest ADOPT candidate of the TTT set** — targets already-trained 4B/8B models
(fit a 3090); caveat: adds inference-time gradient compute, and it is HF/PyTorch (not llama.cpp).

### 10 LaCT — REPRODUCE
Large-Chunk TTT: fast-weight updates over 2K–1M-token chunks → far higher GPU utilization than tiny
per-token TTT; nonlinear state up to ~40% of params. 14B video experiments exceed 24GB, but the LM /
Triton-kernel path is runnable; the GPU-utilization idea is directly relevant to one 3090. MIT.

### 11 Memory Caching (RNNs with Growing Memory) — INSPIRE / PARK
Behrouz et al. (Google). Cache checkpoints of the recurrent hidden state and combine/select them so a
fixed-size recurrent memory's *effective* capacity grows with the sequence; 4 variants (incl. gated
aggregation, sparse selective). Evaluated 760M@30B tok and 1.3B@100B tok, from scratch. **This is the
paper the lab's long-horizon "cache GDN states" idea derives from — but applying it to Qwen GDN states
is `PROPOSED`, not `PUBLISHED` (§2).** OFFICIAL_CODE_NOT_FOUND; from-scratch pretraining is out of
scope for one 3090. PARK direct reproduction; INSPIRE the idea.

### 12 Hybrid Associative Memories (HAM) — INSPIRE / RESEARCH
RNN compresses the full sequence; attention stores **only** tokens the RNN cannot predict → data-
dependent KV growth via one continuous threshold. Conceptually the strongest KV-savings lever here.
OFFICIAL_CODE_NOT_FOUND; scale unspecified.

### 13 Mela — INSPIRE / RESEARCH
Test-time memory consolidation with two update frequencies (gist vs episodic), spread across early
decoder layers ("MemStack", no extra tokens). 4K pretrain context, CLAIMS graceful extension beyond.
OFFICIAL_CODE_NOT_FOUND; single-author; scale UNKNOWN.

### 14 Mamba in the Llama — ADAPT
Distill a Transformer into a hybrid Mamba by **reusing attention projections** as SSM projections and
keeping a fraction of attention layers; 3-stage pipeline. Released 3B/8B distilled checkpoints
(Apache-2.0). **3B distillation is plausible on one 3090; released checkpoints run directly.**

### 15 RADLADS — ADAPT
Convert softmax-attention **Qwen2.5** into RWKV-style recurrent decoders via 3-stage distillation;
released RAD-RWKV6/7 checkpoints (Apache-2.0). Large conversions want multi-GPU; released checkpoints
run on a 3090. Directly Qwen-relevant lineage.

### 16 TPTT — INSPIRE / ADAPT
Inject linearized attention (LiZA) + memory-as-gate (MaG) using **DeltaNet/DeltaProduct** into a frozen
Transformer, aligned by LoRA. Evaluated ~270M–7B (Qwen2.5-1.5B, Gemma3-270m, Mistral-7B…) on MMLU.
Single-author preprint; gains reported mainly ~1B/MMLU → treat as adaptable reference, LoRA feasible on
a 3090. Apache-2.0.

### 17 LoLCATs — ADOPT / REPRODUCE
Two-step linearization: attention-transfer (Hedgehog/T2R feature maps + sliding window) then LoRA to
recover quality. ~0.2% params / ~0.4% tokens of prior methods; "a couple hours on one 40GB A100".
7B–8B fits a 3090 with LoRA (70B/405B out of scope). Peer-reviewed (ICLR'25), mature official code.
**Official repo = HazyResearch/lolcats** (OpenSparseLLMs is Liger, not LoLCATs).

### 18 Liger — ADOPT / REPRODUCE
Repurpose pretrained **key-matrix weights** to build gates → gated linear recurrence **without new
params**; recovers **93%** of base at **0.02%** pretraining tokens; uses FLA Triton kernels; README
converts e.g. Qwen3-8B. 1B–8B fits a 3090. Peer-reviewed (ICML'25). **The most directly applicable
Qwen-linearization path that reuses tooling the lab would already pull in (FLA).**

---

## §7 Qwen hybrid architecture — verified from official config.json (see RNN_ARCHITECTURE_MATRIX + RNN_STATE_MODEL)

Both **`Qwen/Qwen3.5-0.8B`** (24 layers, 18 linear + 6 full, hidden 1024) and **`Qwen/Qwen3.6-27B`**
(64 layers, 48 linear + 16 full, hidden 5120) exist officially and are **dense-FFN Gated-DeltaNet +
gated-attention hybrids** (`model_type: qwen3_5_text`, `full_attention_interval: 4`,
`attn_output_gate: true`, `linear_conv_kernel_dim: 4`), **not MoE**. Qwen3.6-27B's `text_config`
exactly matches the local `fp16/base` on disk. Older lineage: `Qwen/Qwen3-Next-80B-A3B`
(`model_type: qwen3_next`, MoE) — same GDN concept, larger. **The smallest official Qwen GDN hybrid is
Qwen3.5-0.8B.** (Third-party "sparse MoE" descriptions of Qwen3.5-0.8B are contradicted by its config.)
