# RNN_MEMORY_CACHING_SPEC — reproduction specification (RNN-04)

Mechanism-level specification for reproducing **Memory Caching** from the *primary source*, reconciled
BEFORE any code (packet §3). Every algorithmic claim below is traced to a paper equation; the last
section is the equation→code binding the RNN-04 implementation actually uses. This document is the
authority for the implementation — *not* any blog/summary/community repo.

## 0. Source + epistemic disposition (§4)

- **PRIMARY SOURCE:** Behrouz, Li, Deng, Zhong, Razaviyayn, Mirrokni. *"Memory Caching: RNNs with Growing
  Memory."* arXiv:**2602.24281**v1 [cs.LG], 27 Feb 2026 (Google; ICML 2026). CC BY 4.0.
  PDF `https://arxiv.org/pdf/2602.24281`, HTML `https://arxiv.org/html/2602.24281v1`.
- **OFFICIAL_CODE = OFFICIAL_CODE_NOT_FOUND.** No "code available at", no GitHub URL, no artifact/
  reproducibility link anywhere in the PDF (abstract, body, appendices), on the arXiv abstract page, or on
  the alphaXiv mirror. Author site `abehrouz.github.io` lists the paper without a code link. This matches
  the R0/R1 ledger finding. Any community re-implementation (a PyPI/torch effort was mentioned on social
  media but not verified) is disposition **COMMUNITY_IMPLEMENTATION** and MUST NOT silently define the
  algorithm — the equations below govern.
- **Epistemic labels used downstream:** the paper's mechanism is **PUBLISHED**; our implementation is an
  independent **REPRODUCED/ADAPTED** effort. We do **not** claim paper parity — our substrate (small
  pure-PyTorch DeltaNet-family linear memory) and task scale differ from the paper's 760M/1.3B LMs, so
  results are labelled **ADAPTED** (mechanism transported to a toy substrate), never "matches Table X".

## 1. Recurrent state / memory definition (§ paper Sec. 2)

The memory module `M(·)` is an associative network whose **parameters are the recurrent state**. With
`L_M` layers: `L_M = 1` ⇒ **linear / matrix-valued memory** (linear-attention / DeltaNet style, state =
a `d×d` matrix `M`); `L_M ≥ 2` ⇒ **deep memory** (Titans/DLA-style MLP). Paper default deep memory
(App. B): 2-layer MLP, expansion 4, GELU, residual+LayerNorm per chunk `M(x)=x+W₁(W₂x)`.

Base recurrences the paper builds on (proof-of-concept bases = **Linear Attention, SWLA, DLA, Titans**;
DeltaNet/GatedDeltaNet are cited as *motivation only*):

| base | update | read |
|---|---|---|
| Linear Attention (Eq. 2) | `M_t = M_{t-1} + v_t φ(k_t)ᵀ` | `y_i = M_i φ(q_i) / Z_i` |
| Miras attentional-bias (Eq. 3) | `M_{t+1} = argmin_M L(M(k_t);v_t) + Ret(M;M_t)` | — |
| SWLA c=2 (Eq. 26–27) | `M_t = α_t M_{t-1} + Σ_{i=t-c+1}^{t} β_i(t) v_i k_iᵀ` | `y_t = M_t q_t` |
| DLA (Eq. 30–31) | `M_t = M_{t-1} − β_t ∇L(M_{t-1};k_t,v_t)`, `L=−⟨M(k_t),v_t⟩` | `y_t = M_t(q_t)` |
| Titans (Eq. 34–36) | `M_t = α_t M_{t-1} − S_t`, `S_t = η_t S_{t-1} − θ_t ∇L`, `L=‖M(k_t)−v_t‖²` | `y_t = M_t(q_t)` |

**RNN-04 substrate choice:** a small **pure-PyTorch DeltaNet-family linear memory** (delta rule
`S ← S + β(v − Sᵀk)kᵀ`; `read o = Sᵀq`), state = `d_k×d_v` matrix. Rationale: (a) it is a faithful member
of the linear-memory class the paper's Residual/GRM/Soup equations are derived for (Eqs. 12–13); (b) it is
the exact state shape of the lab's Qwen **Gated DeltaNet** kernel, so the §22 transplant-plausibility
argument is concrete; (c) fully transparent for aggregation unit tests (§19). No CUDA/Triton (§7).

## 2. Checkpoint creation rule + segmentation (§ paper Sec. 3, Eq. 4; Sec. 4.2)

Sequence split into segments `S⁽¹⁾…S⁽ᴺ⁾`; each segment run by its own memory recurrence
`M_t⁽ˢ⁾ = f(M_{t-1}⁽ˢ⁾; k_t, v_t)` (Eq. 4). **A checkpoint = the final memory state of each completed
segment**, `{M_{L⁽ˢ⁾}⁽ˢ⁾}`. Segmentation is **position-based (fixed), not content-based**:
- **Constant-size** (default, paper segment len **256** at 4K ctx; sweep {16,32,64,128,256,512}): all
  segments length `C=L/N`.
- **Logarithmic:** segment lengths = powers of two from the binary expansion of `L`; `N ≤ log₂L`.

State carry-over across segments (Sec. 3.4) is a **choice**: (1) *checkpointing / warm-start*
`M₀⁽ˢ⁾ = M_{L⁽ˢ⁻¹⁾}⁽ˢ⁻¹⁾` (one continuous optimization, checkpointed), or (2) *independent compressors*
`M₀⁽ˢ⁾` independent. RNN-04 tests **both** (predeclared) — see §13 of this spec.

## 3. Cache growth + complexity (§ paper Sec. 3.1, 4.2)

- #checkpoints = #completed segments = **N**, `1 ≤ N ≤ L`. Per-state footprint `p` params (constant).
  Total cache footprint `O(N·p)`.
- Update compute `O(L)` unchanged; retrieval `O(N)` reads/token (`O(p·N)`); **total `O(L + p·N·L) ≈ O(N·L)`**.
  `N=1` → `O(L)` (RNN); `N=L` → `O(L²)` (attention). Constant segments `O(pL²/C)`; log segments `O(pL log L)`.
- This is the paper's **interpolation between O(L) RNN and O(L²) Transformer** — the axis RNN-04 sweeps (§14).

## 4. Aggregation equations — the CORE (§ paper Sec. 3, Eqs. 5–17)

General read (Eq. 5): `y_t = Agg({M_{L⁽¹⁾}⁽¹⁾(·),…,M_{L⁽ˢ⁻¹⁾}⁽ˢ⁻¹⁾(·)}; M_t⁽ˢ⁾(·); q_t)`.

**(i) Residual Memory** (Eq. 6–7): `y_t = M_t⁽ˢ⁾(q_t) + Σ_{i=1}^{s-1} M_{L⁽ⁱ⁾}⁽ⁱ⁾(q_t)`.
No learned agg params. Query-dependent reads, but **for LINEAR memory this COLLAPSES** to a single
full-sequence state — `Σ_i M⁽ⁱ⁾ q = (Σ_i M⁽ⁱ⁾) q` and the summed cache equals plain linear attention over
the whole sequence (Eqs. 12–13). ⇒ **zero capacity gain for linear memory**; only helps deep/non-linear
memory. RNN-04 uses this as a **falsifiable correctness prediction** (arm B0), not as the improvement arm.

**(ii) Gated Residual Memory (GRM)** — the paper's key novelty (Eq. 8–10):
```
y_t = γ_t⁽ˢ⁾ M_t⁽ˢ⁾(q_t) + Σ_{i=1}^{s-1} γ_t⁽ⁱ⁾ M_{L⁽ⁱ⁾}⁽ⁱ⁾(q_t)         (Eq. 9)
γ_t⁽ⁱ⁾ = ⟨u_t, MeanPooling(S⁽ⁱ⁾)⟩,   u_t = x_t W_u                       (Eq. 10)
```
γ normalized by **softmax over segments** (line 241); `0≤γ≤1`; scalar-per-(token,segment). Learned param
**`W_u`** (connector); optional param-free `u_t=q_t`. Because γ is **query/context-dependent**, GRM does
**NOT collapse** for linear memory (constant γ ⇒ GRM ≡ Residual). This is the mechanism that can beat a
single state ⇒ **RNN-04 arm B**.

**(iii) Memory Soup** (Eq. 14–15): average the *parameters* of cached memories,
`M_t := {Σ_i γ_t⁽ⁱ⁾ W_c⁽ⁱ⁾}`, then `y_t = M_t(q_t)`; same context-aware γ (Eq. 10). **For linear memory,
Soup ≡ GRM** (souping-then-read = read-then-combine by linearity); diverges only for deep memory. Deferred
(§12/§20) — implemented only if a deep-memory extension is reached.

**(iv) Sparse Selective Caching (SSC)** (Eq. 16–17): MoE-style Top-k router over cached memories,
```
r_t⁽ⁱ⁾ = ⟨u_t, MeanPooling(S⁽ⁱ⁾)⟩,  R_t = argTopk({r_t⁽ⁱ⁾});
y_t = γ_t⁽ˢ⁾ M_t⁽ˢ⁾(q_t) + Σ_{i∈R_t} γ_t⁽ⁱ⁾ M_{L⁽ⁱ⁾}⁽ⁱ⁾(q_t)              (Eq. 17)
```
Selection is **learned + content-based** (relevance = `⟨u_t, meanpool⟩`), NOT recency/random. Learned
param `W_u`. Active set k ≪ N ⇒ retrieval `O(k)`. RNN-04 implements SSC only *after* GRM shows signal
(§12), and pairs it with the **random-selection control (arm D, §11)** to separate "selection policy helps"
from "more state exists".

## 5. Gate details (§ paper Sec. 3.1, Table 5 ablation)

`γ_t⁽ⁱ⁾ = softmax_i(⟨u_t, MeanPool(S⁽ⁱ⁾)⟩)`, `u_t = x_t W_u`. Scalar per (token, segment). **Init of `W_u`
NOT specified** by the paper ⇒ lab choice (we use small-normal init, recorded). Ablation Table 5: removing
context-dependence, removing gating (→Residual), linear-vs-deep memory, and sharing u&q each *degrade*
recall — so the context-aware gate is load-bearing.

## 6. Training-time vs post-training (§ paper Sec. 4.3 "Memory Caching as Post Training")

- **Trained-with-MC:** GRM, Soup, SSC (depend on learned `W_u`/router); Residual changes the forward pass.
  Tables 1–4 models are trained end-to-end with MC.
- **Training-free / POST_TRAINING_MC (verbatim, Sec. 4.3):** *"Memory caching can also be applied after
  pre-training … at inference, we cache the state of the memory after each segment … For decoding, we use
  **moving average of the past cached memory without learnable weights.** … even this simple technique can
  enhance the length extrapolation capability of recurrent models significantly."* ⇒ a **parameter-free
  moving-average aggregation** on a **fixed pretrained** recurrent model. This is the variant closest to a
  future Qwen transplant ⇒ **HIGH PRIORITY** (§18): same frozen base weights, memory OFF (single state) vs
  memory ON (moving-average of cached states), **no weight updates**.

## 7. Stop-gradient / detach (§ paper — SILENT)

The paper **never** mentions stop-gradient/detach on cached states. Sec. 3.4 discusses only state
*warm-start* vs *independent* init, not gradient flow. ⇒ RNN-04 decides explicitly: **trained arms
backprop END-TO-END through cached-segment states** (the paper trains with MC enabled — Tables 1–4 —
so gradients must reach the writes the recall reads from; a detach starves exactly that pathway, empirically
confirmed: detached GRM stayed at chance). The **frozen POST_TRAINING_MC variant** (§18) carries no
gradient, so it is effectively detached there. Recorded as an ADAPTED choice, not paper-derived. Per-forward
cached states are batched per example (no RNN-08b cross-sample leakage — each example owns its own states).

## 8. Initialization / normalization

Memory init/bias `M₀` (Eq. 18) acts as a key-projection bias; softmax-normalized gate; deep memory uses
LayerNorm+residual per chunk. RNN-04 linear substrate: `S₀ = 0`; gate `W_u ~ N(0, 0.02²)`; segment reads
optionally RMS-normalized before gating (recorded).

## 9. Objectives (§ paper App. B)

Outer: autoregressive LM (AdamW, cosine, wd 0.1). Inner memory objectives per base: LA/DLA
`L=−⟨M(k),v⟩`; Titans `L=‖M(k)−v‖²`. **RNN-04 outer objective:** cross-entropy on the *answer/value
token positions only* of the MQAR task (label = −100 elsewhere) — a clean recall objective.

## 10. Experimental tasks reused (§ paper Sec. 5.5)

Paper uses **MQAR (Arora 2024a, "we follow …", Fig. 5, 5 seeds)**, S-NIAH-1/2/3 @4K/8K/16K (Table 2),
in-context retrieval (Table 3), LM+commonsense (Table 1), LongBench (Table 4). The exact MQAR grid
(#pairs/seq-len/vocab) is deferred by the paper to the Zoology/Arora setup. **RNN-04 reproduces MQAR at
toy scale** (our own pinned grid, §14) — the associative-recall axis the primary question targets.

## 11. Complexity accounting the reproduction must report (§16)

Separate **storage** (state bytes, cache bytes = N·p, temp buffers, peak VRAM) from **compute** (update
time, checkpoint-creation time, aggregation/read time, total inference time). Report per number-of-cached-
states, per §15/§16 — never infer efficiency from footprint alone (RNN-08b lesson).

## 12. Algorithm boxes

The paper has **no Algorithm pseudocode boxes** — method is defined by Eqs. 1–36 only. Transcribed above.

---

## EQUATION → CODE BINDING (authoritative for `ops/rnn_mc_*.py`)

| paper eq | quantity | RNN-04 code symbol / function |
|---|---|---|
| Eq. 4 | per-segment recurrence `M_t⁽ˢ⁾=f(M_{t-1};k,v)` | `DeltaNetMemory.run_segment()` → returns final state `S⁽ˢ⁾` |
| — (checkpoint) | cache final segment states `{S⁽ˢ⁾}` | `list[Tensor]` `cached_states`, each `[d_k,d_v]` fp32 |
| Eq. 7 | Residual read (arm B0) | `agg_residual(states, q) = Σ_i Sᵢᵀ q` |
| Eq. 9 | GRM read (arm B) | `agg_grm(states, q, gamma)` = `Σ_i γ_i (Sᵢᵀ q)` |
| Eq. 10 | context-aware gate | `gate_logits = (x@W_u) · meanpool(seg_k)`, `γ = softmax_i` |
| Eq. 14–15 | Memory Soup | `agg_soup` = `(Σ_i γ_i S_i)ᵀ q` (≡ GRM for linear; deferred §20) |
| Eq. 16–17 | SSC Top-k router | `agg_ssc(states,q,gamma,k)` + `router_topk` (arm C-sel); random variant = arm D |
| Sec. 4.3 | POST_TRAINING_MC | `agg_moving_average(states,q)` = `(mean_i S_i)ᵀ q`, param-free, frozen base (§18) |

**Arm map (packet §11–§12):** A=`BASE_RNN` single state (N=1). B0=Residual (predicted ≡ A for linear —
falsifiable). B=GRM (learned context gate). C=equal-memory control (single state with ≈N× more bytes /
larger `d`, same total footprint as B). D=random-selection SSC control (same k, random R_t). Post=frozen-
base moving-average (§18). Order: A → B → **GATE** → {SSC, D, Soup} only if B shows signal (§12).
