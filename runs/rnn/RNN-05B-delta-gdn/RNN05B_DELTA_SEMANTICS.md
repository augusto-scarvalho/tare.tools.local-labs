# RNN05B_DELTA_SEMANTICS — Linear-Attention vs DeltaNet vs Gated-DeltaNet

Written **before** the substrate implementation (packet §3). Maps the three recurrences' state-update
and read equations to the **pinned upstream sources** already in the repo, then to the local functions that
implement them. Do not read old RNN-04/05A notes as the semantic authority — the authority is the two
sources below, both of which live in this repo/session and both of which agree.

## Pinned upstream sources (authority)

1. **`GDN_KERNEL.md`** (this repo). The ggml / llama.cpp Gated-DeltaNet recurrence used by the deploy Qwen3.6
   hybrid, convention `out = Sᵀx`, scalar gate `d_t = exp(g_t)`, `scale = 1/√d_v`, validated chunk-parallel ==
   sequential to ~1e-16 (`M1`) and ported to ggml CPU/CUDA (`M2`/`M3`). The per-token recurrence:
   ```
   S   <- d_t * S                    # gated decay of the whole state
   u_t  = beta_t * (v_t - Sᵀ k_t)    # delta / error correction against the DECAYED state
   S   <- S + k_t ⊗ u_t              # rank-1 write
   o_t  = scale * Sᵀ q_t             # readout from the UPDATED state
   ```
2. **`scratchpad/modeling_qwen3_next.py`** (HF Qwen3-Next, the GDN family this lab deploys). Two independent
   reference paths that we port verbatim (single-head toy):
   - `torch_recurrent_gated_delta_rule` (L479–490) — the sequential **ground truth**:
     ```
     S      = S * exp(g_t)                       # decay  (g_t ≤ 0)
     kv_mem = (S * k_t[...,None]).sum(dim=-2)    # = Sᵀ k_t
     delta  = (v_t - kv_mem) * beta_t
     S      = S + k_t[...,None] * delta[...,None,:]   # S += k ⊗ delta
     o_t    = (S * q_t[...,None]).sum(dim=-2)    # = Sᵀ q_t
     ```
   - `torch_chunk_gated_delta_rule` (L373–451) — the chunk-parallel WY form (our fast path + parity counterpart).
   State shape `[d_k, d_v]`; `q,k` are L2-normalized (`use_qk_l2norm_in_kernel`); `scale = 1/√d_k` on `q`;
   `g_t` is a **per-head scalar** log-decay (Mamba2/GDN `A_log` style, `g_t ≤ 0`); `beta_t ∈ (0,1)`.

The two sources agree exactly (ggml `out=Sᵀx`, scale `1/√d_v` vs HF `1/√d_k`; identical for `d_k=d_v`,
our toy). RNN-01 confirms the real Qwen linear-layer cache is **two** objects — `recurrent_states
[1,H,d_k,d_v]` + `conv_states [1,conv_dim,kernel]` — which is exactly the complete state RNN-05A could not
serialize (matrix only). RNN-05B closes that gap.

## The nested substrate family (one recurrence, three modes)

All three are the *same* loop with two switches — `use_delta` (subtract the state's own prediction) and
`use_gate` (decay). This makes them a clean, matched family whose ONLY differences are the two mechanisms
under test.

| substrate | decay `d_t` | write `u_t` | additive? | historical-state recoverability |
|---|---|---|---|---|
| **LA** (RNN-04/05A) | `1` | `v_t`                     | **yes**, `S = Σ_i k_i⊗v_i` | final state = Σ of segment states (collapses) |
| **DN** (DeltaNet)  | `1` | `β_t (v_t − Sᵀk_t)`       | **no**, write depends on running `S` | not a simple sum of segment states |
| **GDN** (Gated DN) | `exp(g_t)` | `β_t (v_t − Sᵀk_t)` | **no**, write depends on `S` *and* decay reweights history | not a simple sum; history is decayed |

Read (all three): `o_t = Sᵀ q_t` (after the write). `q,k` L2-normalized; `q` scaled by `1/√d_k`.

**Why this matters for the science.** RNN-05A's negative used LA, whose state is *additive*: the online/final
state already equals the sum of all segment states, so cached partial states add no recoverable information
(hypothesis **H2**). DN/GDN writes depend on the running state (delta) and, for GDN, are decayed — so a
historical checkpoint can hold associations the final state has overwritten/forgotten (hypothesis **H3**).
RNN-05B tests whether that structural difference changes the frozen-backbone Memory-Caching verdict, and
whether Memory Caching only ever helps with backbone–memory co-adaptation (**H1**).

## Equation → upstream → local-function map

| quantity | paper/upstream | local (`ops/rnn_delta_substrate.py`) |
|---|---|---|
| projections q,k,v | Qwen `Qwen3NextGatedDeltaNet` in-proj + conv | `DeltaBlock._proj` (single depthwise causal conv over `[q;k;v]`, kernel 4, SiLU) |
| conv boundary state | Qwen `conv_states [1,conv_dim,k]` (RNN-01) | `DeltaBlock._proj(..., conv_state=)` left-context buffer `[B,conv_dim,k-1]` |
| L2 norm on q,k | `l2norm(q/k)` | `F.normalize(...,dim=-1)` in `_proj` |
| decay `g_t` | `A_log`/`-softplus` per-head scalar | `g = -softplus(w_g(x))` (GDN); `g=0` (DN/LA) |
| `beta_t` | `sigmoid` | `beta = sigmoid(w_b(x))` (DN/GDN); `beta=1` (LA) |
| sequential recurrence | `torch_recurrent_gated_delta_rule` | `delta_scan(mode=...)` (ground truth) |
| chunk-parallel recurrence | `torch_chunk_gated_delta_rule` | `delta_chunked` (DN/GDN) / `la_chunked` (LA) |
| read `o_t=Sᵀq_t` | `(S*q).sum(d_k)` | inside `delta_scan` / kernels |
| recurrent state `S` | `last_recurrent_state [1,H,d_k,d_v]` | `[B,d_k,d_v]` matrix (checkpointable) |

## Parity gate (packet §5–6), since FLA is **not installed** in the venv

The independent oracle is our own **sequential scan** (ported line-for-line from
`torch_recurrent_gated_delta_rule`) versus our own **chunk-parallel** path (ported from
`torch_chunk_gated_delta_rule`). They share no code. Required: single step, small sequence, chunked
sequence, and incremental recurrent decode all agree (`DELTANET_REFERENCE_PARITY`,
`GATED_DELTANET_REFERENCE_PARITY`). If parity fails, STOP before Memory Caching (§5).

## Memory Caching on top (unchanged machinery)

Segment the sequence; cache each completed segment's final recurrent state `S`; aggregate reads across
cached + online states with the RNN-04 pure functions (`read_states`, `agg_moving_average`, `agg_grm`).
The ONLY change from RNN-05A is the substrate underneath the cache (LA → DN/GDN). Independent-compressor
(reset `S` per segment) and continuous (warm-start `S`) lifecycles are both exercised, now including the
**conv boundary state** so the checkpoint is the *complete* module state (packet §7).
