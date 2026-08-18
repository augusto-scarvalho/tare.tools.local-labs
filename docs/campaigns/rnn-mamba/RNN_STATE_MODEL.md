# RNN_STATE_MODEL — Qwen3.5 Gated-DeltaNet recurrent state (archaeology)

Packet: RNN Foundation R0/R1 · 2026-08-10. Covers **RNN-01** (state characterization), **RNN-02**
(checkpoint/restore), **RNN-03** (branching). Raw evidence: `runs/rnn/RNN-01-gdn-state/`. Tool:
`ops/rnn_gdn_state_probe.py`.

**Epistemic labels.** State *shapes/dtypes/mechanics* below are `OBSERVED` from the actual transformers
`qwen3_5` source and a runtime capture on a **structurally-faithful surrogate** (a `Qwen3_5TextModel`
whose per-linear-layer GDN head dims are set EXACTLY to real Qwen3.5-0.8B; only depth/width/vocab were
shrunk). Per-layer state tensors are therefore shape/dtype-identical to the real model. Byte totals for
real 0.8B / 27B are `OBSERVED`(per-layer) × `COMPUTED`(real layer counts). Checkpoint/restore verdicts
are properties of the cache mechanics and are **weight-independent**, so they transfer to the real model.

Why a surrogate (§9/§12/§14): the state *semantics* are weight-independent, the official 0.8B checkpoint
is a VLM (`Qwen3_5ForConditionalGeneration`) wrapper, and the packet mandates frugality (no multi-hour
GPU, no unnecessary downloads). Capturing real-weight state *magnitudes* on a downloaded Qwen3.5-0.8B is
listed as an optional cheap follow-up — it would not change any shape/byte/semantics conclusion here.

---

## 1. Where the state lives (source excerpts)

Path: `transformers/models/qwen3_5/modeling_qwen3_5.py` (transformers 5.12.1; installed in
`/home/augus/sglang-venv`). Mirror of the reasoned lines:

- **`class Qwen3_5GatedDeltaNet(nn.Module)` (L371)** — the linear-attention (GDN) mixer. Dims (L374-397):
  ```
  self.num_v_heads = config.linear_num_value_heads      # 16 (0.8B) / 48 (27B)
  self.num_k_heads = config.linear_num_key_heads         # 16
  self.head_k_dim  = config.linear_key_head_dim          # 128
  self.head_v_dim  = config.linear_value_head_dim         # 128
  self.key_dim   = head_k_dim * num_k_heads               # 2048
  self.value_dim = head_v_dim * num_v_heads               # 2048 (0.8B) / 6144 (27B)
  self.conv_dim  = self.key_dim * 2 + self.value_dim       # 6144 (0.8B) / 10240 (27B)
  self.conv1d = nn.Conv1d(conv_dim, conv_dim, kernel_size=conv_kernel_dim=4, groups=conv_dim)  # depthwise
  ```
- **Two cached tensors per linear layer** (L457-458, read from cache):
  ```
  conv_state       = cache_params.layers[layer_idx].conv_states       # short-conv window
  recurrent_state  = cache_params.layers[layer_idx].recurrent_states  # gated-delta matrix memory
  ```
- **Recurrent state allocation** (`torch_recurrent_gated_delta_rule`, L346-347 / chunk path L298-299):
  ```
  last_recurrent_state = torch.zeros(batch_size, num_heads, k_head_dim, v_head_dim, dtype=value.dtype)
  ```
- **The update rule** (`torch_recurrent_gated_delta_rule`, L359-363):
  ```
  last_recurrent_state = last_recurrent_state * g_t                       # gated decay
  kv_mem = (last_recurrent_state * k_t.unsqueeze(-1)).sum(dim=-2)
  last_recurrent_state = last_recurrent_state + k_t.unsqueeze(-1) * delta.unsqueeze(-2)   # rank-1 update
  core_attn_out[:, :, i] = (last_recurrent_state * q_t.unsqueeze(-1)).sum(dim=-2)         # readout
  ```
- **Cache writes** (L487, L550): `cache_params.update_conv_state(...)` and
  `cache_params.update_recurrent_state(last_recurrent_state, layer_idx)`.
- **Prefill vs decode paths** (L524-541): `recurrent_gated_delta_rule` when `seq_len==1` cached-decode;
  `chunk_gated_delta_rule` for prefill / multi-token. Pure-torch fallbacks are used here because
  `flash-linear-attention` and `causal-conv1d` are **not installed** (runtime logged the fallback).
- Full-attention layers (`class Qwen3_5Attention`, L645) use an ordinary growing key/value cache.
- Cache object: **`DynamicCache`**; linear layers materialize as **`LinearAttentionLayer`**
  (`cache_type=DynamicCache, layer_obj=LinearAttentionLayer`, observed at runtime).

**Cross-check against the lab's own llama.cpp kernel.** `GDN_KERNEL.md` documents the deploy fork's
ggml recurrence as `S<-d_t*S; u_t=beta_t*(v_t - Sᵀk_t); S<-S+k_t⊗u_t; o_t=scale*Sᵀq_t`. This is the
**same gated delta rule** as the HF `torch_recurrent_gated_delta_rule` above — two independent
implementations (PyTorch reference vs the lab's chunk-parallel CUDA kernel, validated to ~1e-16) of one
mechanism. That mutual agreement is strong evidence we have the recurrence right.

## 2. State shapes, dtypes, byte accounting (OBSERVED per-layer; COMPUTED totals)

Per **linear (GDN) layer**, batch=1 (runtime capture, dims = real Qwen3.5-0.8B):

| tensor | shape | dtype | numel | bytes |
|---|---|---|---:|---:|
| `conv_states` | `[1, 6144, 4]` | (model dtype; fp32 in probe / bf16 deployed) | 24 576 | 98 304 (fp32) / 49 152 (bf16) |
| `recurrent_states` | `[1, 16, 128, 128]` | **fp32** (`mamba_ssm_dtype`) | 262 144 | **1 048 576 (1.0 MiB)** |

Full-attention layers hold ordinary `keys`/`values` `[1, n_kv, seq, head_dim]` that **grow O(N)**.

**Per-request GDN state (recurrent, fp32; constant in sequence length):**

| Model | linear layers | recurrent/layer | **total recurrent state** |
|---|---:|---:|---:|
| surrogate (8 layers, 6 linear) | 6 | 1.0 MiB | **6.28 MiB** (OBSERVED) |
| **Qwen3.5-0.8B** | 18 | 1.0 MiB | **18.84 MiB** (COMPUTED) |
| **Qwen3.6-27B** | 48 | 3.0 MiB | **147.75 MiB** (COMPUTED) |

Key consequences:
- **The Gated-DeltaNet memory is a fixed, sequence-length-independent budget** (~19 MiB for 0.8B, ~148
  MiB for the 27B deploy model, recurrent-dominated). Only the ¼ full-attention layers carry a
  growing KV cache. This is the whole appeal of the hybrid for long context — and the quantitative
  basis for any future "spend B bytes on cached recurrent state vs KV" comparison (RNN-33).
- The recurrent matrix is a **learned associative memory `S ∈ R^{H×d_k×d_v}`**, updated by the delta
  rule; combining/averaging such states across a sequence (Memory Caching) is defensible *in principle*
  but for Qwen specifically remains **`PROPOSED`**, not established.

## 3. Reset / ownership / chunk semantics (OBSERVED)
- **Initialization / reset:** state is `torch.zeros(...)` at sequence start (no cache) — a fresh request
  begins from a zero associative memory. There is no cross-request carryover unless a cache is explicitly
  passed. Request-ownership is therefore just "own your `DynamicCache`."
- **Batch dimension:** leading dim is batch; per-request isolation = separate cache (or separate batch
  row). No global/batch-shared recurrent state.
- **Chunk semantics:** prefill uses `chunk_gated_delta_rule` (processes the sequence in chunks, carrying
  `initial_state`); cached single-token decode uses the per-step recurrent kernel. Both read/write the
  same `conv_states`/`recurrent_states`. Prefill and decode paths are numerically consistent to ~1e-6
  (see §4), i.e. the chunked prefill and the step recurrence compute the same state.

## 4. RNN-02 checkpoint/restore + RNN-03 branching (OBSERVED; weight-independent)
Deterministic CPU fp32, greedy, `runs/rnn/RNN-01-gdn-state/rnn02_checkpoint_restore.json` /
`rnn03_branching.json`:

| test | max|Δ| | verdict |
|---|---:|---|
| checkpoint → destroy runtime state → restore → continue (hidden + state) | **0.0** | **BIT_EXACT** |
| disk round-trip of state via `torch.save`/`torch.load` | **0.0** | **BIT_EXACT** |
| cached-incremental decode vs full recompute (prefill kernel vs step kernel) | 7.75e-6 | **NUMERICALLY_EQUIVALENT** (≤1e-4) |
| RNN-03 branch A vs branch B separation | 5.66 | branches independent |
| RNN-03 each branch independently restored & reproduced | **0.0** | **BIT_EXACT** |

**Interpretation.** The Gated-DeltaNet recurrent state is a **cleanly checkpointable, restorable,
serializable, and forkable object.** Save/restore is bit-exact; forking a prefix into independent
continuations and restoring each is bit-exact. The only non-zero delta is the expected ~1e-6 fp
difference between the chunk-prefill and per-step-decode kernels (non-associative fp summation), not a
semantic difference. **This clears the RNN-02/03 gate** that the packet requires before any
Memory-Caching-style state manipulation — while we explicitly do NOT start that work (§11/§12).

## 5. What was NOT established (honesty)
- No real-weight run (state *magnitudes/statistics* on the actual Qwen3.5-0.8B weights) — optional
  follow-up; would not change shapes/bytes/semantics.
- No claim that averaging/combining Qwen GDN states is *useful* (that is `PROPOSED`; RNN-06 territory).
- Byte totals assume batch=1, recurrent in fp32, conv in the stated dtype; deployed conv dtype is bf16.
- The full NVIDIA/RULER benchmark was not run against any model (deferred; harness discipline qualified
  separately, `runs/rnn/RNN-00B-ruler/`).
