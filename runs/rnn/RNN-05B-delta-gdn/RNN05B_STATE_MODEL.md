# RNN05B_STATE_MODEL — complete sequence-owned state of the DeltaNet / Gated-DeltaNet block

RNN-05A qualified only the recurrent **matrix** state and was explicitly scoped `FULL_MODULE_* =
NOT_QUALIFIED` because the causal-conv boundary was never reset or serialized. RNN-05B enumerates and
serializes **all** sequence-owned mutable state, so the checkpoint is the *complete* module state. The
machine-readable form is `RNN05B_STATE_INVENTORY.json` (emitted by
`rnn_delta_substrate.state_inventory`). This maps 1:1 to the real Qwen linear-layer cache measured in
RNN-01 (`{recurrent_states, conv_states}`).

## Sequence-owned mutable state (what a checkpoint MUST contain)

| component | shape (per req) | bytes | Qwen analog (RNN-01) | reset / serialize |
|---|---|---:|---|---|
| **recurrent_state_S** | `[d_k, d_v]` = `[64,64]` | 16384 | `recurrent_states [1,H,128,128]` | zero (independent) / carry (continuous); raw fp32 round-trip |
| **conv_state** | `[conv_dim, K−1]` = `[192,3]` | 2304 | `conv_states [1,conv_dim,kernel]` | last `K−1` projected columns; raw fp32 round-trip |
| **complete live state** | — | **18688** | per-layer `{recurrent, conv}` | both together = the full checkpoint |

(`d_k=d_v=64` chosen so the delta rule — rank ≈ `d_k` — has capacity for the MQAR task; see the substrate
selftest for the exact numbers, which the harness re-emits into `RNN05B_STATE_INVENTORY.json`.)

## NOT sequence-owned (correctly excluded from the cache)

- **position/offset** — the toy uses no RoPE/positional recurrence, so there is no positional state to carry
  (noted explicitly so the inventory is exhaustive, not silently incomplete).
- **normalization** — the final `LayerNorm` is a *parameter* (weight/bias), not a per-sequence buffer; the
  `q,k` L2-normalization is stateless. Neither is part of the runtime cache. (This is exactly the class of
  "normalization-related mutable state, if any" the packet §6 asks to investigate — here: none.)

## Why this closes the RNN-05A gap

RNN-05A's lifecycle proofs ran on `S` *after* q/k/v had been precomputed by a causal conv over the **full**
sequence, so the continuation implicitly re-used future-aware features. RNN-05B's
`checkpoint_restore_full_module` (packet §7) instead:

1. runs the **complete** block on the prefix, producing `{S, conv_state}`;
2. serializes both to bytes, deletes the runtime tensors, and restores from bytes;
3. feeds **only** the continuation tokens — the conv rebuilds its outputs from the **restored** `conv_state`
   boundary (never recomputing prefix q/k/v), and the recurrence continues from the restored `S`;
4. compares against the uninterrupted run.

Result (substrate self-test, all three substrates): **BIT_EXACT** — continuation output and final state both
`0.0` max-abs difference; the `S` round-trip is bit-exact. So `FULL_MODULE_CHECKPOINT_RESTORE` is now
**qualified** for real DeltaNet and Gated-DeltaNet, not just the matrix. This is the concrete precondition
the Qwen-GDN transplant gate (§25) requires and the specific thing RNN-05A could not deliver.
