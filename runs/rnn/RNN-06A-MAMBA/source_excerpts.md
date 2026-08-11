# RNN-06A-MAMBA — Source excerpts (DISCOVERY_ONLY)

Backend under qualification (exact executing bytes):
`transformers/models/mamba2/modeling_mamba2.py`
sha256 `83685d785c0df6578fefca8d5a2ed382d70e651ffc83c00b8f40b64807a04fdb`
(transformers 4.48.3). Line numbers below are from this file. These excerpts are
factual source reads made before any outcome-bearing run; they justify
`STATE_CONTRACT.json`.

## Naive path is the one that executes (no kernels)

```
43  if is_mamba_2_ssm_available():  ... else: ... = None, None, None
49  if is_causal_conv1d_available(): ... else: ... = None, None
54  is_fast_path_available = all((selective_state_update, mamba_chunk_scan_combined,
                                  mamba_split_conv1d_scan_combined, causal_conv1d_fn,
                                  causal_conv1d_update,))
...
663 def forward(self, hidden_states, cache_params=None, cache_position=None, attention_mask=None):
664     if is_fast_path_available and "cuda" in self.in_proj.weight.device.type:
665         return self.cuda_kernels_forward(...)
...
670     return self.torch_forward(...)
```
`mamba_ssm`/`causal_conv1d` are absent (verified live) => `is_fast_path_available=False`
=> `torch_forward` executes. Matches the P0 pin.

## Complete sequence-owned state = Mamba2Cache {conv_states, ssm_states}; no position

```
133 class Mamba2Cache:
173     self.conv_states = torch.zeros(num_hidden_layers, batch_size,
                intermediate_size + 2*n_groups*state_size, conv_kernel_size, ...)
181     self.ssm_states = torch.zeros(num_hidden_layers, batch_size,
                num_heads, head_dim, state_size, ...)
191     def update_conv_state(self, layer_idx, new_conv_state, cache_init=False):
194         if cache_init: self.conv_states[layer_idx] = new_conv_state.to(...)      # prefill: in-place slice write
197         else:          self.conv_states[layer_idx] = ...roll(shifts=-1, dims=-1)  # decode: in-place roll+
198                        self.conv_states[layer_idx][:, :, -1] = new_conv_state[:, 0, :]  #        write token 0 only
201     def update_ssm_state(self, layer_idx, new_ssm_state):
202         self.ssm_states[layer_idx] = new_ssm_state.to(...)                        # in-place slice write
205     def reset(self):
206         self.conv_states.zero_(); self.ssm_states.zero_()
```
There is **no** `seqlen_offset`/position attribute on `Mamba2Cache` (contrast Mamba-1).
Updates are IN-PLACE slice writes => a snapshot MUST `.clone()`; a plain reference is
mutated by later steps. `reset()` zeroes both => fresh-state contract is exactly zeros.

## Decode branch is single-token only (drives the continuation contract)

```
478 if cache_params is not None and cache_position is not None and cache_position[0] > 0:   # DECODE
479     cache_params.update_conv_state(..., new_conv_state=hidden_states_B_C, cache_init=False)  # uses token 0
...
516     dt = dt[:, 0, :][:, None, ...]        # <-- only token 0 of the segment
...
567     y = y.reshape(batch_size, -1)[:, None, ...]   # <-- emits exactly ONE output token
568 else:                                    # PREFILL (cache_position[0]==0): chunked naive-ssd over whole prefix
```
Consequence: a k-token continuation segment cannot be processed in one decode call; it
requires k sequential single-token `step`s. Prefill handles the whole prefix at once.

## Position is caller-managed; only cache_position[0] is read

`cache_position` is consumed only as `cache_position[0]` (lines 478, 510, 615): `==0` =>
prefill/init path, `>0` => single-token decode. The harness supplies it per call; it is
not part of the serialized module state.

## Model builds a cache when none is passed; requires cache_position when one is

```
946 if cache_params is None:
947     cache_params = Mamba2Cache(self.config, inputs_embeds.size(0), device=..., dtype=...)
950     cache_position = torch.arange(0, self.config.conv_kernel, device=...)
951 elif cache_position is None:
956     raise ValueError("You have to specify the `cache_position` manually when use_cache=True and cache_params is passed ...")
```
So the harness constructs `Mamba2Cache` explicitly and passes `cache_position` on every
call (prefill: first elem 0; decode: first elem = offset>0).

## Config (AntonV/mamba2-1.3b-hf @ 703e19a4)

num_hidden_layers=48, hidden_size=2048, expand=2 (intermediate=4096), n_groups=1,
state_size=128, num_heads=64, head_dim=64, conv_kernel=4 (conv_dim=4352),
chunk_size=256, vocab=50288, tie_word_embeddings=true, use_conv_bias=true, bf16.
