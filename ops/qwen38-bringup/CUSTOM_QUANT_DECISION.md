# Custom quantization of Qwen3.8-27B — decision record

**Verdict (2026-08-16): NO-GO.** Use a community imatrix GGUF. Re-evaluate only if a trigger below fires.

## Why NO-GO now
1. **Good community quants already calibrate on our workload.** Unsloth UD-Q4_K_XL (Dynamic V3.0
   imatrix) and — notably — **bartowski's Qwen3.8-27B-GGUF, whose imatrix corpus explicitly includes
   tool-calling and reasoning conversations** (Fable research). A code+tool-schema custom imatrix has
   little headroom to beat that.
2. **The 4-bit knee is already high.** IQ4_XS ≈ 94% of full precision (Qwen3.6-27B KLD studies,
   assumed to transfer); Q4→Q5 gains are marginal. Small target for a custom recipe to claim.
3. **64GB RAM makes the BF16 (54.7GB) workflow feasible but tight**, plus ~85GB transient disk on a
   WSL VHDX we already know is painful to compact (memory `wsl-disk-and-compaction`). `llama-quantize`
   mmaps from disk (doesn't need the full model resident) and `llama-imatrix` offloads to the 3090, so
   it *works* — but cost ≈ 1–2 days incl. validation for an unproven gain.

## Flip to GO only if a trigger fires (measure, don't guess — owner rule `sweep-first-and-squeeze`)
1. **Quant-induced failure:** the agent shows tool-call JSON malformation or code-correctness
   regressions at Q4/Q5 that DISAPPEAR when the same traces are re-run at Q8_0 (offloaded, slow, A/B only).
2. **MTP starvation:** `gguf-dump` shows community quants drop the MTP tensors below ~Q6 AND measured
   acceptance is materially under the 0.76–0.82 reference — a custom recipe keeps MTP + embed/output at Q8.
3. **Missing size-point:** you need the ~18–18.5GB point (max quality that still fits 256k KV resident)
   and nobody ships it.
4. **Calibration mismatch:** KLD measured on OUR OWN agent transcripts is significantly worse for the
   community imatrix than for a quick custom-imatrix test quant of one size.

## Recipe (only if triggered)
- **imatrix corpus:** 50–100MB = our real agent transcripts (system prompt + tool schemas + tool
  outputs) + repo code in our languages + ~20% general text (avoid overfitting).
- **Tensor bit allocation:** `output.weight`/embeddings ≥ Q6_K; the 16 full-attention layers' attn
  tensors and the GDN conv/state-projection tensors one tier above base (GDN state-path quant
  sensitivity is UNVERIFIED — test both); **MTP tensors Q8_0**; base mix Q4_K → target 17.5–18.5GB.
- **Validate** against the community quant on: needle recall (`kv_recall_sweep.sh`-style), tool-call
  JSON validity, and a code correctness set — must WIN, not tie, to justify carrying our own artifact.

## Cheaper alternative before ever custom-quantizing
A/B **bartowski imatrix Q4 vs Unsloth UD-Q4_K_XL** on our traces first. If bartowski's tool-calling
calibration already wins, the custom-quant case is dead.
