# A2 — performance levers for the dense-27B deploy model (fable-tc-l1.0)

Two levers probed 2026-08-06 on the deploy candidate `fable-tc-l1.0-Q4_K_M.gguf` (dense 27B, hybrid
GDN, 64 layers, Q4_K_M = 16 GB, fits FULLY in the 3090's 24 GB). Context: the project's decode stack
was tuned on the **MoE**; the **dense** is a different regime, so MoE conclusions don't all transfer.

## The biggest lever is already realized
`l1.0`'s **concision** (−55% reasoning tokens on math, −23% creative) is the dominant end-to-end
speedup — in wall-clock ("time to useful answer") it is ~2× vs plain Fable regardless of t/s — and
**MTP** (`--spec-type draft-mtp`, the head is preserved) adds +122–133% decode on top. Everything
below is incremental.

## Lever 1 — VRAM memory-bandwidth: HELPS the dense (unlike the MoE) ✅

A prior VRAM-OC A/B was **null on the MoE** (decode there is CPU-expert-streaming-bound). That result
is MoE-specific. The dense l1.0 keeps all weights in VRAM, so its decode is **weight-bandwidth-bound**.

Measured by locking the memory clock via `nvidia-smi -lmc` (no Afterburner needed) and running
`llama-bench` (decode tg128, -ngl 99 -fa 1):

| mem clock | decode (tg128) | prefill (pp512) |
|---|---|---|
| 9751 MHz | **51.9 t/s** | 1337 t/s |
| 5001 MHz | **25.7 t/s** | 1171 t/s |
| ratio | **2.02×** (clock 1.95×) | 1.14× |

**Decode scales ~linearly with memory bandwidth** (2.0× for 1.95× clock) — textbook bandwidth-bound.
Prefill barely moves (compute-bound). **Practical takeaway: a VRAM memory-clock OC helps dense decode
~proportionally.** +250 is already applied; pushing to ~+350/+400 in Afterburner (staying below the
+500 GDDR6X EDR threshold that regressed earlier) should buy a few % more decode. Reset clocks with
`nvidia-smi -rmc` after any A/B.

### Corollary — dense with CPU offload is a non-starter for decode
Measured the f16 (51 GB) at partial offload (`-ngl 24`, ~24/64 layers on GPU, ~2/3 on CPU):
**decode = 1.96 t/s** (prefill 26.8 t/s) — ~26× slower than the fully-in-VRAM Q4's 51.9 t/s. A dense
model uses every weight every token, so the CPU-resident half (streaming from ~89 GB/s system RAM)
dominates. This is why the deploy uses Q4 (fits fully in VRAM) and why MoE — not dense — is what the
project offloads.

## Lever 2 — imatrix re-quant: NULL at Q4_K_M ✗ (informative negative)

imatrix ("importance matrix") calibration allocates quant precision to the weights that matter most,
computed by running the model over a calibration corpus. The deploy Q4 was quantized **without** it.

Rebuilt the pipeline from source (the fp16 merge intermediate had been cleaned): regenerate fp16 l1.0
(`a2_merge_raw.py --lam 1.0` from base/tc/fable) → f16 GGUF → `llama-imatrix` over a 1.2 MB
**diverse + domain-matched** corpus (wikitext + Alpaca + our own stored l1.0 creative/reasoning texts,
564 chunks) → `llama-quantize --imatrix` → Q4_K_M. Held-out perplexity (wikitext-2 **test**, disjoint
from the calibration's train split):

| Q4_K_M | held-out PPL |
|---|---|
| no imatrix (deploy) | 5.8681 ± 0.067 |
| with imatrix | 5.8689 ± 0.067 |

**Δ = +0.0008 — statistically identical** (within the error bars). imatrix gave **no measurable Q4
quality gain**. Expected: imatrix's benefit is large at sub-4-bit (IQ1/IQ2/IQ3) but negligible at
Q4_K_M, where the naive quant is already near-lossless. **Do not bother imatrix-ing the Q4 deploy.**
The `l1.0-imatrix.dat` (+ `imatrix_corpus.txt`) are kept in `models/merges/` — reusable IF we ever
make a low-bit quant of l1.0 (the only regime where imatrix would pay). fp16/f16 intermediates deleted
(~119 GB reclaimed; regenerable from source in ~30 min).

## Levers already tested elsewhere (null/negative on the dense)
chunked GDN kernel (M4: −2–4% E2E at B=1), n-gram spec (S3: mtp-alone optimal), asymmetric/sub-4-bit
KV (A3: dense blocked by full-precision GDN recurrent state), CUDA-graphs/MMQ/ub2048 (already default).

## Deploy config (unchanged)
```
llama-server -m models/merges/fable-tc-l1.0-Q4_K_M.gguf \
    -ngl 99 -fa on -c 49152 --spec-type draft-mtp
```
Dense hybrid serves well to ~48–64k ctx; Q4_K_M is the measured sweet spot; MTP on.
