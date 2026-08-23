# LAB-IMG-002 result

## Decision: FIT PASS / MECHANISM PASS / QUALITY REJECT

The exact four-file FP16 SDXL source subset was verified at revision
`462165984030d82259a11f4367a4eed129e94a7b` (6,938,011,430 bytes). The model
loaded in 1.70 s, produced all 4/4 768x768 PNGs, and the identical-seed replay
was byte-identical. Generation took 5.25-5.69 s per image and peaked at 12,403
MiB total GPU memory used with the embedding service resident.

The unchanged semantic gate scored only 3/13 clauses (23.08%) and 0/3 complete
cases. It produced an unrelated laboratory scene instead of the requested
three-line poster, an illegible generic dashboard, and a crowded composition
whose requested colors and object positions were mostly wrong.

## Matched comparison

SDXL is roughly 9.5x faster than the 30-step quantized Qwen-Image arm and uses
about 673 MiB less peak inference memory, but Qwen-Image scored 10/13 clauses
versus SDXL's 3/13. Qwen-Image is therefore the retained image-generation
candidate from this bounded panel; neither model clears the frozen quality gate
or blind-human promotion boundary.

The remaining original FLUX candidate cannot be run unattended because the Hub
repository requires accepted access conditions and this machine has no active
Hugging Face authentication. Feature caching is not opened because the frozen
workload uses independent prompts/seeds rather than repeated denoising features;
Qwen's NF4/BF16 arm and SDXL's FP16 arm already exercise the applicable
quantization/mixed-precision levers. The M-B mechanism comparison is closed with
Qwen retained on quality and SDXL retained only as a high-speed baseline.
