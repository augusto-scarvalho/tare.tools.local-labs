# TRAIN-00B short training bakeoff result

Verdict: `REJECTED`

Across 60 steps per arm, the repository's custom GaLore implementation used
7.41 GiB at 0.91 step/s and diverged from loss 0.4647 to 623.8651. Full AdamW
used 7.20 GiB at 2.57 step/s and ended at 0.2960. LoKr PEFT used 4.04 GiB at
2.90 step/s and ended at 0.3370. GaLore therefore missed its memory,
throughput and convergence gates.

Provenance is complete. Receipt fingerprint:
`cdbde7410ea0b9dae9bebd2d68db43b84e090873071e196ff2c3d4a234f170ac`.
This is a micro-bakeoff of the custom optimizer, not a general result about the
GaLore paper or long-horizon training.

