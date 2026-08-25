# REP-02B precision-tail result

Verdict: `REJECTED`

After correcting the comparator to use the same next-token decode from cloned
post-prefill caches, precision-tail-64 reduced simulated-INT4 logit MSE at 4096
tokens from 0.019287 to 0.015503, a 19.62% reduction against the 50% gate. The
needle passed and the analytical packed-storage estimate was 70.8%, but no
packed codec or VRAM saving was measured.

Provenance is complete. Receipt fingerprint:
`9242f9143e31846db3b6c60e3b77e8869f6b870f5cc177a743d440e809af928b`.
The original REP-02 numbers are `SUPERSEDED_INVALID_COMPARATOR`.

