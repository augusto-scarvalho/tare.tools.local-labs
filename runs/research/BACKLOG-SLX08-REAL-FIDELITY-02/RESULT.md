# BACKLOG-SLX08-REAL-FIDELITY-02 result

Executor outcome: `SLX08_FIDELITY_NEGATIVE_WITH_RETAINED_CONTEXTS_R2` pending
independent review.

The physical run completed all 12 frozen real-Qwen cells, retained exactly 36
context vectors, restored the serving baseline and independently reopened the
298,304-byte safetensors bundle. The fidelity result itself passed: recomputed
median corrected-context cosine was `0.9954493343830109`.

The frozen all-gates claim nevertheless failed because the scorer required
absolute agreement within `1e-9`. All 36 retained tensor hashes matched, but
11/12 cosine projections differed after reopening by only `5.96e-8` to
`2.38e-7`, consistent with FP32 reduction-order variation. The resulting
projection-match rate was `1/12` rather than `12/12`.

This packet is immutable negative evidence under its preregistered gate. It
does not establish a model/mechanism fidelity failure. Independent review must
decide whether this is a gate false negative. Any correction requires a
successor with a numerically justified tolerance frozen before rescoring the
same retained bytes; no new GPU inference is necessary.

No TTFT, runtime integration, production, quality, or solely-index-causal claim
is allowed.
