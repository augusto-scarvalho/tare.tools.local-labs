# SLX-09B 2:4 mask result

Verdict: `REJECTED`

The provenance-complete rerun reproduced the narrow calibration result. Wanda
achieved exact 2:4 conformity and improved MSE by 87.69% versus magnitude
pruning, but logit cosine similarity was 0.77734 against the 0.90 gate. The
weights remained dense tensors with zeros; no sparse packing or throughput was
measured.

Receipt fingerprint:
`e4b2c98cf10c35c6c379852f73cacb4c35874bb91c7020eec95c91e67c472d3e`.
The conclusion is limited to this zero-shot mask and calibration input.

