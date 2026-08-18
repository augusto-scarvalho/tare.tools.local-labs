# Campaign A1 — Windowed & Adaptive Multi-Token Prediction (MTP) ⚡

## Overview
Investigated speculative decoding using Multi-Token Prediction (MTP) heads across context depths up to 262k tokens, testing whether restricting draft attention to a local sliding window (`windowed-MTP`) resolves the draft KV-cache overhead on hybrid architectures (GDN/SSM + Transformer).

## Key Files & Artifacts
- [`A1_WINDOWED_MTP.md`](A1_WINDOWED_MTP.md): Full consolidated scientific record and empirical measurements.
- `tools/gates/verify_mtp.py`: MTP verification and acceptance rate evaluator.
- `tools/gates/verify_mtp_long.py`: Extended long-context draft verification.

## Core Conclusion
**CLOSED / NEGATIVE on Windowing**: The draft KV-cache overhead does not dominate at any reachable context length ($\le 262k$). MTP decode advantage monotonically increases with depth (+176% at 262k). Operationalized full un-windowed MTP with $n_{max}=3$ for production.
