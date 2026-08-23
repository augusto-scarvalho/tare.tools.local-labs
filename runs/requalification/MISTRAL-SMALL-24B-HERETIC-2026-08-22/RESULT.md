# Mistral Small 3.2 24B Heretic compact qualification result

Decision: **HOLD_FIT / AGENT 7 OF 8**  
Date: 2026-08-22

- Artifact identity: exact 14,333,923,776 bytes and SHA-256 match.
- Agent gate: the initial `sequential` HTTP 400 was an invalid short fixture ID. The frozen compatible
  rerun passed 7/8 with zero blind retry; the remaining miss did not call status after unknown transfer.
- Cache compatibility: the first three cases passed 3/3 at 16,384, while the long fixture tokenized to
  30,057 tokens. At the required 32,768 allocation the server left **4,048 MiB free**, 48 MiB below the
  frozen 4,096 MiB reserve, so the final cache gate was not opened.
- GSM and MBPP were correctly not spent.

The model remains useful as its existing bounded writing judge but is not promoted as a 3090 worker.
Corrected agent SHA-256: `4be6f69502080d62f5c3d20bfb204de7f4754de87e6ea0b898ef2ff8263f65e6`.
