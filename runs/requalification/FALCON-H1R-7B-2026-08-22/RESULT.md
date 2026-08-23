# Falcon-H1R-7B official Q8 — compact qualification result

Date: 2026-08-22  
Decision: **HOLD_ROLE / NOT PROMOTED**

## Outcome

Falcon-H1R-7B is technically viable on the RTX 3090 and strong on the compact tool/math panel, but
its bounded coding termination failed the frozen gate. Keep the official Q8 artifact and `lmctl`
profile as a research option; do not replace the 27B incumbent or open the context expansion.

| Gate | Result | Evidence |
|---|---:|---|
| Official artifact | PASS | 8,069,003,296 bytes; SHA-256 exactly `4c96b2...898a` |
| Residency | PASS | 10,048 MiB total GPU use with embedding active; 14,275 MiB free |
| Bounded smoke | PASS | 4/4 non-empty, 4/4 natural stop at recommended temperature 0.6/top-p 0.95 |
| Agent/tool | PASS | corrected endpoint run 8/8, dispatchable, no blind irreversible retry |
| Historical GSM failures | PASS | 4/5 strict; 5/5 format; no truncation |
| `Mbpp/260` at 2,048 | **FAIL** | empty content, `finish_reason=length`, 2,048 generated tokens |
| Context expansion | NOT OPENED | dependency-gated on all compact gates |

The non-promotional 4,096-token diagnostic reproduced the same coding failure: empty content,
`finish_reason=length`, 4,096 generated tokens, 55.95 seconds. This is a stable reasoning-budget/
termination failure on the discriminator, not a short 2k cutoff accident.

## Additional observations

- Model metadata reports 7,585,648,736 parameters, native context 262,144 and Q8_0 size
  8,063,277,440 bytes excluding GGUF overhead. The qualified server used a conservative 32,768-token
  context.
- The smoke's strict integer and exact-OK prompts were followed, but the conceptual hybrid answer
  incorrectly described a globally shared/pruned KV cache instead of distinguishing fixed Mamba state
  from the remaining attention KV. This weakens the qualitative result independently of the hard gate.
- The Falcon-LLM License applies; this technical result is not a legal/deployment clearance.
- The first agent invocation accidentally doubled `/v1` and received HTTP 404 for every request. It is
  preserved as `agent_INVALID_DOUBLE_V1.json` and excluded. The corrected invocation passed 8/8.

## Receipts

- `smoke.json`: `5839040fccb0f33b0760011135ffe4416694edad2eaebd617caa5ddc2b0d5596`
- `agent.json`: `8d6b1e6300bfa7d969795e512dee041a8a07841e501304073d29960df36eaa31`
- `gsm5/GSM8K_MANIFEST.json`: `dc6de4ea85282348f1c2abc9a68f1e887c22eded8e0675094064cddf39887620`
- `mbpp260/records.jsonl`: `2a526a72e47cfca8a8ae8014196dac7e00a8f3d1a0e20bd546cba05e38df5e16`
- `mbpp260_DIAGNOSTIC_4096/records.jsonl`: `dd32a6a22fda49fc25d2f4af6b53c1dae480b78e960226ab847e8fb359ae1df3`

Final machine state: canonical service restored on 8080, embedding healthy on 8081, SERVE lock
coherent, GPU power limit 420 W.

