# GPT-OSS 20B Q4 compact qualification result

Decision: **HOLD_AGENT**  
Date: 2026-08-22

The exact 11,624,759,488-byte artifact matched its upstream SHA-256 and left 11,874 MiB free at 32,768
context. The agent suite scored 6/8 with no blind irreversible retry: it emitted only the Lisbon half of
the required parallel weather calls and failed to dispatch status inspection after an unknown transfer.
The frozen 7/8 gate failed, so cache, GSM and MBPP were not spent.

A separately frozen mitigation then added only the qualified irreversible-recovery policy. It repaired
the target case in 5/5 seeds and in the full matrix, with zero blind retries. The full score nevertheless
remained 6/8: `parallel` and `multi_turn` emitted no tool calls. The mitigation therefore also stopped at
the agent gate; the decision remains **HOLD_AGENT**.

Agent SHA-256: `14e05e6e63cb4b42cea4929ee99140001f1d8c42735074c8f7fc83b432a17537`.
Policy-mitigation SHA-256: `7ff5696d146db019f4a6685552ef5674b85c75fa4cf43a84da8f4d93f7de98c6`.
