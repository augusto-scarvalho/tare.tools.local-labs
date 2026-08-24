# 57,344-token reserve runtime A/B

Frozen after the candidate envelope identified 57,344 as the largest tested context preserving at
least 4,096 MiB free and before this A/B.

- Binary: canonical candidate `b10165-71676e46c`.
- Context: 57,344; q4_0/q4_0 KV; one slot; flash attention; batch 2,048.
- Control: MTP n3 with ubatch 512.
- Challenger: MTP n4 with ubatch 1,024.
- Three deterministic equivalence probes and three counterbalanced short/long repetitions per arm.
- Hard gates and recommendation threshold are unchanged from `RESERVE_PRE_REGISTRATION.md`.

No production mutation is automatic.
