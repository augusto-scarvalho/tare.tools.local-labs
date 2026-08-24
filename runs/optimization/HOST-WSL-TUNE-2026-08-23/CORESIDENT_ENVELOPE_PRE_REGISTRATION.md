# Candidate context envelope with embedding resident

Frozen after restoring port 8081 and observing only 1,735 MiB free with the canonical 131,072-token
text service.

- Keep the restored embedding service healthy on port 8081 throughout.
- Stop only `llm-inference.service`, enter LAB mode, and test canonical candidate `b10165-71676e46c`.
- Fixed text runtime: MTP n3, ubatch 512, q4_0/q4_0 KV, one slot.
- Context ladder: 32,768; 40,960; 43,008; 45,056; 49,152.
- Passing floor: at least 4,096 MiB free after text-model load, with port 8081 still healthy.
- Restore SERVE mode and the original 131,072-token service after measurement.

No deployment default changes automatically.
