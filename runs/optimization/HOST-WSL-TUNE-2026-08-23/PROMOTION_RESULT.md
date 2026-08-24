# Promotion result: b10165-71676e46c

## Outcome

PASS. The qualified canonical build was promoted to the active text service on
2026-08-23 at 14:23:07 -03:00. The rollback was not triggered.

## Active deployment

- Versioned root: `/home/augus/opt/slop.cpp/b10165-71676e46c`
- Executable: `/home/augus/opt/slop.cpp/b10165-71676e46c/bin/llama-server`
- Version: `b10165 (71676e46c)`
- SHA-256: `efb2f06c19d26605a1934c0a9ed5b65dd69034e8765f2d29d0426b7a011cfbe2`
- systemd drop-in: `/etc/systemd/system/llm-inference.service.d/binary-b10165.conf`
- Active text PID during the canary: `199264`
- Effective context: `131072`
- Effective speculative mode: `draft-mtp`, maximum draft length `3`
- Ubatch was not overridden and remains at the qualified implicit value of `512`.

The effective command line also preserved the q4_0 K/V caches, one parallel slot,
32 context checkpoints, flash attention, all GPU layers, metrics, Jinja, and
`--no-mmproj`.

## Gate evidence

- Text `/health`: HTTP 200.
- Embedding `/health`: HTTP 200.
- OpenAI-compatible text fingerprint: `b10165-71676e46c`.
- All three deterministic output hashes matched the qualified baseline exactly.
- Embedding request returned 768 dimensions.
- The executable mapped by the live PID had the qualified SHA-256.
- `llm-inference.service`: active/running, `NRestarts=0`, result success.
- `llm-embedding.service`: active/running on its original PID, `NRestarts=0`, result success.
- Stability observation: 112.3 seconds with the same text PID and both health checks at 200.
- Final GPU state: 22613 MiB used, 1710 MiB free, 31 C, 36.37 W.
- Free VRAM stayed above the pre-registered 1607 MiB floor.
- No matching text-service errors, WSL kernel GPU alerts, or Windows NVIDIA/Display events appeared since promotion.

## Rollback path retained

Remove only `/etc/systemd/system/llm-inference.service.d/binary-b10165.conf`, run
`systemctl daemon-reload`, and restart `llm-inference.service`. This restores the
original unit executable at `/home/augus/src/slop.cpp/build/bin/llama-server`.

No commit or push was performed.
