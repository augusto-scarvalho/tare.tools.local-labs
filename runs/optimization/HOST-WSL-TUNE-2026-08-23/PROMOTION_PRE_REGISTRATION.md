# Promotion pre-registration: b10165-71676e46c

Promotion authorized at 2026-08-23T14:21:29-03:00.

## Scope

- Replace only the active text-server executable with the previously qualified canonical build.
- Preserve model, alias, port, context size 131072, flash attention, GPU layers, metrics, Jinja, MTP n3, implicit ubatch 512, and all other environment-provided arguments.
- Do not restart or reconfigure `llm-embedding.service` on port 8081.

## Baseline

- Old binary: `/home/augus/src/slop.cpp/build/bin/llama-server`
- Old version: `b9863 (5e7f6271c)`
- Old SHA-256: `73a623fad5bd632cebc71e32e4c399621332b3a566ab0cc1d46d4906ffd28b93`
- Candidate source SHA-256: `efb2f06c19d26605a1934c0a9ed5b65dd69034e8765f2d29d0426b7a011cfbe2`
- Text health: HTTP 200
- Embedding health: HTTP 200
- Text-service restarts: 0
- GPU memory before promotion: 22588 MiB used, 1735 MiB free

## Canary gates

1. Deployed binary has the qualified SHA-256 and reports `b10165 (71676e46c)`.
2. Text health on 8080 and embedding health on 8081 return HTTP 200.
3. Text API reports system fingerprint `b10165-71676e46c`.
4. Three deterministic content hashes equal the qualified baseline hashes:
   - `f3c870d45ce7f02c7f12160caf72ec95f9958aaf0eba20754fc8921526c8eb49`
   - `60787c7b9416b4d164feaaa7e29200a0afa3455220ef966465fa07ae341229cc`
   - `9bb1a13763d8c274fc557bc15851c87a9c2ea2ad7de55f590a8e0efd7b02b30a`
5. Embedding endpoint remains functional and returns dimension 768.
6. `NRestarts=0`, no new GPU/kernel fault signatures, and at least 1607 MiB free VRAM.
7. Same text-server PID remains healthy through a 70-second stability window.

## Automatic rollback

If any gate fails, remove only `/etc/systemd/system/llm-inference.service.d/binary-b10165.conf`, reload systemd, restart `llm-inference.service`, and verify the old fingerprint and both health endpoints.
