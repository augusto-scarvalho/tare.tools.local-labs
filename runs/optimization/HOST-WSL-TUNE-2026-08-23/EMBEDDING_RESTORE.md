# Embedding endpoint restoration plan

Captured before mutation on 2026-08-23.

- Port 8081 was down and no embedding systemd unit existed.
- The last authoritative handoff recorded a manual process using the deployed `llama-server`,
  `nomic-embed-text-v1.5.Q8_0.gguf`, context 32,768, parallel 8, embedding mode, and mean pooling.
- Restore that exact argv as `llm-embedding.service` with bounded restart behavior.
- Keep the unit only if ports 8080 and 8081 are healthy together and the text service remains stable.
- Rollback: `systemctl disable --now llm-embedding.service`, remove the unit file, and reload systemd.
