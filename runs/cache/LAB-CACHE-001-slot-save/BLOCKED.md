# LAB-CACHE-001 explicit slot save/restore — blocked launch

The probe and reproducible IQ4_XS launcher were implemented and self-checked, but the required launch
could not be authorized in this session.

Evidence on 2026-08-21:

- `llm-inference.service` owns port 8080 and declares `Restart=always`.
- Directly terminating its llama-server PID caused systemd to restore the exact baseline immediately.
- `systemctl stop llm-inference.service` returned `Interactive authentication required`.
- `sudo -n systemctl stop llm-inference.service` returned `sudo: a password is required`.
- The restored unit is active/running and `/health` reports `{"status":"ok"}`.

Running a second copy is not safe on the 24 GB GPU while the baseline consumes about 18.5 GB. Therefore
slot file save/erase/restore, clean mmap on/off, and engine-swap experiments remain blocked until the service
can be paused through its authorized control path. No result is inferred from this blocker.
