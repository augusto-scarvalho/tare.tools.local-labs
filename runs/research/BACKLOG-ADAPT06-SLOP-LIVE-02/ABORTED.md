# Execution aborted fail-closed

Both LoRA conversions succeeded, but the unprivileged `systemctl stop` did not retire the original daemon. The subsequent health request therefore reached the original route, whose empty `/lora-adapters` response triggered the materialization abort before any baselines. The `finally` path left both 8080 and 8081 healthy. A successor must control the system unit as WSL root and prove 8080 is down before launch.
