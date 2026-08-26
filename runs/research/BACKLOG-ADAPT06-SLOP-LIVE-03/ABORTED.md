# Execution aborted fail-closed

Root service handoff succeeded and both LoRAs converted, but the active `fable-tc-l1.0` GGUF is not the Qwen3.5-0.8B base on which the adapters were trained. llama.cpp rejected `blk.0.ffn_down.weight` shape mismatch before opening port 8080. The original systemd service and embedding service were restored healthy. A successor must convert and serve the frozen local Qwen3.5-0.8B base, and must pass both LoRAs as one comma-separated `--lora` argument as required by this binary.
