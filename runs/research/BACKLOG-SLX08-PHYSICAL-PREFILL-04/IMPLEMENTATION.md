# BACKLOG-SLX08-PHYSICAL-PREFILL-04 implementation

Authorized by the user on 2026-08-29. This is a private-fork experiment and is
not a production or upstream-submission proposal.

The experimental `slop.cpp` server accepts the native completion request field
`slx08_selected_block_prefill`. The ON route is available only when the server
starts with `SLOP_EXPERIMENTAL_SLX08=1`. It rejects media, non-256-token-aligned
prompts, odd block counts and prompts shorter than four blocks. For a valid ON
request it retains alternating token blocks plus the final block, compacts
those tokens before model prefill and returns per-request route, block and token
telemetry. Explicit OFF requests preserve the full dense prompt and return the
same telemetry schema.

No CUDA kernel was added. The bounded treatment is server-side physical token
block compaction before the existing dense prefill. It must not be described as
a general block-sparse attention implementation.

## Bound implementation

- `C:/projects/slop.cpp/tools/server/server-context.cpp`: `85b55865ae6740cab1fe43b1298aedec539b2bed5b9b65b4c2046bcf0e80e246`
- `C:/projects/slop.cpp/tools/server/server-task.cpp`: `31e21f122a6c016114a1864a77d5acdd9c18e173372168b59bf04355a69b7656`
- `C:/projects/slop.cpp/tools/server/server-task.h`: `7682087acf996973958eebf1577078185fb66789135314a9ca2610bac9e13232`
- WSL build binary `/home/augus/build/slop-slx08/bin/llama-server`: `4395a601202ec76bcaef1d10db97849a92b311d8c31e4afce4d8b961609807a1`
- Runner: `tools/research/run_slx08_physical_prefill_r4.py`
- Fixture tests: `tests/test_slx08_physical_prefill_r4.py`

The build used CUDA architecture 86 and completed target `llama-server` at
private-fork base commit `34b3dac7c` with dirty source identity recorded by the
binary as build 10166.
