# Aborted before GPU measurement

Real Qwen symbols and physical Huffman/INT4 buffers were produced, but Triton's first JIT compilation failed because the nested kernel resolved `tl` through module globals while it had been imported into function-local scope. No timing, receipt or result was emitted. Systemd remained `MainPID=29428`, `NRestarts=0`, and embedding health returned HTTP 200.

After transition to `BLOCKED`, only the compiler namespace binding was corrected. A successor must rerun encoding, exact decoding and all timing batches from scratch under a new implementation digest.
