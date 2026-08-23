# LAB-SERVE-001d result

Decision: **BLOCKED_RUNTIME_CRASH / NO TPOT CONCLUSION**  
Date: 2026-08-22

The frozen MTP-on first block completed all five concurrency cells and both length diagnostics. The
first paired MTP-off N=4 cell completed its request workload, but the llama.cpp server then aborted
with a CUDA illegal-memory-access error in `ggml_backend_cuda_synchronize` before SGLang's final
`/server_info` collection. The cell is invalid and the MTP-on cells are unpaired evidence only.

A frozen one-variable infrastructure amendment disabled CUDA graphs symmetrically and required two
valid MTP-off N=4 recovery cells before restarting the campaign. The first recovery cell reproduced
the same illegal-memory-access abort; the second had no live endpoint. The recovery gate therefore
failed, and no further blocks were spent.

No concurrency crossover or MTP effect is inferred from this campaign. The prior qualified
LAB-SERVE-001b/001c conclusions remain authoritative within their tested topologies. Raw valid and
invalid receipts, exact argv, logs and the amendment are retained in this directory.

