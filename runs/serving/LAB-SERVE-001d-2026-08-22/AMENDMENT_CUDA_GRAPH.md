# LAB-SERVE-001d infrastructure amendment — disable CUDA graphs symmetrically

Status: **FROZEN AFTER INVALID INFRA ATTEMPT, BEFORE RETRY**  
Date: 2026-08-22

The first MTP-off N=4 cell completed its request workload but the server then aborted with
`CUDA error: an illegal memory access was encountered` in `ggml_backend_cuda_synchronize`; the log
showed hundreds of graph reuses. SGLang consequently failed its final `/server_info` request and the
cell was correctly recorded invalid. The completed MTP-on cells and the failed off cell are retained
as an invalid infrastructure attempt and are not used in the paired analysis.

One symmetric infrastructure change is admitted: set `GGML_CUDA_DISABLE_GRAPHS=1` in both MTP arms.
Model, engine binary, common flags, concurrency matrix, workload, schedule, gates and the sole MTP arm
difference remain frozen. A recovery gate first runs two fresh MTP-off N=4 cells. Only two valid cells
allow the complete five-block paired campaign to restart from rep 1 in a new `qualified/` directory.

