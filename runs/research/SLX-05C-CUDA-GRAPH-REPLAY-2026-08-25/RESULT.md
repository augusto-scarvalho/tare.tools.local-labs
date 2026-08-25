# SLX-05C result

Status: `REJECTED_OR_UNVERIFIED`  
Promotion: none  
Successor: `SLX-05D`

SLX-05C completed all five eager matrix cells with a fixed restored cache and
complete provenance, but CUDA Graph capture/replay was rejected in every cell.
The immediate cause was an in-place restore of cache tensors created under
`torch.inference_mode()` being issued outside that mode immediately before
capture/replay. No graph timing or semantic comparison was produced.

The raw receipt and its fingerprint
`a79c8fe583d8fa41eb63f2c7c567eb6967807e2f5cf2c576953209685b156a6f`
are preserved under `raw/receipt.json`. This run is evidence of an execution
path defect, not evidence about CUDA Graph performance.
