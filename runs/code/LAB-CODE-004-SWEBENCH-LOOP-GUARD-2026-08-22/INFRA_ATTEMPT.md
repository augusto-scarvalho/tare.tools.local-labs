# Pre-model infrastructure attempt

The first launch attempt produced three `CalledProcessError` records before any model call because
WSL resolved Docker Desktop's disabled-integration shim at `/Docker/host/bin/docker`. No trajectory
contains a model response and these records are excluded from every gate and score.

The correction prepends Docker Desktop's official Linux CLI directory while retaining the same
loopback engine endpoint. Valid generation writes to the distinct
`model-pilot-qwen38-loop-guard-valid/` directory; the invalid receipts are preserved separately in
`model-pilot-qwen38-loop-guard/`.
