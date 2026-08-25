# SLX-05B execution failure

Status: `INVALID_IMPLEMENTATION`  
Promotion: none  
Successor: `SLX-05C`

The preregistered command exited with status 1 during the first `(batch=1,
context=128)` cell, before a receipt was written. The failure was a CUDA device
assert raised by `StaticLayer.update()` after repeated warmup decodes:

```text
index_copy_(): index 129 is out of bounds for dimension 2 with size 129
CUDA error: device-side assert triggered
```

Root cause: Transformers 5.3.0.dev0 `StaticCache` advances an internal
`cumulative_length` tensor and Qwen3.5 additionally mutates convolutional and
recurrent state. Supplying `cache_position` did not make repeated observations
overwrite a frozen position. Consequently SLX-05B did not satisfy its own
fixed-cache preregistration and is not scientific evidence for or against CUDA
Graph replay.

The canonical inference service remained active with the same PID and zero
restarts after the failed process exited. SLX-05C preregisters a post-prefill
state snapshot restored outside every timed observation.
