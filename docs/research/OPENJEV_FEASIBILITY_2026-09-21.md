# OpenJEV: integration, memory and routing diagnostic

Status: measured bounded diagnostic; **UNQUALIFIED** for automatic routing.
No fine-tuning, fleet promotion, service configuration or research-backlog state
was changed. User explicitly requested CPU/GPU/hybrid feasibility and coexistence.

## Frozen source and method

- Model: [AlexWortega/openjev](https://huggingface.co/AlexWortega/openjev), revision
  `f004f37e52695d6ddfb914a64dbf93942839ba1e`, `qwen3.5-4b-nli-v2`.
- Original BF16 checkpoint: 9,078,635,984 bytes; all four downloaded files total
  9,098,629,601 bytes. This is not the quantized generative Qwen from the 19 ms post.
- Hardware: aaaaa, i7-13700K, RTX 3090 24 GiB, WSL2 Linux, four PyTorch threads.
  CPU uses FP32 because this CPU has no AVX512-BF16. CUDA/hybrid use BF16.
- Python 3.12.3, torch 2.14.0+cu126, transformers 5.17.0, accelerate 1.14.0.
  The existing ComfyUI Python was used read-only; accelerate was installed only
  in an experiment-local target. No ComfyUI environment package was changed.
- Three NLI controls, followed by the exact frozen six-case OS/Laya joint-pair
  protocol. No prompt tuning. Three hypotheses per decision, sequential pairs,
  no output generation, no shared prefix optimization. Complete input only.
- GPU probes run under `gpu_lease_run.py`, finite five-second admission wait,
  600-second placement deadline and 180-second case/load alarm. The existing
  process-tree supervisor retains the lease until descendants exit.
- Monitored reserves: 16 GiB available system RAM and 4 GiB free VRAM;
  GPU probing started with an empty ComfyUI queue. Peaks below are observed,
  not guarantees for larger inputs. GPU sampling is once per second; PyTorch
  allocator peaks additionally retain peaks between samples.

## Results

| Placement | Process RSS peak | OpenJEV GPU footprint | Warm decision median | Cold load + verification | NLI / pairs |
| --- | ---: | ---: | ---: | ---: | --- |
| CPU FP32 | 17.96 GiB | No CUDA tensors | 29.25 s | 8.67 s | 3/3; 6/6 |
| Full CUDA BF16 | 4.39 GiB | 9.12 GiB sampled increase | 0.535 s | 11.19 s | 3/3; 6/6 |
| Hybrid BF16, CPU weight offload | 9.59 GiB | 2.48 GiB sampled increase | 4.094 s | 7.35 s | 3/3; 6/6 |

GPU device baseline was 579 MiB. Full-CUDA total used memory peaked at 9,918 MiB;
hybrid at 3,118 MiB. Full-CUDA PyTorch peak allocated/reserved memory was
8.641/8.809 GiB; hybrid was 2.027/2.240 GiB. RAM RSS includes memory-mapped
checkpoint pages and is not additive to the OS's available-memory counter.
CPU held 16.91 GiB of parameter tensors; BF16 holds 8.455 GiB in total.

CPU warm decisions ranged 28.96–30.65 seconds, GPU 0.525–0.541 seconds, hybrid
3.872–4.208 seconds. These are complete three-hypothesis decisions, not per-option
times. Initialization, process launch, transport, lease wait and Qwen switching
are outside the warm decision column. Uninstalled optimized causal-convolution
and linear-attention kernels used the framework reference implementations.

Hybrid means CPU **weight storage** with transfer for GPU execution. It does
not mean CPU computation for the first 28 layers. Its initial attempt failed
before NLI inference because an input was sent to an Accelerate `meta` weight
placeholder. The failure remains in `HYBRID.json`; `HYBRID_2.json` changes real
input placement only. No failed result was overwritten or retuned away.

## Can it coexist with Qwen3.8?

**CPU OpenJEV + GPU Qwen3.8: observed working.** While the FP32 helper remained
resident, a normal gateway call loaded Qwen and returned two output tokens.
The helper then completed all six decisions while Qwen stayed resident. This
proves residency and bounded inference availability, not a concurrent-throughput
or long-context stress test. Available WSL RAM stayed at least 23.31 GiB.

With Qwen loaded, device memory used was 18,978 MiB and free was 5,345 MiB:

- Full BF16 OpenJEV cannot fit in the remaining VRAM even from weights alone.
- Hybrid's observed 2,539 MiB increment would leave about 2,806 MiB free by
  simple subtraction, below the configured 4,096 MiB reserve. This is capacity
  arithmetic, not a simultaneous run. Larger contexts/other processes can worsen it.
- The existing exclusive lease does not admit independent concurrent GPU
  inference anyway. GPU/hybrid helper execution must be serialized, including
  unloading the text backend; warm helper time does not include that penalty.

The practical coexistence route today is CPU OpenJEV on aaaaa. GPU-only is the
fast standalone option. Shared GPU serving or quantizing the NLI checkpoint is
a separate follow-up; the generative model's GGUF cannot substitute for its
trained NLI classification head.

## Quality interpretation and integration

All three placements passed the basic three-class NLI controls and selected
the expected pair in both orders of all three synthetic comparisons: cheaper
accepted delivery, quality floor and adequate lower effort. Laya scored 4/6
on the same frozen strategy cases; OpenJEV scored 6/6. This is a small diagnostic,
not proof of general superiority, calibrated confidence or real monetary savings.

Kernel's optional `compute_plane.openjev_choice_worker` implements the existing
Choice interface. Its CPU CLI rejects less than 36 GiB available RAM before
loading and supports the existing finite assessment timeout. OS policy version 3
can select it without changing tools, flows or role admission. Automatic adoption
remains disabled; use shadow observations and fresh held-out tasks first.

The remote staged checkpoint and source live under
`/home/augus/experiments/openjev-20260921`. No always-running helper service was
installed. Model weights occupy about 8.47 GiB of disk and were retained for reuse.

## Recovery and evidence

Gateway PID remained **381697**, restart count **0**. All helper descendants
were reaped; GPU lease is free, ComfyUI queue empty, and GPU memory returned to
579 MiB. Text residency was restored to its initial empty state; the gateway
will load Qwen on the next ordinary request.

Evidence is under [evidence/openjev-20260921](evidence/openjev-20260921/):
`PRE_REGISTRATION.md`, pinned `checkpoint.json`, `PROTOCOL.json`, complete
`CPU.json`, `CUDA.json`, failed `HYBRID.json`, repaired `HYBRID_2.json`,
`HYBRID_REPAIR.md`, prior worker source, and `RECOVERY.json`.
Tests: Kernel Choice/OpenJEV tests and lab GPU-admission tests run without ML
downloads or GPU use. A successful screen cannot promote a model.
