# SLX-02 APEX4 RTX 3090 qualification - result

## Verdict

`BLOCKED_PUBLISHED_CHECKPOINT`; no port study opens.

The official accuracy package built for `sm_86` and its documented grouped
kernel matrix passed numerical assertions. The only downloaded public model
artifact could not be loaded because both published safetensors shards are
internally inconsistent. The separate end-to-end performance package cited by
the project was not present in the accuracy repository, so kernel timing cannot
satisfy the promotion gate.

## Frozen identities

- Source: `APEX4-W4A4/APEX4-W4A4`
- Source commit: `6ffc9ea07b7da8fc0b0d8937ccd0012878ae337d`
- Candidate: `APEX4-W4A4/Qwen2.5-7b-g128`
- Candidate revision: `3be1cefb76f45c734ad8f4102394cadd5cf6a691`
- Host: NVIDIA RTX 3090, compute capability 8.6, driver 591.86
- Environment: CPython 3.11.15, PyTorch 2.5.1+cu124, CUDA toolkit 12.4
- Virtual environment: `/home/augus/.venvs/apex4-20260824`
- Local checkpoint: `/home/augus/models/apex4/qwen2.5-7b-g128`

## Sequential results

| Phase | Result | Evidence |
|---|---|---|
| Source/package audit | `PASS_WITH_SCOPE_LIMIT` | Accuracy repository has kernels, evaluation, and checkpoints; quantization/calibration code is absent and throughput is attributed to a separate vLLM package |
| Build | `PASS` | Extension compiled with `CUDA_HOME=/usr/local/cuda-12.4` and `TORCH_CUDA_ARCH_LIST=8.6` |
| Auxiliary very-few-stages test | `UNSUPPORTED_COMBINATION` | Test requested `thread_k=64, thread_n=256, groupsize=64`, for which the released dispatcher has no implementation |
| Documented `test_groups` | `PASS` | 60 configurations completed their numerical assertions; 100 timed iterations per configuration |
| Checkpoint identity | `PASS` | Local file sizes exactly match the Hub revision's published file sizes |
| Checkpoint load | `FAIL_SOURCE_ARTIFACT` | Both shards raise `safetensors_rust.SafetensorError: MetadataIncompleteBuffer` before GPU execution |
| WikiText-2 perplexity | `NOT_RUN` | Load failure prevents a valid model instance |
| End-to-end throughput | `NOT_AVAILABLE` | Relevant performance package is separate from the released accuracy package |
| Service restoration | `PASS` | Inference and embedding services active; exact no-thinking canary returned `apex4-baseline-restored-ok` |

The timing values emitted by `test_groups` are not benchmark evidence: the
canonical Fable service was intentionally still active during the build and
kernel-correctness phase, producing high timing variance. Only the numerical
assertions are admitted from that phase.

## Published-artifact failure localization

The two downloaded shards have the exact byte counts advertised by the frozen
Hub revision:

- shard 1: 1,088,290,816 bytes
- shard 2: 1,087,881,216 bytes

Despite that identity match, `safe_open` fails on both. Header inspection also
shows offsets beyond the available payload: shard 1 contains an offset ending
at 4,557,221,844, and shard 2 ends at 1,089,994,752 while its file is only
1,087,881,216 bytes. This is not a partial local download that retrying can
repair; the published blobs at the pinned revision are inconsistent.

## Host changes and restoration

CUDA 12.4 compiler and development libraries were installed side-by-side at
`/usr/local/cuda-12.4`; the existing CUDA 13.3 installation was not replaced.
Only `llm-inference.service` was stopped for the attempted model evaluation.
The embedding service on port 8081 remained active throughout, and the Fable
service was restarted and verified afterward. Fan Control and MSI Afterburner
were untouched.

## Decision

- Do not port APEX4 into `slop.cpp`.
- Do not download the other three public checkpoints: the first preregistered
  artifact failed before the causal question could be exercised.
- Reopen only if the publisher replaces the checkpoint with loadable,
  revision-pinned shards and releases a reproducible end-to-end package whose
  relevant gain clears the standing promotion threshold.
