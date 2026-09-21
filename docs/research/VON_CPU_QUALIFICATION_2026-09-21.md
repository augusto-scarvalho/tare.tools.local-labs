# Von 1.0: CPU qualification screen

Follow-up: [SDK and input-contract review](VON_INPUT_CONTRACT_REVIEW_2026-09-21.md)
confirms parity with the official Option-Marker implementation on these requests
and evaluates the separate SDK-default NLI path and descriptive inputs. This
original screen is specific to Option-Marker and our routing inputs; it is not
a general capability ranking. Original receipts below remain unchanged.

**Disposition: usable through an optional experimental Kernel worker; rejected
for automatic model/effort or flow routing on this screen.** Existing evaluator
configuration, GPU services and production routes were not changed.

## What ran

Model [wfzyx/von-1.0](https://huggingface.co/wfzyx/von-1.0), revision
`aa2fdc9630ecdadef32c56073553b3a69bed38bf`; upstream source
[`2656a69`](https://github.com/wfzyx/von/tree/2656a69be2ef0ebf2f3058302e52791f99cbd8de).
Apache-2.0 source retained in Kernel. Complete trained `option_marker.pt` verified
against the HF SHA-256; no quantization, training, synthetic tensor substitution
or hosted model call.

Acer Windows, Python 3.12, Torch 2.8.0+cpu, Transformers 4.57.6, FP32, two threads.
The adapter uses upstream prompt packing and scoring head/forward code, strict
weights loading and the artifact's temperature 2.2. It constructs the encoder
from config to avoid an unnecessary second 1.5 GB checkpoint, translates the
explicit RoPE frequencies for Transformers 4.57.6, and reads the serialized fast
tokenizer directly. CPU SDPA, no compilation. This is a qualification of that
integration, not a reproduction of the author's GPU benchmark or newer SDK stack.

All 15 preregistered requests were reused unchanged: six earlier joint strategy
cases, six LitJEV boundary cases, three Laya task-demand cases. There was no prompt
tuning, retrying failures or relabeling. The 12 routing rows cover six unique
synthetic scenarios in two orders; they are not 12 independent production tasks.

## Results

| Screen | Correct | Finding |
| --- | ---: | --- |
| Model/effort implementation + audit pairing | 3/6 | Always picked first strategy |
| Missing evidence / comparable cohorts / time | 1/6 | Failed four required abstentions |
| Task demand, including Portuguese | 2/3 | Portuguese concurrency task misclassified as insufficient |
| Total | **6/15** | Does not pass frozen quality gate |
| Semantic agreement across option orders | **0/6 groups** | No routing group was stable |

One concrete failure: both efforts accepted 93/100 deliveries, costing $0.80
versus $1.40 per accepted delivery. The model chose the first option in both
orders; when the expensive option came first its probability was **98.1%**.
That concentrated distribution did not indicate a correct decision.

All responses were valid with complete distributions, 72–696 actual input tokens,
zero generated tokens and no truncation. Loaded-model per-call median **3.062 s**,
range **0.844–8.750 s**, nearest-rank p95 **8.750 s**. Cold load including checkpoint
hashing was **19.516 s**; model process total **83.391 s**. Longer strategy requests
took 6.14–8.75 s. The preregistered 5-second p95 gate failed.

## Memory and protocol canary

The initial supervisor sampled the small Windows virtualenv launcher rather than
its interpreter child. Its 4 MiB peak in `RESOURCE.json` is **invalid model memory
evidence** and is preserved as a failed measurement.

A separately recorded follow-up, frozen before execution, repeated only the
English mechanical case through the actual stdin/stdout worker. It is excluded
from quality counts. The supervisor observed the whole descendant tree:

- Interpreter peak working set: **3,275,030,528 bytes (3.05 GiB)**.
- Sampled simultaneous process-tree RSS peak: **3,224,956,928 bytes (3.00 GiB)**.
- Whole invocation: **21.985 s**; valid answer, exit 0, no surviving processes.
- Stored weights: **1,581,316,050 bytes (1.47 GiB)**, plus ~3.6 MB config/tokenizer.

The follow-up demonstrates the short-request footprint; it does not establish
the original long-request batch's peak, which remains unmeasured. No GPU memory
was allocated by this CPU-only Torch build. Memory was reclaimed when the process
exited; only model artifacts remain on disk. CPU residency is feasible on acer,
but continuous residency and competing workloads were not qualified.

## Comparison and public claims

On the same six original strategy requests, previous observations were Laya 4/6,
OpenJEV CPU 6/6 and LitJEV over resident Qwen 6/6; Von scored 3/6. On the six boundary
requests, resident Qwen scored 4/6 versus Von's 1/6. Those models ran on different
devices/stacks, so these figures do not establish matched-compute speedups. No new
Jev API comparison was performed. Historical results and limitations are recorded
in [the LitJEV report](LITJEV_RESIDENT_QWEN_2026-09-21.md).

The author's headline accuracy, benchmark-table accuracy and stated calibration
temperatures refer to inconsistent or insufficiently distinguished configurations.
The actual marker calibration is 2.2. SDK usage is character-based and counts one
output per answer, and its multiple-question method iterates over questions; we
did not inherit those as measured tokens or one-pass multi-question claims.
Public performance claims were not used as our acceptance evidence.

## Delivered and validation

- Kernel optional `compute_plane.von_choice_worker` with pinned artifacts, CPU
  placement, actual token usage, input marker/size guards and existing protocol.
- Reproducible `tools/analysis/qualify_von.py`, frozen protocol and raw results.
- 33 Kernel assessment/adapter tests and 2 qualification-harness tests passed.
- Automatic routing remains unchanged. No quality improvements, real cost savings,
  broad model ranking or production qualification are claimed.

Evidence: [preregistration](evidence/von-20260921/PRE_REGISTRATION.md),
[protocol](evidence/von-20260921/PROTOCOL.json),
[results](evidence/von-20260921/RESULT.json),
[corrected memory/protocol canary](evidence/von-20260921/RESOURCE_CANARY.json).
The external supervisor and logs are also retained under
`.reconciliation/evidence-store/reports/von-20260921`.
