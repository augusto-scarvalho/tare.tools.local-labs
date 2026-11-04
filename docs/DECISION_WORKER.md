# Guarded decision worker

`tools/serving/decision_worker.py CONFIG.json` reads one bounded choice request
from stdin and writes one JSON answer or controlled failure. It never loads or
switches a GPU model. Resident Qwen inference requires the existing gateway and
shared GPU lease; admission failures may use the serialized, RAM-admitted CPU
OpenJEV fallback. An uncertain GPU submission never starts another evaluator.

## CPU admission

The config requires `cpu_max_total_tokens`, an integer from 1 to 24576. The
current flow deployment uses **2500**. This limits the sum of all sequential NLI
pairs, including the repeated premise, question, and options. It is an
operational resource cap, not a quality score or a universal latency guarantee.
Missing or invalid limits return `ASSESSMENT_EXECUTION_CONFIG_INVALID`.

The worker checks available RAM (36 GiB minimum), takes the existing CPU lock,
verifies pinned tokenizer/config hashes, and counts the unchanged pairs without
truncation before loading model weights. Requests above the cap return
`ASSESSMENT_CAPACITY_EXCEEDED` at `CPU_INPUT_PREFLIGHT`; explicit flow selection
remains available. Accepted requests still verify the complete model checkpoint
before inference. The 52-second CPU deadline remains enforced.

The cap follows the retained CPU probe: approximately 2050 aggregate tokens
took 29 seconds to score, plus an 8.67-second cold load. The real dogfood request
had six pairs totaling 3026 tokens and reproduced the old empty-output exit 124
after 52.35 seconds. These observations justify a conservative admission bound;
they do not qualify arbitrary prompts or guarantee completion below the cap.

## Failure receipts

CPU timeout and interruption write a bounded JSON failure before hard process
exit: `ASSESSMENT_TIMEOUT`/124 or `ASSESSMENT_CANCELLED`/128+signal. `stage` is
`CPU_INPUT_PREFLIGHT`, `CPU_MODEL_LOAD`, or `CPU_SCORING`. The protocol uses stdout
descriptor 1 even while third-party loading output is redirected to stderr.
CPU contention returns `ASSESSMENT_RESOURCE_BUSY`; RAM, checkpoint, and runtime
version failures retain their controlled reason codes. Consumers must inspect
structured failures even when the process exit code is nonzero.

## Deployment

Deploy the worker, matching Kernel `compute_plane/openjev_choice_worker.py`,
and config as one versioned directory. Copy the existing import roots and lease
paths; add `"cpu_max_total_tokens": 2500` to the copied config. Point the flow
assessment policy at the complete successor only after fixture validation.
Do not overwrite a prior worker version, change GPU service state, or relax the
52-second deadline to make a failed interactive request appear successful.

The dogfood correction changes only the flow assessment policy. Other policies
that still reference an older worker keep their existing behavior until migrated.
No daemon, checkpoint download, new model qualification, or automatic retry is
introduced by this correction.
