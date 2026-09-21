# Von documentation audit — frozen before inference

Reproduce pinned upstream OptionMarkerBackend on CPU using Transformers 5.17.0
(the checkpoint version), unmodified SDK source and official constructors. Local
artifact paths only, offline execution. No GPU or service changes. Preserve old
results; package upgrades are isolated in an overlay, leaving Laya environment intact.

Marker: 15 exact original requests for implementation parity, 15 semantically
projected requests with natural-language criteria and one decision question,
10 public classification controls selected by indices before seeing predictions.
NLI (the documented API default): same 15 projected + 10 public controls. 65
forward API calls maximum, no retries or result-driven prompt edits. English and
Portuguese demand requests are unchanged. Original/new formats are separate arms.

The projection preserves cohort, acceptance count, implementation/audit identities,
known/unknown total cost and time. It removes repetitive provenance scaffolding,
clarifies missing-cost abstention and uses descriptive option text. This is a
post-hoc input-contract experiment, not a held-out model improvement benchmark.
Ten public controls are a software/model sanity check, not a benchmark reproduction.

Record full input tokens and refuse SDK truncation before inference. All candidates
must be present. Marker probability parity tolerance 0.0002 accounts for rounding;
record actual deltas, not merely matching winners. Compare tokenizer ID sequences
between 4.57 adapter and reference when diagnosing any parity failure. Disclose
backend differences: marker calibration comes from local file, NLI public method
uses its default temperature argument (1.0), despite reading calibration metadata.

Report quality counts, option-order semantic agreement and abstention separately.
No production promotion regardless of the outcome. Each backend gets one process,
600-second deadline, two CPU threads, >=6 GiB RAM at start, terminate below 2 GiB
available RAM, track interpreter descendants. Preserve every partial or failed run.
