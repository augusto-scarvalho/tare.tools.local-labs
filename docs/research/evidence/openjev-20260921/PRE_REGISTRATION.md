# OpenJEV alternative: bounded integration and resource feasibility

User-authorized 2026-09-21. No fleet promotion or automatic routing activation.
Checkpoint: AlexWortega/openjev at f004f37e52695d6ddfb914a64dbf93942839ba1e,
qwen3.5-4b-nli-v2. Original BF16 checkpoint, text-only inputs, no new training.
Download only config/tokenizer/model files; preserve all failures and raw observations.

Implementation: optional Kernel NLI/Choice worker, Local Labs bounded probe and
receipt. Existing OS shadow Choice interface reused. Pair scores are normalized
entailment scores, not calibrated probabilities of successful Work. Retain raw
three-class logits and NLI probabilities in probe evidence.

Cases: 3 obvious NLI controls (entailment/contradiction/neutral), then the frozen
6-case Laya joint-strategy diagnostic from OS. No prompt tuning after results.
Inputs must be fully tokenized without truncation; maximum 4096 tokens per pair.
At most 9 assessed cases per placement, CPU first, then standalone CUDA, then
hybrid (4 final text layers on CUDA) if supported and within resource limits.
A failed dependency/setup attempt is not an inference attempt and remains recorded.
CPU probe uses 4 threads; GPU probes require the same exclusive GPU lease used
by serving. 600-second deadline per placement, at most 180 seconds per case.
Reserve 16 GiB available host RAM and 4 GiB free VRAM; refuse GPU admission
when ComfyUI queue/lease is occupied. Stop owned child on sustained memory breach.
Never stop training, ComfyUI, or other user jobs. No service restart/config change.

Measure cold load/hash time, loaded parameter bytes, sampled RSS/host available
RAM, CUDA allocated/reserved/peak and nvidia-smi used/free memory, complete case
latency and request token counts. Record environment and model SHA-256.
GPU results are conditional on current idle admission. Preserve initial empty
text residency after tests. CPU/Qwen coexistence: one bounded Qwen request while
CPU helper is resident; record observed text-model VRAM and health. Concurrent
GPU execution is not admitted by today's exclusive lease; capacity arithmetic
alone cannot qualify simultaneous inference. Report this boundary explicitly.

Success for adapter integration: correct label mapping, finite logits, zero
output tokens, both orders preserved, 3/3 NLI controls. Joint routing diagnostic
requires 6/6; passing is still UNQUALIFIED. Failed routing quality does not erase
valid footprint measurements. No GPU fit/latency/savings claims from proxies.

Pre-execution precision selection: aaaaa is i7-13700K/AVX2 without AVX512-BF16.
Use float32 for CPU execution; BF16 for CUDA and hybrid. Report each separately.
Reuse the installed ComfyUI Python read-only (torch 2.14.0+cu126, transformers
5.17.0); put accelerate 1.14.0 in an experiment-local target, never modify the
ComfyUI environment. No NLI inference has run at this point.
