# LitJEV method on the resident Qwen3.8

Status: measured diagnostic; **UNQUALIFIED**, shadow only. The transport and
six original cases passed; the extended screen failed unknown-cost abstention.
No production route, services, package environments or model weights changed.

## What was actually evaluated

Upstream [LitJEV](https://github.com/zhengxuyu/litjev) revision
`e7fb109a7466da9709028eb9c4e9f16eaeb4e2a3`, using its actual `prompting.py` and
`scoring.py` with a bounded native slop.cpp transport. The request states,
questions and choices of the earlier OpenJEV/Laya protocol were unchanged.
Upstream source, license and NOTICE are retained with the evidence.

This evaluates the single-question fast decision method on the existing Qwen
GGUF. It does **not** reproduce the upstream Transformers/BF16 engine, batched
cached branches, learned confidence head, slow thinking path or H100 timings.
The prefix uses the installed qualified Qwen chat template with thinking off;
the question suffix and restricted normalization are upstream code.

All model inference went through the existing port-8080 gateway and its GPU
lease. Loopback port18080 was used only for template rendering/tokenization.
No second model instance was loaded. Each space-prefixed answer code was checked
as one distinct token at the complete suffix boundary. Every candidate appeared
in the returned top128 **pre-sampling** logprobs. A missing candidate would fail
rather than become zero probability. No grammar, logit bias or answer forcing.
Restricted softmax over candidate logprobs equals softmax over their logits
because the shared vocabulary normalization cancels.

The native API requires **one sampled output token per call** to expose the
first-position logprobs; that sampled token is not the decision. The winner is
computed from the complete candidate distribution. This is not zero-output-token
inference. Cache reuse was disabled and all prompts were processed completely.

## Results on aaaaa (RTX 3090)

| Diagnostic | Correct | Median warm native completion call | Input length |
| --- | ---: | ---: | ---: |
| Frozen OpenJEV/Laya comparisons | 6/6 | 0.876 s | 742–756 tokens |
| Fresh synthetic boundary comparisons | 4/6 | 0.523 s | 334–368 tokens |

Original-case range: 0.861–1.232 s. Fresh-case range: 0.467–0.558 s.
These times include the local HTTP completion request, but exclude template
rendering/tokenization, caller SSH, admission under competing workloads and cold
model loading. Loading/profile acquisition took 12.07 s. The entire probe,
including one literal-choice control, loading and recovery, took 21.92 s.

All six scenario groups chose the same semantic answer in both option orders;
probabilities changed with order. Twelve cases are only six unique synthetic
scenarios, not twelve independent real work tasks. The control chose beta as
specified. All responses were valid, untruncated and one output token long.

The original six-case diagnostic previously gave Laya 4/6 and OpenJEV 6/6.
OpenJEV CPU warm median was 29.25 s for its three sequential NLI hypotheses.
This comparison concerns different models and methods; no broad superiority,
matched-compute speedup or real monetary savings are established.

## Failure that matters

When one adequate pair had unknown cost and another had a known cost, the new
fixture required abstention: it asks for the least expensive adequate pair,
which cannot be established from these inputs. Qwen preferred the known-price
pair in both orders. Its normalized probability for that choice was 85.4% in
one order and 46.4% in the other (abstention was 42.9% in the latter).

Choosing a price-known pair may be defensible under a different policy, but it
fails this preregistered least-cost evidence rule. Do not rewrite the labels or
call probability concentration calibrated confidence. Production should enforce
cost comparability and missing-evidence rules deterministically and/or define
an explicit alternative objective before admitting such recommendations.

It correctly abstained for mismatched cohorts and chose faster complete delivery
when cost and acceptance were equal, in both orders. No tuning or extra attempts
followed the failures.

## Resource footprint and recovery

Model: existing qwen38 Qwen3.8-27B UD-Q4_K_XL; build `b10165-71676e46c`,
context32768, one slot. Full live profile and configured artifact identity are
in the receipt; the GGUF was not rehashed by this probe.

GPU usage peaked at **19,017 MiB (18.57 GiB)** with **5,306 MiB (5.18 GiB)**
free. This is whole-device use with Qwen, sampled about every 200 ms, not isolated
LitJEV overhead or a guarantee for long contexts/concurrent embeddings.
No second set of model weights was resident. Available WSL RAM stayed at least
36.48 GiB. The embedding service remained inactive throughout; this probe does
not qualify concurrent index/search embeddings.

Initial and final text residency were empty. The canonical lease supervisor
unloaded only this probe's Qwen, reaped its child and confirmed lease release.
Device memory returned to 604 MiB. ComfyUI queue stayed empty and its service PID
and restart count were unchanged. The embedding service was unchanged.

## Disposition

Promising candidate for evaluation when Qwen is already resident and the GPU is
admitted. OpenJEV CPU remains the resource fallback candidate when images/training
own the GPU and RAM allows it; deterministic routing remains the last fallback.
This probe does not implement that production selection chain or promote either
model. Required before automatic decisions: explicit missing-evidence admission
rules and broader task evaluation. Native zero-output logits and multi-question
cache branching are separate optimizations, not prerequisites for this result.

Evidence: [preregistration](evidence/litjev-20260921/PRE_REGISTRATION.md),
[complete receipt](evidence/litjev-20260921/RESULT.json),
[independent arithmetic/invariant checks](evidence/litjev-20260921/VERIFICATION.json).
