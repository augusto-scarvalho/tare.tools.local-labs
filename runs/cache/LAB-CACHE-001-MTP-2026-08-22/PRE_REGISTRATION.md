# LAB-CACHE-001 persistent slot and MTP rollback closure

**Status:** `PREREGISTERED`  
**Date:** 2026-08-22  
**Question:** does the deployed Qwen3.8 cache remain exactly correct across cancellation/reuse and explicit
slot save/erase/restore, including when embedded MTP speculation is active?

## Frozen substrate

- model: `/home/augus/models/qwen38-27b/unsloth/Qwen3.8-27B-UD-Q4_K_XL.gguf`
- bytes: `17,923,394,624`
- SHA-256: `bee238bbeb3dc0a34bde4d0dedbaee1f98c009e8bb4226f03070054c12fb1372`
- engine: `/home/augus/src/slop.cpp/build/bin/llama-server`, build `b9863-5e7f6271c`
- template/runtime: deployed template behavior, full GPU offload, FlashAttention, q4_0/q4_0 KV, one slot,
  131,072 context tokens, 32 context checkpoints
- maintenance endpoint: port 8092; embedding port 8081 remains untouched
- slot directory: `/home/augus/lab-slot-cache-20260822`

The only launch-factor changes from the restored baseline are port 8092, explicit `--slot-save-path`, and
the preregistered speculation arm.

## Arms and order

Use the same fixed nonce `mtp-rollback-20260822-a` and 256-token generation budget:

1. no-spec cache/cancel/reuse reference;
2. MTP `--spec-type draft-mtp --spec-draft-n-max 3` cache/cancel/reuse;
3. on the live MTP arm, explicit slot save → erase → restore → completion.

## Required invariants

- all four cache cases pass in both arms;
- every warm response is byte-identical to its cold response and both satisfy the known-answer oracle;
- cancellation returns the slot to idle and the post-cancel response matches its cold oracle;
- corresponding no-spec and MTP cold/warm responses are byte-identical;
- no-spec responses contain no draft counters, while MTP responses show `draft_n > 0` and
  `draft_n_accepted >= 0`;
- slot save and restore report positive `n_saved`/`n_restored`, restored completion is byte-identical to the
  cold completion, oracle-correct, and uses restored cache (`cache_n > 0`);
- no server crash, cross-session contamination, or unexplained slot action error.

Any failed correctness or lifecycle invariant keeps LAB-CACHE-001 partial and blocks cache/checkpoint
promotion. Performance is descriptive only. Preserve raw responses and the saved slot file, stop port 8092,
and restore the exact service baseline before ending the tranche.

## Failure-localization amendment

Frozen after the MTP arm produced an oracle failure at 24.5k and before localization runs. The no-spec arm
was correct; MTP returned only `!`, accepted zero drafts, and behaved identically cold/warm. Slot lifecycle
was mechanically successful under MTP but persisted the same already-wrong output, while no-spec
save/restore passed fully.

To determine whether this is limited to raw `/completion` or affects the production chat path, run the
paired four-family context suite at 8k/16k/24k/32k, one replicate, with identical seed and server controls:

1. no-spec reference;
2. MTP n3.

Require byte-equivalent exact answers across arms and all 16 cells correct. Any MTP-only miss establishes a
general chat-path correctness cliff and requires a deployment rollback recommendation. If chat remains
perfect, classify the defect more narrowly as raw-completion/checkpoint interaction and keep MTP deployment
under a scoped warning pending a raw-context depth sweep.

## Raw-context depth amendment

Frozen after both chat arms passed 16/16 with byte-identical answers and before the raw sweep. Run the
same untemplated known-answer prompt at 8k/16k/20k/24k/32k using two shapes: the repetitive archival
filler implicated by the failure and deterministic diverse NIAH filler. Compare no-spec with MTP n3,
128 generated tokens, greedy decoding, seed 0, cache disabled, and record full outputs/draft counters.

Require 10/10 oracle-correct in each arm. If n3 alone fails, run MTP n1 and n2 at the nearest passing and
failing depths to determine whether the defect scales with draft depth. A shape-specific failure is a
raw-prompt distribution cliff; failures across both shapes are a general raw-endpoint depth cliff. This
localization is diagnostic and cannot turn the failed preregistered MTP cache arm into a pass.

## Raw-trigger dissection amendment

Frozen after both raw depth arms passed 10/10. Re-run the exact failed long-case prompt cold at generation
budgets 64 and 128 in no-spec and MTP n3, both unchanged and with only a final `Answer:` cue appended.
If the original fails only under MTP while the cue passes, classify this as a raw untemplated prompt-format
cliff rather than a general context-depth cliff. Then test the unchanged 64-token prompt once at MTP n1
and n2 to determine whether lowering draft depth avoids it. All trigger work remains diagnostic.

## Stateful-sequence reproduction amendment

Frozen after a clean MTP n3 instance passed all four trigger cells. Restart MTP n3, record the original
128-token trigger before the cache suite, execute the exact four-case cache suite with the fixed nonce and
256-token budget, then record the same trigger again with cache disabled. A pre-pass, repeated long-case
failure during the suite, and post-failure establishes sequence-dependent slot/server-state corruption.
If the suite instead passes, retain the first failure as non-reproduced and do not claim a state transition.

## Intermittency amendment

Frozen after reproduction 2 passed 4/4 and both post-sequence triggers passed. Repeat MTP explicit slot
save/erase/restore once on the clean live arm, then run three additional fresh-server replicas of the exact
four-case cache sequence (`rep3` through `rep5`) with the same fixed nonce and controls. Report the full
five-run distribution including the original failure; do not discard or supersede it. Any further oracle,
identity, cancellation, or lifecycle failure keeps the MTP rollback/cache path blocked. Even 4/5 passing
is insufficient for promotion because the invariant is correctness, not mean accuracy.
