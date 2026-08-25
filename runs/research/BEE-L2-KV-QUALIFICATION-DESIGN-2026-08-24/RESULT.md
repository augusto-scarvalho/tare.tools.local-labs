# BEE-L2 codec-independent KV qualification design - result

## Verdict

`DESIGN_COMPLETE`; execution remains `BLOCKED_NO_CANDIDATE_REPRESENTATION`.

BeeLlama's useful contribution here is the shape of the qualification contract,
not a codec to copy. The local repository already has strong retrieval, task,
cache-lifecycle, placement, throughput, and VRAM oracles. The missing primitive
was a fail-closed full-distribution scorer. That scorer is now implemented and
self-checked in `tools/analysis/kv_qualification_metrics.py`.

## Required receipt contract

Every arm must record four distinct fields:

1. `requested`: user/config intent;
2. `resolved`: validated semantic choice after capability checks;
3. `realized`: physical storage type, layout, byte count, backend, and route;
4. `exercised`: counters proving that the realized representation handled the
   evaluated tokens.

Missing or inconsistent fields are a qualification failure. A silent fallback
cannot be scored as the requested arm.

## Frozen comparison structure

The pack is codec-independent. For any future candidate `C`, run three arms on
the same pinned model, engine, prompts, seeds, and hardware:

- `F16`: high-precision reference;
- `Q4`: incumbent symmetric `q4_0/q4_0` KV;
- `C`: candidate representation.

MTP is off for distribution/quality measurements and evaluated later as an
orthogonal lifecycle interaction. Start at 8k; deeper allocations are admitted
only after the earlier phase passes.

## Dependency-gated phases

| Phase | Existing/new oracle | Minimum evidence | Fail-fast gate |
|---|---|---|---|
| L2-0 route attestation | new requested/resolved/realized/exercised receipt | startup plus nonzero exercised-token counters | any fallback or route ambiguity stops the arm |
| L2-1 distribution | `kv_qualification_metrics.py` over full-vocabulary normalized log probabilities | fixed token positions spanning ordinary prose, code, math, and repeated-fact interference | reject top-k approximations, unpaired positions, non-finite values, or candidate shift worse than incumbent Q4 on median and p95 JS |
| L2-2 language loss | pinned perplexity runner | WikiText-2 or another immutable text panel, all three arms | candidate loss regression must be no worse than Q4 plus the measured repeat-control tolerance |
| L2-3 effective context | `context_suite_v2.py`, then official RULER only for survivors | retrieval, multikey, multihop, aggregation at 8k/32k/64k/128k with paired seeds | zero regression against Q4 at passed cells; replicate every failure three times |
| L2-4 bounded tasks | existing GSM8K and code harnesses | fixed paired subsets, greedy, no speculation | candidate non-inferior to Q4 under the standing task-specific tolerance and no new truncation mode |
| L2-5 lifecycle | `cache_correctness_v2.py`, slot save/erase/restore, cancellation, later MTP rollback | cold/warm equality, known-answer oracle, slot isolation, restore parity | any contamination, stale reuse, partial publication, or restore mismatch rejects candidate |
| L2-6 systems | existing KV/energy/placement collectors | six counterbalanced fresh-process pairs, VRAM bytes, prefill/decode, energy | promote only if capacity or speed materially improves without earlier quality/lifecycle loss |

Full-distribution means the complete normalized vocabulary vector at each frozen
position. Server top-probability fields are insufficient and are explicitly
rejected by the scorer.

## Promotion rule

A candidate is promotable only if it:

- proves its physical route and exercised token count;
- is not worse than incumbent Q4 on distribution, loss, context, task, and
  lifecycle gates;
- improves usable context capacity by at least 10%, or improves relevant
  end-to-end throughput by at least 10% beyond the paired noise envelope;
- preserves FlashAttention/GPU routing and the 4 GiB operational reserve where
  that reserve applies.

Kernel-only improvement, memory allocation without usable quality, or a codec
that relies on undocumented fallback does not qualify.

## Reuse inventory

- `tools/benchmarks/context_suite_v2.py`: paired synthetic context families.
- `tools/benchmarks/ruler_local_eval.py`: official RULER execution path.
- `tools/probes/cache_correctness_v2.py`: reuse/divergence/cancellation/long
  context lifecycle.
- `tools/probes/slot_save_restore_probe.py`: explicit persistence boundary.
- `tools/scripts_sh/kv-quant-bench.sh`: placement and throughput scaffolding.
- `tools/analysis/kv_qualification_metrics.py`: new full-distribution metrics.

## Execution trigger

Do not allocate another GPU matrix until a candidate supplies an immutable
format/layout identity and a backend route that can emit the required physical
receipt. At that point L2-0 and L2-1 are the first executable gates; no fork
default changes are authorized by this design.
