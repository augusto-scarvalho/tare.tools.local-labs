# SLX-01A lifecycle and route-receipt gap audit - result

## Frozen subject

- `slop.cpp` head: `71676e46cef1c141a8dfee7dbd8df50bf65e7169`
- Audit mode: source and receipt inspection only; no fork or service mutation.

## Existing coverage

- `tools/scripts_sh/bless_fork.sh` proves three bounded gates: B2b KV host pin
  engagement, MTP comparison, and non-degenerate GPU/host KV smoke behavior.
- `tests/test-recurrent-state-rollback.cpp` covers recurrent rollback,
  checkpoint restore into fresh and dirty contexts, replay, and logits parity.
- `tools/server/tests/unit/test_slot_save.py` covers slot save/restore/erase,
  cross-slot non-corruption, and multimodal rejection semantics.
- Local-labs cache/cancel/reuse receipts cover deterministic cold/warm behavior
  on selected serving tuples, including a preserved historical MTP failure.

## Missing evidence required by the transcript

- No single receipt joins `requested`, `resolved`, `realized`, and `exercised`.
- No generic `fallback_reason` contract connects planner choice to runtime route.
- No exposed `safe_restorable_prefix` versus committed prefix invariant.
- No deterministic combined matrix spans MTP depth, slots, cancellation phase,
  save/restore destination, graphs, placement, and admission pressure.
- Existing route proof is mechanism-specific log matching rather than a common
  schema.

## Smallest next packet

`SLX-01B-EFFECTIVE-ROUTE-RECEIPT` should first define an append-only test receipt
for mechanisms already observable without changing runtime behavior. It must
use current planner/log/counter sources and explicitly report `UNKNOWN` where a
layer is not observable. Only after this shadow packet identifies a concrete
missing source should a narrowly scoped fork instrumentation change be proposed.

## Verdict

`SLX_01A = GAP_CONFIRMED`. Current correctness tests are valuable but do not
satisfy the proposed end-to-end effective-route contract. A broad lifecycle
implementation is not authorized by this audit.
