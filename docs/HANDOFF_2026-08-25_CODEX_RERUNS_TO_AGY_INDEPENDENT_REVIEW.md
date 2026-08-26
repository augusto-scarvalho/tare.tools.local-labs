# Handoff: Codex reruns to AGY independent review

Date: 2026-08-25  
Executor: Codex  
Required independent reviewer: AGY  
Repository state: dirty pre-existing workspace; no commit or push performed

## Review boundary

Codex preregistered, implemented and executed the two successor packets below. Both intentionally stop at `EXECUTED`. Codex did not author `REVIEW.json`, did not move either packet to `VERIFIED`, `PROMOTED` or `REJECTED`, and did not issue a claim code.

AGY must audit raw evidence independently. The narrative `RESULT.md` files are handoff summaries, not review evidence by themselves.

## Packet 1: BACKLOG-ADAPT-REQUAL-02

Purpose: replace the contaminated reused-model evaluation in `BACKLOG-ADAPT-REQUAL-01` with a distinct WSL process and freshly loaded clean base for every arm.

Bound artifacts:

| Artifact | SHA-256 |
|---|---|
| `PRE_REGISTRATION.md` | `9b528e0f9542778fd8dbbc2dcddf730a3e63f46aa130a2b5849e1f99181f0ffd` |
| implementation digest in `PIPELINE.json` | `221d65416c5d2eb9e4cd1402fee6c87a585397e8c1e24c2a1602338661ae521b` |
| `raw/receipt.json` | `8bc38d1f2cb5ef60f53ddb989e5c0aa1104b81359efbc5a2c8e4bbd0d92bc876` |
| `RESULT.md` | `567f32089d9119d666a4100fac43d16de22cd47f4f15e0a4baa16295411abe2d` |

Executor-reported facts requiring independent confirmation:

- 17/17 workers reported zero pre-existing PEFT/tuner modules.
- Forward and reverse smoke orders were semantically byte-identical for `base`, `lokr_1ep` and `target_mlp_only`.
- The complete set has 672 records: 14 arms × 48 samples.
- Every arm has 32 unique frozen GSM8K IDs and 16 unique protected-QA IDs.
- All seven preregistered gates passed.
- `target_mlp_only` scored 10/32 math and 3/16 QA, not the earlier contaminated 15/32 and 4/16.
- Highest descriptive joint observation was `disjoint_composite` at 12/32 math and 5/16 QA; no finalist-selection rule was preregistered, so this is not a promotion.

Required review checks:

1. Recompute all 26 adapter hashes and the base/dataset identities from the ledgers.
2. Inspect all 17 worker receipts, including clean-base counts and distinct process IDs.
3. Independently compare the semantic projections for both smoke orders.
4. Re-score `raw/samples.jsonl` without trusting stored `correct` flags.
5. Confirm that no performance or finalist claim is smuggled into the artifact-requalification claim.
6. Inspect `raw/service_maintenance.json`. The raw `exec_start_restored=false` value comes from comparing a systemd rendering containing volatile PID/timestamp fields; initial/final executable and `argv[]` should be compared directly.

Admissible dispositions:

- If evidence and gates survive review: authorize `ARTIFACT_REQUALIFIED_R2` and the next state allowed by the FSM.
- Otherwise: authorize `ARTIFACT_REQUALIFICATION_R2_REJECTED` with exact failed checks.

Even on successful requalification, do not revive `target_mlp_only` as a finalist. Downstream training requires a separately preregistered selection decision.

## Packet 2: BACKLOG-CUDAGRAPH-SERVING-02

Purpose: replace the invalid first-request/second-request comparison in `BACKLOG-CUDAGRAPH-SERVING-01` with explicit CUDA Graph OFF and ON server processes.

Bound artifacts:

| Artifact | SHA-256 |
|---|---|
| `PRE_REGISTRATION.md` | `6ce9a46106bc591d024e3c89d7d0c0bcaf43cb6a50323b050031866957fc7e33` |
| implementation digest in `PIPELINE.json` | `4adac4f71008f3d0b7d1317ceb0f63cc7e9acd1a804048384e362626053543fc` |
| `raw/receipt.json` | `eaa409f1620b45cb9bc939223f4407aed2b3be5615e5f4bd0027a00539f225f6` |
| `RESULT.md` | `bc4ea23d65d3ebc84cb38f1fd7240125de33040a284259a1a62dbd2ffc939a9a` |

Executor-reported facts requiring independent confirmation:

- Frozen order was OFF, ON, ON, OFF with four discarded warmups and 15 recorded responses per block.
- OFF PIDs contained `GGML_CUDA_DISABLE_GRAPHS=1`; ON PIDs did not contain the variable.
- All blocks used the same binary hash, executable, `argv`, model and request shape.
- There were 30 complete prompt pairs, 30/30 exact semantic matches and exactly 64 completion tokens per observation.
- ON was faster in 27/30 pairs, but median paired speedup was only `1.036998x` against a preregistered `1.10x` minimum.
- OFF/ON p95 values were 1460.224/1439.080 ms, yielding `-0.01447997` regression.
- Six gates passed and `paired_speedup` failed.
- Persistent serving returned active with exact executable/arguments, `NRestarts=0`, and healthy 8080/8081 endpoints.

Required review checks:

1. Confirm the frozen binary, CUDA library and model identities.
2. Inspect every block PID environment in `raw/treatment_controls.json`; do not infer treatment from request order.
3. Confirm ABBA prompt pairing and independently recompute all 30 ratios, p50 and p95 metrics from `raw/samples.jsonl`.
4. Verify semantic equality from content, reasoning content, finish reason and token count.
5. Inspect all four journal logs and the persistent-service recovery record.
6. Enforce the preregistered effect-size threshold. A directionally positive 3.70% median effect cannot pass a 10% gate.

Expected disposition if the evidence survives review: `SERVING_CUDAGRAPH_CAUSAL_REJECTED_R2`, because the mandatory speed gate failed. The review may preserve the bounded descriptive observation that ON was modestly faster on this frozen tuple, but must supersede the earlier 1.5115x causal claim.

## Repository checks observed by executor

- `python -m pytest -q`: 92 passed.
- `python tools/analysis/backlog_pipeline.py gate`: PASS.
- `llm-inference.service`: active/running after both experiments.
- `http://127.0.0.1:8080/health`: `{"status":"ok"}`.
- `http://127.0.0.1:8081/health`: `{"status":"ok"}`.

AGY should rerun the read-only checks and record its own outputs. Any review must cryptographically bind the current packet artifacts and must not reuse the unauthenticated `REVIEW.json` files from the superseded round.
