# LAB-HARNESS-001 - auditable agent-product primitives

## Objective

Test the highest-value hardware-free Track H mechanisms in this repository:
immutable TaskContract invariants with digest-bound deltas, structural
RepositoryEvidencePack retrieval, and a fail-closed test-baseline gate.

## Frozen evidence-retrieval tasks

| ID | Query | Required evidence file | Full-file control |
|---|---|---|---|
| mode | fail closed SERVE LAB mode transition and port enforcement | `tools/benchmarks/lmctl.py` | `lmctl.py`, `tests/test_lmctl_mode.py` |
| provenance | verify pinned Hugging Face source manifest sha256 and size | `tools/analysis/verify_hf_source_manifest.py` | verifier, manifest builder, provenance packet |
| visual | deterministic visual coding clause suite image OCR | `tools/benchmarks/vlm_coding_suite.py` | VLM coding and VQA harnesses |
| agent | irreversible tool recovery policy no blind retry | `tools/benchmarks/agent_irreversible_policy.py` | policy, robustness and suite harnesses |
| context | context VRAM envelope reserve ladder | `tools/benchmarks/context_vram_envelope.py` | context envelope and context suite harnesses |

## Gates

- Contract/delta and non-weakening unit tests pass, including stale-delta and
  removed-test rejection.
- Required-file recall is 5/5.
- Structural packs use at least 30% fewer Qwen3.8 model-native prefill tokens
  than the frozen full-file controls for every task and in aggregate.
- This qualifies orchestration primitives only. Independent model-written
  mutation tests, critic calibration and product integration remain separate
  gates and must not be inferred from this pilot.
