# BACKLOG-ADAPT01-BROAD-ARTIFACT-EVAL-03 preregistration

Task: Complete the broad LoKr artifact evaluation with the actual 48-task protected-QA panel
Evidence class: `artifact_requalification`

## Hypothesis

The immutable 384-step `lokr_3ep_lr1e4` artifact has higher accuracy than its
clean Qwen3.5-0.8B base on the frozen 256-task teacher-disjoint GSM8K panel,
with a paired-bootstrap 95% lower bound above zero, while losing no more than
five percentage points on the actual frozen 48-task protected-QA panel.

R2 already materialized both math arms and the first ten QA tasks, but its
synthetic `f01..f48` identifier construction selected only ten rows. This
continuation imports those immutable samples by hash and generates only the 38
missing QA rows per arm. It does not repeat or selectively replace R2 math.

## Frozen inputs

- `runs/research/BACKLOG-ADAPT01-BROAD-ARTIFACT-EVAL-02/raw/worker.json`
- `runs/research/BACKLOG-ADAPT01-BROAD-ARTIFACT-EVAL-02/raw/receipt.json`
- `runs/research/BACKLOG-ADAPT-MECHANISMS-RERUN-01/raw/mechanisms/adapt01/lokr_3ep_lr1e4/adapter`
- `workloads/gsm8k.jsonl`
- `runs/requalification/QWEN38-HAUHAUCS-NORMAL-QA-2026-08-23/tasks.jsonl`

- Admission SHA-256: `d1a3ac2ae7ea792845de41b1d8c40e3227fa074b250db6a7f27b0dde894afee8`.
- R2 worker SHA-256: `930172bc74c565ab93e9afe36a54c23fb7e84e7dc0b2b66a15b1dba6b1e5a38b`.
- R2 receipt SHA-256: `c64faac1231dcbba3b0c7e2ae9071ce6e503ebf62766e6c53b9b95b416ca4780`.
- R2 preregistration SHA-256: `477e3ece2ff5c70250c45a9c91160f563c36fa10d5344059268c84b99939b196`.
- R2 implementation SHA-256: `7978351859fc4c03a534088339b9e58c91074fdc43ee6bb1c5cef99a87aee022`.
- Adapter weights SHA-256: `7f6d082243f6b406259791dc15a65e4b092b48597fad9b68018d507872ad8fa7`;
  config SHA-256: `08cf4d254e2a6c9aba9d34ba6a0c76926b478d7cd0ad771062acefb71a31d934`.
- GSM8K SHA-256: `68a72276898a45dacb893154477621ff3d05ae7e043a10820644b691d8b63d77`;
  QA corpus SHA-256: `56434ebd53ff9f5adb477bd902566e28644fe9ea88619834f81417a06c66b84f`.
- Frozen math-ID SHA-256: `78a3b7ef26cdf932b79eb6f64dfb576d66770d30a5ce0fd3251536cb6e76901f`.
- Actual ordered 48-task QA-ID SHA-256: `5377ee57e27a3480fdad26c05cc7cc13b7e177c69abdda77795f898d43df45f3`.

## Command

```powershell
python tools/research/run_adapt01_broad_artifact_eval_r3.py --outdir runs/research/BACKLOG-ADAPT01-BROAD-ARTIFACT-EVAL-03
```

## Factors

- Arms: clean base and the immutable LoKr adapter, greedy decoding, identical
  QA prompts and `max_new_tokens=128`.
- Imported evidence per arm: 256 math plus 10 QA samples from the frozen R2
  worker. Fresh continuation per arm: the remaining 38 QA tasks, giving 304
  samples per arm and 608 total.
- Math estimand and 20,000-replicate paired bootstrap remain exactly R2's.
- QA retention is `max(0, base_accuracy - adapter_accuracy)` across all 48
  actual QA tasks. Every merged sample is rescored from raw text.
- Runtime: RTX 3090 under WSL. The qualified-model gateway may be stopped only
  for VRAM and must be restored with the same service identity; port 8081 must
  remain healthy throughout.

## Acceptance gates

- `continuation_integrity`: `source_worker_hash_verified eq True`
- `artifact_identity`: `artifact_hashes_verified eq True`
- `panel_isolation`: `teacher_disjoint_math_tasks eq 256`
- `evaluation_coverage`: `paired_math_and_qa_generations eq 608`
- `broad_gain`: `paired_bootstrap_95ci_lower_math_gain gt 0.0`
- `protected_retention`: `protected_qa_regression le 0.05`
- `independent_score`: `independent_rescore_match eq True`
- `service_recovery`: `service_and_embedding_restored eq True`

## Abort conditions

- Any frozen hash, math panel or ordered QA-ID hash differs.
- R2 does not contain exactly 256 math and ten non-overlapping QA samples per arm.
- A fresh QA output overlaps an imported task or either merged arm is not 256+48.
- Independent rescoring differs from recorded correctness.
- The embedding endpoint becomes unhealthy, serving cannot be restored, or the
  implementation cannot obtain enough VRAM after bounded service maintenance.
- No receipt or claim is emitted after a partial worker, malformed sample or
  incomplete evidence map.

## Allowed claims

- `ADAPT01_384_ARTIFACT_BROAD_GAIN_R3`
- `ADAPT01_384_ARTIFACT_BROAD_GAIN_NOT_CONFIRMED_R3`

Claims outside these codes are forbidden even if a metric looks favorable.
