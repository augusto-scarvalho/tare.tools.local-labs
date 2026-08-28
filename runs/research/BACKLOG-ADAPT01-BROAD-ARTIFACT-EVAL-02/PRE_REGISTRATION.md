# BACKLOG-ADAPT01-BROAD-ARTIFACT-EVAL-02 preregistration

Task: Evaluate the fresh 384-step LoKr artifact on a broad teacher-disjoint panel under GPU co-tenancy  
Evidence class: `artifact_requalification`  
Executor: Codex executor  
Date: 2026-08-26

## Hypothesis

The immutable `lokr_3ep_lr1e4` artifact trained for 384 steps has higher GSM8K accuracy than its clean base on a broad panel absent from every teacher trace, while losing no more than five percentage points on the complete protected-QA panel.

This tests artifact behavior, not training repeatability. It runs concurrently with `BACKLOG-ADAPT-TRACE-DISTILL-05`; timing and throughput are explicitly non-evidence.

## Frozen inputs

- Admission SHA-256 `1691233003e4ab97b9986c035e78581e4ef04b565f766dd3e19a1449e4d81b8c`.
- Parent receipt SHA-256 `dd975197993bab7943ea2407a664f20fda927bb6fc581714eba093f7e93be0c6`.
- Adapter weights SHA-256 `7f6d082243f6b406259791dc15a65e4b092b48597fad9b68018d507872ad8fa7`; config SHA-256 `08cf4d254e2a6c9aba9d34ba6a0c76926b478d7cd0ad771062acefb71a31d934`; training metrics SHA-256 `09d6295f934843fa85cd2a4757a1b045695e1b83c8a56c19c73c3d8bbecc0a9c`.
- GSM8K SHA-256 `68a72276898a45dacb893154477621ff3d05ae7e043a10820644b691d8b63d77`; teacher artifact SHA-256 `dc5cabe44c92e48b0e832881ef27ebad4047b140928c9a12678e0c0c6660006e`; QA SHA-256 `56434ebd53ff9f5adb477bd902566e28644fe9ea88619834f81417a06c66b84f`.
- Base weights/config/tokenizer SHA-256: `c2b1e5a17d9c1e27685d92ed9b382911ebb99955ecd89052d1721241adfbab6c`, `b90b86f35c8e6925ef74ee04d0e758f0a845c83a42089ad82bbaa948de9b4204`, `fe000e3ed39ed12b8d2481d527d44f93c65d37e87645d2dcc80d1bf9d50d2927`.
- Held-out IDs: the 256 lowest numeric GSM8K IDs absent from all 200 teacher rows; compact canonical JSON SHA-256 `78a3b7ef26cdf932b79eb6f64dfb576d66770d30a5ce0fd3251536cb6e76901f`.

## Command

```powershell
python tools/research/run_adapt01_broad_artifact_eval_r2.py --outdir runs/research/BACKLOG-ADAPT01-BROAD-ARTIFACT-EVAL-02 --outer-runner-pid 42416
```

## Factors

- Clean base and adapter are evaluated sequentially in one isolated process, with greedy decoding and identical prompts.
- Panels: 256 teacher-disjoint GSM8K prompts at 192 maximum new tokens and all 48 QA prompts at 128 maximum new tokens, for 608 total paired generations.
- Primary uncertainty: 20,000-replicate paired prompt bootstrap of adapter-minus-base math accuracy.
- QA retention: adapter accuracy minus base accuracy, with regression represented as `max(0, base - adapter)`.
- The outer R5 runner must be alive and `8080` unavailable at start and end; `8081` must remain healthy. This runner never starts or stops a service.

## Acceptance gates

- `artifact_identity`: `artifact_hashes_verified eq True`
- `panel_isolation`: `teacher_disjoint_math_tasks eq 256`
- `evaluation_coverage`: `paired_math_and_qa_generations eq 608`
- `broad_gain`: `paired_bootstrap_95ci_lower_math_gain gt 0.0`
- `protected_retention`: `protected_qa_regression le 0.05`
- `co_tenancy_boundary`: `outer_runner_alive_and_embedding_healthy eq True`

## Abort conditions

- Any frozen hash or held-out panel identity differs.
- The outer R5 runner is absent, `8080` is serving, or `8081` is unhealthy before model load.
- VRAM free is below 12 GiB before the second model process starts.
- Any panel is incomplete, base and adapter prompts differ, or independent rescoring disagrees.

## Allowed claims

- `ADAPT01_384_ARTIFACT_BROAD_GAIN_R2`
- `ADAPT01_384_ARTIFACT_BROAD_GAIN_NOT_CONFIRMED_R2`

No training-repeatability, performance, production, or broader generalization claim is allowed. Independent review remains required.
