# LAB-CODE-003 SWE-bench Verified infrastructure pilot

Status: **PREREGISTERED**  
Date: 2026-08-22

## Objective

Open the next coding tier only if the official repository-level evaluator is reproducible on this
machine. This stage qualifies infrastructure; it does not yet claim model quality.

## Frozen identity

- Official harness: `SWE-bench/SWE-bench` commit
  `7a21e05772954cc81471ae19d56f436cecf43c54`.
- Dataset: `SWE-bench/SWE-bench_Verified`, revision
  `78f471bf655a3137b2e8a75af1501690ec009ec3`, 500 human-verified instances.
- Evaluation must use the official containerized harness. Generated patches are never executed
  directly on Windows or the host WSL environment.

## Dependency gates

1. Create an isolated WSL virtual environment and install the exact harness commit.
2. Load the exact dataset revision, confirm 500 unique `instance_id` values, retain schema and a
   deterministic content hash.
3. Docker server must be reachable from the Ubuntu-24.04 distro.
4. Run one official gold-patch evaluation. The instance is the lexicographically first dataset ID;
   it is selected before observing any score.
5. Only a clean gold pass opens model generation. Docker absence, image failure, dataset drift or a
   gold failure yields `BLOCKED_INFRASTRUCTURE`; no model result is produced.

## Future model pilot, gated off until stages 1-4 pass

Ten deterministic instances, selected by evenly spaced indices after sorting by `instance_id`, one
greedy trajectory each. The agent/harness contract, context budget, tool set, step cap and patch
extraction must be frozen in a separate packet before inference. No leaderboard submission is
authorized.

