# LAB-CODE-003 SWE-bench Verified result

Decision: **COMPLETE / 5 OF 10 RESOLVED / 50% BOUNDED PILOT**  
Date: 2026-08-22

## Superseded infrastructure blocker

The earlier `BLOCKED_INFRASTRUCTURE / MODEL GENERATION NOT OPENED` conclusion is **SUPERSEDED**.
Docker Desktop 4.82.0's native Ubuntu integration repeatedly failed inside its backend with
`Wsl/Service/0x8007274c`. A temporary loopback-only Docker API (`tcp://127.0.0.1:2375`) allowed the
official Linux client and harness to run without restarting WSL or touching the embedding service.

The mandatory official gold evaluation then resolved `astropy__astropy-12907`: 1/1 resolved, zero
infrastructure/ambiguous failures, zero errors and zero unstopped containers. Gold report SHA-256:
`cdc6d0bde290d00eba42aadd9249cd6d2f7992d41d64eb2dd48b18e848b247c4`.

After evaluation, Docker Desktop was stopped, port 2375 was closed, its settings matched the pre-run
backup byte-for-byte, and the temporary WSL group/socket/client bridge was removed. The pinned
mini-SWE-agent package remains installed intentionally for reproducible future runs.

## Frozen model pilot

- official SWE-bench harness commit `7a21e05772954cc81471ae19d56f436cecf43c54`, package 5.0.2;
- exact 500-row dataset revision `78f471bf655a3137b2e8a75af1501690ec009ec3`, fingerprint
  `2cee1d06dbc301e8`, canonical content SHA-256
  `84385d3374a0c37b692a72ee57509fba15e5cce896671944e1348d62a4a8f4de`;
- official mini-SWE-agent commit `25941c89cfbc91eb40b3f8756348c91d9977d57e`, package 2.4.6;
- incumbent Qwen3.8-27B UD-Q4_K_XL through llama.cpp build 9863, thinking off, temperature 0,
  seed 42, maximum 2,048 output tokens/call;
- one trajectory per frozen ID, 40 calls and one-hour wall-clock limit, official per-instance Docker
  image and patch-only submission protocol.

The exact contract and hashes are in `MODEL_PILOT_PACKET.md`.

## Outcome

| Outcome | Count |
|---|---:|
| Frozen instances | 10 |
| Submitted nonempty patches | 5 |
| Empty patches after `LimitsExceeded` at exactly 40 calls | 5 |
| Officially resolved patches | 5 |
| Officially unresolved nonempty patches | 0 |
| Infrastructure, ambiguous or evaluator errors | 0 |
| Unstopped containers | 0 |

The five submitted patches all resolved their official tests: `astropy__astropy-12907`,
`django__django-11603`, `django__django-14752`, `pytest-dev__pytest-5262`, and
`sympy__sympy-24661`. The five empty cases are model failures under the frozen budget, not omitted
denominators: `django__django-13401`, `django__django-16256`, `matplotlib__matplotlib-25311`,
`sphinx-doc__sphinx-11510`, and `sympy__sympy-14531`.

The bounded pilot score is therefore **5/10 = 50% resolved**. It is not a 500-case leaderboard result
and no leaderboard submission was made. The striking split is submission efficiency rather than patch
correctness: conditional on submitting, resolution was 5/5; half the cases exhausted 40 calls first.

## Artifacts

- predictions: `model-pilot-qwen38/preds.json`, SHA-256
  `9d2eeaf3360deca355ef2706e492b8a18edada3aaf0e0198900ae36b5fa48d6a`;
- official report: `model-pilot-qwen38/qwen38-27b.lab-code-003-qwen38-pilot.json`, SHA-256
  `6754ba01bedac3c059ac13c0e5a69d9c86050703568eeb1f81971f52d90d1d3a`;
- ten complete mini-SWE-agent trajectories under `model-pilot-qwen38/<instance_id>/`;
- revision-pinned runner: `tools/benchmarks/mini_swe_verified_pilot.py`.

Next useful experiment: a separately preregistered efficiency arm that changes one lever at a time
(step budget, output budget, or a concise submit reminder). Do not retroactively rescore this pilot or
increase only the five failed cases.
