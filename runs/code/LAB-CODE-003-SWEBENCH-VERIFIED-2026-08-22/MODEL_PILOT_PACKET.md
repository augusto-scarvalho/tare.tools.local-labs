# LAB-CODE-003 model pilot packet

Status: **FROZEN / OPENED BY CLEAN GOLD GATE**  
Date: 2026-08-22

## Gate and identities

- The official gold probe resolved `astropy__astropy-12907` with zero infrastructure failures.
  Report SHA-256: `cdc6d0bde290d00eba42aadd9249cd6d2f7992d41d64eb2dd48b18e848b247c4`.
- Dataset: `SWE-bench/SWE-bench_Verified` revision
  `78f471bf655a3137b2e8a75af1501690ec009ec3`, fingerprint `2cee1d06dbc301e8`, canonical
  content SHA-256 `84385d3374a0c37b692a72ee57509fba15e5cce896671944e1348d62a4a8f4de`.
- Agent: official `SWE-agent/mini-swe-agent` commit
  `25941c89cfbc91eb40b3f8756348c91d9977d57e`, package `mini-swe-agent==2.4.6`, isolated
  environment `/home/augus/mini-swe-agent-venv`; initial `pip freeze` SHA-256
  `c815ffa9755925299980a3f45cdfd25274221501955cf08755ed764e55192f05`.
- Model artifact: `unsloth/Qwen3.8-27B-GGUF` revision
  `f1bfb127c64f7072bdd2cad55f258b9c8b2910fe`, expected upstream SHA-256
  `bee238bbeb3dc0a34bde4d0dedbaee1f98c009e8bb4226f03070054c12fb1372`, local size
  `17,923,394,624` bytes. Local full hash remains pending under LAB-PROV-001.
- Runtime: llama.cpp commit `5e7f6271c06b9104862ab799278a1b7f1323a449`, build 9863, service alias
  `qwen38-27b`, 131,072-token allocation, q4_0 KV, MTP draft n=3.

## Frozen agent contract

- Base prompt/tool contract: mini-SWE-agent's pinned `config/benchmarks/swebench.yaml`.
- One worker; selected IDs are the ten ordered values in `dataset_manifest.json`; one trajectory each.
- Greedy sampling: temperature 0, seed 42, thinking disabled, maximum 2,048 output tokens per call.
- Bash-only tool in an official per-instance Docker image; `/testbed` working directory; command timeout
  120 seconds; container lifetime 75 minutes.
- Maximum 40 model calls and 3 consecutive format errors; one-hour wall clock per instance.
- Submission is accepted only through mini-SWE-agent's exact patch protocol. Empty patches and any exit other
  than `Submitted` fail closed.
- The first selected instance is a protocol qualification gate. If it cannot perform structured tool use and
  submit a nonempty patch, the remaining nine are not spent. Otherwise all ten proceed unchanged.
- Evaluation uses the official SWE-bench containerized harness; no leaderboard submission is authorized.

Configuration: `mini_swe_qwen38.yaml`, SHA-256
`9b1303e167047d8f1088f9536f87de8d4e7387dd979baa9db88806183a81701c`. Runner:
`tools/benchmarks/mini_swe_verified_pilot.py`, SHA-256
`cbc8f363d1666057d01afca7fbfbd44e7af7ab75bfdc91c3d1af0b949a41018a`.
