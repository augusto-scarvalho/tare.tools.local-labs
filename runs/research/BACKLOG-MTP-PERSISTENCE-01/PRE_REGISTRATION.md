# BACKLOG-MTP-PERSISTENCE-01 preregistration

Task: Root-cause intermittent MTP cache persistence failure  
Evidence class: `mechanism_research`  
Executor: Codex executor  
Date: 2026-08-26

## Hypothesis

The historical first-instance MTP save/restore failure is caused by persisting a
slot before the speculative draft lifecycle has completed one accepted-draft
cycle in that fresh server process. A discarded deterministic MTP warmup followed
by slot erase before cache population will synchronize the draft/base lifecycle
and eliminate restore mismatches.

The hypothesis is falsified if no unprimed MTP failure occurs in 20 fresh-process
observations, if any of 20 primed observations fails, or if the no-spec controls
fail. A clean batch without an unprimed reproduction is not called a root cause.

## Frozen inputs

- Historical result: `runs/cache/LAB-CACHE-001-MTP-2026-08-22/RESULT.md`, 3,686 bytes, SHA-256 `5dd8cd202fafcd021a16eebdb4cbf9885d6ef080862179c5507619409a018c47`.
- Blocker revalidation: `docs/research/BLOCKER_REVALIDATION_2026-08-24.md`, 3,413 bytes, SHA-256 `d5888a2ee3c11711113fbc0a1596a00caa21cc31bfc553d9dd87cb0144aba658`.
- Fleet registry: `config/qualified_model_fleet.json`, 9,783 bytes, SHA-256 `042fedf5907f031fb9993c03058f3cc9c8fe2c8d75a3235ea4b5e11c7412cd82`.
- Reference persistence probe: `tools/probes/slot_save_restore_probe.py`, 4,705 bytes, SHA-256 `b24b5f3e2f4ef6bd8687763791a3c78d33ae91e9a129997aa4a16295b3cb81cd`.
- Reference cache probe: `tools/probes/cache_correctness_v2.py`, 10,166 bytes, SHA-256 `4d23c2effc666912af5166085c1d9dd756a5be885cb5571a1af80345fc44a7fb`.
- Historical failing raw files are immutable and remain under `runs/cache/LAB-CACHE-001-MTP-2026-08-22/`.
- Server binary: `/home/augus/opt/slop.cpp/b10165-71676e46c/bin/llama-server`.
- Model: `/home/augus/models/qwen38-27b/unsloth/Qwen3.8-27B-UD-Q4_K_XL.gguf`, 17,923,394,624 bytes, SHA-256 `bee238bbeb3dc0a34bde4d0dedbaee1f98c009e8bb4226f03070054c12fb1372`.
- Runtime: 32,768 context, q4_0 K/V, one slot, FlashAttention, greedy seed 20260826. MTP uses `draft-mtp` with depth 3.
- GPU: NVIDIA RTX 3090, 24,576 MiB.

## Command

```powershell
python tools/research/run_mtp_persistence_first_instance.py --outdir runs/research/BACKLOG-MTP-PERSISTENCE-01
```

The autonomous supervisor runs this command only after the fleet screen exits.
Every observation uses a fresh transient systemd unit and a unique slot-save
directory. The persistent inference service is stopped only through systemd and
is restored in a `finally` block. Port 8081 is never stopped.

## Factors

- No-spec controls: four fresh server processes, each performing one long-prompt slot save, erase, restore and oracle completion.
- MTP treatment matrix: ten balanced `cold, warm, warm, cold` blocks, totaling 20 unprimed and 20 primed fresh server processes.
- Unprimed arm: the persistence sequence is the first inference lifecycle after server health.
- Primed arm: one discarded short completion must report at least one accepted draft token; the slot is then erased before the identical persistence sequence.
- Frozen prompt: deterministic repeated archive text ending in code word `MAGNOLIA`, sized below the 32,768-token runtime limit.
- Each persistence sequence requires successful save/erase/restore API responses, nonzero saved/restored token counts, a restored cache hit, byte-identical cold/restored completions and the `MAGNOLIA` oracle.
- Transient endpoint: `127.0.0.1:18080`; unique unit and save directory per observation.
- Server logs, argv, environment, slot-file byte counts and hashes, GPU snapshots, response bodies and timing/draft counters are retained. The uniquely attributed temporary slot files are removed after hashing to avoid retaining roughly 22 GiB of duplicate cache images.

## Acceptance gates

- `original_failure`: `original_failure_reproduced eq True`
- `controls`: `invariant_controls_pass eq True`
- `fixed_repeats`: `successful_fixed_path_repeats ge 20`
- `semantic_parity`: `post_fix_mismatch_rate eq 0`

`original_failure_reproduced` is true only when at least one unprimed MTP
observation fails the frozen lifecycle/oracle invariant while all no-spec
controls pass. `invariant_controls_pass` additionally requires all 44 fresh
units, the balanced treatment counts, priming materiality and service recovery.

## Abort conditions

- Any frozen source, binary or model identity differs.
- The persistent gateway or embedding endpoint is unhealthy before maintenance.
- The persistent service cannot be stopped through systemd, the temporary port is occupied, or a reserved transient unit already exists.
- A primed arm records zero accepted draft tokens; this invalidates treatment materiality.
- The model reports context overflow for the frozen prompt.
- Port 8081 is unhealthy at any block boundary.
- A transient unit cannot be stopped or its slot directory cannot be attributed uniquely.
- The original gateway service, initial resident model, executable identity or embedding health cannot be restored.

An oracle mismatch is evidence and does not abort the remaining safe arms.

## Allowed claims

- `MTP_PERSISTENCE_ROOT_CAUSED`
- `MTP_PERSISTENCE_HYPOTHESIS_REJECTED`

Claims outside these codes are forbidden even if a metric looks favorable.
The executor stops at `EXECUTED`; an independent actor must review the packet.
