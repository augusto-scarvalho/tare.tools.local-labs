# HANDOFF — RNN-06T2 Official-Mamba Lifecycle Requalification + Recovery Confirmation

## Git / run identity

- **START HEAD:** `06013bd3c6f4e509ae521e3024953274df3b0f15` (RNN-06T final)
- **FINAL HEAD:** `429d980bc62742e1951e03bcdd6e9c7c007215ac`
- **Branch:** `master`
- **Dirty state at handoff:** working tree carries only pre-existing untracked `.harness/artifacts*`
  and historical handoff files; all RNN-06T2 payload is committed. **NOTHING PUSHED.**
- **Model / revision:** `state-spaces/mamba2-1.3b` @ `c5b59d00ec85d313adea86a08cad2a43c962dd3b`
- **Backend:** official `mamba_ssm` 2.2.4 + causal_conv1d 1.5.0.post8 (cu12torch2.6cxx11abiFALSE-cp312)
  + triton 3.2.0 + torch 2.6.0+cu124 (CUDA 12.4, cxx11abi=False), bf16, RTX 3090 (driver 591.86),
  chunk_size 256, 48 layers, 52,002,816 state bytes/seq. Fast path proven firing (kernel counters,
  `fallbackPathCalls=0`), not merely installed.

## CURRENT × RESEARCH × PROPOSED

- **CURRENT (this train, executed + qualified):** the official Mamba-2 fixed-batch single-pass
  historical-recovery lifecycle contract, and adaptive `MAX_CONFIDENCE` recovery over in-run
  snapshots, are prospectively qualified on the real pretrained checkpoint. Corrected economics show
  the marginal recovery cost is inside a freshly preregistered envelope.
- **RESEARCH (context, not re-run):** RNN-06T is reclassified by independent audit —
  `OFFICIAL_MAMBA_LIFECYCLE_STRICT_PREREG = NOT_QUALIFIED`, `PROTOCOL_GATE_ORDERING = FAILED`,
  3A/3B/economics/scout = EXPLORATORY_NON_LOAD_BEARING (see
  `runs/rnn/RNN-06T/AUDIT_RECONCILIATION.md`, append-only).
- **PROPOSED (NOT executed here):** realistic long-context workload discovery that would induce the
  forgetting regime naturally — deferred; opens only after independent audit accepts RNN-06T2.

## Preregistered gates — exact PASS/FAIL

| gate | verdict | basis |
|---|---|---|
| `OFFICIAL_MAMBA_FIXED_BATCH_LIFECYCLE` | **QUALIFIED** | tests A–J all PASS at fixed batch (BIT_EXACT state, argmax-identical readout) |
| `BATCH_SHAPE_NUMERICAL_PORTABILITY` | **OUT_OF_SCOPE_NOT_QUALIFIED** | diagnostic batch1-vs-batchB diff 0.5 > TOL 0.03; preregistered out-of-scope |
| `SINGLE_PASS_HISTORICAL_CAPTURE_T0R` | **QUALIFIED** | 5/5 boundary checks match independent replay, 0 failures |
| `HISTORICAL_RECOVERY_NARROW` | **QUALIFIED** | FIXED_SLOT_76 vs FINAL +0.573 CI[0.505,0.646] net=110 robust 4/4 |
| `ADAPTIVE_SELECTION_NARROW` | **DIRECTIONAL** | MAX_CONF vs FIXED_76 +0.104 CI[0.047,0.161] robust 2/4 (not required) |
| `WIDE_TARGET_RECOVERY_T1R` | **QUALIFIED** | MAX_CONF vs FINAL +0.542 CI[0.474,0.609] net=104 robust 4/4 |
| `ADAPTIVE_SELECTION_T1R` | **QUALIFIED** | MAX_CONF vs strongest carried fixed (slot153) +0.313 CI[0.240,0.375] robust 3/4 |
| `END_TO_END_RECOVERY_UTILITY_T1R` | **QUALIFIED** | RECOVERY−FINAL_STEP p95 +192.7 ms ≤ 250 ms envelope; quality Δ0.542 net=104 |

## Protocol deviations

None affecting a gate. One pre-outcome debugging fix to the T0R runner (continuation offsets for
tests D/G; committed `6e07785`→fix) before any T0R outcome. Economics ran 2 process starts (not more)
to bound wall-clock; warm steady-state (the load-bearing per-query serving cost) is characterized by
80 pooled warm samples.

## Failures / false-greens caught / negative evidence

- **False-green fixed in T0R (relative to RNN-06T):** the `C_branch_fork` independence assertion in
  RNN-06T was `bool(not torch.equal(predP, predQ) or True)` — unconditionally True. RNN-06T2 replaces
  it with a real fork test (fresh parent reconstruction + cross non-interference). Also fixed:
  reset-only-zero (now reuse-equivalence), roundtrip-hash-only (now roundtrip+continuation).
- **Negative/out-of-scope evidence preserved:** batch1-vs-batchB state diff = 0.5 (not "benign";
  out-of-scope for the fixed-batch contract).
- **Narrow adaptive not over-claimed:** `ADAPTIVE_SELECTION_NARROW = DIRECTIONAL` (robustness < 3/4),
  reported honestly; not required to qualify.
- **Every fixed control exposed**, including the weak ones (wide slot38 acc 0.219).
- **Cross-process bf16 nondeterminism (disclosed):** in-process same-path replay is BIT_EXACT (T0R
  test H, 0 boundary failures), but across separate process starts kernel-autotuning flips ~1–3/192
  borderline argmaxes (wide FINAL 0.266↔0.271; adaptive Δ +0.323↔+0.313). Qualification margins
  (recovery Δ 0.542, adaptive Δ ~0.31) dwarf this noise; verdicts identical across both starts. Not
  seed screening — both starts qualified; the committed results are the re-run with complete
  Section-12 counters.
- **Economics premium (corrected):** RECOVERY−FINAL_STEP p95 +192.7 ms ≤ 250 ms; the 972 ms
  RECOVERY−FINAL_FUSED is the orthogonal step-vs-fused cost, not the recovery premium.

## Source excerpts (path / function)

- `ops/rnn_06t2_t0r.py::main` — tests A–J; C uses `L.run_trajectory` to reconstruct a fresh parent
  and `L.state_hash` equality (no tautology); E uses `full_traj_with_cache` on a reset cache vs a
  fresh cache; F continues after `serialize_state`/`deserialize_state` and compares to a no-roundtrip
  continuation.
- `ops/rnn_06t2_t1r.py::capture_readouts` — single-pass in-run capture (`L.run_trajectory`) + K+1
  `L.readout`; real boundary checks vs independent replay; `recovery_harm` / `paired_vs` for tables.
- `ops/rnn_06t2_t1r.py::main(wide_qual)` — TAU_TIE carried-controls, strongest-carried adaptive gate.
- `ops/rnn_06t2_econ.py` — `final_fused_equiv` / `final_step_equiv` / `recovery_equiv` all return a
  constrained scored answer; `ops/rnn_06t2_econ_decide.py` mints utility on p95(RECOVERY−FINAL_STEP).

## Committed diffs (append-only; no amend/rebase of outcome-bearing history)

`bfe4ed8` audit reconciliation · `6e07785` prereg+T0R runner (+debug fix) · `e281329` T0R results ·
`cd294a2` T1R prereg+challenges+runner · `a65b7fb` T1R recovery results + econ tooling ·
`429d980bc62742e1951e03bcdd6e9c7c007215ac` T1R decision + economics + handoff + bundle.

## Artifact hashes

- Qualification-set SHA-256 (T0R): `ca92cfad0d0aac4ae20aa8612f259c559ad592415a71797561b3e5909103cafe`
- Narrow qual set: `34d276ce…` · Wide calib: `dc4010f1…` · Wide qual: `97f303a2…` (all disjoint_all=True)
- Bundle ZIP SHA-256: `52fcf4d00430bb8b24da3c2cfd8b5a4c1c2473c701b2939acbd0f633e4a35426 (outer-envelope hash; not self-contained in the archive)`

## Reproduction commands

```
wsl -d Ubuntu-24.04
source /home/augus/rnn06_env/bin/activate && cd /mnt/c/projects/local-model-lifecycle
python ops/rnn_06t2_t0r.py
python ops/rnn_06t2_t1r_challenges.py
python ops/rnn_06t2_t1r.py narrow && python ops/rnn_06t2_t1r.py wide_calib && python ops/rnn_06t2_t1r.py wide_qual
python ops/rnn_06t2_econ.py 0 && python ops/rnn_06t2_econ.py 1 && python ops/rnn_06t2_econ_decide.py
python ops/rnn_06t2_bundle.py
```

## Authority / effect status

Record + prospective qualification. Supersedes the historical RNN-06T lifecycle mint for gating
downstream recovery work. No production/deploy effect. Nothing pushed.

## Exactly one next recommendation (NOT executed)

**Open a realistic-workload discovery train** on this same official Mamba-2 checkpoint: find a natural
long-context operating point (e.g. a real document-QA or multi-turn workload) that induces the
forgetting regime *without* synthetic sentinel/DS construction, and test whether in-run snapshot
recovery + `MAX_CONFIDENCE` still adds value there. This is deferred and opens only after independent
audit accepts RNN-06T2.
