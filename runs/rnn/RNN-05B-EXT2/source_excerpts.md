# RNN-05B-EXT2 — Source excerpts (paths, functions, line context)

All excerpts from `ops/rnn_05b_ext2.py` (committed at HEAD `5abeab4`). The substrate/generator building blocks
are imported READ-ONLY from `ops/rnn_delta_substrate.py`, `ops/rnn_mc_substrate.py`, `ops/rnn_mc_bench.py`
(UNCHANGED — no recurrence/kernel edits).

## 1. Control-flow invariant — BASE qualification persisted + verified STRICTLY before any MC (§6)
`run()`, `ops/rnn_05b_ext2.py:1058-1076`. MC (P3) begins only after this block; the two `return`s guarantee no
MC/reader code executes on a BLOCKED or unverifiable qualification.
```
1058  qpath = os.path.join(outdir, "BASE_QUALIFICATION.json")
1059  json.dump(base_qual, open(qpath, "w"), indent=2)                     # persist FIRST
...
1064  # ---- MC GATE: LOAD + VERIFY the persisted qualification artifact ----
1065  ok, why = verify_qualification(qpath, grid_sha, cfg.config_sha256(),
1066                                 {mode: {str(bb["seed"]): bb["backbone_sha256"] ...}})
1069  if not ok:  ... return                                              # abort if artifact bad
1072  if base_qual["FIXED_BACKBONE_GRADED_REGION"] == "BLOCKED":
1073      log("[STOP] ... BLOCKED_FIXED_BACKBONE; no MC, no EXT3 (Case A).")
1075      R["outcomes"] = build_outcomes(cfg, R, blocked="BLOCKED_FIXED_BACKBONE")
1076      finalize(...); return                                           # <-- MC (P3, line ~1078+) never reached
```
Executed proof: `rnn05bext2_results.json` has `mc={}`, `snapshot_identity={}`, `ablation={}` (empty) — MC never ran.

## 2. Graded-region gate — a COMMON overlapping graded region, not one cell in a band (§7)
`graded_region_gate()`, `ops/rnn_05b_ext2.py:1196-1218`.
```
1196  def graded_region_gate(cfg, gdn_curves, doses):
1201      mx, mn = max(curve), min(curve)
1202      mid_doses = [doses[i] for i,a in enumerate(curve) if cfg.mid_lo <= a <= cfg.mid_hi]
1203      competent = mx >= cfg.grade_hi        # 0.75
1204      degrades  = mn <= cfg.grade_lo        # 0.45
1205      resolved  = len(mid_doses) >= cfg.min_mid_doses   # >=2
1206      ... seed_graded = competent and degrades and resolved
1211      common = set.intersection(*mid_sets)  # overlap across ALL GDN seeds
1213      qualified = all_graded and len(common) >= 1
```
Executed: every GDN seed `competent=True` but `degrades=False` (min BASE 0.984–0.996, never <=0.45), `mid_doses=[]`
-> `seed_graded=False` -> `FIXED_BACKBONE_GRADED_REGION=BLOCKED`.

## 3. challengeGridSha256 identity + process-stable self-check (§4)
`Ext2Config.challenge_grid()/challenge_grid_sha256()`, `:133-158`; `grid_selfcheck()` (`:174-188`) reconstructs the
EXACT config in a fresh subprocess and re-derives the digest. Recorded identically in PRE_REGISTRATION.md,
machine_config.json, BASE_QUALIFICATION.json, results.meta, outcomes = `66ff24765d17c4fa...`.

## 4. Nested monotonic stress generator (§5) + fixed snapshot positions (§8)
`gap_positions()` (`:200`), `make_base_example()` (`:222`), `materialize()` (`:264`). Distractors fill POST-WRITE
gap positions in a FIXED ascending order shared by all examples; `materialize(dose)` sets the first
`n_distractors_at(dose)` of them -> higher dose is a strict SUPERSET (nested). Writes/queries/targets/positions
are identical across doses (generator self-test: `nested_monotonic=True`, `writes_queries_survive_all_doses=True`,
`no_answer_leak=True`, `memory_bound_design=True`, mean write->query distance 451/512).

## 5. Train ONCE, freeze, SHA-256 (§2) — one recipe, mixture over dose ladder
`train_backbone()` (`:374`, domain-randomized dose per step) then `run()` P1 (`:995-1019`): freeze all params,
`backbone_sha256()`, `torch.save`. The SAME frozen weights are reloaded (`_reload_backbone`, sha-asserted) for
every dose in BASE qualification.

## 6. Target-aware ablation with a valid independent random control (fixes RNN-05B-EXT audit §3)
`target_aware_ablation()`, `:633-720`. Per-target proximal = first snapshot at/after the write segment; the
DROP_RANDOM index is drawn deterministically from indices EXCLUDING the proximal set AND the irrelevant index,
asserted in code (`:657-660`). (NOT EXECUTED this run — gated out by the BLOCK; included for completeness and
validated in the pipeline smoke where random=4, proximal={0,1}, irrelevant=6.)

## 7. SESOI + decision policy (§16, §21)
`_classify_delta_aurc()` (`:1244`) and `build_outcomes()` (`:1268`). PRIMARY SESOI on DELTA_AURC = 0.05; the 3%
margin is `margin_OPERATOR_HEURISTIC` only. Case A path (`:1279-1287`) sets H3_TESTABILITY=BLOCKED_FIXED_BACKBONE,
QWEN_GDN_TRANSPLANT_GATE=DEFER, SYNTHETIC_DENSE_MC=PARK, decision_case="A".
