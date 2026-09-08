# RNN-06D — Recovery Ceiling & Utility Gate

[← Scientific interpretation](scientific-interpretation.md) · [Research-line index](README.md) · [Research Frontier](../../../frontier/RESEARCH_FRONTIER.md)

**Status:** NEXT EXPERIMENT, not executed here.

## D0 — Recovery Ceiling & Snapshot Schedule Qualification

Randomize target write position in held-out examples. Capture a fixed snapshot schedule blind to target location.

Compare:
- FINAL current-state baseline;
- ORACLE-PROXIMAL diagnostic;
- ORACLE-BEST upper bound using gold, diagnostic only;
- FIXED HISTORICAL POOL available to real selectors.

If even ORACLE-BEST fails to beat FINAL by a preregistered material SESOI, park historical snapshot recovery.

## D1 — Parameter-Free Recovery Utility

Only if D0 opens the gate. Compare target-agnostic heuristics such as recency, max confidence, min entropy, margin, preregistered confidence×recency, logit ensemble and final+historical mixes.

Primary utility should account for `recovered − harmed` (equivalent to accuracy delta versus FINAL), plus MiB and latency.

If ceiling is high but heuristics are weak, one trained selector/DART-like experiment may be justified. If heuristics are strong, replicate on official Mamba lineage before Qwen/GDN transfer.

---

**Continue:** [Local Model Lab](../../../research/local-inference/local-model-lab-methodology.md) · [Research Frontier](../../../frontier/RESEARCH_FRONTIER.md) · [Curated Findings](../../../findings/CURATED_FINDINGS.md)
