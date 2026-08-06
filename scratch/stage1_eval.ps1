# A2 Stage-1 eval: for each concise-Fable merge arm, measure (1) concision + GSM8K accuracy vs
# plain Fable, and (2) refusal rate (did the TC delta re-censor the uncensored base?).
# Anchors: fable-plain = uncensored floor; thinkingcap = aligned ceiling for refusal.
# Run from Windows (drives the WSL server). GPU-serial, ~1.5h.
cd C:\projects\local-model-lifecycle
$ErrorActionPreference = 'Continue'

# concision + accuracy: plain Fable (baseline) + the 3 merged arms
$conc = @('fable-plain-q4','fable-tc-l0.4-q4','fable-tc-l0.7-q4','fable-tc-l1.0-q4')
foreach ($m in $conc) {
  Write-Output "=== concision $m ==="
  python a2_concision_bench.py --model $m --workload gsm8k --subset 40 --tag s1 *> "runs/a2/_log_s1_$m.txt"
}

# refusal probe: anchors + all merged arms
$ref = @('fable-plain-q4','thinkingcap-27b-q4','fable-tc-l0.4-q4','fable-tc-l0.7-q4','fable-tc-l1.0-q4')
foreach ($m in $ref) {
  Write-Output "=== refusal $m ==="
  python a2_refusal_probe.py --model $m --tag s1 *> "runs/a2/_log_ref_$m.txt"
}
Write-Output "=== STAGE1 EVAL DONE ==="
