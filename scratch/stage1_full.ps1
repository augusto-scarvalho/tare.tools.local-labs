# A2 Stage-1 FULL campaign (concise-Fable transfer). Drives WSL server from Windows. GPU-serial.
# - Concision+accuracy: GSM8K n=60, tag s1p (plain & l1.0 RESUME the n=8 pilot; l0.4/l0.7 fresh).
#   Headline arms (plain, l1.0) first so the key comparison lands even if interrupted.
# - Refusal: full 20-prompt probe (mild+HARD tier) on all 4 Fable arms + aligned anchors
#   (thinkingcap = aligned ceiling, dense-base = second ceiling). Overwrites the 12-prompt pilot.
# Incremental JSON per arm => crash-safe/resumable. ~2-2.5h.
cd C:\projects\local-model-lifecycle
$ErrorActionPreference = 'Continue'
$stamp = { (Get-Date).ToString('HH:mm:ss') }

Write-Output "[$(& $stamp)] ===== STAGE1 FULL START ====="

# --- concision + accuracy (headline pair first, then fill the sweep) ---
$conc = @('fable-plain-q4','fable-tc-l1.0-q4','fable-tc-l0.7-q4','fable-tc-l0.4-q4')
foreach ($m in $conc) {
  Write-Output "[$(& $stamp)] === concision $m (gsm8k n=60) ==="
  python a2_concision_bench.py --model $m --workload gsm8k --subset 60 --tag s1p *> "runs/a2/_log_conc_$m.txt"
  Write-Output "[$(& $stamp)]   done $m"
}

# --- refusal probe: anchors first (so the discriminating hard tier shows early) ---
$ref = @('thinkingcap-27b-q4','qwen36-27b-dense-q4','fable-plain-q4','fable-tc-l0.4-q4','fable-tc-l0.7-q4','fable-tc-l1.0-q4')
foreach ($m in $ref) {
  Write-Output "[$(& $stamp)] === refusal $m (20 prompts) ==="
  python a2_refusal_probe.py --model $m --tag s1p *> "runs/a2/_log_ref_$m.txt"
  Write-Output "[$(& $stamp)]   done $m"
}

Write-Output "[$(& $stamp)] ===== STAGE1 FULL DONE ====="
