# Market-comparison quality + speed benchmark for fable-tc l1.0.
# DEPLOY CONFIG, speed levers ON (MTP self-draft on the lifecycle fork binary; CUDA graphs +
# MMQ default-on; VRAM OC +350 hardware). Short-context standard benchmarks (HumanEval+, GSM8K)
# so the MoE / long-context fork levers do not apply.
#
# Runs from WINDOWS python (a2_concision_bench shells wsl.exe to serve; client hits 127.0.0.1:8080).
# Durable: everything logs to runs\quality-market\run.log; a2_concision_bench writes each problem
# incrementally and RESUMES the same tag if re-run, so an interrupted job restarts where it stopped.
#
#   powershell -ExecutionPolicy Bypass -File ops\run_market_bench.ps1                # real run (60/200/4096)
#   powershell -ExecutionPolicy Bypass -File ops\run_market_bench.ps1 -HumanEvalN 2 -Gsm8kN 2 -MaxTokens 1024 -Tag test
param(
  [int]$HumanEvalN = 60,
  [int]$Gsm8kN = 200,
  [int]$MaxTokens = 4096,
  [string]$Tag = "market-r0"
)
$ErrorActionPreference = "Continue"
Set-Location C:\projects\local-model-lifecycle
$env:PYTHONPATH = "src"
$MODEL = "fable-tc-l1.0-q4"
$OUT = "runs\quality-market"
New-Item -ItemType Directory -Force -Path $OUT | Out-Null
$log = Join-Path $OUT "run.log"
function Log($m) { $line = "{0}  {1}" -f ([DateTime]::Now.ToString('u')), $m; Add-Content -Path $log -Value $line; Write-Output $line }

Log "===== START tag=$Tag humaneval=$HumanEvalN gsm8k=$Gsm8kN max_tokens=$MaxTokens ====="

# 1) GENERATE (self-serves fable-tc with MTP; records quality inputs + per-problem t/s)
Log "--- generate HumanEval+ (n=$HumanEvalN) ---"
python a2_concision_bench.py --model $MODEL --workload humaneval --subset $HumanEvalN --spec draft-mtp --max-tokens $MaxTokens --tag $Tag *>> $log
Log "--- generate GSM8K (n=$Gsm8kN) ---"
python a2_concision_bench.py --model $MODEL --workload gsm8k --subset $Gsm8kN --spec draft-mtp --max-tokens $MaxTokens --tag $Tag *>> $log

# 2) SCORE HumanEval+ via evalplus (code executed in the WSL sandbox; subset-aware)
Log "--- score HumanEval+ (evalplus, WSL sandbox) ---"
$samplesWsl = "/mnt/c/projects/local-model-lifecycle/runs/a2/${Tag}__${MODEL}__humaneval__samples.jsonl"
$heScore = Join-Path $OUT "humaneval_score.txt"
$heOut = (wsl.exe -d Ubuntu-24.04 -- bash -lc "cd /mnt/c/projects/local-model-lifecycle && /home/augus/evalplus-venv/bin/python3 score_subset.py $samplesWsl" 2>&1 | Out-String)
Set-Content -Path $heScore -Value $heOut -Encoding utf8   # UTF-8 (summarizer is also BOM-aware)
Add-Content -Path $log -Value $heOut

# 3) SUMMARIZE (GSM8K numeric accuracy + decode t/s + concision) -> SUMMARY.md
Log "--- summarize -> SUMMARY.md ---"
python ops\summarize_market_bench.py --tag $Tag --model $MODEL --out (Join-Path $OUT "SUMMARY.md") *>> $log

Log "===== DONE tag=$Tag ====="
