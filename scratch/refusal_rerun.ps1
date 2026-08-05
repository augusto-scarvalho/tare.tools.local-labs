# A2 refusal RERUN com o probe consertado (max_tokens 4096 + regex largo + starved=inconclusivo).
# Skip por braço: se refusal__s1p__<arm>.json ja tem 20 recs COM campo 'starved' (schema novo), pula.
# => relançamento apos kill do reaper e barato (so roda o que falta).
cd C:\projects\local-model-lifecycle
$ErrorActionPreference = 'Continue'
$stamp = { (Get-Date).ToString('HH:mm:ss') }
Write-Output "[$(& $stamp)] ===== REFUSAL RERUN (@4096) START ====="
# ordem p/ fail-fast: Gate 1 (fable-plain + TC) e Gate 2 (l1.0) fecham nos 3 primeiros.
# l0.4 dropado (dominado); fusion-711 = validade externa (raw DavidAU, fora dos stats pareados).
$arms = @('fable-plain-q4','thinkingcap-27b-q4','fable-tc-l1.0-q4','qwen36-27b-dense-q4','fable-tc-l0.7-q4','fable-fusion-711-q4')
foreach ($m in $arms) {
  $jf = "runs\a2\refusal__s1p__$m.json"
  $skip = $false
  if (Test-Path $jf) {
    $chk = python -c "import json,sys;d=json.load(open(sys.argv[1]));print(1 if len(d)==28 and all('verdict' in x for x in d) else 0)" $jf
    if ($chk -eq '1') { $skip = $true }
  }
  if ($skip) { Write-Output "[$(& $stamp)] skip $m (ja feito, schema novo)"; Write-Output "[$(& $stamp)]   done $m"; continue }
  Write-Output "[$(& $stamp)] === refusal $m (20 prompts @4096) ==="
  python a2_refusal_probe.py --model $m --tag s1p *> "runs/a2/_log_ref2_$m.txt"
  Write-Output "[$(& $stamp)]   done $m"
}
Write-Output "[$(& $stamp)] ===== REFUSAL RERUN DONE ====="
