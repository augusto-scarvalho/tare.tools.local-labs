# Gradual undervolt-via-clock-lock sweep for the RTX 3090.
# Run this in an ELEVATED (Admin) PowerShell. Non-admin cannot change GPU clocks.
# For each locked core clock it runs the same llama-bench (WSL) and records
# t/s, peak power/temp/clock, and any NEW nvlddmkm-153 (GPU reset) events.
# It always resets clocks to default at the end. Nothing here is persistent
# (a reboot clears the lock) and nothing touches voltage or the memory clock.
# ASCII-only on purpose (Windows PowerShell 5.1 reads .ps1 as ANSI).

$ErrorActionPreference = 'Continue'
$distro = 'Ubuntu-24.04'
$wuser  = 'augus'
$clocks = @(0, 1860, 1815, 1770, 1725, 1680)   # 0 = reset to stock (control row)

# admin check
$isAdmin = ([Security.Principal.WindowsPrincipal] [Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltinRole]::Administrator)
if (-not $isAdmin) { Write-Host "NOT elevated. Open PowerShell as Administrator and re-run." -ForegroundColor Red; return }

function Get-Num($text, $pat) { if ($text -match $pat) { return $Matches[1] } else { return '?' } }

# stage the bench script into WSL /tmp (strip CRLF via bash ANSI-C quoting)
$benchWin = '/mnt/c/projects/local-model-lifecycle/ops/gpu-stability/uv_bench.sh'
& wsl.exe -d $distro -u $wuser -- bash -lc "sed `$'s/\r//' '$benchWin' > /tmp/uv_bench.sh" | Out-Null
Write-Host "staged bench -> /tmp/uv_bench.sh" -ForegroundColor DarkGray

$results = @()
foreach ($c in $clocks) {
  if ($c -eq 0) { & nvidia-smi -rgc | Out-Null; $lab = 'stock' }
  else          { & nvidia-smi -lgc "$c,$c" | Out-Null; $lab = "lgc$c" }
  Start-Sleep -Seconds 2
  Write-Host ">>> Running bench @ $lab ..." -ForegroundColor Cyan
  $t0  = Get-Date
  $out = & wsl.exe -d $distro -u $wuser -- bash -lc "bash /tmp/uv_bench.sh $lab" 2>&1 | Out-String

  $row = [pscustomobject]@{
    Clock   = $lab
    pp2048  = Get-Num $out 'pp2048\s*\|\s*([\d.]+)'
    tg512   = Get-Num $out 'tg512\s*\|\s*([\d.]+)'
    PeakW   = (Get-Num $out 'PEAK_POWER_W=\s*([\d.]+)')
    PeakC   = (Get-Num $out 'PEAK_TEMP_C=\s*([\d.]+)')
    PeakMHz = (Get-Num $out 'PEAK_SM_MHZ=\s*([\d.]+)')
    RC      = (Get-Num $out 'BENCH_RC=([\d]+)')
    New153  = (Get-WinEvent -FilterHashtable @{LogName='System'; ProviderName='nvlddmkm'; Id=153; StartTime=$t0} -ErrorAction SilentlyContinue | Measure-Object).Count
  }
  $results += $row
  $results | Format-Table -AutoSize | Out-String | Write-Host
  if ($row.New153 -gt 0 -or $row.RC -ne '0') {
    Write-Host ("!! instability at {0} (153={1} rc={2}) - stopping sweep." -f $lab, $row.New153, $row.RC) -ForegroundColor Red
    break
  }
  Start-Sleep -Seconds 8   # brief cooldown between steps
}

& nvidia-smi -rgc | Out-Null
Write-Host "=== SWEEP DONE - clocks reset to default ===" -ForegroundColor Green
$results | Format-Table -AutoSize
$csv = "$env:USERPROFILE\uv_sweep_results.csv"
$results | Export-Csv -NoTypeInformation -Encoding UTF8 $csv
Write-Host "Saved: $csv"
