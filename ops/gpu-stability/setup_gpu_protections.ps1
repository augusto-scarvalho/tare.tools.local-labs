# ============================================================================
#  RTX 3090 stability protections installer  (RUN AS ADMINISTRATOR)
#  1) Persistent core-clock cap at boot  -> kills the high-boost/high-voltage
#     region tied to the "fell off the bus" (Xid-79) crashes, while still
#     letting the card idle down (range form 210,MAX).
#  2) TDR delay bump -> gives long GPU kernels headroom without disabling the
#     recovery safety net (TdrLevel stays default = 3).
#  Fully reversible -- see the REVERT block printed at the end.
#
#  UPDATE 2026-08-04: a paired A/B showed the 1800-MHz cap costs ~0% on the deploy
#  MoE prefill (2387 locked @1800 vs 2311 unlocked @1905 t/s -- transfer-bound, not
#  core-clock-bound; the -4% seen on gpt-oss was a GPU-compute-bound case). So the
#  cap is nearly free on real workloads. It has been SUPERSEDED (2026-08-04) by a GPU
#  undervolt (Afterburner V/F curve, ~1860 MHz @ 850 mV flat, "Apply at startup"),
#  which kills the same high-voltage Xid-79 region while keeping full clocks. The clock
#  cap was DISABLED (task unregistered, -rgc released, boot helper removed; TDR delay
#  KEPT). Undervolt VALIDATED: 10-min sustained soak held 1860 MHz, peak 58C / 274W,
#  decode 93 t/s stable, zero Xid/drop, no early->late drift.
#
#  If you ever need to disable the cap again (KEEP the TDR delay):
#     Unregister-ScheduledTask -TaskName 'RTX3090-ClockLock' -Confirm:$false   # stop boot re-apply
#     nvidia-smi -rgc                                                          # release current lock
#     Remove-Item -Recurse -Force "$env:ProgramData\gpu-tools"                 # remove boot helper
#     # do NOT remove TdrDelay/TdrDdiDelay -- leave the TDR bump in place.
# ============================================================================

# ---- tunables -------------------------------------------------------------
$CoreClockMax = 1800      # boost ceiling (MHz). Conservative & safe; refine via the sweep.
$IdleFloor    = 210       # keep low so the card still idles to P8 (~27W). Do not raise.
$TdrDelaySec  = 10        # seconds (default is 2). Recovery stays ON.
$TaskName     = 'RTX3090-ClockLock'
$ToolDir      = Join-Path $env:ProgramData 'gpu-tools'
$BootScript   = Join-Path $ToolDir 'gpu_clocklock_boot.ps1'
# ---------------------------------------------------------------------------

$isAdmin = ([Security.Principal.WindowsPrincipal] [Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltinRole]::Administrator)
if (-not $isAdmin) { Write-Host "NOT elevated. Open PowerShell as Administrator and re-run." -ForegroundColor Red; return }

$smi = (Get-Command nvidia-smi -ErrorAction SilentlyContinue).Source
if (-not $smi) { $smi = Join-Path $env:SystemRoot 'System32\nvidia-smi.exe' }
if (-not (Test-Path $smi)) { Write-Host "nvidia-smi not found." -ForegroundColor Red; return }

# --- 1a. durable boot helper (retries + logs; driver may lag at startup) ----
New-Item -ItemType Directory -Force -Path $ToolDir | Out-Null
$helper = @'
param([int]$Min=210,[int]$Max=1800)
$log = Join-Path $env:ProgramData 'gpu-tools\gpu_clocklock.log'
$smi = Join-Path $env:SystemRoot 'System32\nvidia-smi.exe'
if (-not (Test-Path $smi)) { $smi = 'nvidia-smi.exe' }
for ($i=1; $i -le 12; $i++) {
  $out = & $smi -lgc "$Min,$Max" 2>&1 | Out-String
  "$(Get-Date -Format o)  try $i  rc=$LASTEXITCODE  $($out.Trim())" | Add-Content -Path $log
  if ($LASTEXITCODE -eq 0 -and $out -notmatch 'permission|Error|not supported') { break }
  Start-Sleep -Seconds 6
}
'@
Set-Content -Path $BootScript -Value $helper -Encoding UTF8
Write-Host "[1] boot helper written -> $BootScript" -ForegroundColor Green

# --- 1b. scheduled task: apply the cap at every startup (as SYSTEM) ---------
$action = New-ScheduledTaskAction -Execute 'powershell.exe' `
          -Argument "-NoProfile -ExecutionPolicy Bypass -File `"$BootScript`" -Min $IdleFloor -Max $CoreClockMax"
$trigger = New-ScheduledTaskTrigger -AtStartup
$trigger.Delay = 'PT30S'
$principal = New-ScheduledTaskPrincipal -UserId 'SYSTEM' -LogonType ServiceAccount -RunLevel Highest
$settings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -StartWhenAvailable
Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false -ErrorAction SilentlyContinue
Register-ScheduledTask -TaskName $TaskName -Action $action -Trigger $trigger -Principal $principal -Settings $settings | Out-Null
Write-Host "[2] boot task '$TaskName' registered (SYSTEM, +30s delay)" -ForegroundColor Green

# --- 1c. apply NOW so it is active without a reboot ------------------------
$now = & $smi -lgc "$IdleFloor,$CoreClockMax" 2>&1 | Out-String
Write-Host "[3] applied now: -lgc $IdleFloor,$CoreClockMax  -> $($now.Trim())" -ForegroundColor Green

# --- 2. TDR delay (needs reboot to take effect) ----------------------------
$gk = 'HKLM:\SYSTEM\CurrentControlSet\Control\GraphicsDrivers'
New-ItemProperty -Path $gk -Name 'TdrDelay'    -Value $TdrDelaySec -PropertyType DWord -Force | Out-Null
New-ItemProperty -Path $gk -Name 'TdrDdiDelay' -Value $TdrDelaySec -PropertyType DWord -Force | Out-Null
Write-Host "[4] TdrDelay=$TdrDelaySec s, TdrDdiDelay=$TdrDelaySec s set (TdrLevel left = 3/recover). REBOOT to activate." -ForegroundColor Green

# --- verify ----------------------------------------------------------------
Write-Host "`n--- current state ---" -ForegroundColor Cyan
& $smi --query-gpu=pstate,clocks.sm,clocks.max.sm,temperature.gpu,power.draw --format=csv
Write-Host "`nProtections installed."
Write-Host "  * Clock cap active now AND every boot. Idle still drops to P8 (~27W)."
Write-Host "  * TDR bump active after next reboot."
Write-Host "`n--- TO REVERT everything ---" -ForegroundColor Yellow
Write-Host "  Unregister-ScheduledTask -TaskName '$TaskName' -Confirm:`$false"
Write-Host "  nvidia-smi -rgc"
Write-Host "  Remove-ItemProperty -Path '$gk' -Name TdrDelay,TdrDdiDelay"
Write-Host "  Remove-Item -Recurse -Force '$ToolDir'   # (then reboot to clear TDR)"
