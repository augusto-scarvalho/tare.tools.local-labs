# Apply the reclaim-enabled .wslconfig, with a backup and a before/after measurement.
# The before/after is the point: it turns "should free memory" into a measured number.
$ErrorActionPreference = 'Stop'
$cfg = Join-Path $env:USERPROFILE '.wslconfig'
$bak = Join-Path $env:USERPROFILE ('.wslconfig.bak-' + (Get-Date -Format 'yyyyMMdd-HHmmss'))

function Mem {
    $p = Get-CimInstance Win32_PerfRawData_PerfOS_Memory
    $g = (& nvidia-smi.exe --query-gpu=memory.used --format=csv,noheader,nounits) -split "`n" | Select-Object -First 1
    $wsl = (Get-Process | Where-Object { $_.Name -match 'vmmem|wsl' } |
            Measure-Object WorkingSet64 -Sum).Sum / 1MB
    [pscustomobject]@{ AvailMB = [int]$p.AvailableMBytes; VramMB = [int]$g; WslMB = [int]$wsl }
}

$before = Mem
"BEFORE  avail={0}MB  vram={1}MB  wsl={2}MB" -f $before.AvailMB, $before.VramMB, $before.WslMB

if (Test-Path $cfg) { Copy-Item $cfg $bak; "backup -> $bak" } else { "no previous .wslconfig" }

@'
[wsl2]
memory=40GB
processors=20
swap=16GB
networkingMode=mirrored

[experimental]
autoMemoryReclaim=gradual
'@ | Set-Content -Encoding ascii $cfg
"written -> $cfg"

# Shutdown is what makes the new limits take effect. It also frees the balloon, which
# is exactly the effect being measured below.
wsl --shutdown
Start-Sleep -Seconds 8

$after = Mem
"AFTER   avail={0}MB  vram={1}MB  wsl={2}MB" -f $after.AvailMB, $after.VramMB, $after.WslMB
"RECLAIMED {0}MB of RAM" -f ($after.AvailMB - $before.AvailMB)
