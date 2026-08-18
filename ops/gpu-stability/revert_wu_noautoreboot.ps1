# Revert: Windows Update "no auto-restart with logged-on users" policy
# Applied 2026-08-12 to stop the Update Orchestrator from auto-rebooting long GPU runs
# (a planned WU restart at 01:29 on 2026-08-12 killed the RNN-06C run; NOT a blackout).
#
# BACKUP STATE AT APPLY TIME: the key
#   HKLM:\SOFTWARE\Policies\Microsoft\Windows\WindowsUpdate\AU
# did NOT exist before; value NoAutoRebootWithLoggedOnUsers was ABSENT.
# Revert therefore removes the value (and the AU key if left empty).
#
# Run elevated:  powershell -ExecutionPolicy Bypass -File ops\revert_wu_noautoreboot.ps1

$key = 'HKLM:\SOFTWARE\Policies\Microsoft\Windows\WindowsUpdate\AU'
if (Test-Path $key) {
    Remove-ItemProperty -Path $key -Name 'NoAutoRebootWithLoggedOnUsers' -ErrorAction SilentlyContinue
    $remaining = (Get-Item $key).GetValueNames()
    if (-not $remaining -or $remaining.Count -eq 0) {
        Remove-Item -Path $key -Force
        Write-Output "Reverted: removed NoAutoRebootWithLoggedOnUsers and the empty AU key."
    } else {
        Write-Output "Reverted: removed NoAutoRebootWithLoggedOnUsers (AU key kept; other values present)."
    }
} else {
    Write-Output "Nothing to revert: AU key not present."
}
