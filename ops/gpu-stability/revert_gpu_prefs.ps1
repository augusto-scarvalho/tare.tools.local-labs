# Revert the 2026-08-02 GPU-preference change: the 11 apps forced onto the iGPU
# (UHD 770, GpuPreference=1) go back to "Let Windows decide" (the entry is removed).
# Run in PowerShell:  powershell -ExecutionPolicy Bypass -File revert_gpu_prefs.ps1
# Takes effect when each app is next restarted. Games (AcCoreConsole=2) are untouched.

$k = 'HKCU:\Software\Microsoft\DirectX\UserGpuPreferences'
$apps = @(
  'C:\Program Files\BraveSoftware\Brave-Browser\Application\brave.exe',
  'C:\Program Files\WindowsApps\OpenAI.ChatGPT-Desktop_1.2026.190.0_x64__2p2nqsd0c76g0\app\ChatGPT Classic.exe',
  'C:\Users\augus\AppData\Local\Programs\Microsoft VS Code\Code.exe',
  'C:\Users\augus\AppData\Local\Programs\cursor\Cursor.exe',
  'C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe',
  'C:\Program Files (x86)\Microsoft\EdgeWebView\Application\150.0.4078.105\msedgewebview2.exe',
  'C:\Program Files (x86)\Microsoft\EdgeWebView\Application\150.0.4078.83\msedgewebview2.exe',
  'C:\Users\augus\AppData\Local\Programs\nordpass\NordPass.exe',
  'C:\Users\augus\AppData\Local\Playnite\Playnite.DesktopApp.exe',
  'C:\Program Files (x86)\Steam\bin\cef\cef.win64\steamwebhelper.exe',
  'C:\Program Files\WindowsApps\5319275A.WhatsAppDesktop_2.2628.101.0_x64__cv1g1gvanyjgm\WhatsApp.Root.exe'
)
$removed = 0
foreach ($a in $apps) {
  $p = Get-ItemProperty -Path $k -Name $a -ErrorAction SilentlyContinue
  if ($p) { Remove-ItemProperty -Path $k -Name $a -Force; Write-Output "reverted (removed): $a"; $removed++ }
  else    { Write-Output "already absent:      $a" }
}
Write-Output "--- reverted $removed of $($apps.Count) apps to 'Let Windows decide' ---"
Write-Output "Restart each app (or reboot) for it to take effect."

# FULL RESTORE alternative (put the key back exactly as it was at backup time):
#   reg import "C:\projects\tare.tools.local-labs\ops\gpu_prefs_backup.reg"
# (Note: gpu_prefs_backup.reg is the POST-change snapshot; this revert script is the
#  cleaner undo. To capture a pre-anything baseline in future, export BEFORE editing.)
