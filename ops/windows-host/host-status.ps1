# Writes C:\ProgramData\tare\host-status.json for the WSL model gateway: Windows free RAM and how long the
# desktop has gone without keyboard/mouse input. Runs every minute inside the logged-on user's session
# (GetLastInputInfo only sees its own session; Windows Home has no quser). A stale file means "unknown".
# Install on the Windows host (as its desktop user, elevated shell), copying this file to C:\ProgramData\tare:
#   icacls C:\ProgramData\tare /grant "${env:COMPUTERNAME}\${env:USERNAME}:(OI)(CI)M" /T
#   $a = New-ScheduledTaskAction -Execute conhost.exe -Argument '--headless powershell.exe -NoProfile -NonInteractive -WindowStyle Hidden -ExecutionPolicy Bypass -File C:\ProgramData\tare\host-status.ps1'
#   $t = (New-ScheduledTaskTrigger -AtLogOn -User $env:USERNAME), (New-ScheduledTaskTrigger -Once -At (Get-Date) -RepetitionInterval (New-TimeSpan -Minutes 1))
#   Register-ScheduledTask Tare-HostStatus -Action $a -Trigger $t -Principal (New-ScheduledTaskPrincipal -UserId $env:USERNAME -LogonType Interactive) -Settings (New-ScheduledTaskSettingsSet -ExecutionTimeLimit (New-TimeSpan -Minutes 1) -MultipleInstances IgnoreNew)
Add-Type -Namespace Tare -Name Input -MemberDefinition @'
[StructLayout(LayoutKind.Sequential)] public struct LASTINPUTINFO { public uint cbSize; public uint dwTime; }
[DllImport("user32.dll")] static extern bool GetLastInputInfo(ref LASTINPUTINFO info);
public static uint IdleMilliseconds() {
    var info = new LASTINPUTINFO(); info.cbSize = (uint)Marshal.SizeOf(info);
    GetLastInputInfo(ref info); return unchecked((uint)Environment.TickCount - info.dwTime);
}
'@
$os = Get-CimInstance Win32_OperatingSystem
$status = [ordered]@{
    ts = [DateTimeOffset]::UtcNow.ToUnixTimeSeconds()
    windows_free_gb = [math]::Round($os.FreePhysicalMemory / 1MB, 2)
    windows_total_gb = [math]::Round($os.TotalVisibleMemorySize / 1MB, 2)
    desktop_idle_seconds = [int]([Tare.Input]::IdleMilliseconds() / 1000)
}
$dir = 'C:\ProgramData\tare'
New-Item -ItemType Directory -Force $dir | Out-Null
($status | ConvertTo-Json -Compress) | Set-Content -Encoding ascii "$dir\host-status.json.tmp"
Move-Item -Force "$dir\host-status.json.tmp" "$dir\host-status.json"
