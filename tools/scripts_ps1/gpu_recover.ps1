# Elevated GPU recovery: restart the RTX 3090 device to try clearing a "GPU is lost" bus wedge
# without a reboot. Logs the result so the non-elevated caller can read it.
$log = 'C:\projects\tare.tools.local-labs\scratch\gpu_restart.log'
$id  = 'PCI\VEN_10DE&DEV_2204&SUBSYS_39873842&REV_A1\4&2635B274&0&0008'
"[$(Get-Date -Format HH:mm:ss)] restarting device $id" | Out-File $log
pnputil /restart-device $id *>> $log
"--- sleep 8 ---" | Out-File -Append $log
Start-Sleep 8
"--- nvidia-smi after restart ---" | Out-File -Append $log
& nvidia-smi *>> $log
"[$(Get-Date -Format HH:mm:ss)] done" | Out-File -Append $log
