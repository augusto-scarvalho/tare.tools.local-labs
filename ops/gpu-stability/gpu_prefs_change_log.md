# GPU-preference change — 2026-08-02

## What & why
To free RTX 3090 VRAM for the LLM, 11 heavy Chromium/Electron desktop apps were set to render
on the **Intel UHD 770 iGPU** (Windows per-app GPU preference = *Power saving*), instead of the
3090. Context: the desktop was holding ~9.3 GB of the 3090's 24 GB VRAM; moving app rendering to
the iGPU (which uses system RAM) reclaims some of that so the model can sit more resident on the
GPU (lower `--n-cpu-moe` → faster decode). See STATUS.md §E1 and memory `placement-is-the-decode-lever`.

## Exactly what changed
Registry key `HKCU\Software\Microsoft\DirectX\UserGpuPreferences`, added value `GpuPreference=1;`
(1 = Power saving = iGPU) for these 11 executables:

- brave.exe, msedge.exe, msedgewebview2.exe (×2 versions)
- Code.exe (VS Code), Cursor.exe
- ChatGPT Classic.exe, WhatsApp.Root.exe, NordPass.exe, Playnite.DesktopApp.exe
- steamwebhelper.exe

**Untouched on purpose:** games (`AcCoreConsole.exe` = GpuPreference=2 → 3090), and `crosvm.exe`
(the WSL2 VM — its CUDA compute MUST stay on the 3090).

## Caveats
- Takes effect only when each app is **restarted** (or on reboot).
- **Partial** VRAM win: app render surfaces move to the iGPU, but the desktop compositor (DWM)
  stays on the 3090 while the monitor is plugged into it. Full ~9 GB needs the display itself on
  the iGPU (a USB-C→HDMI/DP adapter — this board has no plain HDMI/DP out).
- Trades VRAM for a few GB of system RAM (iGPU has no VRAM). Keep an eye on Windows-available vs
  the 16 GB reserve.
- The versioned WindowsApps paths (ChatGPT, WhatsApp, WebView2) reset when those apps update —
  re-run the set command if so.

## How to revert
- Cleanest: `powershell -ExecutionPolicy Bypass -File ops\revert_gpu_prefs.ps1` (removes the 11
  entries → "Let Windows decide").
- Or per app in the GUI: Settings → System → Display → Graphics.
- `ops\gpu_prefs_backup.reg` is a `reg export` snapshot of the key taken right AFTER the change
  (full state, for reference / `reg import` restore).
