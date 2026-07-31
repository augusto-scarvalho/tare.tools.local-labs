"""What the owner's own desktop costs, measured — the basis for the reserve.

The envelope's 16 GB RAM / 4 GB VRAM reserve was DECLARED, not measured. This module
measures what the working set actually costs, so the reserve can be checked against
reality instead of intuition.

Grouped by application family because per-process numbers mislead badly here: Chromium
apps (browser, Cursor, VSCode, WhatsApp, Electron anything) spawn dozens of helper
processes, and reading any single one understates the family by an order of magnitude.
"""
from __future__ import annotations

import collections
import json
import subprocess

# Families, not process names. A user does not run "chrome.exe" x37, they run a browser.
FAMILIES: dict[str, tuple[str, ...]] = {
    "browser":      ("chrome", "msedge", "firefox", "brave", "opera", "vivaldi"),
    "cursor":       ("cursor",),
    "vscode":       ("code", "code - insiders"),
    "whatsapp":     ("whatsapp",),
    "office":       ("soffice", "soffice.bin", "winword", "excel", "powerpnt", "onenote"),
    "terminal":     ("windowsterminal", "powershell", "pwsh", "cmd", "conhost", "wt"),
    "wsl":          ("wsl", "wslhost", "wslservice", "vmmem", "vmmemwsl"),
    "docker":       ("docker", "com.docker.backend", "dockerd"),
    "llama":        ("llama-server", "llama-bench"),
    "python":       ("python", "python3", "py"),
    "communication": ("discord", "slack", "teams", "telegram", "zoom"),
}


def _family(name: str) -> str:
    n = name.lower().removesuffix(".exe")
    for fam, needles in FAMILIES.items():
        if any(n == x or n.startswith(x) for x in needles):
            return fam
    return "other"


def snapshot() -> dict:
    """Per-family RAM working set, plus totals. Uses .NET process enumeration through
    PowerShell rather than a performance counter: counter PATHS are localised and this
    host is pt-BR -- the same trap that once made a guard read the machine as having
    zero RAM."""
    ps = ("Get-Process | Select-Object Name,WorkingSet64,Id | "
          "ConvertTo-Json -Compress")
    out = subprocess.run(["powershell.exe", "-NoProfile", "-Command", ps],
                         capture_output=True, text=True, timeout=60)
    procs = json.loads(out.stdout or "[]")
    if isinstance(procs, dict):
        procs = [procs]

    by_family: dict[str, dict] = collections.defaultdict(
        lambda: {"mb": 0.0, "processes": 0})
    for p in procs:
        fam = _family(p.get("Name", ""))
        by_family[fam]["mb"] += (p.get("WorkingSet64") or 0) / (1024 * 1024)
        by_family[fam]["processes"] += 1

    gpu = subprocess.run(
        ["nvidia-smi.exe", "--query-gpu=memory.total,memory.used", "--format=csv,noheader,nounits"],
        capture_output=True, text=True, timeout=30)
    total_mb = used_mb = None
    if gpu.returncode == 0 and gpu.stdout.strip():
        total_mb, used_mb = (int(x) for x in gpu.stdout.strip().splitlines()[0].split(","))

    ram = subprocess.run(
        ["powershell.exe", "-NoProfile", "-Command",
         "$os=Get-CimInstance Win32_OperatingSystem;"
         "$c=Get-CimInstance Win32_ComputerSystem;"
         "$p=Get-CimInstance Win32_PerfRawData_PerfOS_Memory;"
         "[pscustomobject]@{TotalMB=[int]($c.TotalPhysicalMemory/1MB);"
         "AvailMB=[int]$p.AvailableMBytes;"
         "CommittedMB=[int]($os.TotalVirtualMemorySize/1KB - $os.FreeVirtualMemory/1KB)}"
         "| ConvertTo-Json -Compress"],
        capture_output=True, text=True, timeout=60)
    mem = json.loads(ram.stdout or "{}")

    families = {k: {"mb": round(v["mb"], 1), "processes": v["processes"]}
                for k, v in sorted(by_family.items(), key=lambda kv: -kv[1]["mb"])}
    return {
        "families": families,
        "working_set_total_mb": round(sum(v["mb"] for v in families.values()), 1),
        "ram_total_mb": mem.get("TotalMB"),
        "ram_available_mb": mem.get("AvailMB"),
        "ram_committed_mb": mem.get("CommittedMB"),
        "vram_total_mb": total_mb,
        "vram_used_mb": used_mb,
    }


if __name__ == "__main__":
    s = snapshot()
    print(f"{'family':<16}{'RAM MB':>10}{'procs':>7}")
    print("-" * 34)
    for fam, v in s["families"].items():
        if v["mb"] >= 50:
            print(f"{fam:<16}{v['mb']:>10.0f}{v['processes']:>7}")
    print("-" * 34)
    print(f"{'sum working set':<16}{s['working_set_total_mb']:>10.0f}")
    print()
    print(f"RAM   total {s['ram_total_mb']} MB | available {s['ram_available_mb']} MB "
          f"| committed {s['ram_committed_mb']} MB")
    print(f"VRAM  total {s['vram_total_mb']} MB | used {s['vram_used_mb']} MB")
