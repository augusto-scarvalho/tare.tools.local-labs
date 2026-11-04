"""Host readings — GPU and RAM, as the WINDOWS side sees them.

The envelope protects the machine the owner works on. WSL's view of memory is not
that machine's view, so every reading here is taken on Windows even though the
runtime lives in WSL.

Two rules learned the hard way (see STATUS.md):
  * never read RAM through a localised performance-counter path;
  * a reading that cannot be taken raises, it does not return zero. A guard that
    silently reads 0 refuses every configuration and says nothing.
"""
from __future__ import annotations

import subprocess
from dataclasses import dataclass


class HostReadError(RuntimeError):
    """A required host metric could not be read. Never downgraded to a default."""


@dataclass(frozen=True)
class HostSample:
    vram_total_mb: int
    vram_used_mb: int
    ram_available_mb: int
    # GPU core utilisation, percent busy over nvidia-smi's sampling window (§B3). The
    # ENVELOPE never reads this -- it is instrumentation, not a reserve -- so it carries a
    # default and construction sites that only care about memory (the guard self-check)
    # need not supply it. `sample()` always populates it from a real read; the default is
    # for hand-built samples only, and 0 there means "not measured", never a silent floor.
    gpu_util_pct: int = 0

    @property
    def vram_free_mb(self) -> int:
        return self.vram_total_mb - self.vram_used_mb


def _run(cmd: list[str], timeout: int = 15) -> str:
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except (OSError, subprocess.SubprocessError) as exc:
        raise HostReadError(f"{cmd[0]} failed: {type(exc).__name__}: {exc}") from exc
    if p.returncode != 0:
        raise HostReadError(f"{cmd[0]} rc={p.returncode}: {(p.stderr or '').strip()[:200]}")
    return p.stdout


# `.exe` on purpose, and it is load-bearing.
#
# The envelope protects the WINDOWS host, so reading WSL's own memory would answer the
# wrong question. The `.exe` suffix pins these to the real Windows binaries regardless of
# which side the collector runs on: under WSL2, interop resolves `nvidia-smi.exe` and
# `powershell.exe` to the Windows executables; under native Windows Python they are those
# executables directly. One code path, correct from either side -- which is what keeps the
# measurement describing the Windows host and not the guest.
#
# (Corrected 2026-07-31: this comment used to open "There is no Python on the Windows side
# of this desktop." There is -- Python 3.12.10 -- and the harness may now run from it. The
# suffix reasoning above never depended on that false premise.)
_NVIDIA_SMI = "nvidia-smi.exe"
_POWERSHELL = "powershell.exe"


def read_gpu() -> tuple[int, int, int]:
    """(total_mb, used_mb, util_pct) for GPU 0.

    `utilization.gpu` is percent of the last sampling window a kernel was resident on the
    SMs. In the bandwidth-bound decode regime it is the DISCRIMINATING signal for §B3: at a
    good placement the card computes back-to-back (util high, idle low); at heavy offload it
    stalls on PCIe waiting for the next expert (util low, idle high). One extra CSV column
    on a query the guard already runs every second -- free instrumentation."""
    out = _run([_NVIDIA_SMI, "--query-gpu=memory.total,memory.used,utilization.gpu",
                "--format=csv,noheader,nounits"])
    line = out.strip().splitlines()[0]
    total, used, util = (int(x.strip()) for x in line.split(","))
    return total, used, util


def read_ram_available_mb() -> int:
    """Available RAM in MB, via CIM.

    NOT `Get-Counter '\\Memory\\Available MBytes'`: counter PATHS are localised, so on
    a pt-BR host that path does not exist, the call returns null, and null coerced to
    int becomes 0 -- which reads as "this machine has no RAM". CIM property names are
    always English, so this one travels.

    'Available' rather than 'Free' is deliberate: available counts the standby cache,
    which Windows reclaims on demand. Free-only reads as starved on a perfectly usable
    box, and a guard built on it rejects everything.
    """
    out = _run([_POWERSHELL, "-NoProfile", "-Command",
                "(Get-CimInstance Win32_PerfRawData_PerfOS_Memory).AvailableMBytes"])
    text = out.strip()
    if not text:
        raise HostReadError("AvailableMBytes returned empty - refusing to guess")
    return int(text.splitlines()[0])


def sample() -> HostSample:
    total, used, util = read_gpu()
    return HostSample(vram_total_mb=total, vram_used_mb=used,
                      ram_available_mb=read_ram_available_mb(), gpu_util_pct=util)


if __name__ == "__main__":  # self-check: python -m model_lifecycle.collectors.host
    s = sample()
    print(f"vram {s.vram_free_mb}/{s.vram_total_mb} MB free | ram {s.ram_available_mb} MB "
          f"available | gpu {s.gpu_util_pct}% busy")
    assert s.vram_total_mb > 0 and s.ram_available_mb > 0, "host readings must be positive"
    assert 0 <= s.gpu_util_pct <= 100, "gpu utilisation must be a percent"
    print("host collector self-check OK")
