"""Envelope guard — the reserve is a hard constraint, not a tunable.

A configuration that wins on tokens/s and takes the desktop down produced a NEGATIVE
result: it cost real work. So the guard runs alongside every benchmark and a breach
disqualifies, with the peak recorded.

The subtlety this file exists for: **load-transient pressure is not steady-state
cost.** Streaming a 21 GB model, plus the previous model's page cache not yet
released, drives available RAM below the reserve before the GPU is even touched. A
naive guard rejects those configurations for the act of loading rather than the cost
of running -- observed live on 2026-07-25, three of four configs rejected with the GPU
never loaded.

Two mechanisms answer that, and neither weakens the VRAM check:
  * a breach must be SUSTAINED over `breach_samples` consecutive reads;
  * `wait_for_recovery()` blocks until the host is back above a watermark before the
    next configuration starts, instead of a fixed sleep that was never long enough.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

from ..collectors.host import HostSample, sample


@dataclass(frozen=True)
class Envelope:
    reserve_vram_mb: int = 4096
    reserve_ram_mb: int = 16384
    # A single dip while a 21 GB file streams is not the machine being starved.
    breach_samples: int = 3
    sample_interval_s: float = 1.0


@dataclass
class Watch:
    """Live state of one guarded run. `min_*` are PEAKS of pressure, not averages:
    a mean that looks fine while a spike hit 23.5 GB is a config that kills the box
    on a bad day.

    PHASE-AWARE, and this is the correction that took three live runs to reach.
    Loading a 21 GB GGUF drives available RAM through the floor for as long as the
    read lasts -- measured: 10013 MB against a 16384 MB reserve, with the GPU never
    touched and not one request sent. Enforcing the RAM floor during that window
    disqualifies configurations for the ACT OF LOADING rather than the cost of
    running, which is not the question this platform asks.

    Raising `breach_samples` until it stops hurting would be calibrating the alarm to
    be quiet -- that is how an alarm becomes decoration. Instead:

      * during load  -> only VRAM can disqualify. RAM is RECORDED, not enforced.
      * once healthy -> both enforce, with the sustained-breach rule.

    VRAM is enforced throughout with no exception: it has no cache transient. If it
    is gone, it is gone.

    Load-phase RAM pressure is kept as `load_min_ram_mb` because it is real data --
    it is precisely what the plan's Operational suite calls cold start and swap cost.
    Not enforced, never discarded.
    """
    envelope: Envelope
    min_free_vram_mb: int
    min_available_ram_mb: int
    breach: str | None = None
    enforce_ram: bool = False          # flipped by mark_healthy(); load phase is exempt
    load_min_ram_mb: int | None = None
    _vram_strikes: int = field(default=0, repr=False)
    _ram_strikes: int = field(default=0, repr=False)

    def mark_healthy(self) -> None:
        """The server answered /health: loading is over, steady state begins."""
        self.load_min_ram_mb = self.min_available_ram_mb
        self.enforce_ram = True
        self._ram_strikes = 0          # load-phase strikes must not carry over

    def observe(self, s: HostSample) -> bool:
        """Feed one sample. Returns True while the run may continue."""
        self.min_free_vram_mb = min(self.min_free_vram_mb, s.vram_free_mb)
        self.min_available_ram_mb = min(self.min_available_ram_mb, s.ram_available_mb)

        if s.vram_free_mb < self.envelope.reserve_vram_mb:
            self._vram_strikes += 1
        else:
            self._vram_strikes = 0
        if self.enforce_ram and s.ram_available_mb < self.envelope.reserve_ram_mb:
            self._ram_strikes += 1
        else:
            self._ram_strikes = 0

        if self._vram_strikes >= self.envelope.breach_samples:
            self.breach = (f"vram {s.vram_free_mb}MB < {self.envelope.reserve_vram_mb}MB "
                           f"for {self._vram_strikes} samples")
        elif self.enforce_ram and self._ram_strikes >= self.envelope.breach_samples:
            self.breach = (f"ram {s.ram_available_mb}MB < {self.envelope.reserve_ram_mb}MB "
                           f"for {self._ram_strikes} samples")
        return self.breach is None


def start_watch(env: Envelope) -> Watch:
    s = sample()
    return Watch(envelope=env, min_free_vram_mb=s.vram_free_mb,
                 min_available_ram_mb=s.ram_available_mb)


def precheck(env: Envelope) -> str | None:
    """Refuse BEFORE loading if the host is already inside the reserve. Cheaper than
    discovering it at token 400, and it stops a busy desktop being ambushed.
    Returns a reason string, or None when it is safe to start."""
    s = sample()
    if s.vram_free_mb < env.reserve_vram_mb:
        return f"host already inside VRAM reserve ({s.vram_free_mb}MB free)"
    if s.ram_available_mb < env.reserve_ram_mb:
        return f"host already inside RAM reserve ({s.ram_available_mb}MB available)"
    return None


class HostDidNotRecover(RuntimeError):
    """The host never returned to a safe level between configurations.

    Raised rather than returned-and-ignored, because the silent version is what
    produced the worst class of bad data here: a sweep that proceeds into a shrunken
    envelope rejects the NEXT configuration for its predecessor's footprint, and the
    verdict then describes the queue position rather than the configuration. Measured
    live: `ctx8192 ncmoe8 Q4` passed in one run and was rejected at 10611MB available
    in the next, with identical parameters.
    """


def wait_for_recovery(env: Envelope, *, headroom_mb: int = 4096,
                      timeout_s: float = 180.0) -> bool:
    """Block until the host has recovered past the reserve plus a margin.

    Replaces the fixed `sleep 5` that was never enough after a 21 GB read: the page
    cache from the previous model had not been released when the next config started,
    so a configuration was punished for its predecessor's footprint. Returns False on
    timeout -- the caller records that rather than pretending recovery happened.
    """
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        s = sample()
        if (s.vram_free_mb >= env.reserve_vram_mb + headroom_mb
                and s.ram_available_mb >= env.reserve_ram_mb + headroom_mb):
            return True
        time.sleep(env.sample_interval_s)
    return False


if __name__ == "__main__":
    env = Envelope()
    print("precheck:", precheck(env) or "clear")
    w = start_watch(env)
    ok = w.observe(sample())
    assert ok and w.breach is None, "an idle host must not read as a breach"

    # RAM starvation during LOAD must not disqualify: measured 10013MB while a 21 GB
    # model streamed in, GPU untouched, no request sent.
    loading = HostSample(vram_total_mb=24576, vram_used_mb=2000, ram_available_mb=100)
    for _ in range(env.breach_samples + 3):
        assert w.observe(loading), "RAM floor must not fire before the server is healthy"
    assert w.breach is None and w.min_available_ram_mb == 100, "load pressure still RECORDED"

    # ...and once healthy, the same pressure does disqualify, but only when sustained.
    w.mark_healthy()
    assert w.load_min_ram_mb == 100, "load-phase pressure kept as its own datum"
    for i in range(env.breach_samples - 1):
        assert w.observe(loading), f"breach declared too early at strike {i + 1}"
    assert not w.observe(loading), "sustained RAM breach must fire after healthy"
    print("guard: ram fires only after healthy ->", w.breach)

    # VRAM has no cache transient: it is enforced from the first moment, load or not.
    w2 = start_watch(env)
    vram_gone = HostSample(vram_total_mb=24576, vram_used_mb=24576 - 100, ram_available_mb=40000)
    for i in range(env.breach_samples - 1):
        assert w2.observe(vram_gone), f"vram breach too early at strike {i + 1}"
    assert not w2.observe(vram_gone), "VRAM must disqualify even during load"
    assert not w2.enforce_ram, "and it must do so without waiting for health"
    print("guard: vram fires during load    ->", w2.breach)
    print("guard self-check OK")
