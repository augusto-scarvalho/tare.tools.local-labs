"""Infrastructure workload: load → warm-up → N timed requests → teardown.

This is the runner the whole platform's numbers come from. It produces, per
configuration: load time, TTFT distribution, prompt/generation throughput, end-to-end
latency, pass rate, and the guard's peak pressure -- with a REJECTED verdict recorded
as a result rather than a failure, because it maps the edge of the envelope.
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field

from ..analysis.statistics import describe
from ..collectors.host import sample
from ..collectors.request import chat_stream
from ..control_plane.guard import Envelope, Watch, precheck, start_watch, wait_for_recovery
from ..servers.llama_cpp import LlamaCppAdapter, ServerProfile


@dataclass
class RunResult:
    config_id: str
    verdict: str                      # OK | REJECTED | ERROR | SKIPPED
    reason: str | None = None
    load_seconds: float | None = None
    pass_rate: float = 0.0      # requests that completed and generated tokens
    answer_rate: float = 0.0    # requests that produced usable CONTENT
    requests: int = 0
    failures: list[str] = field(default_factory=list)
    ttft: dict | None = None
    total: dict | None = None
    gen_tps: dict | None = None
    prompt_tps: dict | None = None
    min_free_vram_mb: int | None = None
    min_available_ram_mb: int | None = None
    load_min_ram_mb: int | None = None   # pressure during LOAD: recorded, never enforced
    host_recovered: bool = True          # False => the NEXT run started in a shrunken envelope
    gen_tps_lower_bound_count: int = 0   # how many rates are UNDERSTATED (no TTFT boundary)
    cached_prefill_count: int = 0        # prefills served from KV cache: NOT real prefill
    stderr_tail: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        d = dict(self.__dict__)
        return d


class _GuardThread(threading.Thread):
    """Samples the host while a run is in flight. Runs in its own thread so a slow
    request never starves the guard -- the machine must be watched during exactly the
    window when it is under pressure."""

    def __init__(self, watch: Watch, interval_s: float):
        super().__init__(daemon=True)
        self.watch = watch
        self.interval = interval_s
        self._stop = threading.Event()

    def run(self) -> None:
        while not self._stop.is_set():
            try:
                if not self.watch.observe(sample()):
                    return          # breach latched; the caller reads watch.breach
            except Exception:       # a read failure must not kill the run silently
                pass
            self._stop.wait(self.interval)

    def halt(self) -> None:
        self._stop.set()


def run_config(adapter: LlamaCppAdapter, profile: ServerProfile, *, config_id: str,
               prompt: str, repetitions: int = 5, max_tokens: int = 256,
               envelope: Envelope | None = None,
               warmup: bool = True) -> RunResult:
    env = envelope or Envelope()
    result: RunResult | None = None

    if reason := precheck(env):
        return RunResult(config_id=config_id, verdict="SKIPPED", reason=reason)

    handle = adapter.start(profile)
    watch = start_watch(env)
    guard = _GuardThread(watch, env.sample_interval_s)
    guard.start()

    try:
        if not adapter.wait_until_healthy(handle):
            return RunResult(config_id=config_id,
                             verdict="REJECTED" if watch.breach else "ERROR",
                             reason=watch.breach or "server never became healthy",
                             min_free_vram_mb=watch.min_free_vram_mb,
                             min_available_ram_mb=watch.min_available_ram_mb,
                             stderr_tail=handle.stderr_tail)

        # Loading is over: from here the RAM floor applies too. Before this point only
        # VRAM could disqualify, because a 21 GB read starves RAM by construction.
        watch.mark_healthy()

        if warmup:
            # Discarded on purpose: the first request pays for lazy allocations and
            # would otherwise contaminate the TTFT distribution it is meant to describe.
            chat_stream(handle.base_url, prompt, max_tokens=32)

        ttfts: list[float] = []
        totals: list[float] = []
        gtps: list[float] = []
        ptps: list[float] = []
        failures: list[str] = []

        answered = 0
        completed = 0
        lower_bound_tps = 0   # rates measured without a TTFT boundary (understated)
        cached_prefills = 0   # repetitions whose prompt was served from the KV cache
        for rep in range(repetitions):
            if watch.breach:
                break
            # Each repetition gets a UNIQUE prefix. llama-server caches the KV of an
            # identical prompt, so with a fixed prompt only repetition 1 does real
            # prefill and the rest report a near-zero prompt_ms -- turning the prefill
            # rate into a measurement of the cache. The prefix is short and constant in
            # shape, so generation is unaffected.
            # The unique prefix is NOT sufficient, and the tripwire below caught it: on
            # gpt-oss-20b every one of 12 configurations reported a cached prefill and
            # `prompt_tps` came back null, because the harmony chat template puts a ~84
            # token system preamble BEFORE the user content -- so the common prefix a
            # user-side tag can never reach. The Qwen templates are short enough that
            # reuse never triggered, which is why 0 of 126 configurations on those models
            # were affected and this went unseen until a third geometry was measured.
            #
            # cache_prompt=False is the root fix. It is a no-op on every run already
            # taken (all had cache_n == 0), so the day's decompositions stay comparable.
            r = chat_stream(handle.base_url, f"[req {rep}] {prompt}",
                            max_tokens=max_tokens, cache_prompt=False)
            if not r.ok:
                failures.append(r.error or "unknown")
                continue
            completed += 1
            if r.answered:
                answered += 1
            else:
                # Timing is kept: real tokens at real speed. Only the ANSWER is missing,
                # and that is recorded separately instead of discarding the measurement.
                failures.append(r.error or "no answer")
            if r.ttft_s is not None:
                ttfts.append(r.ttft_s)
            if r.total_s is not None:
                totals.append(r.total_s)
            if (g := r.generation_tps) is not None:
                gtps.append(g)
                if r.generation_tps_is_lower_bound:
                    lower_bound_tps += 1
            # Trust prefill only when the server actually did it. Recorded rather than
            # assumed: the unique prefix above is supposed to defeat the cache, and a
            # non-zero count here is the evidence that it stopped working.
            if r.cache_n:
                cached_prefills += 1
            elif (p := r.prompt_tps) is not None:
                ptps.append(p)

        # Count COMPLETED requests, not TTFTs. A starved thinking model never emits a
        # first CONTENT token, so its ttft is legitimately None -- keying the verdict
        # on the TTFT list would file a perfectly good throughput measurement as ERROR,
        # which is the same conflation this fix exists to remove.
        result = RunResult(
            config_id=config_id,
            verdict="REJECTED" if watch.breach else ("OK" if completed else "ERROR"),
            reason=watch.breach or (None if completed else "no request completed"),
            load_seconds=handle.load_seconds,
            requests=repetitions,
            pass_rate=completed / repetitions if repetitions else 0.0,
            answer_rate=answered / repetitions if repetitions else 0.0,
            failures=failures,
            ttft=describe(ttfts).as_dict() if ttfts else None,
            total=describe(totals).as_dict() if totals else None,
            gen_tps=describe(gtps).as_dict() if gtps else None,
            prompt_tps=describe(ptps).as_dict() if ptps else None,
            min_free_vram_mb=watch.min_free_vram_mb,
            min_available_ram_mb=watch.min_available_ram_mb,
            load_min_ram_mb=watch.load_min_ram_mb,
            gen_tps_lower_bound_count=lower_bound_tps,
            cached_prefill_count=cached_prefills,
        )
        return result
    finally:
        guard.halt()
        adapter.stop(handle)
        # Wait for the host to actually recover before the next configuration, rather
        # than a fixed sleep. WSL2 returns guest memory SLOWLY (measured: ~24 GB came
        # back on its own, minutes later), so this is a race between sweep cadence and
        # reclaim cadence -- NOT a leak, as an earlier version of this note claimed.
        # WSL2 returns guest memory SLOWLY (measured: ~24 GB came back on its own,
        # minutes later), so this is a race between sweep cadence and reclaim cadence,
        # NOT a leak as an earlier note claimed. The outcome is RECORDED on the run:
        # silently proceeding is what made a config inherit its predecessor's
        # footprint and be rejected for it -- a verdict about queue position rather
        # than about the configuration.
        recovered = wait_for_recovery(env)
        if not recovered and result is not None:
            result.host_recovered = False
            result.failures.append(
                "host did not recover before the next config - later verdicts suspect")
