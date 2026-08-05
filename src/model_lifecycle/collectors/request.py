"""Per-request timing, with TTFT measured properly.

TTFT is its own metric, not a by-product: two configurations with identical tokens/s
can start replying in 2 s or in 15 s, and the second one feels broken. Getting it
right requires a STREAMING request -- the first token's arrival time is unavailable
from a buffered response, which is why this file exists instead of a `requests.post`.
"""
from __future__ import annotations

import json
import time
import urllib.request
from dataclasses import dataclass


@dataclass
class RequestResult:
    # Two different questions, deliberately separated after the first live run:
    #   ok       -- did the request complete and generate tokens? (INFRASTRUCTURE)
    #   answered -- did it produce usable content?               (QUALITY)
    # A thinking model that burns its whole budget reasoning generated real tokens at
    # a real speed on a really loaded server: the throughput measurement is VALID and
    # must not be thrown away because the task failed. Discarding it was conflating
    # the instrument with the task.
    ok: bool
    answered: bool = True
    # The assembled content. Until quality became a response variable this collector only
    # kept the LENGTH of the text (`seen_text`) and discarded the text itself, because
    # nothing measured needed it. Additive: every existing caller ignores this field.
    text: str = ""
    # The assembled REASONING trace, separate from the answer (A2 / long-to-short). On a
    # thinking model run with `--reasoning-format deepseek`, llama-server streams the think
    # block as `delta.reasoning_content`, DISTINCT from `delta.content` -- so this is the
    # raw <think> text with the tags already stripped by the server. Additive and empty on
    # every non-thinking run and on any caller that does not set --reasoning-format. The
    # A2 concision metric is the TOKEN count of this text (via `count_tokens` -> /tokenize),
    # not its length: chars-per-token drifts with content, so a char ratio is not a token
    # ratio. Kept as text, not a count, for the same reason every field here is raw -- a
    # stored count cannot be re-tokenized if the tokenizer or the definition moves.
    reasoning_text: str = ""
    ttft_s: float | None = None          # T(first content token) - T(send)
    total_s: float | None = None
    completion_tokens: int = 0
    prompt_tokens: int = 0
    error: str | None = None

    # Server-reported timings (llama.cpp `timings` block, present on every streamed
    # response). These are the AUTHORITATIVE split between prefill and generation: the
    # server knows exactly where prompt processing ended, and from outside that boundary
    # is invisible on a thinking model that never emits a content token.
    #
    # Stored RAW -- counts and milliseconds, never rates. That is the direct fix for
    # this project's worst design defect: `gen_tps` was a DERIVED value written into the
    # raw payload at collection time, so when its formula turned out to be wrong the
    # affected runs could not be repaired by recomputation and had to be re-measured.
    # Raw counts can always be re-divided; a stored quotient cannot be un-divided.
    prompt_n: int | None = None
    prompt_ms: float | None = None
    predicted_n: int | None = None
    predicted_ms: float | None = None
    # Prompt-cache hits. Load-bearing across repetitions: the server reuses the KV of an
    # identical prompt, so repetition 2 and 3 can have near-zero prefill. Averaging
    # prefill over repetitions without this would report a prefill rate that mostly
    # measures the cache.
    #
    # SETTLED 2026-07-25, by an out-of-range value rather than by reading the source:
    # `cache_n` is counted BESIDE `prompt_n`, not inside it. A sweep recorded
    # `cache_n/prompt_n = 222%`, which is impossible if prompt_n were the total — the
    # gpt-oss chat template is ~84 tokens, all cached, leaving a much smaller processed
    # remainder. So `prompt_n/prompt_ms` is an honest rate for the work actually done, and
    # the cached FRACTION of the request is `cache_n / (cache_n + prompt_n)`.
    cache_n: int | None = None

    # Speculative-decoding counts (A4 / IDEAS_BACKLOG §63.4). Present in the server
    # `timings` block ONLY when speculation actually ran -- llama.cpp's
    # `result_timings::to_json()` (server-task.cpp) emits `draft_n`/`draft_n_accepted`
    # guarded by `if (draft_n > 0)`, so a no-spec arm reports NEITHER key (they stay
    # None here, not 0). That distinction is load-bearing: a 0 would mean "spec ran and
    # accepted nothing", None means "spec was off", and `accept_rate` must not divide the
    # first by a real denominator nor the second by zero.
    #
    # Stored RAW for the same reason every count above is: acceptance and accept-length
    # are quotients whose right definition (see the properties) was pinned down only by
    # source-reading, and a stored quotient cannot be re-divided if the formula moves.
    # These are the ONLY machine-readable spec fields the server exposes -- mean accept
    # length, per-position acceptance and drafter time (`t_draft_us`) live in the stderr
    # log (upstream #24536), never the JSON, so the harness derives what it can from these
    # two plus predicted_n and leaves the rest to the log.
    draft_n: int | None = None
    draft_n_accepted: int | None = None

    @property
    def generation_s(self) -> float | None:
        """Fallback window when the server reports no timings. See `generation_tps`.

        The whole request, for EVERY run. Uniform by necessity, not by laziness.

        The obvious window is `total - ttft`, and it is WRONG for a thinking model.
        `ttft` is the time to the first *content* token, which arrives only after all
        the reasoning is done -- so subtracting it removes the reasoning TIME from the
        denominator while `completion_tokens` still counts the reasoning TOKENS in the
        numerator. Measured 2026-07-25: that produced 611.8 and 599.2 t/s for a 35B MoE
        on a 3090, roughly 8x reality, and only for the two runs that happened to
        answer. It silently reordered a ranking -- an inflated config "won" purely by
        having replied.

        Using the full request for everyone understates the rate (prefill is inside
        it), but understates it the SAME WAY for every candidate, and comparability is
        what a ranking needs. TTFT remains its own metric, which is where the
        prefill-vs-generation split belongs.
        """
        if self.total_s is None:
            return None
        return max(self.total_s, 1e-9)

    @property
    def generation_tps_is_lower_bound(self) -> bool:
        """True only when falling back to wall-clock, where prefill sits inside the
        window. With server timings the rate is exact, and saying otherwise would make
        the flag meaningless by always being set."""
        return not self.predicted_ms and self.total_s is not None

    @property
    def generation_tps(self) -> float | None:
        """Tokens per second of ACTUAL generation, reasoning included.

        Reasoning tokens are generated by the same path at the same cost, so an
        infrastructure benchmark counts them. A quality benchmark must not -- which is
        why `answered` exists alongside this number rather than inside it.

        Prefers the server's own `predicted_ms`, which excludes prefill exactly. The
        wall-clock fallback below is comparable across candidates but blunt: it dilutes
        any generation-side difference with prompt-processing time, which is precisely
        how an A/B ends up unable to see the effect it was built to measure.
        """
        if self.predicted_ms and self.predicted_n:
            return self.predicted_n / (self.predicted_ms / 1000.0)
        g = self.generation_s
        if not g or not self.completion_tokens:
            return None
        return self.completion_tokens / g

    @property
    def prompt_tps(self) -> float | None:
        """Prefill rate. Prompt processing and generation are different bottlenecks and
        are never averaged into one number: a model can be fast at one and slow at the
        other.

        The TTFT-derived fallback CANNOT be computed on a thinking model -- `ttft_s` is
        the time to the first *content* token and stays None when the budget is spent on
        reasoning, so prefill silently went unmeasured on every run of this fleet.
        `prompt_ms` is reported regardless of whether an answer ever appeared.
        """
        if self.prompt_ms and self.prompt_n:
            return self.prompt_n / (self.prompt_ms / 1000.0)
        if self.ttft_s is None or not self.prompt_tokens or self.ttft_s <= 0:
            return None
        return self.prompt_tokens / self.ttft_s

    @property
    def tpot_ms(self) -> float | None:
        """Time per output token (ms), decode phase only -- the reciprocal of
        `generation_tps` in ms. Named separately because TPOT is the standard serving
        metric (vLLM/MLPerf) and the reader should not have to invert a t/s to get the
        per-token latency a config's interactivity is judged on. Uses the server's
        `predicted_ms/predicted_n`, which excludes prefill exactly; the wall-clock
        fallback is deliberately NOT offered here -- a TPOT diluted with prompt time is
        worse than no TPOT, because it looks like a decode number and is not one.

        NB this is the MEAN inter-token time. The p50/p95/p99 ITL distribution needs the
        per-token timing stream (`timings_per_token`), which this collector does not
        request; at batch-1 on this box decode is near-deterministic (CV ~0.006) so the
        mean carries the interactivity story and the tail adds little (A4 scope note)."""
        if self.predicted_ms and self.predicted_n:
            return self.predicted_ms / self.predicted_n
        return None

    @property
    def accept_rate(self) -> float | None:
        """Draft acceptance rate alpha = accepted / drafted (A4). This is the intrinsic,
        hardware-independent quality of the drafter and the EXACT quantity the server
        logs as `draft acceptance` (`draft_ratio = n_draft_accepted / n_draft_total`,
        server-context.cpp) -- verified against source AND against the server's own logged
        value to 5 decimals (a4_spec_metrics_probe.py: JSON 187/268 == logged 0.69776).
        None when spec was off (draft_n absent) so a no-spec arm reads as 'no acceptance
        to report' rather than a spurious 0/0.

        NB there is deliberately NO `mean_accept_len` (tau) property here. tau = tokens
        advanced per target forward pass = 1 + n_draft_accepted/n_draft_verif_steps
        (the §63.4 'accepted draft tokens per verification', vLLM/Leviathan). It is NOT
        derivable from the JSON alone: n_draft_verif_steps is log-only (upstream #24536),
        and recovering it from predicted_n needs a boundary constant that VARIES per run
        -- the probe measured predicted_n - accepted - 1 = 68 while the true step count
        was 67, so a predicted_n derivation gives tau 3.75 vs the server's 3.79 (~1% wrong,
        and it was wrong in the FIRST implementation until the probe caught it). The robust
        exact relation is tau = 1 + gamma*alpha (gamma = --spec-draft-n-max), because the
        drafter always proposes gamma per step so n_draft_verif_steps = draft_n/gamma;
        that identity reproduces the log to the decimal (1 + 4*0.69776 = 3.79). tau is
        therefore computed where gamma is known (ab_isolate's spec block) from this alpha,
        never from predicted_n, and validated against the log by the gate."""
        if self.draft_n and self.draft_n_accepted is not None:
            return self.draft_n_accepted / self.draft_n
        return None


def _timing_fields(t: dict) -> dict:
    """Server timings -> RequestResult fields. One place, because the starved path and
    the answered path both need them and the starved path is exactly where they matter
    most: it is the run with no TTFT, where wall-clock cannot separate prefill from
    generation at all."""
    return {"prompt_n": t.get("prompt_n"), "prompt_ms": t.get("prompt_ms"),
            "predicted_n": t.get("predicted_n"), "predicted_ms": t.get("predicted_ms"),
            "cache_n": t.get("cache_n"),
            # Absent (None), not 0, when speculation was off -- the server omits the keys.
            "draft_n": t.get("draft_n"),
            "draft_n_accepted": t.get("draft_n_accepted")}


def chat_stream(base_url: str, prompt: str, *, max_tokens: int = 256,
                temperature: float = 0.0, timeout_s: float = 300.0,
                model: str = "local",
                cache_prompt: bool | None = None) -> RequestResult:
    """One streaming chat completion, timed.

    `max_tokens` matters more than it looks on a thinking model: measured on this
    fleet, a tight budget makes the model spend everything on reasoning_content and
    return an EMPTY content with finish_reason=length -- which looks like a broken
    model and is a starved one. Keep it generous, or record the floor.
    """
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    # Only sent when asked for, so callers that have been measuring against the server
    # default keep measuring against the server default. A prefill benchmark wants
    # cache_prompt=False -- the chat template is a common prefix on every request, so the
    # server reuses it and the measurement stops describing the length that was asked for.
    # Everything else here leaves it alone rather than silently changing an instrument
    # that several completed runs were taken with.
    if cache_prompt is not None:
        payload["cache_prompt"] = cache_prompt
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(f"{base_url}/v1/chat/completions", data=body,
                                 headers={"Content-Type": "application/json"})

    t0 = time.monotonic()
    ttft: float | None = None
    completion = prompt_toks = 0
    seen_text = 0
    chunks: list[str] = []
    # Measured, not inferred. First run against the live model returned empty content
    # three times; the code called it "starved?" as a guess. These two fields turn the
    # guess into evidence: reasoning_chars shows the budget WAS spent, and
    # finish_reason=length shows it ran out before answering.
    reasoning_chars = 0
    reasoning_pieces: list[str] = []
    finish_reason: str | None = None
    timings: dict = {}
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            for raw in resp:
                line = raw.decode("utf-8", "replace").strip()
                if not line.startswith("data:"):
                    continue
                payload = line[5:].strip()
                if payload == "[DONE]":
                    break
                try:
                    ev = json.loads(payload)
                except ValueError:
                    continue
                # Arrives on the final chunk. Verified live against llama-server 9859:
                # prompt_n/prompt_ms/predicted_n/predicted_ms/cache_n.
                if t := ev.get("timings"):
                    timings = t
                if usage := ev.get("usage"):
                    completion = usage.get("completion_tokens", completion)
                    prompt_toks = usage.get("prompt_tokens", prompt_toks)
                for ch in ev.get("choices") or []:
                    if ch.get("finish_reason"):
                        finish_reason = ch["finish_reason"]
                    delta = ch.get("delta") or {}
                    rc = delta.get("reasoning_content") or ""
                    if rc:
                        reasoning_chars += len(rc)
                        reasoning_pieces.append(rc)
                    piece = delta.get("content") or ""
                    if piece:
                        seen_text += len(piece)
                        chunks.append(piece)
                        # First CONTENT token, deliberately: reasoning_content arrives
                        # earlier on thinking models, and timing to that would flatter
                        # them against models that answer directly.
                        if ttft is None:
                            ttft = time.monotonic() - t0
        total = time.monotonic() - t0
    except Exception as exc:  # noqa: BLE001 - a failed request is a RESULT, not a crash
        return RequestResult(ok=False, total_s=time.monotonic() - t0,
                             error=f"{type(exc).__name__}: {exc}")

    if seen_text == 0:
        # Empty content is a real, recordable outcome -- never silently a success.
        # Measured 2026-07-25 against Qwen3.6-35B-A3B: max_tokens=300 on a ~120-word
        # request produced empty content three times out of three, with the budget
        # consumed by reasoning. The distinction below is the one that matters
        # operationally: STARVED is a harness mistake (raise the budget), DEAD is a
        # model or config problem. Reporting them as one thing sends you hunting the
        # wrong bug.
        starved = finish_reason == "length" or reasoning_chars > 0
        why = (f"starved: budget spent on reasoning "
               f"(finish_reason={finish_reason}, reasoning_chars={reasoning_chars}, "
               f"completion_tokens={completion}) - raise max_tokens"
               if starved else
               f"empty content with no reasoning (finish_reason={finish_reason})")
        # ok=True when tokens were actually produced: the server worked, the timing is
        # real. answered=False records that the ANSWER never came.
        produced = completion > 0 or reasoning_chars > 0
        return RequestResult(ok=produced, answered=False, text="".join(chunks),
                             reasoning_text="".join(reasoning_pieces),
                             ttft_s=ttft, total_s=total,
                             completion_tokens=completion, prompt_tokens=prompt_toks,
                             error=why, **_timing_fields(timings))
    return RequestResult(ok=True, text="".join(chunks),
                         reasoning_text="".join(reasoning_pieces),
                         ttft_s=ttft, total_s=total,
                         completion_tokens=completion, prompt_tokens=prompt_toks,
                         **_timing_fields(timings))

def count_tokens(base_url: str, text: str, *, timeout_s: float = 30.0) -> int | None:
    """Exact token count of `text` under the SERVER'S tokenizer, via `/tokenize`.

    Why the server and not a local tokenizer: the A2 concision metric compares two
    DIFFERENT model files (base vs ThinkingCap), and a reasoning-token reduction is only
    honest if both sides are counted by the tokenizer that actually generated the tokens.
    The two GGUFs share a byte-identical tokenizer here (verified: same chat_template
    sha256, same BOS/EOS), so the count is comparable across arms -- but routing it through
    each arm's own live server keeps that a checked fact, not an assumption, and needs no
    tokenizer files on the Windows side.

    Empty text is 0, not None -- a problem where the model emitted no reasoning is a real
    zero, not a missing measurement. None is reserved for an actual /tokenize failure, so a
    transport error can never be silently averaged in as 'zero reasoning'.

    NB this counts the tokens of the reasoning TEXT (think tags already stripped by
    `--reasoning-format deepseek`); it is therefore a hair below the server's internally
    generated reasoning-token count, which also spent the `<think>`/`</think>` delimiter
    tokens. That offset is a small per-response constant, identical for both arms, so it
    cancels in the paired reduction -- the quantity the experiment reports.
    """
    if not text:
        return 0
    body = json.dumps({"content": text}).encode("utf-8")
    req = urllib.request.Request(f"{base_url}/tokenize", data=body,
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            toks = json.loads(resp.read().decode("utf-8", "replace")).get("tokens")
        # Default /tokenize returns a flat list of ids; with_pieces would return dicts.
        return len(toks) if isinstance(toks, list) else None
    except Exception:  # noqa: BLE001 -- a failed count is None (missing), never a fake 0
        return None


if __name__ == "__main__":
    # Exact rates when the server reports timings, and the lower-bound flag must clear.
    r = RequestResult(ok=True, total_s=20.0, completion_tokens=1500, prompt_tokens=64,
                      prompt_n=64, prompt_ms=200.0, predicted_n=1500,
                      predicted_ms=18000.0, cache_n=0)
    assert abs(r.generation_tps - 83.333) < 0.01, r.generation_tps
    assert abs(r.prompt_tps - 320.0) < 0.01, r.prompt_tps
    assert not r.generation_tps_is_lower_bound, "server timings are exact"

    # The case that motivated all of this: a thinking model that never answered. No
    # TTFT, so the old prompt_tps was None -- prefill went unmeasured on every run.
    starved = RequestResult(ok=True, answered=False, ttft_s=None, total_s=20.0,
                            completion_tokens=1500, prompt_tokens=64,
                            prompt_n=64, prompt_ms=200.0, predicted_n=1500,
                            predicted_ms=18000.0)
    assert starved.prompt_tps is not None, "prefill must be measurable without a TTFT"
    assert abs(starved.generation_tps - 83.333) < 0.01

    # Fallback: no server timings -> wall-clock, and the flag must say so.
    old = RequestResult(ok=True, ttft_s=None, total_s=20.0, completion_tokens=1500)
    assert abs(old.generation_tps - 75.0) < 0.01, old.generation_tps
    assert old.generation_tps_is_lower_bound, "wall-clock includes prefill"
    assert old.prompt_tps is None

    assert _timing_fields({})["prompt_n"] is None, "absent timings must not crash"

    # A2 concision: reasoning_text is additive and defaults empty, so every non-thinking
    # caller and every run without --reasoning-format is unaffected.
    assert RequestResult(ok=True).reasoning_text == "", "reasoning must default empty, not None"
    assert RequestResult(ok=True, reasoning_text="x").reasoning_text == "x"

    # A4 spec-decode metrics. No draft fields -> no acceptance to report, and TPOT is the
    # decode reciprocal regardless of spec.
    assert r.accept_rate is None, "no spec -> no acceptance"
    assert abs(r.tpot_ms - 12.0) < 1e-6, r.tpot_ms   # 18000ms / 1500 tok

    # Spec ran: alpha = accepted/drafted, EXACT (matches the server's logged draft_ratio).
    # The probe's real numbers, so the self-check pins the exact value the box produced:
    # 187 accepted / 268 drafted = 0.69776..., the value server-context.cpp logged.
    spec = RequestResult(ok=True, total_s=20.0, completion_tokens=256,
                         predicted_n=256, predicted_ms=2113.72,
                         draft_n=268, draft_n_accepted=187)
    assert abs(spec.accept_rate - 187 / 268) < 1e-12, spec.accept_rate
    assert abs(spec.accept_rate - 0.69776) < 5e-5, spec.accept_rate
    # draft_n absent but accepted present must NOT fabricate a rate (spec off is spec off).
    assert RequestResult(ok=True, draft_n=None, draft_n_accepted=None).accept_rate is None
    # draft_n == 0 is also 'no acceptance' (never divide by zero).
    assert RequestResult(ok=True, draft_n=0, draft_n_accepted=0).accept_rate is None
    # The two spec fields must round-trip through the timings extractor, and be absent
    # (None) when the server omits them.
    got = _timing_fields({"draft_n": 268, "draft_n_accepted": 187})
    assert got["draft_n"] == 268 and got["draft_n_accepted"] == 187
    assert _timing_fields({"predicted_n": 5})["draft_n"] is None, "no spec -> None"

    print("request collector self-check OK "
          f"(exact gen={r.generation_tps:.1f} t/s, prefill={r.prompt_tps:.1f} t/s, "
          f"alpha={spec.accept_rate:.5f}, tpot={r.tpot_ms:.1f}ms)")
