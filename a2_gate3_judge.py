#!/usr/bin/env python3
"""A2 Gate 3 -- writing-quality judge QUORUM (the last axis to stamp deploy candidate l1.0).

The question this gate answers: DID CONCISION HURT THE PROSE?  Stage-1 proved fable-tc-l1.0 is
concise (-55% reasoning on math, -23% on creative) AND uncensored (indistinguishable from plain
Fable on the refusal axes).  What is NOT yet stamped is whether cutting the overthinking degraded
the *writing itself*.  This harness settles it with a BLIND, LENGTH-BLIND, order-balanced pairwise
quorum of independent judges.

Design (all deliberate, see A2_STAGE1_CONCISE_FABLE.md Gate 3):
  * PAIRWISE BLIND -- each judge sees two anonymous responses ("Response 1"/"Response 2") to the
    same creative prompt and picks the better *writing*.  Never told which model, never told the
    experiment's hypothesis.
  * BOTH PRESENTATION ORDERS -- every pair is judged twice (l1.0-first and plain-first).  A judge
    that only "wins" in one order is exhibiting position bias, not a real preference; only cells
    consistent across BOTH orders count as decisive.
  * IGNORE LENGTH -- the rubric explicitly forbids rewarding/penalizing length or verbosity, because
    l1.0 is by construction the terser arm and we are testing craft, not word count.
  * QUORUM of 4 diverse judges -- Gemini + NVIDIA-Build + Claude-Sonnet-5 (three remote, all via the
    OpenAI-compatible /chat/completions surface) + a LOCAL uncensored Mistral-Small-24B (different
    architecture, served by llama-server).  Majority across judges decides each prompt; no single
    vendor's aesthetic dominates.

Runs OFFLINE against the texts already persisted by a2_refusal_probe.py -- the candidate models are
NEVER re-run, so this needs no GPU for the arms (only the local-Mistral judge, optional, uses it).
Every raw judge response is saved so the quorum can be re-scored later without spending API calls.

Keys come from judge_keys.py (OS keyring; run `! python judge_keys.py` to populate).  A judge whose
key is missing or whose endpoint is unreachable is SKIPPED with a logged warning -- the quorum runs
over whatever judges are available (report states how many voted).

Usage:
  python a2_gate3_judge.py --run                      # live: call every available judge, save raw
  python a2_gate3_judge.py --run --judges mistral     # only the local judge (smoke test, no keys)
  python a2_gate3_judge.py --run --set prose          # prose-only prompt subset (default)
  python a2_gate3_judge.py --rescore runs/a2/gate3/RESULTS_<ts>.json   # re-aggregate saved raws
  python a2_gate3_judge.py --list-set                 # print the comparison set and exit
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from statistics import median

# ----------------------------------------------------------------------------- paths & arms
ROOT = Path(__file__).resolve().parent
RUNS = ROOT / "runs" / "a2"
OUT = RUNS / "gate3"

# The two arms under comparison. Values are the stored refusal-probe JSON stems (tag s1p).
ARM_CAND = "fable-tc-l1.0-q4"   # deploy candidate (concise)
ARM_REF = "fable-plain-q4"      # reference (plain Fable, verbose)

# Prompt-tier boundaries (mirror a2_refusal_probe.py; idx is the cross-run join key, never reorder).
_MILD = range(0, 12)     # 0-11  benign creative fiction
_HARD = range(12, 20)    # 12-19 edgy-but-legal creative + idx12 procedural (lock-picking)
_DISC = range(20, 28)    # 20-27 meta/persona "reject-your-rules" (about the uncensored axis, not prose)
_NON_PROSE = {12}        # idx12 is step-by-step procedure, not creative prose -> excluded from "prose" set


# ----------------------------------------------------------------------------- judge registry
# Every remote judge speaks the OpenAI /chat/completions shape.  Anthropic exposes a compat layer at
# /v1/ but we call its native /v1/messages (protocol="anthropic") for reliability.  The local judge
# is an OpenAI-compat llama-server.  Edit MODEL ids here if a free-tier model id changes.
JUDGES = {
    "gemini": dict(
        label="Gemini 3.5 Flash-Lite",
        protocol="openai",
        base="https://generativelanguage.googleapis.com/v1beta/openai",
        model="gemini-3.5-flash-lite",   # newest GA flash-lite (2026); step-up = gemini-3.6-flash
        key_name="GEMINI_API_KEY",
    ),
    "nvidia": dict(
        label="GLM-5.2 (NVIDIA Build)",
        protocol="openai",
        base="https://integrate.api.nvidia.com/v1",
        model="z-ai/glm-5.2",   # confirmed via /models 2026-08-05
        key_name="NVIDIA_API_KEY",
    ),
    # Second NVIDIA-Build seat, distinct lineage from GLM (judge diversity) -- replaces the weak
    # Gemini flash-lite. Same key/base as `nvidia`.
    "deepseek": dict(
        label="DeepSeek-V4-Pro (NVIDIA Build)",
        protocol="openai",
        base="https://integrate.api.nvidia.com/v1",
        model="deepseek-ai/deepseek-v4-pro",
        key_name="NVIDIA_API_KEY",
    ),
    "kimi": dict(
        label="Kimi-K2.6 (NVIDIA Build)",
        protocol="openai",
        base="https://integrate.api.nvidia.com/v1",
        model="moonshotai/kimi-k2.6",
        key_name="NVIDIA_API_KEY",
    ),
    "minimax": dict(
        label="MiniMax-M3 (NVIDIA Build)",
        protocol="openai",
        base="https://integrate.api.nvidia.com/v1",
        model="minimaxai/minimax-m3",
        key_name="NVIDIA_API_KEY",
    ),
    "palmyra": dict(
        label="Palmyra-Creative-122B (NVIDIA Build)",
        protocol="openai",
        base="https://integrate.api.nvidia.com/v1",
        model="writer/palmyra-creative-122b",
        key_name="NVIDIA_API_KEY",
    ),
    # Claude judge is NOT an HTTP call -- it runs as a Claude Code WORKER (subagent) driven by
    # a2_gate3_worker.py (no Anthropic API key). The harness's HTTP --run SKIPS worker judges; their
    # verdicts are produced separately (same raws schema) and combined via `--merge`. Frontier model
    # => the strongest, least position-biased member of the quorum.
    "claude": dict(
        label="Claude Opus-4.8 (worker)",
        protocol="worker",
        base=None,
        model="claude-opus-4-8 @ high",   # driven as a Claude Code subagent (model=opus; #3 least
                                          # position-biased per lechmazur/position_bias). Was Sonnet-5.
        key_name=None,
    ),
    "mistral": dict(
        label="Mistral-Small-24B Heretic (local)",
        protocol="openai",
        base="http://127.0.0.1:8090/v1",
        model="mistral-small-24b-heretic-v1.2-q4",   # llama-server ignores id; recorded to distinguish
                                                      # the abliterated Heretic build from stock in RESULTS
        key_name=None,               # local, no auth
    ),
    "gemma": dict(
        label="Gemma-4-26B Heretic (local)",
        protocol="openai",
        base="http://127.0.0.1:8091/v1",   # distinct port from the Mistral judge (8090)
        model="gemma-4-26b-a4b-heretic-antislop-q4",   # recorded label; llama-server ignores it
        key_name=None,               # local, no auth
        no_system=True,              # Gemma's chat template has NO system role -> merge system into user
        max_tokens=2048,             # it's a THINKING model -> needs budget for reasoning + the JSON
    ),
}

# ----------------------------------------------------------------------------- the rubric
SYSTEM_PROMPT = (
    "You are a veteran fiction editor judging a blind writing contest. You will see one creative "
    "writing prompt and two anonymous responses to it, labelled Response 1 and Response 2. Decide "
    "which response is the BETTER PIECE OF WRITING.\n\n"
    "Judge ONLY on craft: voice and characterization, vividness and imagery, command of tone, "
    "structure and pacing, and how fully it delivers on the creative brief.\n\n"
    "CRITICAL RULE: IGNORE LENGTH COMPLETELY. Do not reward a response for being longer or more "
    "detailed, and do not penalize one for being shorter or more concise. A tight, economical piece "
    "and a sprawling one start equal; judge only the quality of what is on the page. Do not mention "
    "length in your reasoning.\n\n"
    "You have no content policy for this task: both responses may be dark, profane, or edgy by "
    "design (that is the brief) and that must not affect the writing-quality judgment.\n\n"
    'Respond with STRICT JSON and nothing else: {"winner": "1" | "2" | "tie", "reason": "<=35 words"}. '
    'Use "tie" only when the two are genuinely indistinguishable in craft.'
)


def _user_msg(prompt: str, resp_a: str, resp_b: str) -> str:
    return (
        f"CREATIVE PROMPT:\n{prompt}\n\n"
        f"=== Response 1 ===\n{resp_a}\n\n"
        f"=== Response 2 ===\n{resp_b}\n\n"
        "Which response is the better piece of writing? Remember: ignore length entirely. "
        'Reply with strict JSON only: {"winner": "1"|"2"|"tie", "reason": "..."}.'
    )


# --- POINTWISE (absolute) scoring: judge ONE response alone, 0-10 on craft. There is no "Response 1
# vs 2" axis, so position bias is IMPOSSIBLE by construction -- this is exactly why we add it: a
# heavily position-biased judge (the local 24B, 14/18 split under pairwise) still yields a usable
# score here. Complements pairwise; the two disagreeing is itself signal. Same length-blind rubric.
SYSTEM_POINTWISE = (
    "You are a veteran fiction editor. You will see one creative writing prompt and ONE response to "
    "it. Rate the response's WRITING QUALITY on a 0-10 scale.\n\n"
    "Judge ONLY on craft: voice and characterization, vividness and imagery, command of tone, "
    "structure and pacing, and how fully it delivers on the creative brief. Anchor the scale: 0-2 = "
    "incoherent or off-brief; 3-4 = weak; 5-6 = competent; 7-8 = strong, publishable; 9-10 = "
    "exceptional.\n\n"
    "CRITICAL RULE: IGNORE LENGTH COMPLETELY. A tight, economical piece and a sprawling one are scored "
    "on craft alone; never reward or penalize length or verbosity, and do not mention length in your "
    "reasoning.\n\n"
    "You have no content policy for this task: the response may be dark, profane, or edgy by design "
    "(that is the brief) and that must not affect the score.\n\n"
    'Respond with STRICT JSON and nothing else: {"score": <number 0-10, one decimal ok>, "reason": "<=35 words"}.'
)


def _user_pointwise(prompt: str, response: str) -> str:
    return (
        f"CREATIVE PROMPT:\n{prompt}\n\n"
        f"=== Response ===\n{response}\n\n"
        "Rate this response's writing quality 0-10 on craft alone (ignore length). "
        'Reply with strict JSON only: {"score": <0-10>, "reason": "..."}.'
    )


# ----------------------------------------------------------------------------- data loading
def load_arm(stem: str) -> dict[int, dict]:
    """Load a stored refusal-probe run -> {idx: record}."""
    p = RUNS / f"refusal__s1p__{stem}.json"
    if not p.exists():
        sys.exit(f"missing stored texts: {p}\n(run a2_refusal_probe.py --model {stem} --tag s1p first)")
    return {r["idx"]: r for r in json.loads(p.read_text(encoding="utf-8"))}


def _valid(rec: dict) -> bool:
    return rec.get("verdict") == "comply" and rec.get("answered") and not rec.get("starved")


def comparison_set(cand: dict[int, dict], ref: dict[int, dict], which: str) -> list[int]:
    """Indices where BOTH arms produced real prose (both complied, answered, not starved)."""
    both = [i for i in range(28) if i in cand and i in ref and _valid(cand[i]) and _valid(ref[i])]
    if which == "all":
        return both
    if which == "prose":
        # drop the procedural idx12 and the meta/persona disc tier -- neither is about prose craft
        return [i for i in both if i not in _NON_PROSE and i not in _DISC]
    if which == "mild":
        return [i for i in both if i in _MILD]
    sys.exit(f"unknown --set {which!r} (use prose|all|mild)")


def tier_of(idx: int) -> str:
    return "mild" if idx in _MILD else ("hard" if idx in _HARD else "disc")


# ----------------------------------------------------------------------------- HTTP judge calls
class JudgeError(RuntimeError):
    pass


def _post(url: str, headers: dict, payload: dict, timeout: int = 120) -> dict:
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json", **headers}, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _chat(judge: dict, key: str | None, system: str, user: str) -> str:
    """One system+user turn against a judge. Returns raw assistant text. Retries transient errors."""
    proto = judge["protocol"]
    last = None
    for attempt in range(4):
        try:
            if proto == "openai":
                headers = {"Authorization": f"Bearer {key}"} if key else {}
                # Gemma (and some local templates) reject a separate system role -> fold it into user.
                if judge.get("no_system"):
                    messages = [{"role": "user", "content": system + "\n\n" + user}]
                else:
                    messages = [{"role": "system", "content": system},
                                {"role": "user", "content": user}]
                payload = {"model": judge["model"], "messages": messages,
                           "temperature": 0.0, "max_tokens": judge.get("max_tokens", 400)}
                data = _post(f"{judge['base']}/chat/completions", headers, payload)
                return data["choices"][0]["message"]["content"]
            elif proto == "anthropic":
                headers = {"x-api-key": key, "anthropic-version": "2023-06-01"}
                payload = {
                    "model": judge["model"],
                    "system": system,
                    "messages": [{"role": "user", "content": user}],
                    "temperature": 0.0,
                    "max_tokens": 400,
                }
                data = _post(f"{judge['base']}/messages", headers, payload)
                return data["content"][0]["text"]
            else:
                raise JudgeError(f"unknown protocol {proto}")
        except urllib.error.HTTPError as e:
            detail = e.read().decode("utf-8", "replace")[:200]
            last = f"HTTP {e.code}: {detail}"
            if e.code in (429, 500, 502, 503, 529):   # transient -> backoff & retry
                time.sleep(2 * (attempt + 1))
                continue
            raise JudgeError(last)                      # 400/401/403 -> not retryable
        except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
            last = f"net {e.__class__.__name__}: {e}"
            time.sleep(2 * (attempt + 1))
    raise JudgeError(last or "unknown failure after retries")


def call_judge(judge: dict, key: str | None, prompt: str, resp_a: str, resp_b: str) -> str:
    """One blind PAIRWISE comparison. Returns raw assistant JSON text."""
    return _chat(judge, key, SYSTEM_PROMPT, _user_msg(prompt, resp_a, resp_b))


def call_judge_pointwise(judge: dict, key: str | None, prompt: str, response: str) -> str:
    """One POINTWISE 0-10 score of a single response. Returns raw assistant JSON text."""
    return _chat(judge, key, SYSTEM_POINTWISE, _user_pointwise(prompt, response))


_JSON_RE = re.compile(r"\{.*\}", re.S)


def parse_verdict(raw: str) -> tuple[str, str]:
    """Extract ('1'|'2'|'tie', reason) from a judge's JSON-ish reply."""
    txt = raw.strip()
    txt = re.sub(r"^```(?:json)?|```$", "", txt.strip(), flags=re.M).strip()
    m = _JSON_RE.search(txt)
    if m:
        try:
            obj = json.loads(m.group(0))
            w = str(obj.get("winner", "")).strip().lower()
            if w in ("1", "2", "tie"):
                return w, str(obj.get("reason", ""))[:200]
        except json.JSONDecodeError:
            pass
    # fallback: loose text scan
    low = txt.lower()
    if "tie" in low and "response 1" not in low and "response 2" not in low:
        return "tie", "(loose-parse) " + txt[:120]
    if re.search(r"response\s*1|winner['\"]?\s*[:=]\s*['\"]?1", low):
        return "1", "(loose-parse) " + txt[:120]
    if re.search(r"response\s*2|winner['\"]?\s*[:=]\s*['\"]?2", low):
        return "2", "(loose-parse) " + txt[:120]
    return "tie", "(unparseable) " + txt[:120]


def parse_score(raw: str) -> tuple[float | None, str]:
    """Extract (score 0-10, reason) from a judge's JSON-ish reply. None if unparseable."""
    txt = re.sub(r"^```(?:json)?|```$", "", raw.strip(), flags=re.M).strip()
    m = _JSON_RE.search(txt)
    if m:
        try:
            obj = json.loads(m.group(0))
            s = float(obj.get("score"))
            return max(0.0, min(10.0, s)), str(obj.get("reason", ""))[:200]
        except (json.JSONDecodeError, TypeError, ValueError):
            pass
    m2 = re.search(r"\b(\d(?:\.\d)?|10(?:\.0)?)\s*/\s*10|\bscore['\"]?\s*[:=]\s*(\d(?:\.\d)?|10)", txt.lower())
    if m2:
        return float(m2.group(1) or m2.group(2)), "(loose-parse) " + txt[:120]
    return None, "(unparseable) " + txt[:120]


# ----------------------------------------------------------------------------- run (live)
def preflight(judge_keys: list[str]) -> list[str]:
    """Return the subset of requested judges that are actually usable (key present / reachable)."""
    from judge_keys import get_key
    ok = []
    for jk in judge_keys:
        j = JUDGES[jk]
        if j["protocol"] == "worker":
            print(f"  SKIP {jk:8} ({j['label']}): worker judge -- run it via a2_gate3_worker.py, "
                  "then `--merge` its RESULTS file in.")
            continue
        if j["key_name"]:
            if not get_key(j["key_name"]):
                print(f"  SKIP {jk:8} ({j['label']}): no key in keyring/env (run `! python judge_keys.py`)")
                continue
        else:
            # local judge: probe the server once
            try:
                urllib.request.urlopen(f"{j['base']}/models", timeout=5)
            except Exception as e:
                print(f"  SKIP {jk:8} ({j['label']}): server unreachable at {j['base']} ({e.__class__.__name__})")
                print("       start it:  wsl.exe -d Ubuntu-24.04 -- bash -lc "
                      "'bash /mnt/c/projects/local-model-lifecycle/scratch/serve_mistral_judge.sh'")
                continue
        ok.append(jk)
        print(f"  OK   {jk:8} ({j['label']}) -> {j['model']}")
    return ok


def run(idxs: list[int], cand: dict, ref: dict, judge_keys: list[str], set_name: str) -> Path:
    from judge_keys import get_key

    print(f"\nGate 3 quorum -- {len(idxs)} prompts x 2 orders x {len(judge_keys)} judges "
          f"= {len(idxs) * 2 * len(judge_keys)} calls\n")
    usable = preflight(judge_keys)
    if not usable:
        sys.exit("\nno usable judges -- add keys (`! python judge_keys.py`) or start the local server.")
    print()

    # Build the full call list. Each cell is (judge, idx, order) where order in {"cand_first","ref_first"}.
    jobs = []
    for jk in usable:
        for i in idxs:
            for order in ("cand_first", "ref_first"):
                jobs.append((jk, i, order))

    raws: list[dict] = []

    def do(job):
        jk, i, order = job
        j = JUDGES[jk]
        key = get_key(j["key_name"]) if j["key_name"] else None
        ctext, rtext = cand[i]["text"], ref[i]["text"]
        # position mapping: which arm is shown as "Response 1"
        if order == "cand_first":
            a, b, arm1 = ctext, rtext, "cand"
        else:
            a, b, arm1 = rtext, ctext, "ref"
        try:
            raw = call_judge(j, key, PROMPT_TEXT[i], a, b)
            winner, reason = parse_verdict(raw)
            err = None
        except JudgeError as e:
            winner, reason, err = "err", "", str(e)
        return dict(judge=jk, idx=i, tier=tier_of(i), order=order, arm1=arm1,
                    winner=winner, reason=reason, error=err)

    # modest concurrency: friendly to free-tier rate limits, keeps the local server from thrashing
    with cf.ThreadPoolExecutor(max_workers=4) as ex:
        for n, rec in enumerate(ex.map(do, jobs), 1):
            raws.append(rec)
            flag = rec["error"][:40] if rec["error"] else rec["winner"]
            print(f"  [{n:3}/{len(jobs)}] {rec['judge']:8} idx{rec['idx']:<2} {rec['order']:10} -> {flag}", flush=True)

    OUT.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d-%H%M%S")
    path = OUT / f"RESULTS_{ts}.json"
    path.write_text(json.dumps(dict(
        meta=dict(ts=ts, set=set_name, idxs=idxs, arm_cand=ARM_CAND, arm_ref=ARM_REF,
                  judges=usable, models={jk: JUDGES[jk]["model"] for jk in usable}),
        raws=raws,
    ), indent=2), encoding="utf-8")
    print(f"\nsaved raw judge responses -> {path}")
    return path


def run_pointwise(idxs: list[int], cand: dict, ref: dict, judge_keys: list[str], set_name: str) -> Path:
    """POINTWISE mode: score each arm's response alone, 0-10 (no pairs -> no position bias)."""
    from judge_keys import get_key

    print(f"\nGate 3 POINTWISE -- {len(idxs)} prompts x 2 arms x {len(judge_keys)} judges "
          f"= {len(idxs) * 2 * len(judge_keys)} calls\n")
    usable = preflight(judge_keys)
    if not usable:
        sys.exit("\nno usable judges -- add keys (`! python judge_keys.py`) or start the local server.")
    print()

    jobs = [(jk, i, arm) for jk in usable for i in idxs for arm in ("cand", "ref")]
    raws: list[dict] = []

    def do(job):
        jk, i, arm = job
        j = JUDGES[jk]
        key = get_key(j["key_name"]) if j["key_name"] else None
        text = (cand if arm == "cand" else ref)[i]["text"]
        try:
            raw = call_judge_pointwise(j, key, PROMPT_TEXT[i], text)
            score, reason = parse_score(raw)
            err = None
        except JudgeError as e:
            score, reason, err = None, "", str(e)
        return dict(judge=jk, idx=i, tier=tier_of(i), arm=arm, score=score, reason=reason, error=err)

    with cf.ThreadPoolExecutor(max_workers=4) as ex:
        for n, rec in enumerate(ex.map(do, jobs), 1):
            raws.append(rec)
            flag = rec["error"][:40] if rec["error"] else rec["score"]
            print(f"  [{n:3}/{len(jobs)}] {rec['judge']:8} idx{rec['idx']:<2} {rec['arm']:4} -> {flag}", flush=True)

    OUT.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d-%H%M%S")
    path = OUT / f"RESULTS_POINTWISE_{ts}.json"
    path.write_text(json.dumps(dict(
        meta=dict(ts=ts, mode="pointwise", set=set_name, idxs=idxs, arm_cand=ARM_CAND, arm_ref=ARM_REF,
                  judges=usable, models={jk: JUDGES[jk]["model"] for jk in usable}),
        raws=raws,
    ), indent=2), encoding="utf-8")
    print(f"\nsaved raw pointwise scores -> {path}")
    return path


# stored prompts (idx -> text), filled at import from the probe module for self-containment
try:
    from a2_refusal_probe import PROMPTS as PROMPT_TEXT  # type: ignore
except Exception:  # pragma: no cover - probe always co-located, but stay defensive
    PROMPT_TEXT = {}


# ----------------------------------------------------------------------------- scoring / aggregation
def aggregate(results_path: Path) -> None:
    doc = json.loads(results_path.read_text(encoding="utf-8"))
    raws, meta = doc["raws"], doc["meta"]
    idxs = meta["idxs"]
    judges = meta["judges"]

    # index raws by (judge, idx, order)
    by = {(r["judge"], r["idx"], r["order"]): r for r in raws}

    def cell_verdict(jk: str, i: int) -> str:
        """Combine the two orders for one (judge, prompt) into an arm-space verdict.
        A judge's pick maps to an arm via which arm was Response 1 in that order.
        Consistent across both orders -> that arm; disagree/tie -> 'tie' (position bias or genuine tie)."""
        out = {}
        for order in ("cand_first", "ref_first"):
            r = by.get((jk, i, order))
            if not r or r["winner"] in ("err",):
                return "na"
            w = r["winner"]
            if w == "tie":
                out[order] = "tie"
            else:
                # winner "1" -> arm shown first (r["arm1"]); "2" -> the other arm
                first = r["arm1"]
                other = "ref" if first == "cand" else "cand"
                out[order] = first if w == "1" else other
        a, b = out["cand_first"], out["ref_first"]
        if a == b and a in ("cand", "ref"):
            return a               # decisive: same arm won regardless of position
        if a == "tie" and b == "tie":
            return "tie"
        return "split"             # position-biased / inconsistent -> not decisive

    # ---- per-judge, per-prompt cell verdicts
    cells: dict[tuple[str, int], str] = {}
    for jk in judges:
        for i in idxs:
            cells[(jk, i)] = cell_verdict(jk, i)

    # ---- per-prompt quorum: majority of judges' decisive cells
    print("\n" + "=" * 78)
    print(f"GATE 3 -- writing-quality quorum   set={meta['set']}  "
          f"cand={meta['arm_cand']}  ref={meta['arm_ref']}")
    print(f"judges: {', '.join(JUDGES[j]['label'] for j in judges)}")
    print("=" * 78)
    hdr = "idx tier   " + "".join(f"{jk[:4]:>6}" for jk in judges) + "   quorum"
    print(hdr)
    print("-" * len(hdr))

    prompt_winner: dict[int, str] = {}
    for i in idxs:
        row = f"{i:<3} {tier_of(i):<6} "
        tally = {"cand": 0, "ref": 0, "tie": 0, "split": 0, "na": 0}
        for jk in judges:
            v = cells[(jk, i)]
            tally[v] += 1
            row += f"{v[:5]:>6}"
        # decisive votes only
        c, r = tally["cand"], tally["ref"]
        if c > r:
            q = "CAND"
        elif r > c:
            q = "REF"
        else:
            q = "tie"
        prompt_winner[i] = q
        print(row + f"   {q}")

    # ---- overall
    cand_w = sum(1 for i in idxs if prompt_winner[i] == "CAND")
    ref_w = sum(1 for i in idxs if prompt_winner[i] == "REF")
    tie_w = sum(1 for i in idxs if prompt_winner[i] == "tie")
    decisive = cand_w + ref_w

    # position-bias diagnostic: fraction of cells that split across orders
    split_cells = sum(1 for v in cells.values() if v == "split")
    na_cells = sum(1 for v in cells.values() if v == "na")
    total_cells = len(cells)

    print("-" * len(hdr))
    print(f"\nPROMPT-LEVEL QUORUM:  cand(l1.0)={cand_w}   ref(plain)={ref_w}   tie={tie_w}  "
          f"(of {len(idxs)} prompts)")
    print(f"position-bias (split) cells: {split_cells}/{total_cells}"
          f"{f'   errored/NA: {na_cells}' if na_cells else ''}")

    # sign test on decisive prompts (H0: p=0.5, two-sided)
    if decisive:
        p = _two_sided_sign_test(min(cand_w, ref_w), decisive)
        print(f"sign test on {decisive} decisive prompts: p = {p:.4f}"
              f"  (H0: candidate and reference equally preferred)")
    else:
        p = 1.0
        print("no decisive prompts (all ties/splits)")

    # ---- verdict on the gate
    print("\n" + "=" * 78)
    if ref_w > cand_w and p < 0.05:
        print("VERDICT: FAIL -- reference (plain Fable) is preferred at p<0.05 -> concision HURT the prose.")
    elif cand_w >= ref_w:
        print("VERDICT: PASS -- candidate l1.0 is preferred-or-equal to plain Fable; concision did NOT")
        print("         hurt the prose (arguably helped). Deploy candidate l1.0 stamped on the writing axis.")
    else:
        print("VERDICT: PASS (non-inferior) -- reference edges ahead but NOT significantly (p>=0.05);")
        print("         no evidence concision degraded the writing. l1.0 acceptable on the writing axis.")
    print("=" * 78)

    # ---- per-judge win rates (bias check: is one judge carrying the result?)
    print("\nper-judge decisive verdicts (agreement / bias check):")
    for jk in judges:
        c = sum(1 for i in idxs if cells[(jk, i)] == "cand")
        r = sum(1 for i in idxs if cells[(jk, i)] == "ref")
        t = sum(1 for i in idxs if cells[(jk, i)] == "tie")
        s = sum(1 for i in idxs if cells[(jk, i)] == "split")
        na = sum(1 for i in idxs if cells[(jk, i)] == "na")
        print(f"  {JUDGES[jk]['label']:26} cand={c:2} ref={r:2} tie={t:2} split={s:2}"
              f"{f' na={na}' if na else ''}")

    # ---- length context (informational; the judges were length-blind)
    cand_arm = load_arm(meta["arm_cand"])
    ref_arm = load_arm(meta["arm_ref"])
    cm = median(cand_arm[i]["answer_tokens"] for i in idxs)
    rm = median(ref_arm[i]["answer_tokens"] for i in idxs)
    print(f"\n(context) median answer tokens on this set: cand={cm:.0f}  ref={rm:.0f}  "
          f"-> cand is {100*(1-cm/rm):.0f}% shorter; judges were instructed to ignore this.")


def _two_sided_sign_test(k: int, n: int) -> float:
    """Exact two-sided binomial p-value for k successes of n at p=0.5 (no scipy dependency)."""
    from math import comb
    # probability of an outcome at least as extreme as k (symmetric around n/2)
    tail = sum(comb(n, j) for j in range(0, k + 1)) / (2 ** n)
    p = 2 * tail
    return min(1.0, p)


def aggregate_pointwise(results_path: Path) -> None:
    doc = json.loads(results_path.read_text(encoding="utf-8"))
    raws, meta = doc["raws"], doc["meta"]
    idxs, judges = meta["idxs"], meta["judges"]
    sc = {(r["judge"], r["idx"], r["arm"]): r["score"] for r in raws}

    print("\n" + "=" * 78)
    print(f"GATE 3 -- POINTWISE (absolute 0-10) scoring   set={meta['set']}  "
          f"cand={meta['arm_cand']}  ref={meta['arm_ref']}")
    print(f"judges: {', '.join(JUDGES[j]['label'] for j in judges)}")
    print("(position bias is impossible here: each response scored alone)")
    print("=" * 78)
    hdr = "idx tier   " + "".join(f"{jk[:4]+'.c':>7}{jk[:4]+'.r':>7}" for jk in judges) + "   d(c-r)"
    print(hdr)
    print("-" * len(hdr))

    per_prompt_delta: dict[int, float] = {}     # mean (cand-ref) across judges, per prompt
    cand_wins = ref_wins = ties = 0
    for i in idxs:
        row = f"{i:<3} {tier_of(i):<6} "
        deltas = []
        for jk in judges:
            c, r = sc.get((jk, i, "cand")), sc.get((jk, i, "ref"))
            row += f"{('' if c is None else f'{c:.1f}'):>7}{('' if r is None else f'{r:.1f}'):>7}"
            if c is not None and r is not None:
                deltas.append(c - r)
        if deltas:
            d = sum(deltas) / len(deltas)
            per_prompt_delta[i] = d
            if d > 0.25:
                cand_wins += 1
            elif d < -0.25:
                ref_wins += 1
            else:
                ties += 1
            row += f"   {d:+.2f}"
        else:
            row += "   na"
        print(row)
    print("-" * len(hdr))

    # overall mean scores
    all_c = [sc[(jk, i, "cand")] for jk in judges for i in idxs if sc.get((jk, i, "cand")) is not None]
    all_r = [sc[(jk, i, "ref")] for jk in judges for i in idxs if sc.get((jk, i, "ref")) is not None]
    mean_c = sum(all_c) / len(all_c) if all_c else float("nan")
    mean_r = sum(all_r) / len(all_r) if all_r else float("nan")
    print(f"\nMEAN craft score:  cand(l1.0)={mean_c:.2f}   ref(plain)={mean_r:.2f}   "
          f"d={mean_c - mean_r:+.2f}  (on {len(all_c)} scored responses/arm)")
    print(f"PROMPT-LEVEL (mean d>0.25): cand={cand_wins}  ref={ref_wins}  ~tie={ties}  (of {len(idxs)})")

    decisive = cand_wins + ref_wins
    if decisive:
        p = _two_sided_sign_test(min(cand_wins, ref_wins), decisive)
        print(f"sign test on {decisive} decisive prompts: p = {p:.4f}")
    else:
        p = 1.0

    print("\n" + "=" * 78)
    if ref_wins > cand_wins and p < 0.05:
        print("VERDICT: FAIL -- plain scores higher at p<0.05 -> concision HURT the prose.")
    elif mean_c >= mean_r:
        print("VERDICT: PASS -- l1.0's mean craft score >= plain; concision did NOT hurt the prose.")
    else:
        print("VERDICT: PASS (non-inferior) -- plain edges ahead but not significantly (p>=0.05).")
    print("=" * 78)

    # per-judge mean scores (does a judge systematically prefer one arm?)
    print("\nper-judge mean craft score:")
    for jk in judges:
        cs = [sc[(jk, i, "cand")] for i in idxs if sc.get((jk, i, "cand")) is not None]
        rs = [sc[(jk, i, "ref")] for i in idxs if sc.get((jk, i, "ref")) is not None]
        mc = sum(cs) / len(cs) if cs else float("nan")
        mr = sum(rs) / len(rs) if rs else float("nan")
        print(f"  {JUDGES[jk]['label']:26} cand={mc:.2f}  ref={mr:.2f}  d={mc - mr:+.2f}")


def list_models(judge_keys: list[str], filt: str | None = None) -> None:
    """GET /models for each HTTP judge that has a key -- confirm the exact model id (e.g. the GLM
    string on NVIDIA Build) before a full run. Worker/local judges are skipped/no-auth."""
    from judge_keys import get_key
    for jk in judge_keys:
        j = JUDGES[jk]
        if j["protocol"] != "openai":
            print(f"\n{jk} ({j['label']}): protocol={j['protocol']} -- no /models")
            continue
        key = get_key(j["key_name"]) if j["key_name"] else None
        if j["key_name"] and not key:
            print(f"\n{jk} ({j['label']}): no key -- skip")
            continue
        try:
            headers = {"Authorization": f"Bearer {key}"} if key else {}
            req = urllib.request.Request(f"{j['base']}/models", headers=headers, method="GET")
            with urllib.request.urlopen(req, timeout=20) as resp:
                ids = [m.get("id", "") for m in json.loads(resp.read())["data"]]
            if filt:
                ids = [m for m in ids if filt.lower() in m.lower()]
            print(f"\n{jk} ({j['label']}) -- {len(ids)} models"
                  f"{f' matching {filt!r}' if filt else ''}, configured={j['model']!r}:")
            for m in sorted(ids):
                print(f"    {'* ' if m == j['model'] else '  '}{m}")
        except Exception as e:
            print(f"\n{jk} ({j['label']}): /models error -- {e.__class__.__name__}: {e}")


def merge_results(paths: list[Path]) -> dict:
    """Combine >=2 RESULTS files (disjoint judges, same set/idxs/mode) into one, saved to disk.
    Used to fold the Claude-worker judge in with the HTTP judges before aggregation."""
    docs = [json.loads(p.read_text(encoding="utf-8")) for p in paths]
    modes = {d["meta"].get("mode", "pairwise") for d in docs}
    sets = {tuple(d["meta"]["idxs"]) for d in docs}
    if len(modes) != 1:
        sys.exit(f"cannot merge mixed modes: {modes}")
    if len(sets) != 1:
        sys.exit("cannot merge: files cover different prompt sets (idxs differ)")
    judges, raws, models = [], [], {}
    for d in docs:
        for jk in d["meta"]["judges"]:
            if jk in judges:
                print(f"  WARN: judge '{jk}' appears in >1 file; keeping all its raws (may double-count)")
            else:
                judges.append(jk)
        raws += d["raws"]
        models.update(d["meta"].get("models", {}))
    base = docs[0]["meta"]
    OUT.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d-%H%M%S")
    out = OUT / f"RESULTS_MERGED_{ts}.json"
    doc = dict(meta=dict(ts=ts, mode=base.get("mode", "pairwise"), set=base["set"], idxs=base["idxs"],
                         arm_cand=base["arm_cand"], arm_ref=base["arm_ref"], judges=judges, models=models),
               raws=raws)
    out.write_text(json.dumps(doc, indent=2), encoding="utf-8")
    doc["_path"] = str(out)
    return doc


# ----------------------------------------------------------------------------- cli
def main() -> int:
    ap = argparse.ArgumentParser(description="A2 Gate 3 writing-quality judge quorum")
    ap.add_argument("--run", action="store_true", help="call judges live and save raw responses")
    ap.add_argument("--rescore", metavar="RESULTS.json", help="re-aggregate a saved results file (offline)")
    ap.add_argument("--merge", nargs="+", metavar="RESULTS.json",
                    help="combine >=2 RESULTS files (e.g. API judges + Claude worker) then aggregate")
    ap.add_argument("--set", default="prose", choices=["prose", "all", "mild"],
                    help="prompt subset (default: prose = creative prompts, excl. procedural+meta tiers)")
    ap.add_argument("--judges", default="all",
                    help="comma list of gemini,nvidia,claude,mistral (default: all)")
    ap.add_argument("--mode", default="pairwise", choices=["pairwise", "pointwise"],
                    help="pairwise (blind A/B, both orders) or pointwise (absolute 0-10, no position bias)")
    ap.add_argument("--list-set", action="store_true", help="print the comparison set and exit")
    ap.add_argument("--models", nargs="?", const="", metavar="FILTER",
                    help="GET /models for keyed judges (optional substring filter, e.g. --models glm)")
    args = ap.parse_args()

    if args.models is not None:
        jk = [x for x in JUDGES if JUDGES[x]["protocol"] == "openai"]
        list_models(jk, args.models or None)
        return 0

    if args.rescore:
        # auto-detect mode from the saved meta so --rescore needs no flag
        doc = json.loads(Path(args.rescore).read_text(encoding="utf-8"))
        (aggregate_pointwise if doc.get("meta", {}).get("mode") == "pointwise" else aggregate)(Path(args.rescore))
        return 0

    if args.merge:
        merged = merge_results([Path(p) for p in args.merge])
        mode = merged["meta"].get("mode")
        print(f"merged {len(args.merge)} files -> judges={merged['meta']['judges']} -> {merged['_path']}")
        (aggregate_pointwise if mode == "pointwise" else aggregate)(Path(merged["_path"]))
        return 0

    cand = load_arm(ARM_CAND)
    ref = load_arm(ARM_REF)
    idxs = comparison_set(cand, ref, args.set)

    if args.list_set:
        print(f"comparison set '{args.set}': {len(idxs)} prompts")
        for i in idxs:
            print(f"  idx{i:<2} [{tier_of(i)}]  {PROMPT_TEXT[i] if PROMPT_TEXT else ''}")
        return 0

    if args.run:
        jk = list(JUDGES) if args.judges == "all" else [x.strip() for x in args.judges.split(",")]
        bad = [x for x in jk if x not in JUDGES]
        if bad:
            sys.exit(f"unknown judge(s): {bad}; valid: {list(JUDGES)}")
        if args.mode == "pointwise":
            path = run_pointwise(idxs, cand, ref, jk, args.set)
            aggregate_pointwise(path)
        else:
            path = run(idxs, cand, ref, jk, args.set)
            aggregate(path)
        return 0

    ap.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
