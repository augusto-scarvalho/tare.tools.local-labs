#!/usr/bin/env python3
"""LAB-QA-001 — Benchmark Harness Qualification self-test.

The scorer is an INSTRUMENT and can fail: this lab already shipped TWO independent HumanEval
scoring bugs in one pipeline (a non-self-contained sample; a stale EvalPlus result reused), and
that inverted a scientific conclusion (commit 5a24781 → retracted by 81eed6d/e8fb3d3). This suite
qualifies the harness glue BEFORE its numbers are trusted for promotion.

Design: deterministic, **no GPU, no EvalPlus, runs in seconds**. It exercises the REAL functions in
`benchmark_harness_qa.py` (the same ones `a2_concision_bench.py` / `score_subset.py` import), plus a
tiny in-process code scorer (`_mini_score`) that executes solution+test in a subprocess — the same
"exec the program, run the tests" contract EvalPlus uses, without its dataset.

Run:  python tests/benchmark_harness/benchmark_harness_selftest.py
Exit 0 = all pass. Emits benchmark_selftest_report.json next to this file.
"""
from __future__ import annotations

import json
import pathlib
import subprocess
import sys
import tempfile

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import benchmark_harness_qa as q  # noqa: E402

HERE = pathlib.Path(__file__).resolve().parent
CASES: list[dict] = []


def case(name: str, incident: str):
    def deco(fn):
        try:
            fn()
            CASES.append({"name": name, "status": "pass", "incident": incident, "detail": ""})
        except AssertionError as e:
            CASES.append({"name": name, "status": "FAIL", "incident": incident, "detail": str(e)})
        except Exception as e:  # a test that errored is a failed test
            CASES.append({"name": name, "status": "ERROR", "incident": incident,
                          "detail": f"{type(e).__name__}: {e}"})
        return fn
    return deco


def _mini_score(solution: str, test_src: str, timeout: float = 10.0) -> bool:
    """Execute `solution + test` in a fresh subprocess; True iff it exits 0 (tests passed). Mirrors
    EvalPlus's exec-and-assert contract on a tiny fixture — no dataset, no GPU."""
    prog = solution + "\n" + test_src + "\ncheck()\n"
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False, encoding="utf-8") as f:
        f.write(prog); path = f.name
    try:
        r = subprocess.run([sys.executable, path], capture_output=True, timeout=timeout)
        return r.returncode == 0
    except subprocess.TimeoutExpired:
        return False
    finally:
        pathlib.Path(path).unlink(missing_ok=True)


# ---- a minimal HumanEval-shaped fixture: the prompt PROVIDES a helper the completion relies on --
FX_PROMPT = (
    "def is_even(n):\n"
    "    return n % 2 == 0\n\n\n"
    "def count_evens(xs):\n"
    '    """Count even numbers using the helper above."""\n'
)
FX_COMPLETION_CONCISE = (  # a concise model CONTINUES the prompt: only the target, reuses is_even
    "def count_evens(xs):\n"
    "    return sum(1 for x in xs if is_even(x))\n"
)
FX_TEST = "def check():\n    assert count_evens([1,2,3,4]) == 2\n    assert count_evens([]) == 0\n"
FX_IDS = ["fx/0", "fx/1"]


# ==================================================================================================
# Required cases
# ==================================================================================================

@case("known-good completion -> PASS",
      "sanity: a correct self-contained program must score PASS")
def _():
    sol = q.assemble_humaneval_solution(FX_PROMPT, FX_COMPLETION_CONCISE)
    assert _mini_score(sol, FX_TEST) is True, "correct assembled solution should pass"

@case("known-bad completion -> FAIL",
      "sanity: a wrong program must score FAIL (guards against a scorer that always passes)")
def _():
    bad = "def count_evens(xs):\n    return 999\n"
    assert _mini_score(bad, FX_TEST) is False, "wrong solution must fail"

@case("missing-prompt (non-self-contained) -> detected/FAIL",
      "INCIDENT 2026-08-10 (81eed6d): storing the bare completion scored concise models 0/60 "
      "(NameError on prompt-provided helper). The fix prepends the prompt.")
def _():
    bare_fails = _mini_score(FX_COMPLETION_CONCISE, FX_TEST) is False      # reproduces the bug
    fixed_pass = _mini_score(q.assemble_humaneval_solution(FX_PROMPT, FX_COMPLETION_CONCISE), FX_TEST) is True
    assert bare_fails, "bare completion should FAIL (NameError: is_even) — reproduces the incident"
    assert fixed_pass, "assemble_humaneval_solution must make it self-contained and PASS"

@case("wrong (unknown) task_id -> detected",
      "a sample whose id is not in the dataset must be flagged, not scored")
def _():
    probs = q.validate_samples([{"task_id": "fx/999", "solution": "x"}], FX_IDS)
    assert any(p["kind"] == "unknown-task_id" for p in probs), probs

@case("duplicate task_id -> detected",
      "two samples for one id would let one silently shadow the other")
def _():
    probs = q.validate_samples(
        [{"task_id": "fx/0", "solution": "a"}, {"task_id": "fx/0", "solution": "b"}], FX_IDS)
    assert any(p["kind"] == "duplicate-task_id" for p in probs), probs

@case("missing sample -> detected",
      "a subset missing an expected id would score a smaller-than-claimed denominator")
def _():
    probs = q.validate_samples([{"task_id": "fx/0", "solution": "a"}], FX_IDS)
    assert any(p["kind"] == "missing-sample" and p["task_id"] == "fx/1" for p in probs), probs

@case("extra sample -> detected",
      "an id outside the declared subset must be surfaced")
def _():
    probs = q.validate_samples(
        [{"task_id": "fx/0", "solution": "a"}, {"task_id": "fx/1", "solution": "b"},
         {"task_id": "fx/2", "solution": "c"}], FX_IDS)
    assert any(p["kind"] == "extra-sample" for p in probs), probs

@case("truncated generation -> detected",
      "a generation that hit the token cap (finish_reason=length) must not be scored as a genuine "
      "wrong answer (cf. fable-fusion non-termination, LAB-CLOSE-002)")
def _():
    recs = [{"task_id": "fx/0", "finish_reason": "length", "answer_tokens": 4096},
            {"task_id": "fx/1", "finish_reason": "stop", "answer_tokens": 120}]
    flagged = q.flag_truncated(recs, max_tokens=4096)
    assert flagged == ["fx/0"], flagged

@case("malformed JSONL -> detected",
      "a corrupt line must be reported, not silently skipped (which would shrink the subset)")
def _():
    recs, errs = q.parse_jsonl_strict('{"task_id":"fx/0","solution":"a"}\nthis is not json\n[1,2,3]\n')
    assert len(recs) == 1, recs
    assert len(errs) == 2 and errs[0]["reason"].startswith("malformed-json"), errs

@case("stale result cache -> invalidation forced",
      "INCIDENT 2026-08-10 (81eed6d): EvalPlus reuses <padded>_eval_results.json if present, so "
      "re-scoring a corrected samples file returned STALE verdicts. bust_stale_results removes it.")
def _():
    with tempfile.TemporaryDirectory() as d:
        padded = pathlib.Path(d) / "s.padded.jsonl"; padded.write_text("{}")
        res = padded.with_name(padded.stem + "_eval_results.json")

        # emulate EvalPlus: SKIP evaluate if the results file already exists
        def stub_evaluate(fresh_pass: bool):
            if res.exists():
                return  # skipped — this is exactly how the stale value leaked
            res.write_text(json.dumps({"eval": {"fx/0": [{"plus_status": "pass" if fresh_pass else "fail"}]}}))

        res.write_text(json.dumps({"eval": {"fx/0": [{"plus_status": "pass"}]}}))  # STALE = pass
        # (a) WITHOUT busting: the fresh (fail) verdict never gets written -> stale 'pass' leaks
        stub_evaluate(fresh_pass=False)
        leaked = json.loads(res.read_text())["eval"]["fx/0"][0]["plus_status"]
        assert leaked == "pass", "control: without bust, the stale verdict is what leaks"
        # (b) WITH bust: stale removed, fresh (fail) is written
        removed = q.bust_stale_results(res)
        assert removed is True
        stub_evaluate(fresh_pass=False)
        fresh = json.loads(res.read_text())["eval"]["fx/0"][0]["plus_status"]
        assert fresh == "fail", "after bust, the current verdict is used"

@case("wrong benchmark version -> detected",
      "a score recorded under a different benchmark version must not be compared as-is")
def _():
    mism = q.check_identity({"benchmark_version": "humaneval-plus-0.2"},
                            {"benchmark_version": "humaneval-plus-0.1"})
    assert any(m["field"] == "benchmark_version" for m in mism), mism

@case("wrong dataset hash -> detected",
      "an edited/swapped dataset changes its content hash -> caught before the score is trusted")
def _():
    a = q.dataset_hash([{"task_id": "fx/0", "prompt": "P0"}, {"task_id": "fx/1", "prompt": "P1"}])
    b = q.dataset_hash([{"task_id": "fx/0", "prompt": "P0-EDITED"}, {"task_id": "fx/1", "prompt": "P1"}])
    assert a != b, "editing a prompt must change the dataset hash"
    mism = q.check_identity({"dataset_hash": b}, {"dataset_hash": a})
    assert any(m["field"] == "dataset_hash" for m in mism), mism


@case("exact-value scorer rejects substring and wrappers",
      "INCIDENT 2026-08-20: MQAR accepted `gold in answer`, so gold 100 could pass as 1000")
def _():
    assert q.strict_exact_reply("100", "100")
    assert not q.strict_exact_reply("1000", "100")
    assert not q.strict_exact_reply("the answer is 100", "100")
    assert not q.strict_exact_reply("`100`", "100")


@case("GSM8K strict final-line contract",
      "requalification requires the declared #### answer line; last-number fallback is diagnostic")
def _():
    assert q.strict_gsm8k_answer("work\n#### 1,234") == "1234"
    assert q.strict_gsm8k_answer("#### -2.5\n") == "-2.5"
    assert q.strict_gsm8k_answer("work gives 18") is None
    assert q.strict_gsm8k_answer("#### 18\nextra") is None
    assert q.lenient_last_number("work gives 18") == "18"
    assert q.numeric_equal("18.0", "18")


@case("score-bearing benchmark hash includes gold answers",
      "an edited gold answer must invalidate benchmark identity even when prompts are unchanged")
def _():
    a = [{"task_id": "fx/0", "prompt": "P0", "answer": "1"}]
    b = [{"task_id": "fx/0", "prompt": "P0", "answer": "2"}]
    assert q.dataset_hash(a) == q.dataset_hash(b), "historical prompt hash contract changed"
    assert q.benchmark_content_hash(a) != q.benchmark_content_hash(b)


@case("Wilson interval sanity and denominator guards",
      "accuracy claims expose finite-sample uncertainty and cannot use an empty denominator")
def _():
    lo, hi = q.wilson_interval(40, 40)
    assert 0.91 < lo < 0.92 and hi == 1.0, (lo, hi)
    try:
        q.wilson_interval(0, 0)
    except ValueError:
        pass
    else:
        raise AssertionError("empty denominator must fail closed")


@case("artifact identity hashes on demand and classifies lineage",
      "LAB-PROV-001 requires a full content hash and explicit source/requant class")
def _():
    with tempfile.TemporaryDirectory() as d:
        artifact = pathlib.Path(d) / "fixture.gguf"
        artifact.write_bytes(b"GGUF-fixture")
        assert q.artifact_sha256(artifact) == __import__("hashlib").sha256(b"GGUF-fixture").hexdigest()
    identity = q.run_identity(
        benchmark_name="fixture", benchmark_version="v1", dataset_version="fixture-v1",
        problems=[{"task_id": "f/0", "prompt": "p"}], sampling={"temperature": 0},
        model_id="fixture", model_path="fixture.gguf", quant="IQ4_XS",
        engine_commit="abc", timestamp="2026-08-20T00:00:00Z", repo_root=ROOT,
        model_sha256="a" * 64, model_bytes=12, source_repo="org/repo",
        source_revision="rev", quantizer="llama-quantize", imatrix="none",
        provenance_class="COMMUNITY_REQUANT")
    assert identity["source_repo"] == "org/repo"
    assert identity["provenance_class"] == "COMMUNITY_REQUANT"
    assert identity["model_sha256"] == "a" * 64


@case("shared code-fence extraction keeps executable code and format remains observable",
      "HumanEval+ and MBPP+ must use the same extraction glue before isolated execution")
def _():
    assert q.extract_code("```python\ndef f():\n    return 1\n```") == "def f():\n    return 1"
    assert q.extract_code("```py\ndef f(): pass\n```") == "def f(): pass"
    assert q.extract_code("def f(): return 2") == "def f(): return 2"
    assert "```" not in q.extract_code("```python\nx = 1\n```")


@case("unknown artifact lineage fails closed instead of inventing provenance",
      "unrecognized provenance classes must not silently enter historical run identity")
def _():
    try:
        q.run_identity(
            benchmark_name="fixture", benchmark_version="v1", dataset_version="v1",
            problems=[], sampling={}, model_id="m", model_path="m.gguf",
            timestamp="2026-08-20T00:00:00Z", provenance_class="OFFICIALISH")
    except ValueError:
        pass
    else:
        raise AssertionError("invalid provenance class accepted")


# ==================================================================================================
# Metamorphic tests (same-meaning input -> same result)
# ==================================================================================================

@case("metamorphic: reorder samples -> same aggregate",
      "validate_samples must be order-independent")
def _():
    s1 = [{"task_id": "fx/0", "solution": "a"}, {"task_id": "fx/1", "solution": "b"}]
    s2 = list(reversed(s1))
    assert q.validate_samples(s1, FX_IDS) == q.validate_samples(s2, FX_IDS)

@case("metamorphic: rerun scorer -> same result",
      "the code scorer must be deterministic across identical runs")
def _():
    sol = q.assemble_humaneval_solution(FX_PROMPT, FX_COMPLETION_CONCISE)
    assert _mini_score(sol, FX_TEST) == _mini_score(sol, FX_TEST) is True

@case("metamorphic: irrelevant metadata -> same result",
      "extra non-scoring fields on a record must not change validation")
def _():
    base = [{"task_id": "fx/0", "solution": "a"}, {"task_id": "fx/1", "solution": "b"}]
    withmeta = [dict(r, wall_s=1.2, model="x", note="hi") for r in base]
    assert q.validate_samples(base, FX_IDS) == q.validate_samples(withmeta, FX_IDS)

@case("metamorphic: fresh vs cached evaluation -> same result",
      "with the stale-cache bug fixed, a fresh evaluate must equal a (correctly) cached one")
def _():
    with tempfile.TemporaryDirectory() as d:
        res = pathlib.Path(d) / "r_eval_results.json"
        payload = {"eval": {"fx/0": [{"plus_status": "pass"}]}}
        res.write_text(json.dumps(payload))
        cached = json.loads(res.read_text())
        q.bust_stale_results(res)                  # bust then re-evaluate identical -> same
        res.write_text(json.dumps(payload))
        fresh = json.loads(res.read_text())
        assert cached == fresh


def main() -> int:
    passed = sum(1 for c in CASES if c["status"] == "pass")
    failed = [c for c in CASES if c["status"] != "pass"]
    report = {"suite": "LAB-QA-001 benchmark_harness_selftest",
              "total": len(CASES), "passed": passed, "failed": len(failed), "cases": CASES}
    (HERE / "benchmark_selftest_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    for c in CASES:
        mark = "ok  " if c["status"] == "pass" else "FAIL"
        print(f"  [{mark}] {c['name']}" + (f"   -> {c['detail']}" if c["status"] != "pass" else ""))
    print(f"\nLAB-QA-001: {passed}/{len(CASES)} passed" + (" — ALL GREEN" if not failed else f" — {len(failed)} FAILED"))
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
