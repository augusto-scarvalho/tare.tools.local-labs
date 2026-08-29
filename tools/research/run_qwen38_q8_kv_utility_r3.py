#!/usr/bin/env python3
"""Gold-blind semantic rescore of frozen Qwen3.8 F16/Q8 KV outputs."""
from __future__ import annotations

import argparse
import inspect
import json
import pathlib
import random
import statistics
import sys
import time
from typing import Any

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tools.analysis.a2_stats import numeric_equal
from tools.analysis.experiment_provenance import (
    build_provenance, canonical_json_sha256, provenance_complete, sha256_file,
)
from tools.analysis.final_numeric_answer_v2 import extract_concluded_numeric_for_question

TASK_ID = "BACKLOG-QWEN38-Q8-KV-UTILITY-03"
SOURCE = ROOT / "runs/research/BACKLOG-QWEN38-Q8-KV-UTILITY-02"
BOOTSTRAP_SEED = 2026082812
REPLICATES = 20_000
HOST_INPUTS = {
    "config/research_backlog_admissions/BACKLOG-QWEN38-Q8-KV-UTILITY-03.json": "88d49662d735fff3f8830e5dfbbb017dfbcaffcd4064e74409b62551972f7a5d",
    "runs/research/BACKLOG-QWEN38-Q8-KV-UTILITY-03/PRE_REGISTRATION.md": "fcc94b5aaf1d1b2e7f99fea0b9edc5ca0fa8dd0afa4ad604a64579d361a18adc",
    "runs/research/BACKLOG-QWEN38-Q8-KV-UTILITY-02/raw/receipt.json": "c4653adbc30ba652ef64d60130f416b47667e14c97ca726d5b55e309e7af2b20",
    "runs/research/BACKLOG-QWEN38-Q8-KV-UTILITY-02/raw/samples.jsonl": "3801db8e8c45e30bbfa417d03da23a3db0131caa1039b0ea90fd9a59e4090196",
    "runs/research/BACKLOG-QWEN38-Q8-KV-UTILITY-02/raw/actual_scores.json": "1019f48c86e73682c461776e9452371e692a157c93d8564415131342ecdaeef6",
    "runs/research/BACKLOG-QWEN38-Q8-KV-UTILITY-02/REVIEW.json": "d20b52be6597d9112b6daf7593d45b5db7cdd2a9ce10b270358445d0b471f49c",
    "runs/research/BACKLOG-QWEN38-Q8-KV-UTILITY-02/raw/hardware_metrics.json": "47911fb9ff1638928303e54de1f9112f6abe664d7f929faf4c90b78ea1d3d789",
    "runs/research/BACKLOG-QWEN38-Q8-KV-UTILITY-02/raw/effective_route.json": "82e8ef87d15986a2462d3995b72efd4fcf048be7052bfe098dc49d2beee3a73e",
    "runs/research/BACKLOG-QWEN38-Q8-KV-UTILITY-02/raw/recovery_state.json": "115164d321be023b4fba6ec363448749226b31c46594492546bb0744dac6707f",
    "runs/research/BACKLOG-QWEN38-Q8-KV-UTILITY-02/raw/service_identity.json": "cd50bcaca5eb0c15a656b24293bd49c3c04f41e6f949e30f925bd887eac6f864",
    "runs/research/BACKLOG-QWEN38-Q8-KV-UTILITY-02/raw/treatment_controls.json": "e4501c41383bd2add941ce4c1c80218dd602a3eb7b1393e29c3f306d548aa614",
    "workloads/gsm8k.jsonl": "68a72276898a45dacb893154477621ff3d05ae7e043a10820644b691d8b63d77",
    "tools/analysis/final_numeric_answer_v2.py": "51bb1bb3af967eb1d6638f2816d89d90c7251f533f17274ac33efd3e25f7cf35",
    "tests/fixtures/final_numeric_answer_v2_cases.json": "5e6494486f9caec3b40e072896e7b574911f188d4f11db4d6d90056565641184",
    "tests/test_final_numeric_answer_v2.py": "876ff43e96f6320f362e50abef83a877bd43ce2d1c1e97171786a9105c50683a",
}


def write_json(path: pathlib.Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")


def validate_fixtures() -> dict[str, Any]:
    cases = json.loads((ROOT / "tests/fixtures/final_numeric_answer_v2_cases.json").read_text(encoding="utf-8"))
    rows = []
    for case in cases:
        got = extract_concluded_numeric_for_question(case["question"], case["text"])
        rows.append({"id":case["id"], "expected":case["expected"], "actual":got.value,
                     "method":got.method, "pass":got.value == case["expected"] and got.method == case["method"]})
    return {"cases":len(rows), "passed":sum(r["pass"] for r in rows), "rows":rows}


def bootstrap(rows: list[dict[str, Any]]) -> dict[str, Any]:
    arms = {arm:{r["task_id"]:int(r["rescored_correct"]) for r in rows if r["arm"] == arm} for arm in ("f16", "q8")}
    if len(arms["f16"]) != 128 or set(arms["f16"]) != set(arms["q8"]):
        raise ValueError("paired coverage is incomplete")
    diffs = [arms["q8"][task] - arms["f16"][task] for task in sorted(arms["f16"])]
    rng = random.Random(BOOTSTRAP_SEED)
    estimates = sorted(sum(diffs[rng.randrange(128)] for _ in range(128)) / 128 for _ in range(REPLICATES))
    return {"point":statistics.mean(diffs), "replicates":REPLICATES, "seed":BOOTSTRAP_SEED,
            "lower_95":estimates[int(.025 * REPLICATES)], "upper_95":estimates[int(.975 * REPLICATES)],
            "q8_only_correct":sum(x == 1 for x in diffs), "f16_only_correct":sum(x == -1 for x in diffs)}


def execute(outdir: pathlib.Path) -> dict[str, Any]:
    raw = outdir / "raw"
    if any(raw.iterdir()): raise RuntimeError("raw directory is not empty")
    started, mono = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), time.monotonic()
    ledger, inputs = {}, []
    for relative, expected in HOST_INPUTS.items():
        path, actual = ROOT / relative, sha256_file(ROOT / relative)
        if actual != expected: raise ValueError(f"frozen input mismatch: {relative}: {actual} != {expected}")
        ledger[relative] = {"bytes":path.stat().st_size, "sha256":actual}; inputs.append(path)
    fixture = validate_fixtures()
    if fixture["passed"] != fixture["cases"]: raise RuntimeError("external fixture validation failed")
    prompts = {row["task_id"]:row for row in (json.loads(line) for line in (ROOT / "workloads/gsm8k.jsonl").read_text(encoding="utf-8").splitlines() if line.strip())}
    source_rows = [json.loads(line) for line in (SOURCE / "raw/samples.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    rows, seen = [], set()
    for row in source_rows:
        key = (row["arm"], row["task_id"])
        if key in seen: raise ValueError(f"duplicate row: {key}")
        seen.add(key)
        prompt = prompts[row["task_id"]]["prompt"]
        got = extract_concluded_numeric_for_question(prompt, row["content"])
        rows.append({"arm":row["arm"], "task_id":row["task_id"], "gold":row["gold"],
                     "original_extracted":row["extracted"], "original_correct":bool(row["correct"]),
                     "rescored_extracted":got.value, "rescored_method":got.method,
                     "rescored_correct":bool(numeric_equal(got.value, row["gold"])),
                     "http_status":row["http_status"], "new_tokens":row["predicted_n"], "throughput_tps":row["throughput_tps"]})
    if len(rows) != 256: raise ValueError(f"expected 256 rows, got {len(rows)}")
    with (raw / "rescored_samples.jsonl").open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows: stream.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    water = [r for r in rows if r["task_id"] == "gsm8k/111"]
    regressions = {"cases":len(water), "passed":sum(r["rescored_extracted"] is None for r in water),
                   "rows":[{"arm":r["arm"], "task_id":r["task_id"], "expected":None,
                            "actual":r["rescored_extracted"], "pass":r["rescored_extracted"] is None} for r in water]}
    if regressions["cases"] != 2 or regressions["passed"] != 2: raise RuntimeError("retained Q8 regression validation failed")
    comparison = bootstrap(rows); f16 = sum(r["rescored_correct"] for r in rows if r["arm"] == "f16"); q8 = sum(r["rescored_correct"] for r in rows if r["arm"] == "q8")
    old = json.loads((SOURCE / "raw/actual_scores.json").read_text(encoding="utf-8"))
    signature = list(inspect.signature(extract_concluded_numeric_for_question).parameters)
    metrics = {"q8_r2_sources_verified":True, "external_fixture_cases":fixture["cases"],
        "external_fixture_pass_rate":fixture["passed"] / fixture["cases"],
        "retained_regression_pass_rate":regressions["passed"] / regressions["cases"],
        "rescored_requests":len(rows), "f16_correct":f16, "q8_correct":q8, "f16_accuracy":f16/128,
        "q8_accuracy":q8/128, "f16_minus_q8_accuracy":(f16-q8)/128,
        "paired_q8_minus_f16_accuracy":comparison,
        "paired_bootstrap_95ci_lower_q8_minus_f16_accuracy":comparison["lower_95"],
        "f16_extraction_coverage":sum(r["rescored_extracted"] is not None for r in rows if r["arm"] == "f16")/128,
        "q8_extraction_coverage":sum(r["rescored_extracted"] is not None for r in rows if r["arm"] == "q8")/128,
        "changed_labels":sum(r["original_correct"] != r["rescored_correct"] for r in rows),
        "vram_saving_mib":old["vram_saving_mib"], "q8_vs_f16_throughput_ratio":old["q8_vs_f16_throughput_ratio"],
        "service_and_embedding_restored":old["service_and_embedding_restored"],
        "explicit_cache_controls_verified":old["explicit_cache_controls_verified"],
        "scorer_does_not_receive_gold":signature == ["question", "text"]}
    defs = {"source_integrity":("q8_r2_sources_verified","eq",True), "fixture_validation":("external_fixture_pass_rate","eq",1.0),
        "retained_regressions":("retained_regression_pass_rate","eq",1.0), "request_coverage":("rescored_requests","eq",256),
        "utility_noninferiority":("paired_bootstrap_95ci_lower_q8_minus_f16_accuracy","gt",-.05),
        "quality_regression":("f16_minus_q8_accuracy","le",.03), "physical_memory_saving":("vram_saving_mib","ge",500),
        "throughput_non_regression":("q8_vs_f16_throughput_ratio","ge",.95),
        "service_recovery":("service_and_embedding_restored","eq",True), "scorer_blinding":("scorer_does_not_receive_gold","eq",True)}
    gates = {}
    for gate,(metric,op,threshold) in defs.items():
        actual=metrics[metric]; passed=actual==threshold if op=="eq" else actual>=threshold if op=="ge" else actual>threshold if op=="gt" else actual<=threshold
        gates[gate]={"metric":metric,"operator":op,"threshold":threshold,"actual":actual,"pass":passed}
    write_json(raw / "actual_scores.json", metrics); write_json(raw / "external_fixture_validation.json", fixture)
    write_json(raw / "retained_regression_validation.json", regressions)
    write_json(raw / "independent_evaluation.json", {"comparison":comparison,"f16_correct":f16,"q8_correct":q8})
    write_json(raw / "paired_baseline.json", {"baseline":"f16","treatment":"q8","paired_tasks":128,"comparison":comparison})
    write_json(raw / "scorer_hashes.json", {"implementation":ledger["tools/analysis/final_numeric_answer_v2.py"],
               "fixtures":ledger["tests/fixtures/final_numeric_answer_v2_cases.json"],"signature":signature})
    source_receipt=json.loads((SOURCE/"raw/receipt.json").read_text(encoding="utf-8"))
    write_json(raw / "source_execution_receipt.json", {"task_id":SOURCE.name,"receipt_sha256":sha256_file(SOURCE/"raw/receipt.json"),"receipt_fingerprint":source_receipt["receipt_fingerprint"]})
    for name in ("hardware_metrics","effective_route","recovery_state","service_identity","treatment_controls"):
        write_json(raw / f"{name}.json", {"imported_from":f"{SOURCE.name}/raw/{name}.json","sha256":ledger[f"runs/research/{SOURCE.name}/raw/{name}.json"]})
    evidence={"acceptance_gates":"raw/receipt.json","actual_scores":"raw/actual_scores.json","effective_route":"raw/effective_route.json",
        "external_fixture_validation":"raw/external_fixture_validation.json","hardware_metrics":"raw/hardware_metrics.json",
        "independent_evaluation":"raw/independent_evaluation.json","paired_baseline":"raw/paired_baseline.json","provenance":"raw/receipt.json",
        "raw_samples":"raw/rescored_samples.jsonl","receipt_fingerprint":"raw/receipt.json","recovery_state":"raw/recovery_state.json",
        "scorer_hashes":"raw/scorer_hashes.json","service_identity":"raw/service_identity.json",
        "source_execution_receipt":"raw/source_execution_receipt.json","treatment_controls":"raw/treatment_controls.json"}
    evidence_files=sorted(p for p in raw.rglob("*") if p.is_file())
    provenance=build_provenance(script_path=pathlib.Path(__file__).resolve(),started_at_utc=started,started_monotonic=mono,
        input_paths=[*inputs,*evidence_files],packages=[],runtime={"execution_mode":"offline_semantic_rescore","new_inference":False,"service_mutation":False})
    complete,errors=provenance_complete(provenance)
    if not complete: raise RuntimeError(f"incomplete provenance: {errors}")
    receipt={"schema":"local-labs-backlog-receipt-v1","task_id":TASK_ID,"provenance":provenance,"provenance_complete":True,"gates":gates,"evidence":evidence}
    receipt["receipt_fingerprint"]=canonical_json_sha256(receipt); write_json(raw/"receipt.json",receipt)
    failed=[name for name,gate in gates.items() if not gate["pass"]]
    claim="QWEN38_Q8_KV_UTILITY_NONINFERIOR_R3" if not failed else "QWEN38_Q8_KV_UTILITY_NOT_NONINFERIOR_R3"
    (outdir/"RESULT.md").write_text(f"# {TASK_ID} result\n\n`{claim}` pending independent review.\n\n"
        f"F16/Q8 rescored `{f16}/128` and `{q8}/128`; paired delta `{comparison['point']:.6f}`, 95% CI "
        f"`[{comparison['lower_95']:.6f}, {comparison['upper_95']:.6f}]`. Imported physical saving `{metrics['vram_saving_mib']:.1f}` MiB "
        f"at `{metrics['q8_vs_f16_throughput_ratio']:.4f}x` throughput. Failed gates: `{', '.join(failed) if failed else 'none'}`.\n",encoding="utf-8",newline="\n")
    return receipt


def main() -> int:
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument("--outdir",type=pathlib.Path,default=ROOT/"runs/research"/TASK_ID); parser.add_argument("--selfcheck",action="store_true"); args=parser.parse_args()
    if args.selfcheck:
        fixture=validate_fixtures(); assert fixture["cases"]>=15 and fixture["passed"]==fixture["cases"]; return 0
    receipt=execute(args.outdir.resolve()); print(json.dumps(receipt["gates"],separators=(",",":")),flush=True); return 0


if __name__ == "__main__": raise SystemExit(main())
