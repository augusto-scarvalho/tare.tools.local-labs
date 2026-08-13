#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Package the RNN-07A realistic operating-point discovery train (+ RNN-06T2-E1 economics closure).

BUNDLE INVARIANT (enforced): archive payload == manifest payload == SHA256SUMS payload, excluding ONLY
{TRAIN_MANIFEST.json, SHA256SUMS.txt}. Raises on any unmanifested payload. Excludes model weights, HF
caches, venv, compiled caches, .git, and the LongBench v2 data (dataset).
"""
import hashlib
import json
import os
import zipfile

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
ROOT = "RNN-07A-REALISTIC-OPERATING-POINT"
A = "runs/rnn/RNN-07A"
E1 = "runs/rnn/RNN-06T2-E1"
OUT_ZIP = os.path.join(REPO, A, "RNN-07A-REALISTIC-OPERATING-POINT-audit-bundle.zip")
HANDOFF = ".harness/handoff/HANDOFF-rnn-07a-realistic-operating-point.md"
METADATA_NAMES = {f"{ROOT}/TRAIN_MANIFEST.json", f"{ROOT}/SHA256SUMS.txt"}

FILES = {
    f"{ROOT}/HANDOFF.md": HANDOFF,
    f"{ROOT}/git_evidence.txt": f"{A}/git_evidence.txt",
    # ---- RNN-06T2-E1 economics semantic closure ----
    f"{ROOT}/E1_economics/E1_PRE_REGISTRATION.md": f"{E1}/E1_PRE_REGISTRATION.md",
    f"{ROOT}/E1_economics/E1_ECON_run0.json": f"{E1}/E1_ECON_run0.json",
    f"{ROOT}/E1_economics/E1_ECON_run1.json": f"{E1}/E1_ECON_run1.json",
    f"{ROOT}/E1_economics/E1_ECONOMICS.json": f"{E1}/E1_ECONOMICS.json",
    f"{ROOT}/E1_economics/E1_DECISION.md": f"{E1}/E1_DECISION.md",
    # ---- RNN-07A discovery ----
    f"{ROOT}/07A/RNN-07A_PRE_REGISTRATION.md": f"{A}/RNN-07A_PRE_REGISTRATION.md",
    f"{ROOT}/07A/SCOUT_SUMMARY.json": f"{A}/SCOUT_SUMMARY.json",
    f"{ROOT}/07A/SCOUT_16K.json": f"{A}/SCOUT_16K.json",
    f"{ROOT}/07A/SCOUT_32K.json": f"{A}/SCOUT_32K.json",
    f"{ROOT}/07A/RECOVERY_RESULTS.json": f"{A}/RECOVERY_RESULTS.json",
    f"{ROOT}/07A/RNN-07A_DECISION.md": f"{A}/RNN-07A_DECISION.md",
    f"{ROOT}/07A/stdout_scout.log": f"{A}/stdout_scout.log",
    # ---- executed source (runners + deps) ----
    f"{ROOT}/ops/rnn_06t2_e1_econ.py": "ops/rnn_06t2_e1_econ.py",
    f"{ROOT}/ops/rnn_06t2_e1_decide.py": "ops/rnn_06t2_e1_decide.py",
    f"{ROOT}/ops/rnn_07a_lib.py": "ops/rnn_07a_lib.py",
    f"{ROOT}/ops/rnn_07a_scout.py": "ops/rnn_07a_scout.py",
    f"{ROOT}/ops/rnn_07a_recovery.py": "ops/rnn_07a_recovery.py",
    f"{ROOT}/ops/rnn_07a_bundle.py": "ops/rnn_07a_bundle.py",
    f"{ROOT}/ops/rnn_06t_lib.py": "ops/rnn_06t_lib.py",
    f"{ROOT}/ops/rnn_06d_lib.py": "ops/rnn_06d_lib.py",
}


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def gv(rel, key):
    try:
        return json.load(open(os.path.join(REPO, rel))).get(key)
    except Exception:
        return None


def gvn(rel, *keys):
    try:
        d = json.load(open(os.path.join(REPO, rel)))
        for k in keys:
            d = d[k]
        return d
    except Exception:
        return None


def main():
    manifest = {
        "train": ROOT,
        "official_checkpoint": "state-spaces/mamba2-1.3b",
        "revision": "c5b59d00ec85d313adea86a08cad2a43c962dd3b",
        "fast_path": "mamba_ssm 2.2.4 fast path (chunk_scan_combined prefill + selective_state_update decode)",
        "workload": "LongBench v2 (THUDM/LongBench-v2) 4-way MC — dataset EXCLUDED from bundle",
        "decision_states": {
            # RNN-06T2-E1
            "ECONOMICS_OUTPUT_COMPARABILITY_E1": gvn(f"{E1}/E1_ECONOMICS.json", "MINTS", "ECONOMICS_OUTPUT_COMPARABILITY_E1"),
            "MARGINAL_RECOVERY_UTILITY_ON_STEP_PATH_E1": gvn(f"{E1}/E1_ECONOMICS.json", "MINTS", "MARGINAL_RECOVERY_UTILITY_ON_STEP_PATH_E1"),
            "RECOVERY_PATH_VS_FUSED_BASELINE_E1": gvn(f"{E1}/E1_ECONOMICS.json", "MINTS", "RECOVERY_PATH_VS_FUSED_BASELINE_E1", "verdict"),
            "GENERAL_END_TO_END_DEPLOYMENT_UTILITY": gvn(f"{E1}/E1_ECONOMICS.json", "MINTS", "GENERAL_END_TO_END_DEPLOYMENT_UTILITY"),
            # RNN-07A
            "REALISTIC_TASK_COMPETENCE": gv(f"{A}/SCOUT_SUMMARY.json", "REALISTIC_TASK_COMPETENCE"),
            "REALISTIC_FORGETTING_OPERATING_POINT": gv(f"{A}/SCOUT_SUMMARY.json", "REALISTIC_FORGETTING_OPERATING_POINT"),
            "REALISTIC_HISTORICAL_RECOVERY_SIGNAL": gv(f"{A}/RECOVERY_RESULTS.json", "REALISTIC_HISTORICAL_RECOVERY_SIGNAL"),
            "REALISTIC_ADAPTIVE_SELECTION_SIGNAL": gv(f"{A}/RECOVERY_RESULTS.json", "REALISTIC_ADAPTIVE_SELECTION_SIGNAL"),
        },
        "nothing_pushed": True,
        "payload_files": [], "skipped_missing": []}

    present, sha_lines = [], []
    for arc, rel in FILES.items():
        ap = os.path.join(REPO, rel)
        if not os.path.isfile(ap):
            manifest["skipped_missing"].append(rel)
            continue
        digest = sha256_file(ap)
        manifest["payload_files"].append({"arcname": arc, "repo_path": rel,
                                          "size": os.path.getsize(ap), "sha256": digest})
        sha_lines.append(f"{digest}  {arc}")
        present.append((ap, arc))

    manifest_payload = {e["arcname"] for e in manifest["payload_files"]}
    sha_payload = {ln.split("  ", 1)[1] for ln in sha_lines}
    with zipfile.ZipFile(OUT_ZIP, "w", zipfile.ZIP_DEFLATED) as z:
        for ap, arc in present:
            z.write(ap, arc)
        z.writestr(f"{ROOT}/TRAIN_MANIFEST.json", json.dumps(manifest, indent=2).encode())
        z.writestr(f"{ROOT}/SHA256SUMS.txt", ("\n".join(sha_lines) + "\n").encode())

    with zipfile.ZipFile(OUT_ZIP) as z:
        archive_names = set(z.namelist())
    archive_payload = archive_names - METADATA_NAMES
    unmanifested = archive_payload - manifest_payload
    missing = manifest_payload - archive_payload
    sha_mismatch = manifest_payload ^ sha_payload
    if unmanifested or missing or sha_mismatch:
        raise SystemExit(f"BUNDLE INVARIANT VIOLATION: unmanifested={unmanifested} missing={missing} "
                         f"sha_mismatch={sha_mismatch}")
    if not METADATA_NAMES.issubset(archive_names):
        raise SystemExit(f"metadata missing: {METADATA_NAMES - archive_names}")
    print(f"ZIP: {OUT_ZIP}")
    print(f"payload_files: {len(present)} (+ TRAIN_MANIFEST.json + SHA256SUMS.txt)")
    print(f"skipped_missing: {manifest['skipped_missing']}")
    print(f"decision_states: {json.dumps(manifest['decision_states'])}")
    print("BUNDLE_INVARIANT: archive_payload == manifest_payload == sha256sums_payload  OK")
    print(f"ZIP_SHA256: {sha256_file(OUT_ZIP)}")
    print(f"ZIP_BYTES: {os.path.getsize(OUT_ZIP)}")


if __name__ == "__main__":
    main()
