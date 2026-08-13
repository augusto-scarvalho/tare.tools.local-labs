#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Package the RNN-07A-BRIDGE (NoLiMa semi-synthetic controlled bridge) consolidated audit bundle.

BUNDLE INVARIANT (enforced): archive payload == manifest payload == SHA256SUMS payload, excluding ONLY
{TRAIN_MANIFEST.json, SHA256SUMS.txt}. Excludes model weights, HF caches, venv, .git, and all external
datasets (LongBench v2 + NoLiMa; provenance is recorded in EXTERNAL_WORKLOAD_PROVENANCE.json).
"""
import hashlib
import json
import os
import zipfile

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
ROOT = "RNN-07A-BRIDGE-NOLIMA"
A = "runs/rnn/RNN-07A"
OUT_ZIP = os.path.join(REPO, A, "RNN-07A-REALISTIC-OPERATING-POINT-BRIDGE-audit-bundle.zip")
HANDOFF = ".harness/handoff/HANDOFF-rnn-07a-nolima-bridge.md"
METADATA_NAMES = {f"{ROOT}/TRAIN_MANIFEST.json", f"{ROOT}/SHA256SUMS.txt"}

FILES = {
    f"{ROOT}/HANDOFF.md": HANDOFF,
    f"{ROOT}/git_evidence.txt": f"{A}/git_evidence_bridge.txt",
    f"{ROOT}/AUDIT_RECONCILIATION.md": f"{A}/AUDIT_RECONCILIATION.md",
    f"{ROOT}/EXTERNAL_WORKLOAD_PROVENANCE.json": f"{A}/EXTERNAL_WORKLOAD_PROVENANCE.json",
    f"{ROOT}/BRIDGE_PRE_REGISTRATION.md": f"{A}/BRIDGE_PRE_REGISTRATION.md",
    f"{ROOT}/BRIDGE_DECISION.md": f"{A}/BRIDGE_DECISION.md",
    f"{ROOT}/BRIDGE_SHORT_RESULTS.json": f"{A}/BRIDGE_SHORT_RESULTS.json",
    f"{ROOT}/BRIDGE_LONG_RESULTS.json": f"{A}/BRIDGE_LONG_RESULTS.json",
    f"{ROOT}/stdout_bridge_short.log": f"{A}/stdout_bridge_short.log",
    f"{ROOT}/stdout_bridge_long.log": f"{A}/stdout_bridge_long.log",
    # parent natural-workload context (preserved, unchanged)
    f"{ROOT}/parent/RNN-07A_DECISION.md": f"{A}/RNN-07A_DECISION.md",
    f"{ROOT}/parent/SCOUT_SUMMARY.json": f"{A}/SCOUT_SUMMARY.json",
    # executed source
    f"{ROOT}/ops/rnn_07a_bridge_lib.py": "ops/rnn_07a_bridge_lib.py",
    f"{ROOT}/ops/rnn_07a_bridge_short.py": "ops/rnn_07a_bridge_short.py",
    f"{ROOT}/ops/rnn_07a_bridge_long.py": "ops/rnn_07a_bridge_long.py",
    f"{ROOT}/ops/rnn_07a_bridge_bundle.py": "ops/rnn_07a_bridge_bundle.py",
    f"{ROOT}/ops/rnn_07a_lib.py": "ops/rnn_07a_lib.py",
    f"{ROOT}/ops/rnn_06t_lib.py": "ops/rnn_06t_lib.py",
}


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def gv(rel, key, default=None):
    try:
        return json.load(open(os.path.join(REPO, rel))).get(key, default)
    except Exception:
        return default


def main():
    manifest = {
        "train": ROOT,
        "role": "NoLiMa SEMI_SYNTHETIC_CONTROLLED_BRIDGE (never a natural-workload qualification)",
        "official_checkpoint": "state-spaces/mamba2-1.3b",
        "revision": "c5b59d00ec85d313adea86a08cad2a43c962dd3b",
        "external_datasets_excluded": ["THUDM/LongBench-v2", "amodaresi/NoLiMa @ 378115b1"],
        "decision_states": {
            "BRIDGE_SHORT_CONTEXT_COMPETENCE": gv(f"{A}/BRIDGE_SHORT_RESULTS.json", "BRIDGE_SHORT_CONTEXT_COMPETENCE"),
            "BRIDGE_LONG_CONTEXT_DEGRADATION": gv(f"{A}/BRIDGE_LONG_RESULTS.json", "BRIDGE_LONG_CONTEXT_DEGRADATION"),
            "BRIDGE_HISTORICAL_RECOVERY_SIGNAL": gv(f"{A}/BRIDGE_LONG_RESULTS.json", "BRIDGE_HISTORICAL_RECOVERY_SIGNAL"),
            "BRIDGE_ADAPTIVE_SELECTION_SIGNAL": gv(f"{A}/BRIDGE_LONG_RESULTS.json", "BRIDGE_ADAPTIVE_SELECTION_SIGNAL"),
            "PARENT_REALISTIC_TASK_COMPETENCE": gv(f"{A}/SCOUT_SUMMARY.json", "REALISTIC_TASK_COMPETENCE"),
            "PARENT_REALISTIC_FORGETTING_OPERATING_POINT": gv(f"{A}/SCOUT_SUMMARY.json", "REALISTIC_FORGETTING_OPERATING_POINT"),
        },
        "nothing_pushed": True, "payload_files": [], "skipped_missing": []}

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
