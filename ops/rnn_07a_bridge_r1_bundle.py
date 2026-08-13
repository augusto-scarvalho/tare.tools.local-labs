#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Package the RNN-07A-BRIDGE-R1 true-in-run recovery consolidated audit bundle.

BUNDLE INVARIANT (enforced): archive payload == manifest payload == SHA256SUMS payload, excluding ONLY
{TRAIN_MANIFEST.json, SHA256SUMS.txt}. Excludes weights/HF caches/venv/.git and all external datasets.
"""
import hashlib
import json
import os
import zipfile

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
ROOT = "RNN-07A-BRIDGE-R1-INRUN-RECOVERY"
A = "runs/rnn/RNN-07A"
OUT_ZIP = os.path.join(REPO, A, "RNN-07A-BRIDGE-R1-INRUN-RECOVERY-audit-bundle.zip")
HANDOFF = ".harness/handoff/HANDOFF-rnn-07a-bridge-r1-inrun-recovery.md"
METADATA_NAMES = {f"{ROOT}/TRAIN_MANIFEST.json", f"{ROOT}/SHA256SUMS.txt"}

FILES = {
    f"{ROOT}/HANDOFF.md": HANDOFF,
    f"{ROOT}/git_evidence.txt": f"{A}/git_evidence_r1.txt",
    f"{ROOT}/AUDIT_RECONCILIATION_BRIDGE.md": f"{A}/AUDIT_RECONCILIATION_BRIDGE.md",
    f"{ROOT}/R1_PRE_REGISTRATION.md": f"{A}/R1_PRE_REGISTRATION.md",
    f"{ROOT}/EXTERNAL_WORKLOAD_PROVENANCE.json": f"{A}/EXTERNAL_WORKLOAD_PROVENANCE.json",
    f"{ROOT}/R1_RESULTS.json": f"{A}/R1_RESULTS.json",
    f"{ROOT}/R1_REPLAY_INPROCESS.json": f"{A}/R1_REPLAY_INPROCESS.json",
    f"{ROOT}/R1_DECISION.md": f"{A}/R1_DECISION.md",
    f"{ROOT}/stdout_r1.log": f"{A}/stdout_r1.log",
    f"{ROOT}/stdout_r1_replay.log": f"{A}/stdout_r1_replay.log",
    # parent bridge context (preserved, unchanged)
    f"{ROOT}/parent/BRIDGE_DECISION.md": f"{A}/BRIDGE_DECISION.md",
    f"{ROOT}/parent/BRIDGE_LONG_RESULTS.json": f"{A}/BRIDGE_LONG_RESULTS.json",
    # executed source
    f"{ROOT}/ops/rnn_07a_bridge_r1.py": "ops/rnn_07a_bridge_r1.py",
    f"{ROOT}/ops/rnn_07a_bridge_r1_replay.py": "ops/rnn_07a_bridge_r1_replay.py",
    f"{ROOT}/ops/rnn_07a_bridge_r1_bundle.py": "ops/rnn_07a_bridge_r1_bundle.py",
    f"{ROOT}/ops/rnn_07a_bridge_lib.py": "ops/rnn_07a_bridge_lib.py",
    f"{ROOT}/ops/rnn_07a_lib.py": "ops/rnn_07a_lib.py",
    f"{ROOT}/ops/rnn_06t_lib.py": "ops/rnn_06t_lib.py",
}


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def gv(rel, *keys):
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
        "role": "TRUE in-run historical recovery requalification on NoLiMa SEMI_SYNTHETIC_CONTROLLED_BRIDGE",
        "official_checkpoint": "state-spaces/mamba2-1.3b",
        "revision": "c5b59d00ec85d313adea86a08cad2a43c962dd3b",
        "external_datasets_excluded": ["amodaresi/NoLiMa @ 378115b1", "THUDM/LongBench-v2"],
        "capture_semantics": gv(f"{A}/R1_RESULTS.json", "capture_semantics"),
        "decision_states": {
            "HISTORICAL_INFORMATION_PRESENCE_R1": gv(f"{A}/R1_RESULTS.json", "HISTORICAL_INFORMATION_PRESENCE_R1"),
            "TRUE_IN_RUN_COARSE_HISTORICAL_RECOVERY_R1": gv(f"{A}/R1_RESULTS.json", "TRUE_IN_RUN_COARSE_HISTORICAL_RECOVERY_R1"),
            "TRUE_IN_RUN_MAX_CONFIDENCE_R1": gv(f"{A}/R1_RESULTS.json", "TRUE_IN_RUN_MAX_CONFIDENCE_R1"),
            "R1QualificationSetSha256": gv(f"{A}/R1_RESULTS.json", "fresh_set_identity", "R1QualificationSetSha256"),
            "disjoint_from_historical_recovery": gv(f"{A}/R1_RESULTS.json", "fresh_set_identity", "disjoint_from_historical_recovery"),
            "in_process_same_shape_replay_all_match": gv(f"{A}/R1_REPLAY_INPROCESS.json", "IN_PROCESS_SAME_SHAPE_ALL_MATCH"),
            "cross_process_match_to_capture": gv(f"{A}/R1_REPLAY_INPROCESS.json", "CROSS_PROCESS_MATCH_TO_CAPTURE_ALL"),
            "n_recovery": gv(f"{A}/R1_RESULTS.json", "n_recovery"),
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
