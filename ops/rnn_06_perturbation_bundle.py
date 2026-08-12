#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Package the RNN-06 Controlled State-Load Perturbation + Historical Info train (mobile ZIP).

Bundles Backlog 1 (RNN-06B3, EXECUTED, QUALIFIED) + Backlog 2 (RNN-06C, EXECUTED, QUALIFIED) +
train protocol/handoff/git-evidence, with TRAIN_MANIFEST.json + SHA256SUMS.txt.

BUNDLE INVARIANT (enforced): archive payload files == manifest payload files == SHA256SUMS payload
files, excluding ONLY {TRAIN_MANIFEST.json, SHA256SUMS.txt} by explicit rule. Construction RAISES
if any unmanifested payload is present. No writestr-only payloads. Excludes prior-train BLOCKED
markers (superseded; see 06C SUPERSEDED_NOTE.md), weights, HF cache, venv, .git.
"""
import hashlib
import json
import os
import zipfile

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
ROOT = "RNN-06-state-load-perturbation-historical-info-train"
OUT_ZIP = os.path.join(REPO, "runs", "rnn", "RNN-06-PERTURBATION-TRAIN",
                       "RNN-06-state-load-perturbation-historical-info-train-audit-bundle.zip")
METADATA_NAMES = {f"{ROOT}/TRAIN_MANIFEST.json", f"{ROOT}/SHA256SUMS.txt"}

B3 = "runs/rnn/RNN-06B3-MAMBA-CONTROLLED-STATE-LOAD-PERTURBATION"
C6 = "runs/rnn/RNN-06C-MAMBA-HISTORICAL-INFO"
TR = "runs/rnn/RNN-06-PERTURBATION-TRAIN"

FILES = {
    f"{ROOT}/TRAIN_HANDOFF.md": f"{TR}/TRAIN_HANDOFF.md",
    f"{ROOT}/TRAIN_PROTOCOL.md": f"{TR}/TRAIN_PROTOCOL.md",
    f"{ROOT}/git_evidence/git_evidence.txt": f"{TR}/git_evidence.txt",
    # Backlog 1 — RNN-06B3
    f"{ROOT}/RNN-06B3/PRE_REGISTRATION.md": f"{B3}/PRE_REGISTRATION.md",
    f"{ROOT}/RNN-06B3/B3_CALIBRATION.json": f"{B3}/B3_CALIBRATION.json",
    f"{ROOT}/RNN-06B3/B3_CALIBRATION_DECISION.md": f"{B3}/B3_CALIBRATION_DECISION.md",
    f"{ROOT}/RNN-06B3/B3_QUALIFICATION_SPEC.json": f"{B3}/B3_QUALIFICATION_SPEC.json",
    f"{ROOT}/RNN-06B3/B3_STRESS_GRID.json": f"{B3}/B3_STRESS_GRID.json",
    f"{ROOT}/RNN-06B3/B3_RESULTS.json": f"{B3}/B3_RESULTS.json",
    f"{ROOT}/RNN-06B3/B3_CURVES.csv": f"{B3}/B3_CURVES.csv",
    f"{ROOT}/RNN-06B3/B3_DECISION.md": f"{B3}/B3_DECISION.md",
    f"{ROOT}/RNN-06B3/stdout_calibration.log": f"{B3}/stdout_calibration.log",
    f"{ROOT}/RNN-06B3/stdout_base.log": f"{B3}/stdout_base.log",
    # Backlog 2 — RNN-06C (executed)
    f"{ROOT}/RNN-06C/PRE_REGISTRATION.md": f"{C6}/PRE_REGISTRATION.md",
    f"{ROOT}/RNN-06C/HISTORICAL_INFO_SPEC.json": f"{C6}/HISTORICAL_INFO_SPEC.json",
    f"{ROOT}/RNN-06C/HISTORICAL_INFO_RESULTS.json": f"{C6}/HISTORICAL_INFO_RESULTS.json",
    f"{ROOT}/RNN-06C/HISTORICAL_INFO_DECISION.md": f"{C6}/HISTORICAL_INFO_DECISION.md",
    f"{ROOT}/RNN-06C/SUPERSEDED_NOTE.md": f"{C6}/SUPERSEDED_NOTE.md",
    f"{ROOT}/RNN-06C/stdout_base.log": f"{C6}/stdout_base.log",
    # runners / lib
    f"{ROOT}/ops/rnn_06b3_lib.py": "ops/rnn_06b3_lib.py",
    f"{ROOT}/ops/rnn_06b3_calibration.py": "ops/rnn_06b3_calibration.py",
    f"{ROOT}/ops/rnn_06b3_challenges.py": "ops/rnn_06b3_challenges.py",
    f"{ROOT}/ops/rnn_06b3_base.py": "ops/rnn_06b3_base.py",
    f"{ROOT}/ops/rnn_06c_challenges.py": "ops/rnn_06c_challenges.py",
    f"{ROOT}/ops/rnn_06c_base.py": "ops/rnn_06c_base.py",
    f"{ROOT}/ops/rnn_06_perturbation_bundle.py": "ops/rnn_06_perturbation_bundle.py",
}


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    manifest = {"train": ROOT,
                "backlog_1_RNN_06B3": "EXECUTED; STATE_LOAD_FORGETTING_PERTURBATION = QUALIFIED; "
                                      "TRANSITION_SHAPE = GRADED",
                "backlog_2_RNN_06C": "EXECUTED; HISTORICAL_STATE_INFORMATION = QUALIFIED",
                "pinned_chunk_size": 32, "payload_files": [], "skipped_missing": []}
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
        raise SystemExit(f"BUNDLE INVARIANT VIOLATION: unmanifested={unmanifested} "
                         f"missing={missing} sha_mismatch={sha_mismatch}")
    if not METADATA_NAMES.issubset(archive_names):
        raise SystemExit(f"metadata missing: {METADATA_NAMES - archive_names}")

    print(f"ZIP: {OUT_ZIP}")
    print(f"payload_files: {len(present)}  (+ TRAIN_MANIFEST.json + SHA256SUMS.txt)")
    print(f"skipped_missing: {manifest['skipped_missing']}")
    print("BUNDLE_INVARIANT: archive_payload == manifest_payload == sha256sums_payload  OK")
    print(f"ZIP_SHA256: {sha256_file(OUT_ZIP)}")
    print(f"ZIP_BYTES: {os.path.getsize(OUT_ZIP)}")


if __name__ == "__main__":
    main()
