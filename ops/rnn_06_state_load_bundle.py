#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Package the RNN-06 Fixed-Length State Load + Historical Info train (mobile ZIP).

Bundles Backlog 1 (RNN-06B2, EXECUTED) + Backlog 2 (RNN-06C, BLOCKED_BY_06B2 markers) + train
protocol/handoff/git-evidence, with TRAIN_MANIFEST.json + SHA256SUMS.txt.

BUNDLE INVARIANT (fix carried from the prior train's §22): archive payload files == manifest
payload files == SHA256SUMS payload files, excluding ONLY {TRAIN_MANIFEST.json, SHA256SUMS.txt}
by explicit rule. Construction RAISES if any unmanifested payload is present. No writestr-only
payloads: every payload is a real on-disk manifested file (incl. 06C BLOCKED artifacts).

Excludes weights / HF cache / venv / .git. Prints the ZIP's own SHA-256. No GPU/model.
"""
import hashlib
import json
import os
import zipfile

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
ROOT = "RNN-06-state-load-historical-info-train"
OUT_ZIP = os.path.join(REPO, "runs", "rnn", "RNN-06-STATE-LOAD-TRAIN",
                       "RNN-06-state-load-historical-info-train-audit-bundle.zip")
METADATA_NAMES = {f"{ROOT}/TRAIN_MANIFEST.json", f"{ROOT}/SHA256SUMS.txt"}

B2 = "runs/rnn/RNN-06B2-MAMBA-FIXED-LENGTH-STATE-LOAD"
C6 = "runs/rnn/RNN-06C-MAMBA-HISTORICAL-INFO"
TR = "runs/rnn/RNN-06-STATE-LOAD-TRAIN"

FILES = {
    f"{ROOT}/TRAIN_HANDOFF.md": f"{TR}/TRAIN_HANDOFF.md",
    f"{ROOT}/TRAIN_PROTOCOL.md": f"{TR}/TRAIN_PROTOCOL.md",
    f"{ROOT}/git_evidence/git_evidence.txt": f"{TR}/git_evidence.txt",
    # Backlog 1 — RNN-06B2
    f"{ROOT}/RNN-06B2-MAMBA-FIXED-LENGTH-STATE-LOAD/PRE_REGISTRATION.md": f"{B2}/PRE_REGISTRATION.md",
    f"{ROOT}/RNN-06B2-MAMBA-FIXED-LENGTH-STATE-LOAD/B2_CALIBRATION.json": f"{B2}/B2_CALIBRATION.json",
    f"{ROOT}/RNN-06B2-MAMBA-FIXED-LENGTH-STATE-LOAD/B2_CALIBRATION_DECISION.md": f"{B2}/B2_CALIBRATION_DECISION.md",
    f"{ROOT}/RNN-06B2-MAMBA-FIXED-LENGTH-STATE-LOAD/B2_QUALIFICATION_SPEC.json": f"{B2}/B2_QUALIFICATION_SPEC.json",
    f"{ROOT}/RNN-06B2-MAMBA-FIXED-LENGTH-STATE-LOAD/B2_STRESS_GRID.json": f"{B2}/B2_STRESS_GRID.json",
    f"{ROOT}/RNN-06B2-MAMBA-FIXED-LENGTH-STATE-LOAD/B2_RESULTS.json": f"{B2}/B2_RESULTS.json",
    f"{ROOT}/RNN-06B2-MAMBA-FIXED-LENGTH-STATE-LOAD/B2_CURVES.csv": f"{B2}/B2_CURVES.csv",
    f"{ROOT}/RNN-06B2-MAMBA-FIXED-LENGTH-STATE-LOAD/B2_DECISION.md": f"{B2}/B2_DECISION.md",
    f"{ROOT}/RNN-06B2-MAMBA-FIXED-LENGTH-STATE-LOAD/stdout_calibration.log": f"{B2}/stdout_calibration.log",
    f"{ROOT}/RNN-06B2-MAMBA-FIXED-LENGTH-STATE-LOAD/stdout_base.log": f"{B2}/stdout_base.log",
    # Backlog 2 — RNN-06C (blocked)
    f"{ROOT}/RNN-06C-MAMBA-HISTORICAL-INFO/BLOCKED_BY_06B2.md": f"{C6}/BLOCKED_BY_06B2.md",
    f"{ROOT}/RNN-06C-MAMBA-HISTORICAL-INFO/BLOCKED.json": f"{C6}/BLOCKED.json",
    # runners / lib
    f"{ROOT}/ops/rnn_06b2_lib.py": "ops/rnn_06b2_lib.py",
    f"{ROOT}/ops/rnn_06b2_calibration.py": "ops/rnn_06b2_calibration.py",
    f"{ROOT}/ops/rnn_06b2_challenges.py": "ops/rnn_06b2_challenges.py",
    f"{ROOT}/ops/rnn_06b2_base.py": "ops/rnn_06b2_base.py",
    f"{ROOT}/ops/rnn_06_state_load_bundle.py": "ops/rnn_06_state_load_bundle.py",
}


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    manifest = {
        "train": ROOT,
        "backlog_1_RNN_06B2": "EXECUTED; FIXED_LENGTH_STATE_LOAD_REGION = BLOCKED "
                              "(IMMEDIATE_CLIFF + NOT_ROBUST_ACROSS_STRATA)",
        "backlog_2_RNN_06C": "BLOCKED_BY_06B2 (not executed); HISTORICAL_STATE_INFORMATION = NOT_MINTED",
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

    # ---- ENFORCE BUNDLE INVARIANT ----
    with zipfile.ZipFile(OUT_ZIP) as z:
        archive_names = set(z.namelist())
    archive_payload = archive_names - METADATA_NAMES
    unmanifested = archive_payload - manifest_payload
    missing_from_archive = manifest_payload - archive_payload
    sha_mismatch = manifest_payload ^ sha_payload
    if unmanifested or missing_from_archive or sha_mismatch:
        raise SystemExit(
            f"BUNDLE INVARIANT VIOLATION: unmanifested={unmanifested} "
            f"missing_from_archive={missing_from_archive} sha_mismatch={sha_mismatch}")
    if not METADATA_NAMES.issubset(archive_names):
        raise SystemExit(f"metadata files missing: {METADATA_NAMES - archive_names}")

    print(f"ZIP: {OUT_ZIP}")
    print(f"payload_files: {len(present)}  (+ TRAIN_MANIFEST.json + SHA256SUMS.txt metadata)")
    print(f"skipped_missing: {manifest['skipped_missing']}")
    print("BUNDLE_INVARIANT: archive_payload == manifest_payload == sha256sums_payload  OK")
    print(f"ZIP_SHA256: {sha256_file(OUT_ZIP)}")
    print(f"ZIP_BYTES: {os.path.getsize(OUT_ZIP)}")


if __name__ == "__main__":
    main()
