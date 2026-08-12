#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Package the RNN-06D Historical-State Recovery Ceiling + Parameter-Free Utility train (mobile ZIP).

Bundles Backlog 1 (RNN-06D0, EXECUTED, RECOVERY_CEILING = QUALIFIED) + Backlog 2 (RNN-06D1, EXECUTED,
RECOVERY_UTILITY = QUALIFIED_PARAMETER_FREE) + train protocol/handoff/git-evidence, with
TRAIN_MANIFEST.json + SHA256SUMS.txt.

BUNDLE INVARIANT (enforced): archive payload files == manifest payload files == SHA256SUMS payload
files, excluding ONLY {TRAIN_MANIFEST.json, SHA256SUMS.txt} by explicit rule. Construction RAISES if
any unmanifested payload is present. No writestr-only payloads. Excludes model weights, HF cache,
venv, .git.
"""
import hashlib
import json
import os
import zipfile

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
ROOT = "RNN-06D-recovery-utility-train"
D0 = "runs/rnn/RNN-06D"
OUT_ZIP = os.path.join(REPO, D0, "RNN-06D-recovery-utility-train-audit-bundle.zip")
HANDOFF = ".harness/handoff/HANDOFF-rnn-06d-recovery-utility-train.md"
METADATA_NAMES = {f"{ROOT}/TRAIN_MANIFEST.json", f"{ROOT}/SHA256SUMS.txt"}

FILES = {
    f"{ROOT}/TRAIN_PROTOCOL.md": f"{D0}/TRAIN_PROTOCOL.md",
    f"{ROOT}/HANDOFF.md": HANDOFF,
    f"{ROOT}/git_evidence/git_evidence.txt": f"{D0}/git_evidence.txt",
    # Backlog 1 — RNN-06D0
    f"{ROOT}/D0/PRE_REGISTRATION.md": f"{D0}/PRE_REGISTRATION.md",
    f"{ROOT}/D0/D0_CALIBRATION.json": f"{D0}/D0_CALIBRATION.json",
    f"{ROOT}/D0/D0_CALIBRATION_DECISION.md": f"{D0}/D0_CALIBRATION_DECISION.md",
    f"{ROOT}/D0/D0_QUALIFICATION_SPEC.json": f"{D0}/D0_QUALIFICATION_SPEC.json",
    f"{ROOT}/D0/SNAPSHOT_SCHEDULE.json": f"{D0}/SNAPSHOT_SCHEDULE.json",
    f"{ROOT}/D0/RECOVERY_CEILING_RESULTS.json": f"{D0}/RECOVERY_CEILING_RESULTS.json",
    f"{ROOT}/D0/D0_READOUTS.npz": f"{D0}/D0_READOUTS.npz",
    f"{ROOT}/D0/SNAPSHOT_IDENTITY_SAMPLE.json": f"{D0}/SNAPSHOT_IDENTITY_SAMPLE.json",
    f"{ROOT}/D0/MECHANISM_ACTIVATION_D0.json": f"{D0}/MECHANISM_ACTIVATION_D0.json",
    f"{ROOT}/D0/COST_PROFILE_D0.json": f"{D0}/COST_PROFILE_D0.json",
    f"{ROOT}/D0/D0_CURVES.csv": f"{D0}/D0_CURVES.csv",
    f"{ROOT}/D0/D0_DECISION.md": f"{D0}/D0_DECISION.md",
    f"{ROOT}/D0/stdout_calibration.log": f"{D0}/stdout_calibration.log",
    f"{ROOT}/D0/stdout_ceiling.log": f"{D0}/stdout_ceiling.log",
    # Backlog 2 — RNN-06D1
    f"{ROOT}/D1/D1_PRE_REGISTRATION.md": f"{D0}/D1_PRE_REGISTRATION.md",
    f"{ROOT}/D1/RECOVERY_UTILITY_RESULTS.json": f"{D0}/RECOVERY_UTILITY_RESULTS.json",
    f"{ROOT}/D1/MECHANISM_ACTIVATION.json": f"{D0}/MECHANISM_ACTIVATION.json",
    f"{ROOT}/D1/RECOVERY_HARM.csv": f"{D0}/RECOVERY_HARM.csv",
    f"{ROOT}/D1/SELECTOR_DIAGNOSTICS.csv": f"{D0}/SELECTOR_DIAGNOSTICS.csv",
    f"{ROOT}/D1/D1_DECISION.md": f"{D0}/D1_DECISION.md",
    f"{ROOT}/D1/stdout_recovery.log": f"{D0}/stdout_recovery.log",
    # runners / lib
    f"{ROOT}/ops/rnn_06d_lib.py": "ops/rnn_06d_lib.py",
    f"{ROOT}/ops/rnn_06d0_calibration.py": "ops/rnn_06d0_calibration.py",
    f"{ROOT}/ops/rnn_06d0_challenges.py": "ops/rnn_06d0_challenges.py",
    f"{ROOT}/ops/rnn_06d0_ceiling.py": "ops/rnn_06d0_ceiling.py",
    f"{ROOT}/ops/rnn_06d1_recovery.py": "ops/rnn_06d1_recovery.py",
    f"{ROOT}/ops/rnn_06d_bundle.py": "ops/rnn_06d_bundle.py",
}


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    manifest = {"train": ROOT,
                "backlog_1_RNN_06D0": "EXECUTED; RECOVERY_CEILING = QUALIFIED",
                "backlog_2_RNN_06D1": "EXECUTED; RECOVERY_UTILITY = QUALIFIED_PARAMETER_FREE "
                                      "(best MAX_CONFIDENCE)",
                "subject": "AntonV/mamba2-1.3b-hf@703e19a4 tf4.48.3 torch2.6.0+cu124 bf16 cs32 "
                           "is_fast_path_available=False",
                "pinned_chunk_size": 32, "K": 4, "payload_files": [], "skipped_missing": []}
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
