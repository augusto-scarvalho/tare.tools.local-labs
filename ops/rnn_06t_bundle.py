#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Package the RNN-06T Official Mamba Transportability train (mobile ZIP).

BUNDLE INVARIANT (enforced): archive payload == manifest payload == SHA256SUMS payload, excluding
ONLY {TRAIN_MANIFEST.json, SHA256SUMS.txt}. Raises on any unmanifested payload. Excludes model
weights, HF caches, venv, compiled caches, .git.
"""
import hashlib
import json
import os
import zipfile

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
ROOT = "RNN-06T-MAMBA-OFFICIAL"
T = "runs/rnn/RNN-06T"
OUT_ZIP = os.path.join(REPO, T, "RNN-06T-MAMBA-OFFICIAL-audit-bundle.zip")
HANDOFF = ".harness/handoff/HANDOFF-rnn-06t-mamba-official-transport.md"
METADATA_NAMES = {f"{ROOT}/TRAIN_MANIFEST.json", f"{ROOT}/SHA256SUMS.txt"}

FILES = {
    f"{ROOT}/TRAIN_PROTOCOL.md": f"{T}/TRAIN_PROTOCOL.md",
    f"{ROOT}/FINAL_DECISION.md": f"{T}/FINAL_DECISION.md",
    f"{ROOT}/HANDOFF.md": HANDOFF,
    f"{ROOT}/git_evidence/git_evidence.txt": f"{T}/git_evidence.txt",
    f"{ROOT}/AUDIT_RECONCILIATION_RNN-06D.md": f"{T}/../RNN-06D/AUDIT_RECONCILIATION.md",
    # Section 1 backend
    f"{ROOT}/backend/ENVIRONMENT_PROVENANCE.md": f"{T}/ENVIRONMENT_PROVENANCE.md",
    f"{ROOT}/backend/OFFICIAL_MAMBA_ENV.json": f"{T}/OFFICIAL_MAMBA_ENV.json",
    # T0
    f"{ROOT}/T0/T0_PRE_REGISTRATION.md": f"{T}/T0_PRE_REGISTRATION.md",
    f"{ROOT}/T0/OFFICIAL_MAMBA_STATE_CONTRACT.json": f"{T}/OFFICIAL_MAMBA_STATE_CONTRACT.json",
    f"{ROOT}/T0/T0_RESULTS.json": f"{T}/T0_RESULTS.json",
    f"{ROOT}/T0/T0_DECISION.md": f"{T}/T0_DECISION.md",
    f"{ROOT}/T0/stdout_t0.log": f"{T}/stdout_t0.log",
    # 3A
    f"{ROOT}/3A/T1_3A_PRE_REGISTRATION.md": f"{T}/T1_3A_PRE_REGISTRATION.md",
    f"{ROOT}/3A/T1_3A_QUALIFICATION_SPEC.json": f"{T}/T1_3A_QUALIFICATION_SPEC.json",
    f"{ROOT}/3A/T1_3A_RESULTS.json": f"{T}/T1_3A_RESULTS.json",
    f"{ROOT}/3A/T1_3A_READOUTS.npz": f"{T}/T1_3A_READOUTS.npz",
    f"{ROOT}/3A/T1_3A_DECISION.md": f"{T}/T1_3A_DECISION.md",
    f"{ROOT}/3A/stdout_3a.log": f"{T}/stdout_3a.log",
    # 3B
    f"{ROOT}/3B/T1_3B_PRE_REGISTRATION.md": f"{T}/T1_3B_PRE_REGISTRATION.md",
    f"{ROOT}/3B/T1_3B_CALIBRATION_SPEC.json": f"{T}/T1_3B_CALIBRATION_SPEC.json",
    f"{ROOT}/3B/T1_3B_CALIBRATION.json": f"{T}/T1_3B_CALIBRATION.json",
    f"{ROOT}/3B/T1_3B_QUALIFICATION_SPEC.json": f"{T}/T1_3B_QUALIFICATION_SPEC.json",
    f"{ROOT}/3B/T1_3B_RESULTS.json": f"{T}/T1_3B_RESULTS.json",
    f"{ROOT}/3B/T1_3B_READOUTS.npz": f"{T}/T1_3B_READOUTS.npz",
    f"{ROOT}/3B/T1_3B_DECISION.md": f"{T}/T1_3B_DECISION.md",
    f"{ROOT}/3B/stdout_3b_calib.log": f"{T}/stdout_3b_calib.log",
    f"{ROOT}/3B/stdout_3b_qual.log": f"{T}/stdout_3b_qual.log",
    # economics + scout
    f"{ROOT}/economics/T1_ECON_PRE_REGISTRATION.md": f"{T}/T1_ECON_PRE_REGISTRATION.md",
    f"{ROOT}/economics/T1_ECONOMICS.json": f"{T}/T1_ECONOMICS.json",
    f"{ROOT}/economics/stdout_econ.log": f"{T}/stdout_econ.log",
    f"{ROOT}/scout/T1_NONSYNTH_SCOUT.json": f"{T}/T1_NONSYNTH_SCOUT.json",
    f"{ROOT}/scout/stdout_scout.log": f"{T}/stdout_scout.log",
    # runners / lib
    f"{ROOT}/ops/rnn_06t_backend.py": "ops/rnn_06t_backend.py",
    f"{ROOT}/ops/rnn_06t_lib.py": "ops/rnn_06t_lib.py",
    f"{ROOT}/ops/rnn_06t_t0.py": "ops/rnn_06t_t0.py",
    f"{ROOT}/ops/rnn_06t_3a_challenges.py": "ops/rnn_06t_3a_challenges.py",
    f"{ROOT}/ops/rnn_06t_3a.py": "ops/rnn_06t_3a.py",
    f"{ROOT}/ops/rnn_06t_3b_challenges.py": "ops/rnn_06t_3b_challenges.py",
    f"{ROOT}/ops/rnn_06t_3b.py": "ops/rnn_06t_3b.py",
    f"{ROOT}/ops/rnn_06t_econ.py": "ops/rnn_06t_econ.py",
    f"{ROOT}/ops/rnn_06t_scout.py": "ops/rnn_06t_scout.py",
    f"{ROOT}/ops/rnn_06t_bundle.py": "ops/rnn_06t_bundle.py",
}


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    manifest = {"train": ROOT, "official_checkpoint": "state-spaces/mamba2-1.3b",
                "revision": "c5b59d00ec85d313adea86a08cad2a43c962dd3b",
                "fast_path": "mamba_ssm 2.2.4 + causal_conv1d 1.5.0.post8 (cu12torch2.6cxx11abiFALSE-cp312)",
                "decision_states": {
                    "OFFICIAL_MAMBA_FASTPATH": "RUNNABLE", "OFFICIAL_MAMBA_LIFECYCLE": "QUALIFIED",
                    "SINGLE_PASS_HISTORICAL_CAPTURE": "QUALIFIED",
                    "HISTORICAL_RECOVERY_TRANSPORT": "QUALIFIED", "ADAPTIVE_SELECTOR_ADVANTAGE": "DIRECTIONAL",
                    "OFFICIAL_MAMBA_SYNTHETIC_TRANSPORT": "PARTIAL", "WIDE_TARGET_RECOVERY": "QUALIFIED",
                    "ADAPTIVE_SELECTION": "QUALIFIED", "END_TO_END_RECOVERY_UTILITY": "QUALIFIED",
                    "NON_SYNTHETIC_RECOVERY_SCOUT": "NO_SIGNAL"},
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
    print(f"payload_files: {len(present)}  (+ TRAIN_MANIFEST.json + SHA256SUMS.txt)")
    print(f"skipped_missing: {manifest['skipped_missing']}")
    print("BUNDLE_INVARIANT: archive_payload == manifest_payload == sha256sums_payload  OK")
    print(f"ZIP_SHA256: {sha256_file(OUT_ZIP)}")
    print(f"ZIP_BYTES: {os.path.getsize(OUT_ZIP)}")


if __name__ == "__main__":
    main()
