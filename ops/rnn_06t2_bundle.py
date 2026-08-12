#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Package the RNN-06T2 Official-Mamba Requalification + Recovery-Confirmation train (mobile ZIP).

BUNDLE INVARIANT (enforced): archive payload == manifest payload == SHA256SUMS payload, excluding
ONLY {TRAIN_MANIFEST.json, SHA256SUMS.txt}. Raises on any unmanifested payload. Excludes model
weights, HF caches, venv, compiled caches, .git, large unrelated temp files.
"""
import hashlib
import json
import os
import zipfile

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
ROOT = "RNN-06T2-MAMBA-REQUALIFICATION"
T = "runs/rnn/RNN-06T2"
OUT_ZIP = os.path.join(REPO, T, "RNN-06T2-MAMBA-REQUALIFICATION-audit-bundle.zip")
HANDOFF = ".harness/handoff/HANDOFF-rnn-06t2-mamba-requalification-confirmation.md"
METADATA_NAMES = {f"{ROOT}/TRAIN_MANIFEST.json", f"{ROOT}/SHA256SUMS.txt"}

FILES = {
    f"{ROOT}/TRAIN_PROTOCOL.md": f"{T}/TRAIN_PROTOCOL.md",
    f"{ROOT}/FINAL_DECISION.md": f"{T}/FINAL_DECISION.md",
    f"{ROOT}/HANDOFF.md": HANDOFF,
    f"{ROOT}/git_evidence.txt": f"{T}/git_evidence.txt",
    f"{ROOT}/ENVIRONMENT_PROVENANCE.json": f"{T}/ENVIRONMENT_PROVENANCE.json",
    f"{ROOT}/AUDIT_RECONCILIATION_RNN-06T.md": "runs/rnn/RNN-06T/AUDIT_RECONCILIATION.md",
    # T0R
    f"{ROOT}/T0R/T0R_PRE_REGISTRATION.md": f"{T}/T0R_PRE_REGISTRATION.md",
    f"{ROOT}/T0R/T0R_RESULTS.json": f"{T}/T0R_RESULTS.json",
    f"{ROOT}/T0R/T0R_DECISION.md": f"{T}/T0R_DECISION.md",
    f"{ROOT}/T0R/stdout_t0r.log": f"{T}/stdout_t0r.log",
    # T1R prereg + narrow
    f"{ROOT}/T1R/T1R_PRE_REGISTRATION.md": f"{T}/T1R_PRE_REGISTRATION.md",
    f"{ROOT}/T1R/narrow/T1R_NARROW_QUAL_SPEC.json": f"{T}/T1R_NARROW_QUAL_SPEC.json",
    f"{ROOT}/T1R/narrow/T1R_NARROW_RESULTS.json": f"{T}/T1R_NARROW_RESULTS.json",
    f"{ROOT}/T1R/narrow/T1R_NARROW_READOUTS.npz": f"{T}/T1R_NARROW_READOUTS.npz",
    f"{ROOT}/T1R/narrow/stdout_t1r_narrow.log": f"{T}/stdout_t1r_narrow.log",
    # T1R wide
    f"{ROOT}/T1R/wide/T1R_WIDE_CALIB_SPEC.json": f"{T}/T1R_WIDE_CALIB_SPEC.json",
    f"{ROOT}/T1R/wide/T1R_WIDE_CALIBRATION.json": f"{T}/T1R_WIDE_CALIBRATION.json",
    f"{ROOT}/T1R/wide/T1R_WIDE_QUAL_SPEC.json": f"{T}/T1R_WIDE_QUAL_SPEC.json",
    f"{ROOT}/T1R/wide/T1R_WIDE_RESULTS.json": f"{T}/T1R_WIDE_RESULTS.json",
    f"{ROOT}/T1R/wide/T1R_WIDE_READOUTS.npz": f"{T}/T1R_WIDE_READOUTS.npz",
    f"{ROOT}/T1R/wide/T1R_SCORED_VALUE_TOKEN_MAP.json": f"{T}/T1R_SCORED_VALUE_TOKEN_MAP.json",
    f"{ROOT}/T1R/wide/stdout_t1r_widecalib.log": f"{T}/stdout_t1r_widecalib.log",
    f"{ROOT}/T1R/wide/stdout_t1r_widequal.log": f"{T}/stdout_t1r_widequal.log",
    # economics
    f"{ROOT}/economics/T1R_ECON_run0.json": f"{T}/T1R_ECON_run0.json",
    f"{ROOT}/economics/T1R_ECON_run1.json": f"{T}/T1R_ECON_run1.json",
    f"{ROOT}/economics/T1R_ECONOMICS.json": f"{T}/T1R_ECONOMICS.json",
    f"{ROOT}/economics/stdout_econ0.log": f"{T}/stdout_econ0.log",
    f"{ROOT}/economics/stdout_econ1.log": f"{T}/stdout_econ1.log",
    f"{ROOT}/economics/stdout_econ_decide.log": f"{T}/stdout_econ_decide.log",
    f"{ROOT}/T1R/T1R_DECISION.md": f"{T}/T1R_DECISION.md",
    # runners / libs (executed source)
    f"{ROOT}/ops/rnn_06t2_t0r.py": "ops/rnn_06t2_t0r.py",
    f"{ROOT}/ops/rnn_06t2_t1r_challenges.py": "ops/rnn_06t2_t1r_challenges.py",
    f"{ROOT}/ops/rnn_06t2_t1r.py": "ops/rnn_06t2_t1r.py",
    f"{ROOT}/ops/rnn_06t2_econ.py": "ops/rnn_06t2_econ.py",
    f"{ROOT}/ops/rnn_06t2_econ_decide.py": "ops/rnn_06t2_econ_decide.py",
    f"{ROOT}/ops/rnn_06t2_bundle.py": "ops/rnn_06t2_bundle.py",
    f"{ROOT}/ops/rnn_06t_lib.py": "ops/rnn_06t_lib.py",
    f"{ROOT}/ops/rnn_06d_lib.py": "ops/rnn_06d_lib.py",
    f"{ROOT}/ops/rnn_06b2_lib.py": "ops/rnn_06b2_lib.py",
}


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_state(rel, key):
    try:
        return json.load(open(os.path.join(REPO, rel))).get(key)
    except Exception:
        return None


def main():
    t0r = os.path.join(T, "T0R_RESULTS.json")
    widw = os.path.join(T, "T1R_WIDE_RESULTS.json")
    narw = os.path.join(T, "T1R_NARROW_RESULTS.json")
    econ = os.path.join(T, "T1R_ECONOMICS.json")
    manifest = {"train": ROOT, "official_checkpoint": "state-spaces/mamba2-1.3b",
                "revision": "c5b59d00ec85d313adea86a08cad2a43c962dd3b",
                "fast_path": "mamba_ssm 2.2.4 + causal_conv1d 1.5.0.post8 (cu12torch2.6cxx11abiFALSE-cp312)",
                "decision_states": {
                    "OFFICIAL_MAMBA_FIXED_BATCH_LIFECYCLE": load_state(t0r, "OFFICIAL_MAMBA_FIXED_BATCH_LIFECYCLE"),
                    "BATCH_SHAPE_NUMERICAL_PORTABILITY": load_state(t0r, "BATCH_SHAPE_NUMERICAL_PORTABILITY"),
                    "SINGLE_PASS_HISTORICAL_CAPTURE_T0R": load_state(t0r, "SINGLE_PASS_HISTORICAL_CAPTURE_T0R"),
                    "HISTORICAL_RECOVERY_NARROW": load_state(narw, "HISTORICAL_RECOVERY_NARROW"),
                    "ADAPTIVE_SELECTION_NARROW": load_state(narw, "ADAPTIVE_SELECTION_NARROW"),
                    "WIDE_TARGET_RECOVERY_T1R": load_state(widw, "WIDE_TARGET_RECOVERY_T1R"),
                    "ADAPTIVE_SELECTION_T1R": load_state(widw, "ADAPTIVE_SELECTION_T1R"),
                    "END_TO_END_RECOVERY_UTILITY_T1R": load_state(econ, "END_TO_END_RECOVERY_UTILITY_T1R")},
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
