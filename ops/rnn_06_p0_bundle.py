#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Package the RNN-06-P0 audit deliverable into a ZIP with a SHA256 manifest.

Includes protocol/config/identity/result/decision artifacts, curves, the scout
sources, and the handoff. Excludes model weights, HF cache, venv, .git, and any
large temporary files. Emits SHA256SUMS.txt + MANIFEST.json inside the ZIP and
prints the final ZIP's own SHA-256.
"""
import hashlib
import json
import os
import sys
import zipfile

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RUNDIR = os.path.join(REPO, "runs", "rnn", "RNN-06-P0")

# explicit include list (paths relative to REPO). Missing files are skipped with
# a recorded note (do not fabricate artifacts for candidates that never ran).
INCLUDE = [
    "runs/rnn/RNN-06-P0/P0_PROTOCOL.md",
    "runs/rnn/RNN-06-P0/machine_config.json",
    "runs/rnn/RNN-06-P0/MODEL_IDENTITY_DELTANET.json",
    "runs/rnn/RNN-06-P0/MODEL_IDENTITY_MAMBA2.json",
    "runs/rnn/RNN-06-P0/calibration_examples.json",
    "runs/rnn/RNN-06-P0/P0_RESULTS_DELTANET.json",
    "runs/rnn/RNN-06-P0/P0_RESULTS_MAMBA2.json",
    "runs/rnn/RNN-06-P0/P0_CURVES.csv",
    "runs/rnn/RNN-06-P0/P0_DECISION.md",
    "runs/rnn/RNN-06-P0/HANDOFF.md",
    "runs/rnn/RNN-06-P0/git_evidence.txt",
    "runs/rnn/RNN-06-P0/stdout_sweep.log",
    "runs/rnn/RNN-06-P0/stdout_sweep_mamba.log",
    "ops/rnn_06_p0_mqar.py",
    "ops/rnn_06_p0_curves.py",
    "ops/rnn_06_p0_bundle.py",
]


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    out_zip = os.path.join(RUNDIR, "RNN-06-P0-audit-bundle.zip")
    manifest = {"packet": "RNN-06-P0", "files": [], "skipped_missing": []}
    sha_lines = []
    present = []
    for rel in INCLUDE:
        ap = os.path.join(REPO, rel)
        if not os.path.isfile(ap):
            manifest["skipped_missing"].append(rel)
            continue
        size = os.path.getsize(ap)
        digest = sha256_file(ap)
        arc = rel  # keep repo-relative structure inside the zip
        manifest["files"].append({"path": arc, "size": size, "sha256": digest})
        sha_lines.append(f"{digest}  {arc}")
        present.append((ap, arc))

    manifest_bytes = json.dumps(manifest, indent=2).encode()
    sha_bytes = ("\n".join(sha_lines) + "\n").encode()

    with zipfile.ZipFile(out_zip, "w", zipfile.ZIP_DEFLATED) as z:
        for ap, arc in present:
            z.write(ap, arc)
        z.writestr("MANIFEST.json", manifest_bytes)
        z.writestr("SHA256SUMS.txt", sha_bytes)

    zip_sha = sha256_file(out_zip)
    print(f"ZIP: {out_zip}")
    print(f"files_included: {len(present)}")
    print(f"skipped_missing: {manifest['skipped_missing']}")
    print(f"ZIP_SHA256: {zip_sha}")
    print(f"ZIP_BYTES: {os.path.getsize(out_zip)}")


if __name__ == "__main__":
    main()
