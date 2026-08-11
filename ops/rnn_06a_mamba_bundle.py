#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Package the RNN-06A-MAMBA audit deliverable (mobile ZIP).

All protocol/config/identity artifacts + lifecycle results/matrix/decision +
source excerpts + runner + git evidence + handoff, with MANIFEST.json and
SHA256SUMS.txt. Excludes weights / HF cache / venv / .git / large temp files.
Prints the ZIP's own SHA-256. Pure packaging; no GPU/model.
"""
import hashlib
import json
import os
import zipfile

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RUNDIR = os.path.join(REPO, "runs", "rnn", "RNN-06A-MAMBA")

INCLUDE = [
    "runs/rnn/RNN-06A-MAMBA/PRE_REGISTRATION.md",
    "runs/rnn/RNN-06A-MAMBA/STATE_CONTRACT.json",
    "runs/rnn/RNN-06A-MAMBA/machine_config.json",
    "runs/rnn/RNN-06A-MAMBA/MODEL_IDENTITY.json",
    "runs/rnn/RNN-06A-MAMBA/source_excerpts.md",
    "runs/rnn/RNN-06A-MAMBA/LIFECYCLE_RESULTS.json",
    "runs/rnn/RNN-06A-MAMBA/LIFECYCLE_MATRIX.csv",
    "runs/rnn/RNN-06A-MAMBA/LIFECYCLE_DECISION.md",
    "runs/rnn/RNN-06A-MAMBA/stdout_lifecycle.log",
    "runs/rnn/RNN-06A-MAMBA/git_evidence.txt",
    "ops/rnn_06a_mamba_lifecycle.py",
    "ops/rnn_06a_mamba_bundle.py",
    ".harness/handoff/HANDOFF-rnn-06a-mamba.md",
]


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    out_zip = os.path.join(RUNDIR, "RNN-06A-MAMBA-audit-bundle.zip")
    manifest = {"packet": "RNN-06A-MAMBA", "kind": "audit_bundle",
                "verdict": "FROZEN_BACKBONE_LIFECYCLE=NOT_QUALIFIED",
                "files": [], "skipped_missing": []}
    sha_lines, present = [], []
    for rel in INCLUDE:
        ap = os.path.join(REPO, rel)
        if not os.path.isfile(ap):
            manifest["skipped_missing"].append(rel)
            continue
        size = os.path.getsize(ap)
        digest = sha256_file(ap)
        manifest["files"].append({"path": rel, "size": size, "sha256": digest})
        sha_lines.append(f"{digest}  {rel}")
        present.append((ap, rel))

    with zipfile.ZipFile(out_zip, "w", zipfile.ZIP_DEFLATED) as z:
        for ap, rel in present:
            z.write(ap, rel)
        z.writestr("MANIFEST.json", json.dumps(manifest, indent=2).encode())
        z.writestr("SHA256SUMS.txt", ("\n".join(sha_lines) + "\n").encode())

    print(f"ZIP: {out_zip}")
    print(f"files_included: {len(present)}")
    print(f"skipped_missing: {manifest['skipped_missing']}")
    print(f"ZIP_SHA256: {sha256_file(out_zip)}")
    print(f"ZIP_BYTES: {os.path.getsize(out_zip)}")


if __name__ == "__main__":
    main()
