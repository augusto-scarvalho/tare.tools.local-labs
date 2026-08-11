#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Package the FINAL closed RNN-06-P0 audit deliverable.

Superset of ops/rnn_06_p0_bundle.py: all prior P0 audit material PLUS the
append-only closure artifacts (AUDIT_RECONCILIATION.md, closure git-evidence,
final bundle script, and the untracked closure handoff). Emits SHA256SUMS.txt +
MANIFEST.json and prints the ZIP's own SHA-256. Pure packaging; no GPU/model.

Note: P0_CURVES.csv is bundled as its working-tree (CRLF) bytes; the committed
Git blob is LF-normalized (.gitattributes eol=lf). Both hashes are documented in
AUDIT_RECONCILIATION.md §2; the MANIFEST records the actual bundled bytes.
"""
import hashlib
import json
import os
import zipfile

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RUNDIR = os.path.join(REPO, "runs", "rnn", "RNN-06-P0")

INCLUDE = [
    # prior P0 audit material
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
    # append-only closure artifacts
    "runs/rnn/RNN-06-P0/AUDIT_RECONCILIATION.md",
    "runs/rnn/RNN-06-P0/git_evidence_closure.txt",
    "ops/rnn_06_p0_bundle_final.py",
    ".harness/handoff/HANDOFF-rnn-06-p0-final-closure.md",
]


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    out_zip = os.path.join(RUNDIR, "RNN-06-P0-final-audit-bundle.zip")
    manifest = {"packet": "RNN-06-P0", "kind": "final_closure", "files": [],
                "skipped_missing": []}
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
