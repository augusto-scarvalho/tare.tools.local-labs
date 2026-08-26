#!/usr/bin/env python3
"""Complete-output successor for BACKLOG-CTRL01-REAL-TOKEN-03."""
from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tools.research import run_ctrl01_real_token as base


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:8080")
    args = parser.parse_args()
    outdir = (ROOT / args.outdir).resolve()
    receipt = base.run(outdir, args.base_url.rstrip("/"), max_tokens=512)
    receipt["schema"] = "local-labs-ctrl01-real-token-v3"
    receipt["claim_code_pending_independent_review"] = (
        "CTRL01_RUNTIME_QUALIFIED_R3"
        if receipt["verdict"] == "QUALIFIED"
        else "CTRL01_FALSE_POSITIVE_CONFIRMED_R3"
    )
    receipt["provenance"]["command"] = "python tools/research/run_ctrl01_real_token_r3.py --outdir runs/research/BACKLOG-CTRL01-REAL-TOKEN-03"
    receipt["provenance"]["runner_sha256"] = base.sha256_path(pathlib.Path(__file__))
    receipt["provenance"]["max_tokens"] = 512
    receipt.pop("receipt_fingerprint_sha256", None)
    canonical = json.dumps(receipt, sort_keys=True, ensure_ascii=False).encode("utf-8")
    receipt["receipt_fingerprint_sha256"] = hashlib.sha256(canonical).hexdigest()
    base.write_json(outdir / "raw" / "receipt.json", receipt)
    base.write_json(outdir / "raw" / "artifact_hashes.json", {
        **receipt["provenance"]["source_hashes"],
        "runs/research/BACKLOG-CTRL01-REAL-TOKEN-02/ABORTED.md": base.sha256_path(ROOT / "runs/research/BACKLOG-CTRL01-REAL-TOKEN-02/ABORTED.md"),
        "runs/research/BACKLOG-CTRL01-REAL-TOKEN-03/PRE_REGISTRATION.md": receipt["provenance"]["preregistration_sha256"],
        "tools/research/run_ctrl01_real_token.py": base.sha256_path(ROOT / "tools/research/run_ctrl01_real_token.py"),
        "tools/research/run_ctrl01_real_token_r3.py": receipt["provenance"]["runner_sha256"],
        "runs/research/BACKLOG-CTRL01-REAL-TOKEN-03/raw/samples.jsonl": receipt["provenance"]["samples_sha256"],
    })
    base.write_result(outdir, receipt)
    print(json.dumps({"metrics": receipt["metrics"], "gates": receipt["gates"], "verdict": receipt["verdict"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
