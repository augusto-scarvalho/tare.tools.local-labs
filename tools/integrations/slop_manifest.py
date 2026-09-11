#!/usr/bin/env python3
"""Ingest or query a slop.rs run without granting qualification authority."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from model_lifecycle.integrations.slop import (  # noqa: E402
    SlopManifestError,
    ingest_generate_manifest,
    validate_ingest_receipt,
)


def _write_new(path: Path, document: object) -> None:
    data = json.dumps(document, indent=2, ensure_ascii=False, sort_keys=True) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary = Path(stream.name)
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
    except FileExistsError as error:
        raise SlopManifestError(f"refusing to overwrite existing receipt: {path}") from error
    except OSError as error:
        raise SlopManifestError(f"cannot publish receipt {path}: {error}") from error
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="slop_manifest")
    subparsers = parser.add_subparsers(dest="command", required=True)

    ingest = subparsers.add_parser("ingest")
    ingest.add_argument("--manifest", type=Path, required=True)
    ingest.add_argument("--retain-dir", type=Path, required=True)
    ingest.add_argument("--receipt", type=Path, required=True)
    ingest.add_argument("--consumer-repository", type=Path, default=ROOT)
    ingest.add_argument("--skip-artifact-readback", action="store_true")
    ingest.add_argument("--allow-dirty-consumer", action="store_true")

    query = subparsers.add_parser("query")
    query.add_argument("--receipt", type=Path, required=True)
    query.add_argument("--allow-dirty-consumer", action="store_true")

    args = parser.parse_args(argv)
    try:
        if args.command == "ingest":
            receipt = ingest_generate_manifest(
                args.manifest,
                args.retain_dir,
                args.consumer_repository,
                verify_artifacts=not args.skip_artifact_readback,
                require_clean_consumer=not args.allow_dirty_consumer,
            )
            _write_new(args.receipt, receipt)
            print(json.dumps(receipt, ensure_ascii=False, sort_keys=True))
            return 0

        receipt = json.loads(args.receipt.read_text(encoding="utf-8"))
        valid = validate_ingest_receipt(
            receipt,
            require_clean_consumer=not args.allow_dirty_consumer,
        )
        print(json.dumps(valid, ensure_ascii=False, sort_keys=True))
        return 0
    except (OSError, json.JSONDecodeError, SlopManifestError) as error:
        print(f"slop manifest rejected: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
