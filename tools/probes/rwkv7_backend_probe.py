#!/usr/bin/env python3
"""Report the release-bundled RWKV7 backend resolution without loading weights."""

from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import json
import sys
from pathlib import Path
from types import ModuleType

import torch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    args = parser.parse_args()
    package_name = "_rwkv7_backend_probe_release"
    package = ModuleType(package_name)
    package.__package__ = package_name
    package.__path__ = [str(args.model / "inference")]
    sys.modules[package_name] = package
    importlib.import_module(f"{package_name}.runtime")
    dispatch = importlib.import_module(f"{package_name}.kernel_dispatch")
    statuses = {
        name: {
            "available": status.available,
            "version": status.version,
            "reason": status.reason,
        }
        for name, status in dispatch.kernel_backend_status().items()
    }
    print(json.dumps({
        "transformers": importlib.metadata.version("transformers"),
        "tilelang": importlib.metadata.version("tilelang"),
        "statuses": statuses,
        "auto_cuda_resolves_to": dispatch.resolve_backend("auto", torch.device("cuda")),
    }, indent=2))


if __name__ == "__main__":
    main()
