#!/usr/bin/env python3
"""Matched SDXL baseline for LAB-IMG-002."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from diffusers import StableDiffusionXLPipeline

from qwen_image_bench import GpuMonitor, PROMPTS, sha256_file


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--height", type=int, default=768)
    parser.add_argument("--width", type=int, default=768)
    parser.add_argument("--steps", type=int, default=30)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    started = time.perf_counter()
    with GpuMonitor() as monitor:
        pipeline = StableDiffusionXLPipeline.from_pretrained(
            args.model, variant="fp16", use_safetensors=True,
            torch_dtype=torch.float16, local_files_only=True,
        ).to("cuda")
    load = {"seconds": time.perf_counter() - started, **monitor.summary()}

    rows = []
    for case in PROMPTS:
        generator = torch.Generator(device="cuda").manual_seed(case["seed"])
        started = time.perf_counter()
        with GpuMonitor() as monitor:
            image = pipeline(
                prompt=case["text"], negative_prompt="blurry, low quality, distorted text",
                height=args.height, width=args.width, num_inference_steps=args.steps,
                generator=generator,
            ).images[0]
        path = args.output / f"{case['id']}.png"
        image.save(path)
        row = {
            **case, "path": str(path), "seconds": time.perf_counter() - started,
            "sha256": sha256_file(path), "bytes": path.stat().st_size,
            "dimensions": list(image.size), **monitor.summary(),
        }
        rows.append(row)
        print(json.dumps({key: row[key] for key in (
            "id", "seconds", "sha256", "peak_memory_used_mib",
        )}), flush=True)
    report = {
        "schema_version": 1, "model": args.model, "revision": args.revision,
        "torch": torch.__version__, "cuda": torch.version.cuda,
        "device": torch.cuda.get_device_name(0), "height": args.height,
        "width": args.width, "steps": args.steps, "load": load, "images": rows,
        "same_seed_byte_identical": rows[2]["sha256"] == rows[3]["sha256"],
    }
    (args.output / "results.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"load": load, "same_seed_byte_identical": report["same_seed_byte_identical"]}, indent=2))


if __name__ == "__main__":
    main()
