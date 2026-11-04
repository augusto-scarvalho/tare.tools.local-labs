#!/usr/bin/env python3
"""Pinned, bounded Qwen-Image benchmark with GPU-memory receipts."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import threading
import time
from pathlib import Path

import torch
from diffusers import BitsAndBytesConfig, QwenImagePipeline, QwenImageTransformer2DModel
from transformers import BitsAndBytesConfig as TransformersBitsAndBytesConfig
from transformers import Qwen2_5_VLForConditionalGeneration


PROMPTS = [
    {
        "id": "typography",
        "seed": 10161,
        "text": (
            "A clean minimalist laboratory poster, front-facing, dark navy background, "
            "high contrast white sans-serif typography. The poster contains exactly three "
            "lines of text: 'TARE LAB', 'BUILD 10161', and 'STATUS READY'. No other text."
        ),
    },
    {
        "id": "dashboard",
        "seed": 10162,
        "text": (
            "A crisp dark-mode operations dashboard UI screenshot, front-facing and flat, "
            "with exactly three status cards labelled 'QUEUE 7', 'GPU 68%', and 'CACHE OK'. "
            "Professional spacing, high contrast, no other text."
        ),
    },
    {
        "id": "composition",
        "seed": 10163,
        "text": (
            "Simple geometric composition on a pure white background: one red cube on the "
            "left, one blue sphere on the right, and one green triangle centered above them. "
            "No text and no additional objects."
        ),
    },
    {
        "id": "composition_replay",
        "seed": 10163,
        "text": (
            "Simple geometric composition on a pure white background: one red cube on the "
            "left, one blue sphere on the right, and one green triangle centered above them. "
            "No text and no additional objects."
        ),
    },
]


class GpuMonitor:
    def __init__(self) -> None:
        self.stop_event = threading.Event()
        self.samples: list[dict[str, float]] = []
        self.thread = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        while not self.stop_event.is_set():
            try:
                output = subprocess.check_output([
                    "nvidia-smi", "--query-gpu=memory.used,memory.total,power.draw,temperature.gpu",
                    "--format=csv,noheader,nounits",
                ], text=True, timeout=5).strip().split(", ")
                self.samples.append({
                    "time": time.time(), "memory_used_mib": float(output[0]),
                    "memory_total_mib": float(output[1]), "power_w": float(output[2]),
                    "temperature_c": float(output[3]),
                })
            except Exception:
                pass
            self.stop_event.wait(0.25)

    def __enter__(self) -> "GpuMonitor":
        self.thread.start()
        return self

    def __exit__(self, *_: object) -> None:
        self.stop_event.set()
        self.thread.join(timeout=3)

    def summary(self) -> dict[str, float | int | None]:
        return {
            "samples": len(self.samples),
            "peak_memory_used_mib": max((s["memory_used_mib"] for s in self.samples), default=None),
            "peak_power_w": max((s["power_w"] for s in self.samples), default=None),
            "peak_temperature_c": max((s["temperature_c"] for s in self.samples), default=None),
        }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen-Image")
    parser.add_argument("--revision", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--height", type=int, default=768)
    parser.add_argument("--width", type=int, default=768)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--case", action="append", choices=[case["id"] for case in PROMPTS])
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    component_config = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    text_config = TransformersBitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    load_started = time.perf_counter()
    with GpuMonitor() as monitor:
        transformer = QwenImageTransformer2DModel.from_pretrained(
            args.model, subfolder="transformer", revision=args.revision,
            quantization_config=component_config, torch_dtype=torch.bfloat16,
        )
        text_encoder = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            args.model, subfolder="text_encoder", revision=args.revision,
            quantization_config=text_config, torch_dtype=torch.bfloat16,
        )
        pipeline = QwenImagePipeline.from_pretrained(
            args.model, revision=args.revision, transformer=transformer,
            text_encoder=text_encoder, torch_dtype=torch.bfloat16,
        )
        pipeline.enable_model_cpu_offload()
    load = {
        "seconds": time.perf_counter() - load_started,
        **monitor.summary(),
    }

    selected = [case for case in PROMPTS if not args.case or case["id"] in args.case]
    rows = []
    for case in selected:
        generator = torch.Generator(device="cpu").manual_seed(case["seed"])
        started = time.perf_counter()
        with GpuMonitor() as monitor:
            image = pipeline(
                prompt=case["text"], negative_prompt="blurry, low quality, distorted text",
                height=args.height, width=args.width, num_inference_steps=args.steps,
                true_cfg_scale=4.0, generator=generator,
            ).images[0]
        path = args.output / f"{case['id']}.png"
        image.save(path)
        row = {
            **case, "path": str(path), "seconds": time.perf_counter() - started,
            "sha256": sha256_file(path), "bytes": path.stat().st_size,
            "dimensions": list(image.size), **monitor.summary(),
        }
        rows.append(row)
        print(json.dumps({k: row[k] for k in (
            "id", "seconds", "sha256", "peak_memory_used_mib",
        )}), flush=True)

    report = {
        "schema_version": 1, "model": args.model, "revision": args.revision,
        "torch": torch.__version__, "cuda": torch.version.cuda,
        "device": torch.cuda.get_device_name(0), "height": args.height,
        "width": args.width, "steps": args.steps, "load": load, "images": rows,
        "same_seed_byte_identical": (
            next((row["sha256"] for row in rows if row["id"] == "composition"), None)
            == next((row["sha256"] for row in rows if row["id"] == "composition_replay"), object())
        ) if any(row["id"] == "composition_replay" for row in rows) else None,
    }
    (args.output / "results.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "load": load, "same_seed_byte_identical": report["same_seed_byte_identical"],
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
