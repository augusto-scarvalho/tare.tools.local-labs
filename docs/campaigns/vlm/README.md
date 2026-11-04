# Campaign VLM — Vision-Language Models & Multimodal Inference 👁️

## Overview
Evaluated multimodal vision models (Gemma-4 12B/26B vision architectures) for latency, token efficiency, OCR accuracy, and UI comprehension in agentic desktop workflows.

## Key Files & Artifacts
- [`M_A_VLM.md`](M_A_VLM.md): Multimodal architecture, vision encoders, and refusal baseline.
- [`M_A_VLM_PERF.md`](M_A_VLM_PERF.md): Performance profiling and image resolution scaling.
- `tools/probes/vlm_probe.py`: VLM image processing latency probe.
- `tools/benchmarks/vlm_vqa_bench.py`: Visual question answering benchmark.

## Core Conclusion
**CONFIRMED**: Zero-refusal multimodal pipeline established for local visual telemetry and desktop GUI element localization with sub-second image prefill latency.
