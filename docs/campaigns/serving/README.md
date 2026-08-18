# Campaign Serving — Multi-Slot Scheduling & Production Topology ⚡

## Overview
Evaluated multi-slot concurrency, parallel request scheduling, VRAM budget allocation, and micro-batch tuning (`--ubatch-size`) for high-throughput local agent inference.

## Key Files & Artifacts
- [`DEPLOY.md`](DEPLOY.md): The consolidated production deployment runbook and golden configs.
- [`SERVING.md`](SERVING.md): Multi-slot orchestration, load-balancing, and topology limits.

## Core Conclusion
**OPERATIONALIZED**: Setting `--ubatch-size 2048` yields a **2.03x prefill speedup** on long context (128k prompt TTFT reduced from 137.8s to 67.9s) with ~1.6GB VRAM headroom on 24GB GPUs.
