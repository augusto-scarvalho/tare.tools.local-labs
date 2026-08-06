#!/bin/bash
# Download fp16 safetensors for base / ThinkingCap / Fable for the A2 merge+distill work.
# Full repo (these are safetensors-only repos; no GGUF to skip). hf resumes partials on re-run.
set -e
cd /home/augus/models/fp16
HF=/home/augus/.local/bin/hf

echo "[$(date +%H:%M)] base (Qwen/Qwen3.6-27B)..."
"$HF" download Qwen/Qwen3.6-27B --local-dir base

echo "[$(date +%H:%M)] tc (bottlecapai/ThinkingCap-Qwen3.6-27B)..."
"$HF" download bottlecapai/ThinkingCap-Qwen3.6-27B --local-dir tc

echo "[$(date +%H:%M)] fable (DavidAU Fable-Fusion-711 MTP)..."
"$HF" download DavidAU/Qwen3.6-27B-Fable-Fusion-711-Uncensored-Heretic-NM-DAU-MTP --local-dir fable

echo "[$(date +%H:%M)] ALL FP16 DONE"
du -sh base tc fable
