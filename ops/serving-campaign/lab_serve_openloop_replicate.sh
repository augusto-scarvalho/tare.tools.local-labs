#!/usr/bin/env bash
# LAB-SERVE-001c — open-loop paired-block campaign orchestrator (all-WSL; avoids Windows/WSL argv
# mangling). Executes the PRE_REGISTRATION.md design: 3 load points x 2 reps x 2 arms = 12 server-
# starts. Per-rep seed shared by both arms (common arrival schedule §14); arm order alternated per rep.
# HARD 4h deadline guard + per-cell timeout; partials are saved and never silently dropped.
set -u
BIN=/home/augus/src/llama.cpp-master/build/bin/llama-server
MODEL=/home/augus/models/qwen36-35b-a3b-mtp/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf
REPO=/mnt/c/projects/local-model-lifecycle
OUT=$REPO/runs/serving/LAB-SERVE-001c/campaign/raw
DS=$REPO/runs/serving/LAB-SERVE-001c/workload/workload_001c.jsonl
PY=/home/augus/sglang-venv/bin/python
PORT=8080
BASE="-m $MODEL --host 127.0.0.1 --port $PORT -fa on --ctx-size 73728 --parallel 4 --jinja --n-cpu-moe 8 --cache-type-k q8_0 --cache-type-v q8_0"
MTP="--spec-type draft-mtp --spec-draft-n-max 4"
mkdir -p "$OUT"
MANIFEST="$OUT/manifest.jsonl"; : > "$MANIFEST"

# pre-registered load points (req/s) and per-rep seeds / arm order
declare -A RATE=( [low]=0.030 [near]=0.072 [over]=0.110 )
POINTS=(low near over)
declare -A SEED=( [1]=101 [2]=102 )
declare -A ORDER=( [1]="off on" [2]="on off" )
NUM_PROMPTS=12; WARMUP=1
DEADLINE_S=$((4*3600)); CELL_TIMEOUT=1200; CELL_BUDGET=1000   # reserve for a cell before deadline
START=$(date +%s)
STOPPED_ON_BUDGET=false

start_server () {
  local extra="$1"; local log="$2"
  pkill -f llama-server 2>/dev/null; sleep 3
  echo "ARGV: $BIN $BASE $extra" > "$log"
  nohup $BIN $BASE $extra >> "$log" 2>&1 &
  for i in $(seq 1 300); do
    curl -s "http://127.0.0.1:$PORT/health" 2>/dev/null | grep -q '"status"' && return 0
    sleep 1
  done
  return 1
}

probe_topology () {
  local props slots
  props=$(curl -s "http://127.0.0.1:$PORT/props")
  slots=$(curl -s "http://127.0.0.1:$PORT/slots")
  echo "$props" > "$1"; echo "$slots" > "$2"
}

for pt in "${POINTS[@]}"; do
  for rep in 1 2; do
    for arm in ${ORDER[$rep]}; do
      now=$(date +%s); elapsed=$((now-START))
      if (( elapsed + CELL_BUDGET > DEADLINE_S )); then
        echo "BUDGET GUARD: elapsed=${elapsed}s would cross 4h -> STOP"; STOPPED_ON_BUDGET=true; break 3
      fi
      tag="${pt}_rep${rep}_${arm}"
      extra=""; [ "$arm" = "on" ] && extra="$MTP"
      echo "=== CELL $tag  rate=${RATE[$pt]} seed=${SEED[$rep]} (elapsed ${elapsed}s) ==="
      if start_server "$extra" "$OUT/server_${tag}.log"; then
        probe_topology "$OUT/${tag}.props.json" "$OUT/${tag}.slots.json"
        timeout $CELL_TIMEOUT $PY $REPO/ops/lab_serve_bench_openloop.py --tag "$tag" --outdir "$OUT" \
          --dataset-path "$DS" --num-prompts $NUM_PROMPTS --request-rate "${RATE[$pt]}" \
          --warmup $WARMUP --seed "${SEED[$rep]}"
        rc=$?
      else
        echo "SERVER FAILED for $tag"; rc=97
      fi
      echo "{\"tag\":\"$tag\",\"point\":\"$pt\",\"rep\":$rep,\"arm\":\"$arm\",\"rate\":${RATE[$pt]},\"seed\":${SEED[$rep]},\"rc\":$rc}" >> "$MANIFEST"
      pkill -f llama-server 2>/dev/null; sleep 3
    done
  done
done

END=$(date +%s)
echo "{\"start\":$START,\"end\":$END,\"elapsed_s\":$((END-START)),\"stopped_on_budget\":$STOPPED_ON_BUDGET,\"num_prompts\":$NUM_PROMPTS,\"warmup\":$WARMUP,\"capacity_req_s\":0.085,\"rates\":{\"low\":0.030,\"near\":0.072,\"over\":0.110}}" > "$OUT/campaign_meta.json"
echo "CAMPAIGN DONE elapsed=$((END-START))s stopped_on_budget=$STOPPED_ON_BUDGET"
