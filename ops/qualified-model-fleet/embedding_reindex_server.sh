#!/usr/bin/env bash
# On-demand Qwen3-Embedding-4B Q8_0 server on port 8082 for (re)indexing the Library.
# The resident service on 8081 (Q4_K_M, hybrid) keeps serving queries; this one picks the
# fastest mode the GPU can spare right now:
#   full   : >= 7000 MiB free  -> weights on GPU        (~10k tok/s)
#   hybrid : >= 2500 MiB free  -> weights in RAM, GPU compute per 4096-token batch (~4.5k tok/s)
#   cpu    : otherwise         -> pure CPU, P-cores     (~100 tok/s)
# Usage: embedding_reindex_server.sh start [full|hybrid|cpu] | stop | status
set -euo pipefail

model=/home/augus/models/embedding/Qwen3-Embedding-4B-Q8_0.gguf
server=/home/augus/src/slop.cpp/build/bin/llama-server
port=8082
pidfile=/tmp/llm-embedding-reindex.pid
logfile=/tmp/llm-embedding-reindex.log

free_mib() {
    nvidia-smi --query-gpu=memory.total,memory.used --format=csv,noheader,nounits \
        | awk -F', *' '{print $1-$2}'
}

pick_mode() {
    local free; free="$(free_mib)"
    if (( free >= 7000 )); then echo full; elif (( free >= 2500 )); then echo hybrid; else echo cpu; fi
}

case "${1:-status}" in
start)
    if [[ -f "$pidfile" ]] && kill -0 "$(cat "$pidfile")" 2>/dev/null; then
        echo "REINDEX_SERVER_ALREADY_RUNNING pid=$(cat "$pidfile")"; exit 0
    fi
    mode="${2:-$(pick_mode)}"
    common=(-m "$model" --host 0.0.0.0 --port "$port" --embedding --pooling last --parallel 4)
    case "$mode" in
        full)   cmd=(taskset -c 0-15 "$server" "${common[@]}" -ngl 99 --ctx-size 16384 -b 4096 -ub 4096) ;;
        hybrid) cmd=(taskset -c 0-15 "$server" "${common[@]}" -ngl 0 -t 8 --ctx-size 16384 -b 4096 -ub 4096) ;;
        cpu)    cmd=(env CUDA_VISIBLE_DEVICES= taskset -c 0-15 "$server" "${common[@]}" -ngl 0 -t 8 --ctx-size 4096 -b 512 -ub 512) ;;
        *) echo "unknown mode: $mode"; exit 2 ;;
    esac
    nohup "${cmd[@]}" >"$logfile" 2>&1 &
    echo $! >"$pidfile"
    for ((i=1; i<=90; i++)); do
        if curl -fsS --max-time 2 "http://127.0.0.1:$port/health" >/dev/null 2>&1; then
            printf 'REINDEX_SERVER_READY mode=%s port=%s pid=%s free_mib_before=%s\n' "$mode" "$port" "$(cat "$pidfile")" "$(free_mib)"
            exit 0
        fi
        sleep 2
    done
    echo REINDEX_SERVER_START_TIMEOUT; tail -n 20 "$logfile"; exit 1 ;;
stop)
    if [[ -f "$pidfile" ]]; then kill "$(cat "$pidfile")" 2>/dev/null || true; rm -f "$pidfile"; echo REINDEX_SERVER_STOPPED; else echo REINDEX_SERVER_NOT_RUNNING; fi ;;
status)
    if [[ -f "$pidfile" ]] && kill -0 "$(cat "$pidfile")" 2>/dev/null; then
        printf 'RUNNING pid=%s health=%s\n' "$(cat "$pidfile")" "$(curl -s --max-time 2 "http://127.0.0.1:$port/health" || echo down)"
    else echo NOT_RUNNING; fi
    printf 'GPU free_mib=%s suggested_mode=%s\n' "$(free_mib)" "$(pick_mode)" ;;
*) echo "usage: $0 start [full|hybrid|cpu] | stop | status"; exit 2 ;;
esac
