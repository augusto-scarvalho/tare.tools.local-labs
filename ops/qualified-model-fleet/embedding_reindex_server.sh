#!/usr/bin/env bash
# On-demand Qwen3-Embedding-4B Q8_0 server on port 8082 for (re)indexing the Library.
# The resident service on 8081 (Q4_K_M, hybrid) keeps serving queries; this one picks the
# fastest mode the GPU can spare right now:
#   full   : >= 7000 MiB free  -> weights on GPU        (~10k tok/s)
#   hybrid : >= 2500 MiB free  -> weights in RAM, GPU compute per 4096-token batch (~4.5k tok/s)
#   cpu    : otherwise         -> pure CPU, P-cores     (~100 tok/s)
# Usage: embedding_reindex_server.sh start [full|hybrid|cpu] | stop | status
set -euo pipefail

model=${REINDEX_MODEL:-/home/augus/models/embedding/Qwen3-Embedding-4B-Q8_0.gguf}
server=${REINDEX_SERVER:-/home/augus/src/slop.cpp/build/bin/llama-server}
port=8082
runtime=${REINDEX_RUNTIME_DIR:-${XDG_RUNTIME_DIR:-/tmp}/llm-embedding-reindex-$UID}
umask 077
mkdir -p "$runtime"
pidfile=$runtime/server.pid
logfile=$runtime/server.log
# Serialize start/stop; the server must not inherit the control lock.
exec 9>"$runtime/control.lock"
flock -x 9

owned_pid() {
    local pid arg has_server=0 has_model=0
    [[ -f "$pidfile" ]] || return 1
    read -r pid < "$pidfile"
    [[ "$pid" =~ ^[1-9][0-9]*$ && -r "/proc/$pid/cmdline" ]] || return 1
    while IFS= read -r -d '' arg; do
        [[ "$arg" != "$server" ]] || has_server=1
        [[ "$arg" != "$model" ]] || has_model=1
    done < "/proc/$pid/cmdline"
    (( has_server && has_model )) || return 1
    printf '%s\n' "$pid"
}

free_mib() {
    nvidia-smi --query-gpu=memory.total,memory.used --format=csv,noheader,nounits \
        | awk -F', *' '{print $1-$2}'
}

pick_mode() {
    local free; free="$(free_mib 2>/dev/null)" || free=0
    [[ "$free" =~ ^[0-9]+$ ]] || free=0
    if (( free >= 7000 )); then echo full; elif (( free >= 2500 )); then echo hybrid; else echo cpu; fi
}

case "${1:-status}" in
start)
    if pid="$(owned_pid)"; then
        echo "REINDEX_SERVER_ALREADY_RUNNING pid=$pid"; exit 0
    fi
    mode="${2:-$(pick_mode)}"
    common=(-m "$model" --host 0.0.0.0 --port "$port" --embedding --pooling last --parallel 4)
    case "$mode" in
        full)   cmd=(taskset -c 0-15 "$server" "${common[@]}" -ngl 99 --ctx-size 16384 -b 4096 -ub 4096) ;;
        hybrid) cmd=(taskset -c 0-15 "$server" "${common[@]}" -ngl 0 -t 8 --ctx-size 16384 -b 4096 -ub 4096) ;;
        cpu)    cmd=(env CUDA_VISIBLE_DEVICES= taskset -c 0-15 "$server" "${common[@]}" -ngl 0 -t 8 --ctx-size 4096 -b 512 -ub 512) ;;
        *) echo "unknown mode: $mode"; exit 2 ;;
    esac
    nohup "${cmd[@]}" 9>&- >"$logfile" 2>&1 &
    child_pid=$!
    echo "$child_pid" >"$pidfile"
    for ((i=1; i<=90; i++)); do
        if ! kill -0 "$child_pid" 2>/dev/null; then
            rm -f "$pidfile"
            echo REINDEX_SERVER_EXITED; tail -n 20 "$logfile"; exit 1
        fi
        if owned_pid >/dev/null && curl -fsS --max-time 2 "http://127.0.0.1:$port/health" >/dev/null 2>&1; then
            printf 'REINDEX_SERVER_READY mode=%s port=%s pid=%s free_mib_now=%s\n' "$mode" "$port" "$child_pid" "$(free_mib 2>/dev/null || echo unavailable)"
            exit 0
        fi
        sleep 2
    done
    kill "$child_pid" 2>/dev/null || true
    rm -f "$pidfile"
    echo REINDEX_SERVER_START_TIMEOUT; tail -n 20 "$logfile"; exit 1 ;;
stop)
    if pid="$(owned_pid)"; then
        kill "$pid"; rm -f "$pidfile"; echo REINDEX_SERVER_STOPPED
    else
        rm -f "$pidfile"; echo REINDEX_SERVER_NOT_RUNNING
    fi ;;
status)
    if pid="$(owned_pid)"; then
        printf 'RUNNING pid=%s health=%s\n' "$pid" "$(curl -s --max-time 2 "http://127.0.0.1:$port/health" || echo down)"
    else echo NOT_RUNNING; fi
    printf 'GPU free_mib=%s suggested_mode=%s\n' "$(free_mib 2>/dev/null || echo unavailable)" "$(pick_mode)" ;;
*) echo "usage: $0 start [full|hybrid|cpu] | stop | status"; exit 2 ;;
esac
