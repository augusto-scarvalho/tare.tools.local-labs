#!/usr/bin/env bash
# §B5 precondition probe — does the MoE spill experts to DISK during decode on THIS box?
#
# `--pin-hot-experts` (PRs #25932 closed / #26414 open, both UNMERGED) exists to stop the
# OS page cache from EVICTING mmap'd expert weights to disk when a MoE model EXCEEDS RAM.
# Its own PR says: minor benefit when the model fits in RAM; the value is disk-paging under
# over-capacity. This box has 64 GB RAM and qwen36-35B's experts are ~18 GB, so the premise
# may never engage. Rather than build an experimental branch to measure a foregone null,
# measure the PRECONDITION directly: at the HEAVIEST placement (ncmoe=40, every expert layer
# resident on the CPU), does sustained decode take MAJOR PAGE FAULTS or read from disk?
#
#   majflt ~ 0  AND  disk-read ~ 0  during steady decode  =>  no eviction, experts stay
#   resident, and the lever --pin-hot-experts protects has nothing to protect here.
#
# No root, no install, no build: /proc/<pid>/stat majflt and /proc/diskstats are always
# there. vmtouch residency is printed too when available (bonus, not required).
#
# Run entirely inside WSL (a real script file, so shell variables expand normally -- the
# $VAR-empties-out gotcha only bites `wsl.exe -- bash -lc '$VAR'` one-liners):
#   wsl.exe -d Ubuntu-24.04 -- bash /mnt/c/projects/local-model-lifecycle/probe_b5_spill.sh
set -u

BIN=/home/augus/src/llama.cpp-master/build/bin/llama-server
MODEL=/home/augus/models/qwen36-35b-a3b-mtp/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf
PORT=8091
NCMOE=40
MAXTOK=1500

command -v "$BIN" >/dev/null 2>&1 || { echo "REFUSING: no binary at $BIN"; exit 2; }
[ -f "$MODEL" ] || { echo "REFUSING: no model at $MODEL"; exit 2; }

pkill -9 -f "port $PORT" 2>/dev/null
sleep 1

echo "== launching ncmoe=$NCMOE server (all experts on CPU = max resident footprint) =="
"$BIN" -m "$MODEL" -fa on --n-cpu-moe "$NCMOE" --ctx-size 8192 \
  --cache-type-k q8_0 --cache-type-v q8_0 --host 127.0.0.1 --port "$PORT" \
  > /tmp/b5_server.log 2>&1 &
SRV=$!
trap 'kill -9 $SRV 2>/dev/null' EXIT

# wait until /health is ok (up to 180s)
for i in $(seq 1 180); do
  if curl -s "http://127.0.0.1:$PORT/health" 2>/dev/null | grep -q '"ok"\|"status":"ok"'; then
    break
  fi
  kill -0 $SRV 2>/dev/null || { echo "SERVER DIED; tail:"; tail -20 /tmp/b5_server.log; exit 1; }
  sleep 1
done
echo "== server healthy (pid $SRV) =="

# RSS + resident footprint before decode
rss_kb=$(awk '/VmRSS/{print $2}' /proc/$SRV/status)
echo "server VmRSS: $((rss_kb/1024)) MB"
command -v vmtouch >/dev/null 2>&1 && { echo "-- vmtouch model residency --"; vmtouch "$MODEL" | sed 's/^/   /'; } || echo "(vmtouch not installed; relying on majflt + diskstats)"

# baselines
majflt0=$(awk '{print $12}' /proc/$SRV/stat)          # field 12 = majflt (cumulative major faults)
minflt0=$(awk '{print $10}' /proc/$SRV/stat)          # field 10 = minflt
# sum sectors READ across real disks (sd*/nvme*), field 6 of /proc/diskstats
read0=$(awk '$3 ~ /^(sd[a-z]+|nvme[0-9]+n[0-9]+|vd[a-z]+)$/ {s+=$6} END{print s+0}' /proc/diskstats)

echo "== sustained decode ($MAXTOK tokens, greedy) =="
t0=$(date +%s.%N)
curl -s "http://127.0.0.1:$PORT/completion" -H 'Content-Type: application/json' \
  -d "{\"prompt\":\"Explain in detail, step by step with worked numbers, why memory bandwidth rather than raw compute limits token generation on a single consumer GPU, covering arithmetic intensity, batch-size-one, mixture-of-experts routing, PCIe transfer of offloaded experts, and KV-cache growth.\",\"n_predict\":$MAXTOK,\"temperature\":0,\"cache_prompt\":false}" \
  > /tmp/b5_decode.json 2>/dev/null
t1=$(date +%s.%N)

majflt1=$(awk '{print $12}' /proc/$SRV/stat)
minflt1=$(awk '{print $10}' /proc/$SRV/stat)
read1=$(awk '$3 ~ /^(sd[a-z]+|nvme[0-9]+n[0-9]+|vd[a-z]+)$/ {s+=$6} END{print s+0}' /proc/diskstats)

ntok=$(grep -o '"tokens_predicted":[0-9]*' /tmp/b5_decode.json | head -1 | grep -o '[0-9]*')
dt=$(awk "BEGIN{print $t1-$t0}")
echo
echo "============================================================"
echo "§B5 PRECONDITION RESULT (ncmoe=$NCMOE, heaviest placement)"
echo "============================================================"
echo "decode wall:        ${dt}s   tokens: ${ntok:-?}"
echo "MAJOR page faults:  $((majflt1 - majflt0))   <- disk-backed faults during decode"
echo "minor page faults:  $((minflt1 - minflt0))   (resident, no disk)"
echo "disk sectors READ:  $((read1 - read0))       (x512B; whole system, so an upper bound)"
echo
if [ "$((majflt1 - majflt0))" -le 8 ] && [ "$((read1 - read0))" -le 4096 ]; then
  echo "VERDICT: NO disk spill during decode. The eviction the flag prevents does not"
  echo "         occur here (model fits in 64 GB with margin). --pin-hot-experts has"
  echo "         nothing to protect on this box; revisit only for a model that EXCEEDS RAM."
else
  echo "VERDICT: disk activity during decode is NON-TRIVIAL -- the eviction premise MAY"
  echo "         engage here. --pin-hot-experts could be worth building (#26414). Inspect."
fi
