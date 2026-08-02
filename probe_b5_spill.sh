#!/usr/bin/env bash
# §B5 precondition probe (v2, STEADY-STATE) — does the MoE spill experts to DISK during
# decode on THIS box, IN STEADY STATE?
#
# --pin-hot-experts (PRs #25932 closed / #26414 open, both UNMERGED) exists to stop the OS
# page cache from EVICTING mmap'd expert weights to disk when a MoE EXCEEDS RAM. Its own PR:
# minor benefit when the model fits in RAM; value is disk-paging under over-capacity. This
# box has 64 GB RAM and qwen36-35B's experts are ~18 GB, so the premise may never engage.
#
# v1 saw 23 major faults over one 1500-tok decode of a FRESH-loaded model. That is the
# lazy-mmap fault-in tail (llama.cpp mmaps but faults expert pages in on first touch), NOT
# steady-state eviction: 23 faults against ~480k expert-accesses (1500 tok x 40 layers x 8
# experts) is ~0. The discriminating test is back-to-back decodes on the SAME warm server:
#
#   decode 1 majflt > 0, decodes 2..N majflt ~ 0  =>  cold fault-in only; everything now
#     resident and STAYS resident -> no eviction -> the flag has nothing to protect here.
#   decode 2..N majflt stays high              =>  experts genuinely round-trip to disk
#     between uses -> eviction is real -> --pin-hot-experts (#26414) could be worth building.
#
# No root/install/build: /proc/<pid>/stat field 12 = cumulative major faults.
#   MSYS_NO_PATHCONV=1 wsl.exe -d Ubuntu-24.04 -- bash /mnt/c/projects/local-model-lifecycle/probe_b5_spill.sh
set -u

BIN=/home/augus/src/llama.cpp-master/build/bin/llama-server
MODEL=/home/augus/models/qwen36-35b-a3b-mtp/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf
PORT=8091
NCMOE=40
MAXTOK=1500
DECODES=3

[ -x "$BIN" ] || { echo "REFUSING: no binary at $BIN"; exit 2; }
[ -f "$MODEL" ] || { echo "REFUSING: no model at $MODEL"; exit 2; }
pkill -9 -f "port $PORT" 2>/dev/null; sleep 1

echo "== launching ncmoe=$NCMOE server (all experts on CPU = max resident footprint) =="
"$BIN" -m "$MODEL" -fa on --n-cpu-moe "$NCMOE" --ctx-size 8192 \
  --cache-type-k q8_0 --cache-type-v q8_0 --host 127.0.0.1 --port "$PORT" \
  > /tmp/b5_server.log 2>&1 &
SRV=$!
trap 'kill -9 $SRV 2>/dev/null' EXIT
for i in $(seq 1 180); do
  curl -s "http://127.0.0.1:$PORT/health" 2>/dev/null | grep -q 'ok' && break
  kill -0 $SRV 2>/dev/null || { echo "SERVER DIED:"; tail -20 /tmp/b5_server.log; exit 1; }
  sleep 1
done
rss_kb=$(awk '/VmRSS/{print $2}' /proc/$SRV/status)
echo "== server healthy (pid $SRV), VmRSS $((rss_kb/1024)) MB =="
echo

PROMPT='Explain in detail, step by step with worked numbers, why memory bandwidth rather than raw compute limits token generation on a single consumer GPU, covering arithmetic intensity, batch-size-one, mixture-of-experts routing, PCIe transfer of offloaded experts, and KV-cache growth. Be concrete and long.'

printf '%-9s %-8s %-8s %-8s\n' "decode" "majflt" "minflt" "wall_s"
prev_maj=$(awk '{print $12}' /proc/$SRV/stat)
prev_min=$(awk '{print $10}' /proc/$SRV/stat)
maj2plus=0
for d in $(seq 1 $DECODES); do
  t0=$(date +%s.%N)
  curl -s "http://127.0.0.1:$PORT/completion" -H 'Content-Type: application/json' \
    -d "{\"prompt\":\"[pass $d] $PROMPT\",\"n_predict\":$MAXTOK,\"temperature\":0,\"cache_prompt\":false}" \
    > /tmp/b5_decode.json 2>/dev/null
  t1=$(date +%s.%N)
  maj=$(awk '{print $12}' /proc/$SRV/stat); min=$(awk '{print $10}' /proc/$SRV/stat)
  dmaj=$((maj - prev_maj)); dmin=$((min - prev_min))
  printf '%-9s %-8s %-8s %-8s\n' "$d" "$dmaj" "$dmin" "$(awk "BEGIN{printf \"%.1f\", $t1-$t0}")"
  [ "$d" -ge 2 ] && maj2plus=$((maj2plus + dmaj))
  prev_maj=$maj; prev_min=$min
done

echo
echo "============================================================"
echo "§B5 STEADY-STATE VERDICT (ncmoe=$NCMOE, model fits in 64 GB)"
echo "============================================================"
echo "major faults across steady-state decodes 2..$DECODES: $maj2plus"
if [ "$maj2plus" -le 4 ]; then
  echo "-> NO steady-state disk spill. The cold fault-in tail aside, experts stay resident"
  echo "   and DO NOT round-trip to disk. The eviction --pin-hot-experts prevents does not"
  echo "   occur here; the lever has nothing to protect. Revisit only for a MoE that"
  echo "   EXCEEDS this box's RAM. No experimental build (#26414) warranted."
else
  echo "-> experts DO round-trip to disk in steady state ($maj2plus major faults). The"
  echo "   eviction premise engages here; building #26414 to pin hot experts could pay off."
fi
