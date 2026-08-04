#!/usr/bin/env bash
# mmq-vs-cublas-generality.sh — S2 generality check across QUANTS and MODEL FAMILIES (one-shot, not the
# standing gate; the deploy-model gate is mmq-vs-cublas-bench.sh). Answers "did we validate across models?"
#
# RESULT (2026-08-04, isolated arms + cooldown, ub2048, sm_86 undervolt) — verdict UNANIMOUS:
#   model              family   quant   type   MMQ t/s   cuBLAS t/s   verdict
#   qwen36-35b-a3b     Qwen     Q4_K_M  MoE      2529        590       MMQ +329%
#   qwen36-35b-a3b     Qwen     Q5_K_M  MoE      1851*       688       MMQ +169%
#   qwen36-35b-a3b     Qwen     Q6_K    MoE      2140        162       MMQ +1223%
#   gemma-4-26b        Gemma    q4_0    MoE      2526*      1512       MMQ +67%
#   gpt-oss-20b        OpenAI   Q4_K_M  MoE      4727       3287       MMQ +44%
#   granite-4.0-h      IBM      Q4_K_M  MoE      1675        933       MMQ +80%
#   mistral-small-24b  Mistral  Q4_K_M  dense    2088       2186       cuBLAS +4.7%   (large-batch dense residual)
#   (* = thermal first-rep variance on that MMQ cell; mean still wins decisively)
# => EVERY MoE / every quant / every family keeps MMQ (win +44%..+1223%). The one dense residual (cuBLAS edges
#    large-ubatch dense prefill) REPRODUCES on a different family (Mistral +4.7% == Qwen dense +5-11%), proving
#    it's a general GEMM-shape property, not a Qwen quirk. cuBLAS gets relatively worse at higher-bit k-quants
#    (Q6_K MoE collapses). Confirms S2's "keep MMQ default" across 5 labs + 4 quants (S1-level breadth).
# Note: cuBLAS-for-MoE can NaN/assert on some models (#19659); if a cuBLAS cell prints <no output>, that is the
#    crash and itself confirms cuBLAS-MoE is not a valid option.
set -u
MMQ=/home/augus/src/llama.cpp-master/build/bin/llama-bench
CUB=/home/augus/src/llama.cpp-master/build-cublas/bin/llama-bench
R="${R:-6}"; COOL="${COOL:-15}"; UB="${UB:-2048}"
tps() { grep -E '±' | tail -1 | sed -E 's/.*\| +([0-9.]+ . [0-9.]+) \|.*/\1/'; }

cell() {  # label bin model extra
  local out clk
  out=$("$2" -m "$3" $4 -p "$UB" -n 0 -ngl 99 -fa 1 -b "$UB" -ub "$UB" -r "$R" 2>/tmp/berr | tps)
  [ -z "$out" ] && out="<no output: likely #19659 crash -> $(tail -1 /tmp/berr | cut -c1-50)>"
  clk=$(nvidia-smi --query-gpu=clocks.sm --format=csv,noheader | tr -d '\n')
  printf '    %-14s %s   [%s]\n' "$1" "$out" "$clk"; sleep "$COOL"
}
pair() { echo "### $1"; cell MMQ "$MMQ" "$2" "$3"; cell cuBLAS "$CUB" "$2" "$3"; echo; }

M=/home/augus/models
echo "############ QUANT AXIS — deploy MoE qwen36-35b-a3b, ncmoe=8, ub$UB ############"
pair "Q4_K_M (deploy)" "$M/qwen36-35b-a3b/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf" "--n-cpu-moe 8"
pair "Q5_K_M"          "$M/qwen36-35b-a3b/Qwen3.6-35B-A3B-UD-Q5_K_M.gguf" "--n-cpu-moe 8"
pair "Q6_K"            "$M/qwen36-35b-a3b/Qwen3.6-35B-A3B-UD-Q6_K.gguf"   "--n-cpu-moe 8"
echo "############ FAMILY AXIS — other labs / quants, ub$UB ############"
pair "gemma-4-26b MoE (q4_0)"   "$M/gemma-4-26b-a4b/gemma-4-26B_q4_0-it.gguf" "--n-cpu-moe 8"
pair "gpt-oss-20b MoE (Q4_K_M)" "$M/gpt-oss-20b/gpt-oss-20b-Q4_K_M.gguf" "--n-cpu-moe 4"
pair "granite-4-h MoE (Q4_K_M)" "$M/granite-4.0-h-small/granite-4.0-h-small-Q4_K_M.gguf" "--n-cpu-moe 8"
pair "mistral-24b DENSE (Q4_K_M)" "$M/mistral-small-24b/mistralai_Mistral-Small-3.2-24B-Instruct-2506-Q4_K_M.gguf" ""
echo "############ DONE ############"
