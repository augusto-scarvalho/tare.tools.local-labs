#!/usr/bin/env bash
# Phase A discriminating sweeps: MoE q4 (halved KV) + dense 27B q8/q4 (the heavy case).
D=/mnt/c/projects/local-model-lifecycle
CTX="8192 65536 131072 262144"
echo "##### MoE q4_0 (ncmoe=8) #####";  bash $D/phase_a_ctx.sh moe   8  q4_0 $CTX
echo; echo "##### DENSE q8_0 (ngl=99) #####"; bash $D/phase_a_ctx.sh dense 99 q8_0 $CTX
echo; echo "##### DENSE q4_0 (ngl=99) #####"; bash $D/phase_a_ctx.sh dense 99 q4_0 $CTX
