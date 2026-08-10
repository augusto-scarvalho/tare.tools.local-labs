#!/usr/bin/env python
"""Generate RNN_ARCHITECTURE_MATRIX.{json,csv} from verified data (§6). UNKNOWN, never guessed."""
import csv, json, os, sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def gdn_state_bytes(nv, kd, vd, rec_dtype_bytes=4):
    return nv * kd * vd * rec_dtype_bytes  # per linear layer, batch=1, recurrent only


COLS = [
    "model", "architecture_family", "attention_mechanism", "recurrent_mechanism", "state_update",
    "num_layers", "hybrid_ratio_linear_to_full", "recurrent_state_shape_per_linear_layer",
    "recurrent_state_bytes_per_request", "kv_cache_behavior", "parallel_training_form",
    "recurrent_inference_form", "chunkwise_form", "mutable_at_test_time",
    "requires_training_after_insertion", "official_implementation", "verified_from",
]

ROWS = [
    dict(
        model="Qwen3-0.6B (conventional Transformer control)",
        architecture_family="Transformer (dense softmax attention)",
        attention_mechanism="full softmax attention (GQA) at every layer",
        recurrent_mechanism="none", state_update="n/a",
        num_layers="UNKNOWN (not verified this packet)", hybrid_ratio_linear_to_full="0:all",
        recurrent_state_shape_per_linear_layer="n/a",
        recurrent_state_bytes_per_request="0 (no recurrent state; KV grows with sequence)",
        kv_cache_behavior="grows O(N) with sequence length (all layers)",
        parallel_training_form="standard attention", recurrent_inference_form="none (KV cache)",
        chunkwise_form="n/a", mutable_at_test_time="no (KV cache is append-only, not trained)",
        requires_training_after_insertion="n/a (native)",
        official_implementation="transformers qwen3 / HF Qwen/Qwen3-0.6B",
        verified_from="architecture class certain; exact dims UNKNOWN this packet",
    ),
    dict(
        model="Qwen3.5-0.8B (GDN hybrid) [PRIMARY SURROGATE TARGET]",
        architecture_family="Gated DeltaNet + gated-attention hybrid (dense FFN)",
        attention_mechanism="gated full softmax (GQA 8Q/2KV, head_dim 256) at every 4th layer",
        recurrent_mechanism="Gated DeltaNet (linear-attention) at the other 3/4 layers",
        state_update="gated delta rule: S<-g*S; S<-S+k (x) (beta*(v-S^T k)); o=S^T q",
        num_layers="24", hybrid_ratio_linear_to_full="18:6 (3:1)",
        recurrent_state_shape_per_linear_layer="[1, 16, 128, 128] (num_v_heads=16, d_k=d_v=128)",
        recurrent_state_bytes_per_request=(
            f"{gdn_state_bytes(16,128,128)} B/linear-layer (fp32); "
            f"18 layers = {round(gdn_state_bytes(16,128,128)*18/1048576,2)} MiB recurrent, "
            "CONSTANT in sequence length (+conv ~0.05 MiB/layer)"),
        kv_cache_behavior="grows O(N) only on the 6 full-attention layers",
        parallel_training_form="chunk_gated_delta_rule (chunkwise-parallel)",
        recurrent_inference_form="recurrent_gated_delta_rule (per-step, O(1) state)",
        chunkwise_form="yes", mutable_at_test_time="state evolves during forward; weights frozen (not learned)",
        requires_training_after_insertion="n/a (native architecture)",
        official_implementation="transformers qwen3_5 / HF Qwen/Qwen3.5-0.8B",
        verified_from="HF config.json + transformers/models/qwen3_5 source (this packet)",
    ),
    dict(
        model="Qwen3.6-27B (GDN hybrid) [deploy-tier]",
        architecture_family="Gated DeltaNet + gated-attention hybrid (dense FFN)",
        attention_mechanism="gated full softmax (GQA 24Q/4KV, head_dim 256) at every 4th layer",
        recurrent_mechanism="Gated DeltaNet (linear-attention) at the other 3/4 layers",
        state_update="gated delta rule (identical to Qwen3.5-0.8B)",
        num_layers="64", hybrid_ratio_linear_to_full="48:16 (3:1)",
        recurrent_state_shape_per_linear_layer="[1, 48, 128, 128] (num_v_heads=48, d_k=d_v=128)",
        recurrent_state_bytes_per_request=(
            f"{gdn_state_bytes(48,128,128)} B/linear-layer (fp32); "
            f"48 layers = {round(gdn_state_bytes(48,128,128)*48/1048576,2)} MiB recurrent, CONSTANT in seq len"),
        kv_cache_behavior="grows O(N) only on the 16 full-attention layers",
        parallel_training_form="chunk_gated_delta_rule", recurrent_inference_form="recurrent_gated_delta_rule",
        chunkwise_form="yes", mutable_at_test_time="state evolves during forward; weights frozen",
        requires_training_after_insertion="n/a (native)",
        official_implementation="transformers qwen3_5 / HF Qwen/Qwen3.6-27B (== local fp16/base text_config)",
        verified_from="local fp16/base config.json + HF config.json (this packet)",
    ),
    dict(
        model="Gated DeltaNet (reference)",
        architecture_family="linear attention (gated delta rule)",
        attention_mechanism="none (pure linear-attention block) / hybrids add SWA",
        recurrent_mechanism="matrix-valued fast-weight memory, gated delta update",
        state_update="gate (fast erasure) + delta rule (targeted KV update); improves Mamba2",
        num_layers="config-dependent", hybrid_ratio_linear_to_full="varies (pure or +SWA/Mamba2)",
        recurrent_state_shape_per_linear_layer="[num_heads, d_k, d_v] matrix",
        recurrent_state_bytes_per_request="O(num_heads*d_k*d_v), constant in seq len",
        kv_cache_behavior="none (fixed recurrent state replaces KV)",
        parallel_training_form="chunkwise-parallel (paper's algorithm)",
        recurrent_inference_form="yes (O(1) state)", chunkwise_form="yes",
        mutable_at_test_time="state evolves; weights frozen",
        requires_training_after_insertion="from-scratch training",
        official_implementation="NVlabs/GatedDeltaNet (NC) @ b53d6d3; FLA (MIT) @ 7843b32",
        verified_from="arXiv 2412.06464 + repos (this packet)",
    ),
    dict(
        model="TTT (reference, TTT-Linear/MLP)",
        architecture_family="test-time-training RNN (expressive hidden state)",
        attention_mechanism="none", recurrent_mechanism="hidden state IS a model (linear or 2-layer MLP)",
        state_update="self-supervised gradient step per token/chunk (fast weights)",
        num_layers="config-dependent", hybrid_ratio_linear_to_full="0:all (pure)",
        recurrent_state_shape_per_linear_layer="weights of inner model (TTT-Linear: [d,d]; TTT-MLP: 2-layer)",
        recurrent_state_bytes_per_request="O(inner-model params), constant in seq len",
        kv_cache_behavior="none", parallel_training_form="mini-batch TTT (linear complexity, CLAIM)",
        recurrent_inference_form="yes", chunkwise_form="yes (LaCT = large-chunk variant)",
        mutable_at_test_time="YES — the defining property (inner weights learned at test time)",
        requires_training_after_insertion="from-scratch (layer trained); In-Place TTT = drop-in variant",
        official_implementation="test-time-training/ttt-lm-pytorch (MIT) @ cd831db",
        verified_from="arXiv 2407.04620 + repo (this packet)",
    ),
    dict(
        model="RetNet (reference)",
        architecture_family="retention network",
        attention_mechanism="parallel retention (train) equivalent to a decayed attention",
        recurrent_mechanism="decayed retention state", state_update="state <- decay*state + k^T v (per step)",
        num_layers="config-dependent", hybrid_ratio_linear_to_full="0:all",
        recurrent_state_shape_per_linear_layer="[d_k, d_v] per head (decayed)",
        recurrent_state_bytes_per_request="O(d_k*d_v*heads), constant in seq len",
        kv_cache_behavior="none (recurrent form)",
        parallel_training_form="parallel retention", recurrent_inference_form="yes (O(1) state, CLAIM)",
        chunkwise_form="yes (chunkwise-recurrent)", mutable_at_test_time="state evolves; weights frozen",
        requires_training_after_insertion="from-scratch",
        official_implementation="microsoft/unilm/retnet (MIT)",
        verified_from="arXiv 2307.08621 + repo (this packet)",
    ),
    dict(
        model="Mamba2 / SSM (reference)",
        architecture_family="selective state-space model",
        attention_mechanism="none", recurrent_mechanism="selective SSM hidden state + short conv",
        state_update="input-dependent (selective) linear recurrence h<-A*h+B*x; y=C*h",
        num_layers="config-dependent", hybrid_ratio_linear_to_full="0:all (pure) / hybrids keep some attn",
        recurrent_state_shape_per_linear_layer="[d_inner, d_state] SSM state + conv state",
        recurrent_state_bytes_per_request="O(d_inner*d_state), constant in seq len",
        kv_cache_behavior="none", parallel_training_form="parallel scan / chunked",
        recurrent_inference_form="yes (O(1) state)", chunkwise_form="yes",
        mutable_at_test_time="state evolves; weights frozen",
        requires_training_after_insertion="from-scratch; MambaInLlama distills from a Transformer",
        official_implementation="state-spaces/mamba (NOT pinned this packet — SHA UNKNOWN)",
        verified_from="architecture class known; repo SHA UNKNOWN this packet",
    ),
]


def main():
    with open(os.path.join(REPO, "RNN_ARCHITECTURE_MATRIX.json"), "w") as f:
        json.dump(dict(columns=COLS, rows=ROWS,
                       note="Recurrent-state bytes are batch=1, recurrent matrix only (fp32 per "
                            "mamba_ssm_dtype); conv-window state (~0.05-0.08 MiB/linear layer) and "
                            "full-attention KV (grows with sequence) are separate."), f, indent=2)
    with open(os.path.join(REPO, "RNN_ARCHITECTURE_MATRIX.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLS)
        w.writeheader()
        for r in ROWS:
            w.writerow(r)
    print("wrote RNN_ARCHITECTURE_MATRIX.json / .csv with", len(ROWS), "rows")


if __name__ == "__main__":
    main()
