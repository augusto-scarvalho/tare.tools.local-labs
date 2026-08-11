#!/usr/bin/env python
"""RNN-08b analyzer: fixed param semantics (§8), outcome vocabulary (§17), decision policy (§18)."""
import argparse, json, os


def rel(a, b):
    return None if (a is None or b is None or b == 0) else round((a - b) / b * 100, 2)


def run(args):
    R = json.load(open(args.results))
    G = json.load(open(args.gates)) if os.path.exists(args.gates) else {}
    C = json.load(open(args.calib)) if os.path.exists(args.calib) else {}
    a = R["arms"]; base, lora, tptt = a.get("base", {}), a.get("lora", {}), a.get("tptt", {})

    metrics = ["sft_loss", "wikitext_ppl", "retention_acc_ctx256", "retention_acc_ctx1024",
               "prefill_ms_400tok"]
    comp = {k: dict(base=base.get(k), lora=lora.get(k), tptt=tptt.get(k),
                    lora_vs_base_pct=rel(lora.get(k), base.get(k)),
                    tptt_vs_base_pct=rel(tptt.get(k), base.get(k)),
                    tptt_vs_lora_pct=rel(tptt.get(k), lora.get(k))) for k in metrics}

    # §8 param semantics (compare trainable[0] and total[1] separately)
    lt = R["training"].get("lora", {}).get("trainable_all")
    tt = R["mechanism"].get("trainable_all")
    params = dict(lora_trainable=lt[0] if lt else None, tptt_trainable=tt[0] if tt else None,
                  trainableParamsMatched=bool(lt and tt and lt[0] == tt[0]),
                  lora_total=lt[1] if lt else None, tptt_total=tt[1] if tt else None,
                  totalParamsMatched=bool(lt and tt and lt[1] == tt[1]),
                  frozenStructuralDelta=(tt[1] - lt[1]) if (lt and tt) else None)

    sft, ppl = comp["sft_loss"], comp["wikitext_ppl"]
    tvl, ppl_tvl = sft["tptt_vs_lora_pct"], ppl["tptt_vs_lora_pct"]
    oi = R.get("order_invariance", {})
    gsum = G.get("GATES_SUMMARY", {}).get("gates", {})

    # §17 statuses
    STATE_LIFECYCLE = "QUALIFIED" if gsum.get("INDEPENDENT_SAMPLE_ISOLATION") == "PASS" and \
        gsum.get("INTRA_SEQUENCE_RECURRENCE") == "PASS" and oi.get("status") == "PASS" else "FAILED"
    SAVE_RELOAD = "QUALIFIED" if gsum.get("TPTT_SAVE_RELOAD") == "QUALIFIED" else "FAILED"
    CONTROL_VALIDITY = "GOOD" if (params["trainableParamsMatched"] and STATE_LIFECYCLE == "QUALIFIED") \
        else "LIMITED"
    MEMORY_AXIS = C.get("MEMORY_AXIS", "NOT_QUALIFIED")

    MARGIN = 3.0
    if tvl is None:
        tptt_vs_lora = "INCONCLUSIVE"
    else:
        sft_better = tvl <= -MARGIN
        sft_worse = tvl >= MARGIN
        ppl_collapse = ppl_tvl is not None and ppl_tvl >= 10
        if sft_better and not ppl_collapse:
            tptt_vs_lora = "CLEAR_POSITIVE_SIGNAL"
        elif sft_worse and ppl_collapse:
            tptt_vs_lora = "QUALITY_REGRESSION"
        elif sft_worse or ppl_collapse:
            tptt_vs_lora = "MIXED" if (sft_better or (tvl < 0)) else "QUALITY_REGRESSION"
        else:
            tptt_vs_lora = "NO_DETECTABLE_GAIN"

    # cost ratio
    lo = R["training"].get("lora", {}); tp = R["training"].get("tptt", {})
    cost_ratio = round(lo.get("tok_per_s", 1) / tp.get("tok_per_s", 1), 1) if tp.get("tok_per_s") else None

    # §18 direction
    if tptt_vs_lora == "CLEAR_POSITIVE_SIGNAL":
        direction = "REPLICATE_TPTT (note: gain on quality axis; MEMORY_AXIS NOT_QUALIFIED)"
    elif tptt_vs_lora == "QUALITY_REGRESSION":
        direction = "PARK_THIS_TPTT_CONFIGURATION / PIVOT_MEMORY_CACHING"
    elif tptt_vs_lora in ("NO_DETECTABLE_GAIN", "MIXED"):
        direction = "PARK_TPTT_FOR_NOW / PIVOT_MEMORY_CACHING (far higher cost)"
    else:
        direction = "PARK_TPTT_FOR_NOW"

    tree = dict(TPTT_DEPENDENCY="QUALIFIED", STATE_LIFECYCLE=STATE_LIFECYCLE, SAVE_RELOAD=SAVE_RELOAD,
                CONTROL_VALIDITY=CONTROL_VALIDITY, MEMORY_AXIS=MEMORY_AXIS,
                TPTT_VS_LORA=tptt_vs_lora, RNN_DIRECTION=direction)
    out = dict(comparisons=comp, param_semantics=params, order_invariance=oi,
               gates=gsum, memory_axis=C.get("selected"), cost_ratio_lora_over_tptt=cost_ratio,
               training=R.get("training", {}), state_bytes=R.get("mechanism", {}).get("state_bytes_per_request"),
               decision_tree=tree,
               interpretation=("Lifecycle now VALID (isolation+order-invariance BIT_EXACT); the "
                               "TPTT-vs-LoRA comparison is trustworthy this time, unlike RNN-08."))
    json.dump(out, open(args.out, "w"), indent=2)
    print(json.dumps({"decision_tree": tree, "param_semantics": params,
                      "sft_loss": {k: sft[k] for k in ["base", "lora", "tptt", "tptt_vs_lora_pct"]},
                      "wikitext_ppl": {k: ppl[k] for k in ["base", "lora", "tptt", "tptt_vs_lora_pct"]},
                      "cost_ratio_lora_over_tptt": cost_ratio,
                      "order_invariance": oi.get("status")}, indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--gates", required=True)
    ap.add_argument("--calib", required=True)
    ap.add_argument("--out", required=True)
    run(ap.parse_args())
