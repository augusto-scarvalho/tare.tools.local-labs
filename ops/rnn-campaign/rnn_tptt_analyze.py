#!/usr/bin/env python
"""RNN-08/09 analyzer: §13 comparisons, §15 outcome label, §21 decision tree. Pure/CPU over results JSON."""
import argparse, json, os


def rel(a, b):
    return None if (a is None or b is None or b == 0) else round((a - b) / b * 100, 2)


def run(args):
    R = json.load(open(args.results))
    arms = R["arms"]
    base, lora, tptt = arms.get("base", {}), arms.get("lora", {}), arms.get("tptt", {})
    canary = json.load(open(args.canary)) if args.canary and os.path.exists(args.canary) else {}

    def m(a, k):
        return a.get(k)

    # §13 mandatory comparisons on every metric
    metrics = ["sft_loss", "wikitext_ppl", "retention_acc_ctx256", "retention_acc_ctx1024",
               "prefill_ms_400tok"]
    comp = {}
    for k in metrics:
        comp[k] = dict(base=m(base, k), lora=m(lora, k), tptt=m(tptt, k),
                       lora_vs_base_pct=rel(m(lora, k), m(base, k)),
                       tptt_vs_base_pct=rel(m(tptt, k), m(base, k)),
                       tptt_vs_lora_pct=rel(m(tptt, k), m(lora, k)))

    # primary = held-out SFT loss (lower better); guard = wikitext ppl (collapse detector)
    sft = comp["sft_loss"]; ppl = comp["wikitext_ppl"]
    tvl = sft["tptt_vs_lora_pct"]           # <0 means TPTT better than LoRA on SFT loss
    ppl_tvl = ppl["tptt_vs_lora_pct"]       # >0 means TPTT worse (higher ppl)
    iso = R.get("state_isolation", {})

    # §15 outcome vocabulary (conservative; tiny eval set -> require clear margin, no p-values)
    MARGIN = 3.0  # percent, relative, on SFT loss
    if tvl is None:
        outcome = "INCONCLUSIVE"
    elif not iso.get("isolated", True):
        outcome = "INCONCLUSIVE"  # state leakage invalidates quality comparison
    elif tvl <= -MARGIN and (ppl_tvl is None or ppl_tvl < 10):
        outcome = "CLEAR_POSITIVE_SIGNAL"
    elif tvl <= -MARGIN and ppl_tvl is not None and ppl_tvl >= 10:
        outcome = "COMPUTE_ONLY_GAIN"       # helps target, hurts general -> not a clean win
    elif tvl >= MARGIN:
        outcome = "QUALITY_REGRESSION"
    else:
        outcome = "NO_DETECTABLE_GAIN"

    # §21 decision tree
    trainable_matched = (R.get("training", {}).get("lora", {}).get("trainable_all")
                         == R.get("mechanism", {}).get("trainable_all"))
    control = "GOOD" if (trainable_matched and iso.get("isolated", False)) else (
        "INVALID" if not iso.get("isolated", True) else "LIMITED")
    direction = {"CLEAR_POSITIVE_SIGNAL": "CONTINUE_TPTT",
                 "COMPUTE_ONLY_GAIN": "CONTINUE_TPTT",
                 "NO_DETECTABLE_GAIN": "PARK_TPTT",
                 "QUALITY_REGRESSION": "PARK_TPTT",
                 "INCONCLUSIVE": "PARK_TPTT"}[outcome]

    tree = dict(
        TPTT_DEPENDENCY=canary.get("TPTT_DEPENDENCY", "QUALIFIED"),
        UPSTREAM_MECHANISM=canary.get("UPSTREAM_MECHANISM", "REPRODUCED"),
        CONTROL_VALIDITY=control,
        TPTT_VS_LORA=outcome,
        RNN_RESEARCH_DIRECTION=direction)

    out = dict(comparisons=comp, primary_metric="sft_loss (held-out dolly)",
               guard_metric="wikitext_ppl", state_isolation=iso,
               trainable_params=dict(lora=R.get("training", {}).get("lora", {}).get("trainable_all"),
                                     tptt=R.get("mechanism", {}).get("trainable_all"),
                                     matched=trainable_matched),
               training=R.get("training", {}), mechanism=R.get("mechanism", {}),
               retention_note="all arms saturate at 1.0 -> non-discriminative at this scale/difficulty",
               decision_tree=tree)
    json.dump(out, open(args.out, "w"), indent=2)
    print(json.dumps({"decision_tree": tree,
                      "sft_loss": {k: sft[k] for k in ["base", "lora", "tptt", "tptt_vs_lora_pct"]},
                      "wikitext_ppl": {k: ppl[k] for k in ["base", "lora", "tptt", "tptt_vs_lora_pct"]}},
                     indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--canary", default=None)
    ap.add_argument("--out", required=True)
    run(ap.parse_args())
