#!/usr/bin/env python
"""
RNN-08b corrected 3-arm run. Uses the IndependentSequenceTPTT lifecycle: per-batch reset in training
(custom Trainer), per-example reset in every eval. Adds order-invariance smoke. MEMORY_AXIS NOT_QUALIFIED
(calibration) -> quality/control metrics only. float32. Isolated venv.
"""
import argparse, json, os, time, math, platform, random
import torch
from transformers import (AutoTokenizer, AutoConfig, Trainer, TrainingArguments,
                          default_data_collator)
from datasets import load_dataset
import tptt
from rnn_tptt_lifecycle import (build_base, build_lora, build_tptt, IndependentSequenceTPTT,
                                MODEL, LORA)

MAXLEN, SEED, DEV, DT = 256, 42, "cuda", torch.float32


def ntrain(m):
    return (sum(p.numel() for p in m.parameters() if p.requires_grad),
            sum(p.numel() for p in m.parameters()))


class ResetTrainer(Trainer):
    _iso = None
    def training_step(self, model, inputs, *a, **k):
        if self._iso is not None:
            self._iso.reset()
        return super().training_step(model, inputs, *a, **k)


def make_dataset(tok):
    ds = load_dataset("databricks/databricks-dolly-15k", split="train")
    fp = ds._fingerprint
    ds = ds.shuffle(seed=SEED).select(range(576))
    tr, ev = ds.select(range(512)), ds.select(range(512, 576))

    def tk(ex):
        ctx = ("\n" + ex["context"]) if ex.get("context") else ""
        prompt = f"### Instruction:\n{ex['instruction']}{ctx}\n### Response:\n"
        p = tok(prompt, add_special_tokens=False)["input_ids"]
        r = tok(ex["response"] + tok.eos_token, add_special_tokens=False)["input_ids"]
        ids = (p + r)[:MAXLEN]; labels = ([-100] * len(p) + r)[:MAXLEN]
        pad = MAXLEN - len(ids)
        return dict(input_ids=ids + [tok.pad_token_id] * pad,
                    attention_mask=[1] * len(ids) + [0] * pad, labels=labels + [-100] * pad)
    cols = tr.column_names
    return tr.map(tk, remove_columns=cols), ev.map(tk, remove_columns=cols), fp


@torch.no_grad()
def sft_loss(model, eval_ds, iso=None, order=None):
    model.eval()
    idxs = order if order is not None else range(len(eval_ds))
    per = {}
    for i in idxs:
        if iso is not None:
            iso.reset()
        b = eval_ds[i]
        loss = model(input_ids=torch.tensor([b["input_ids"]], device=DEV),
                     attention_mask=torch.tensor([b["attention_mask"]], device=DEV),
                     labels=torch.tensor([b["labels"]], device=DEV)).loss
        per[i] = float(loss) if torch.isfinite(loss) else None
    vals = [v for v in per.values() if v is not None]
    return round(sum(vals) / len(vals), 4), per


@torch.no_grad()
def wikitext_ppl(model, tok, iso=None, n=50, seqlen=512):
    model.eval()
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    ids = tok("\n\n".join(t for t in ds["text"] if t.strip()), add_special_tokens=False)["input_ids"]
    losses = []
    for i in range(n):
        chunk = ids[i * seqlen:(i + 1) * seqlen]
        if len(chunk) < seqlen:
            break
        if iso is not None:
            iso.reset()
        t = torch.tensor([chunk], device=DEV)
        losses.append(float(model(input_ids=t, labels=t).loss))
    return round(math.exp(sum(losses) / len(losses)), 3)


@torch.no_grad()
def retention(model, tok, target_tokens, iso=None, n=20):
    """Historical single-key probe (NON-DISCRIMINATIVE per calibration) — kept for continuity."""
    model.eval()
    rnd = random.Random(1234 + target_tokens)
    filler = tok("The grass is green and the sky is blue. Nothing here. ",
                 add_special_tokens=False)["input_ids"]
    ok = 0
    for _ in range(n):
        if iso is not None:
            iso.reset()
        key = "".join(rnd.choice("ABCDEFGHJKLMNPQRSTUVWXYZ") for _ in range(5))
        val = str(rnd.randint(1000000, 9999999)); dis = str(rnd.randint(1000000, 9999999))
        needle = tok(f" The magic number for {key} is {val}. ", add_special_tokens=False)["input_ids"]
        q = tok(f"\nWhat is the magic number for {key}? Answer:", add_special_tokens=False)["input_ids"]
        body = []
        while len(body) < target_tokens - len(needle) - len(q):
            body += filler
        body = body[:max(0, target_tokens - len(needle) - len(q))]
        d = len(body) // 2
        ctx = (body[:d] + needle + body[d:] + q)[:target_tokens]

        def nll(ans):
            a = tok(" " + ans, add_special_tokens=False)["input_ids"]
            return float(model(input_ids=torch.tensor([ctx + a], device=DEV),
                               labels=torch.tensor([[-100] * len(ctx) + a], device=DEV)).loss)
        if nll(val) < nll(dis):
            ok += 1
    return round(ok / n, 3)


@torch.no_grad()
def prefill_ms(model, tok, iso=None):
    model.eval()
    ids = tok("The quick brown fox " * 100, return_tensors="pt").to(DEV)
    ids = {k: v[:, :400] for k, v in ids.items()}
    torch.cuda.synchronize(); t0 = time.time()
    for _ in range(3):
        if iso is not None:
            iso.reset()
        model(**ids)
    torch.cuda.synchronize()
    return round((time.time() - t0) / 3 * 1000, 1)


def evaluate(model, tok, eval_ds, iso=None):
    sft, per = sft_loss(model, eval_ds, iso)
    return dict(sft_loss=sft, wikitext_ppl=wikitext_ppl(model, tok, iso),
                retention_acc_ctx256=retention(model, tok, 256, iso),
                retention_acc_ctx1024=retention(model, tok, 1024, iso),
                prefill_ms_400tok=prefill_ms(model, tok, iso)), per


def train_arm(model, arm, train_ds, outdir, iso=None):
    args = TrainingArguments(
        output_dir=os.path.join(outdir, f"trainer_{arm}"), per_device_train_batch_size=2,
        gradient_accumulation_steps=4, num_train_epochs=3, learning_rate=2e-4, weight_decay=0.0,
        lr_scheduler_type="cosine", warmup_steps=10, logging_steps=40, save_strategy="no",
        seed=SEED, data_seed=SEED, report_to=[], disable_tqdm=True, dataloader_num_workers=0)
    cbs = [tptt.LiZACallback(model, mode="gradual", initial_weight=0.0, final_weight=0.5,
                             transition_step=96)] if arm == "tptt" else []
    tr = ResetTrainer(model=model, args=args, train_dataset=train_ds,
                      data_collator=default_data_collator, callbacks=cbs)
    tr._iso = iso
    torch.cuda.reset_peak_memory_stats(); t0 = time.time()
    out = tr.train(); wall = time.time() - t0
    toks = out.global_step * 2 * 4 * MAXLEN
    return dict(train_loss=out.training_loss, steps=out.global_step, wall_s=round(wall, 1),
                tokens=toks, tok_per_s=round(toks / wall, 1),
                peak_vram_mb=round(torch.cuda.max_memory_allocated() / 1e6, 1),
                trainable_all=ntrain(model))


def run(args):
    os.makedirs(args.outdir, exist_ok=True)
    torch.manual_seed(SEED); random.seed(SEED)
    tok = AutoTokenizer.from_pretrained(MODEL)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    cfg = AutoConfig.from_pretrained(MODEL)
    train_ds, eval_ds, fp = make_dataset(tok)
    R = dict(meta=dict(model=MODEL, dtype="float32", seed=SEED, maxlen=MAXLEN,
                       dataset_fingerprint=fp, train_n=len(train_ds), eval_n=len(eval_ds),
                       torch=torch.__version__, tptt_commit="242e2140c2af469765b84ab3a7a79668be254cfa",
                       MEMORY_AXIS="NOT_QUALIFIED (calibration: all grid configs base_acc=1.0)"),
             arms={}, training={}, mechanism={}, order_invariance={})

    def snap():
        json.dump(R, open(os.path.join(args.outdir, "rnn08b_results.json"), "w"), indent=2)

    # A BASE
    m = build_base(DEV, DT)
    R["arms"]["base"], _ = evaluate(m, tok, eval_ds, None)
    del m; torch.cuda.empty_cache(); snap()

    # B LoRA-only
    m = build_lora(DEV, DT)
    R["training"]["lora"] = train_arm(m, "lora", train_ds, args.outdir, None)
    R["arms"]["lora"], _ = evaluate(m, tok, eval_ds, None)
    m.save_pretrained(os.path.join(args.outdir, "adapter_lora"))
    del m; torch.cuda.empty_cache(); snap()

    # C TPTT+LoRA (corrected: per-batch + per-example reset)
    m, lc = build_tptt(cfg, DEV, DT)
    iso = IndependentSequenceTPTT(lc)
    R["mechanism"] = dict(liza_modules_replaced=sum(1 for _ in m.modules()
                          if type(_).__name__ == "LiZAttention"),
                          trainable_all=ntrain(m), state_bytes_per_request=5935104,
                          state_note="from gates: 144 tensors, delta state [1,14,64,64] fp32, ~5.66 MiB")
    R["training"]["tptt"] = train_arm(m, "tptt", train_ds, args.outdir, iso)
    tptt_metrics, per_tptt = evaluate(m, tok, eval_ds, iso)
    R["arms"]["tptt"] = tptt_metrics
    # order-invariance smoke (§12): same 3 examples, order ABC vs CAB, WITH per-example reset
    _, perA = sft_loss(m, eval_ds, iso, order=[0, 1, 2])
    _, perB = sft_loss(m, eval_ds, iso, order=[2, 0, 1])
    inv = max(abs(perA[i] - perB[i]) for i in [0, 1, 2])
    R["order_invariance"] = dict(order_ABC={str(k): perA[k] for k in [0, 1, 2]},
                                 order_CAB={str(k): perB[k] for k in [0, 1, 2]},
                                 max_abs_diff=inv, status="PASS" if inv == 0.0 else "FAIL")
    try:
        os.makedirs(os.path.join(args.outdir, "adapter_tptt"), exist_ok=True)
        tptt.save_tptt_safetensors(m, os.path.join(args.outdir, "adapter_tptt"),
                                   "adapter_model.safetensors")
    except Exception as e:
        R["tptt_save_err"] = repr(e)[:200]
    del m; torch.cuda.empty_cache(); snap()
    print(json.dumps(dict(arms=R["arms"], training=R["training"],
                          order_invariance=R["order_invariance"],
                          mechanism={k: R["mechanism"][k] for k in ["liza_modules_replaced",
                                     "trainable_all", "state_bytes_per_request"]}), indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", required=True)
    run(ap.parse_args())
