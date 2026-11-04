#!/usr/bin/env python
"""
RNN-08/09 — BASE vs LoRA-only vs TPTT+LoRA, budget-matched (§1,§9,§11).
Evaluates 3 axes (§12): held-out SFT loss, RULER-style retention (teacher-forced, no generation),
wikitext perplexity. Mechanism + state-isolation proof (§16). Isolated venv only. float32.

Outcome vocabulary is applied in the analyzer, not here. This script produces raw normalized numbers.
"""
import argparse, json, os, time, platform, subprocess, random, copy
import torch
from transformers import (AutoTokenizer, AutoModelForCausalLM, AutoConfig, Trainer,
                          TrainingArguments, default_data_collator, TrainerCallback)
from peft import LoraConfig, get_peft_model
import tptt
from datasets import load_dataset

MODEL = "Qwen/Qwen2.5-0.5B"
MAXLEN = 256
SEED = 42
LORA = dict(r=8, lora_alpha=16, target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            lora_dropout=0.0, bias="none", task_type="CAUSAL_LM")
DEV = "cuda"
DT = torch.float32


def ntrain(m):
    return (sum(p.numel() for p in m.parameters() if p.requires_grad),
            sum(p.numel() for p in m.parameters()))


# ---------------- data ----------------
def format_example(ex):
    ctx = ("\n" + ex["context"]) if ex.get("context") else ""
    prompt = f"### Instruction:\n{ex['instruction']}{ctx}\n### Response:\n"
    return prompt, ex["response"]


def make_dataset(tok):
    ds = load_dataset("databricks/databricks-dolly-15k", split="train")
    fp = ds._fingerprint
    ds = ds.shuffle(seed=SEED).select(range(576))
    train_raw, eval_raw = ds.select(range(512)), ds.select(range(512, 576))

    def tokenize(ex):
        prompt, resp = format_example(ex)
        p_ids = tok(prompt, add_special_tokens=False)["input_ids"]
        r_ids = tok(resp + tok.eos_token, add_special_tokens=False)["input_ids"]
        ids = (p_ids + r_ids)[:MAXLEN]
        labels = ([-100] * len(p_ids) + r_ids)[:MAXLEN]
        pad = MAXLEN - len(ids)
        ids = ids + [tok.pad_token_id] * pad
        labels = labels + [-100] * pad
        attn = [1] * (MAXLEN - pad) + [0] * pad
        return dict(input_ids=ids, attention_mask=attn, labels=labels)

    cols = train_raw.column_names
    return (train_raw.map(tokenize, remove_columns=cols),
            eval_raw.map(tokenize, remove_columns=cols), fp, eval_raw)


# ---------------- models ----------------
def build(arm, cfg):
    base = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=DT)
    if arm == "base":
        return base.to(DEV)
    if arm == "lora":
        return get_peft_model(base, LoraConfig(**LORA)).to(DEV)
    if arm == "tptt":
        m, lcache = tptt.get_tptt_model(base, cfg, operator_mode="delta_rule", mag_weight=0.5,
                                        linear_precision=torch.float32,
                                        use_linear_checkpoint=True, max_chunk_size=32)
        m = get_peft_model(m, LoraConfig(**LORA)).to(DEV)
        m._tptt_lcache = lcache
        return m
    raise ValueError(arm)


def n_liza(model):
    from tptt import LiZAttention
    return sum(1 for _ in model.modules() if isinstance(_, LiZAttention))


# ---------------- training ----------------
def train_arm(model, arm, train_ds, outdir):
    args = TrainingArguments(
        output_dir=os.path.join(outdir, f"trainer_{arm}"),
        per_device_train_batch_size=2, gradient_accumulation_steps=4,
        num_train_epochs=3, learning_rate=2e-4, weight_decay=0.0,
        lr_scheduler_type="cosine", warmup_steps=10, logging_steps=20,
        save_strategy="no", seed=SEED, data_seed=SEED, report_to=[], fp16=False, bf16=False,
        dataloader_num_workers=0, disable_tqdm=True)
    callbacks = []
    if arm == "tptt":
        callbacks.append(tptt.LiZACallback(model, mode="gradual", initial_weight=0.0,
                                           final_weight=0.5, transition_step=96))
    trainer = Trainer(model=model, args=args, train_dataset=train_ds,
                      data_collator=default_data_collator, callbacks=callbacks)
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    out = trainer.train()
    wall = time.time() - t0
    steps = out.global_step
    tokens = steps * 2 * 4 * MAXLEN
    return dict(train_loss=out.training_loss, steps=steps, wall_s=round(wall, 1),
                tokens=tokens, tok_per_s=round(tokens / wall, 1),
                peak_vram_mb=round(torch.cuda.max_memory_allocated() / 1e6, 1),
                trainable_all=ntrain(model))


# ---------------- evals ----------------
@torch.no_grad()
def sft_loss(model, eval_ds):
    model.eval()
    tot, n = 0.0, 0
    for i in range(len(eval_ds)):
        b = eval_ds[i]
        ids = torch.tensor([b["input_ids"]], device=DEV)
        am = torch.tensor([b["attention_mask"]], device=DEV)
        lb = torch.tensor([b["labels"]], device=DEV)
        loss = model(input_ids=ids, attention_mask=am, labels=lb).loss
        if torch.isfinite(loss):
            tot += float(loss); n += 1
    return round(tot / max(n, 1), 4)


@torch.no_grad()
def answer_nll(model, tok, context_ids, answer):
    a_ids = tok(answer, add_special_tokens=False)["input_ids"]
    ids = torch.tensor([context_ids + a_ids], device=DEV)
    labels = torch.tensor([[-100] * len(context_ids) + a_ids], device=DEV)
    return float(model(input_ids=ids, labels=labels).loss)


@torch.no_grad()
def ruler_retention(model, tok, target_tokens, n=20):
    model.eval()
    rnd = random.Random(1234 + target_tokens)
    correct = 0
    filler = tok("The grass is green and the sky is blue. Nothing important here. ",
                 add_special_tokens=False)["input_ids"]
    for _ in range(n):
        key = "".join(rnd.choice("ABCDEFGHJKLMNPQRSTUVWXYZ") for _ in range(5))
        val = str(rnd.randint(1000000, 9999999))
        distract = str(rnd.randint(1000000, 9999999))
        needle = tok(f" The magic number for {key} is {val}. ", add_special_tokens=False)["input_ids"]
        q = tok(f"\nWhat is the magic number for {key}? Answer:", add_special_tokens=False)["input_ids"]
        body = []
        while len(body) < target_tokens - len(needle) - len(q):
            body += filler
        body = body[:max(0, target_tokens - len(needle) - len(q))]
        depth = int(len(body) * 0.5)
        ctx = (body[:depth] + needle + body[depth:] + q)[:target_tokens]
        if answer_nll(model, tok, ctx, " " + val) < answer_nll(model, tok, ctx, " " + distract):
            correct += 1
    return round(correct / n, 3)


@torch.no_grad()
def wikitext_ppl(model, tok, n=50, seqlen=512):
    model.eval()
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n\n".join(t for t in ds["text"] if t.strip())
    all_ids = tok(text, add_special_tokens=False)["input_ids"]
    losses = []
    for i in range(n):
        s = i * seqlen
        chunk = all_ids[s:s + seqlen]
        if len(chunk) < seqlen:
            break
        ids = torch.tensor([chunk], device=DEV)
        losses.append(float(model(input_ids=ids, labels=ids).loss))
    import math
    return round(math.exp(sum(losses) / len(losses)), 3)


@torch.no_grad()
def state_isolation(model, tok):
    """Sample B logits must be identical whether or not sample A ran first (§16)."""
    model.eval()
    A = tok("Alpha alpha alpha unrelated context number 111.", return_tensors="pt").to(DEV)
    B = tok("What is two plus two?", return_tensors="pt").to(DEV)
    lgB0 = model(**B).logits
    _ = model(**A)
    lgB1 = model(**B).logits
    diff = float((lgB0 - lgB1).abs().max())
    return dict(max_abs_diff=diff, isolated=bool(diff == 0.0))


@torch.no_grad()
def infer_latency(model, tok):
    model.eval()
    ids = tok("The quick brown fox " * 100, return_tensors="pt").to(DEV)
    ids = {k: v[:, :400] for k, v in ids.items()}
    torch.cuda.synchronize(); t0 = time.time()
    for _ in range(3):
        model(**ids)
    torch.cuda.synchronize()
    return round((time.time() - t0) / 3 * 1000, 1)


def evaluate(model, tok, eval_ds, tag):
    r = dict(arm=tag)
    for key, fn in [("sft_loss", lambda: sft_loss(model, eval_ds)),
                    ("retention_acc_ctx256", lambda: ruler_retention(model, tok, 256)),
                    ("retention_acc_ctx1024", lambda: ruler_retention(model, tok, 1024)),
                    ("wikitext_ppl", lambda: wikitext_ppl(model, tok)),
                    ("prefill_ms_400tok", lambda: infer_latency(model, tok))]:
        try:
            r[key] = fn()
        except Exception as e:
            r[key] = None; r[key + "_err"] = repr(e)[:150]
            torch.cuda.empty_cache()
    return r


def run(args):
    os.makedirs(args.outdir, exist_ok=True)
    torch.manual_seed(SEED); random.seed(SEED)
    tok = AutoTokenizer.from_pretrained(MODEL)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    cfg = AutoConfig.from_pretrained(MODEL)
    train_ds, eval_ds, fp, _ = make_dataset(tok)

    R = dict(meta=dict(model=MODEL, dtype="float32", seed=SEED, maxlen=MAXLEN,
                       dataset="databricks/databricks-dolly-15k", dataset_fingerprint=fp,
                       train_n=len(train_ds), eval_n=len(eval_ds),
                       torch=torch.__version__, python=platform.python_version(),
                       tptt_commit="242e2140c2af469765b84ab3a7a79668be254cfa"),
             arms={}, training={}, mechanism={}, state_isolation={})

    resultf = os.path.join(args.outdir, "rnn08_results.json")

    def snapshot():
        with open(resultf, "w") as f:
            json.dump(R, f, indent=2)

    # A BASE
    m = build("base", cfg)
    R["arms"]["base"] = evaluate(m, tok, eval_ds, "base")
    del m; torch.cuda.empty_cache(); snapshot()

    # B LoRA-only
    m = build("lora", cfg)
    R["training"]["lora"] = train_arm(m, "lora", train_ds, args.outdir)
    R["arms"]["lora"] = evaluate(m, tok, eval_ds, "lora")
    m.save_pretrained(os.path.join(args.outdir, "adapter_lora"))
    del m; torch.cuda.empty_cache(); snapshot()

    # C TPTT+LoRA
    m = build("tptt", cfg)
    R["mechanism"] = dict(liza_modules_replaced=n_liza(m),
                          trainable_all=ntrain(m),
                          shared_linear_cache=str(type(getattr(m, "_tptt_lcache", None)).__name__))
    R["training"]["tptt"] = train_arm(m, "tptt", train_ds, args.outdir)
    R["state_isolation"] = state_isolation(m, tok)
    R["arms"]["tptt"] = evaluate(m, tok, eval_ds, "tptt")
    try:
        tptt.save_tptt_safetensors(m, os.path.join(args.outdir, "adapter_tptt"),
                                   "adapter_model.safetensors")
    except Exception as e:
        R["tptt_save_err"] = repr(e)[:200]
    del m; torch.cuda.empty_cache()
    snapshot()
    print(json.dumps(dict(arms=R["arms"], training=R["training"], mechanism=R["mechanism"],
                          state_isolation=R["state_isolation"]), indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", required=True)
    run(ap.parse_args())
