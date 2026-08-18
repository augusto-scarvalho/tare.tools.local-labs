#!/usr/bin/env python
"""
RNN-08b — single explicit TPTT state-lifecycle mechanism (§4).

IndependentSequenceTPTT resets the shared TPTT recurrent memory (LCache) at each INDEPENDENT
OUTER-sequence boundary (one training batch / one eval example). Intra-sequence chunk recurrence is
left untouched. reset() runs BETWEEN forwards, so it never detaches or disables gradients of the
current forward's graph. This is the ONE mechanism used everywhere (no ad-hoc reset() calls).
"""
import torch
from transformers import AutoModelForCausalLM
from peft import LoraConfig, get_peft_model
import tptt

MODEL = "Qwen/Qwen2.5-0.5B"
LORA = dict(r=8, lora_alpha=16, target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            lora_dropout=0.0, bias="none", task_type="CAUSAL_LM")
TPTT_KW = dict(operator_mode="delta_rule", mag_weight=0.5, linear_precision=torch.float32,
               use_linear_checkpoint=True, max_chunk_size=32)


def build_base(dev, dt=torch.float32):
    return AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=dt).to(dev)


def build_lora(dev, dt=torch.float32):
    b = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=dt)
    return get_peft_model(b, LoraConfig(**LORA)).to(dev)


def build_tptt(cfg, dev, dt=torch.float32):
    b = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=dt)
    m, lc = tptt.get_tptt_model(b, cfg, **TPTT_KW)
    m = get_peft_model(m, LoraConfig(**LORA)).to(dev)
    return m, lc


def _collect(obj, out):
    if torch.is_tensor(obj):
        if obj.numel() > 0:
            out.append(obj)
    elif isinstance(obj, dict):
        for v in obj.values():
            _collect(v, out)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            _collect(v, out)
    return out


class IndependentSequenceTPTT:
    def __init__(self, lcache):
        self.lc = lcache

    def reset(self):
        self.lc.reset()

    def state_tensors(self):
        return _collect(getattr(self.lc, "inputs_states", None), [])

    def state_report(self):
        ts = self.state_tensors()
        total_bytes = sum(t.numel() * t.element_size() for t in ts)
        sq = sum(float(t.detach().float().pow(2).sum()) for t in ts)
        shapes = [[list(t.shape), str(t.dtype).replace("torch.", "")] for t in ts[:4]]
        return dict(n_tensors=len(ts), total_bytes=total_bytes,
                    l2_norm=round(sq ** 0.5, 4), sample_shapes=shapes)

    def __enter__(self):
        self.reset()
        return self

    def __exit__(self, *a):
        return False
