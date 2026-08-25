# ADAPT-00A adapter-mechanics preflight - result

## Verdict

`PASS`; open the frozen ADAPT-00B geometry matrix.

The official 0.8B base loaded without custom model code, a rank-8 LoRA trained
on the frozen ThinkingCap-derived panel, all preregistered learning/retention/
reload/VRAM gates passed, and the canonical Fable service was restored.

## Frozen identities

- Base: `Qwen/Qwen3.5-0.8B-Base`
- Revision: `dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68`
- Loaded class: `Qwen3_5ForCausalLM`
- Base parameters: 752,393,024
- Teacher receipt SHA-256:
  `dc5cabe44c92e48b0e832881ef27ebad4047b140928c9a12678e0c0c6660006e`
- Prompt snapshot SHA-256:
  `68a72276898a45dacb893154477621ff3d05ae7e043a10820644b691d8b63d77`
- Protected README SHA-256:
  `60b5bcff20064963ee37e8c98b6805db491d5270ce528fa8c201ffd01fb45ea5`
- Environment: CPython 3.11.15, PyTorch 2.5.1+cu124,
  Transformers 5.15.1, PEFT 0.20.0
- Raw machine-readable receipt: `raw/metrics.json`

## Configuration

- 200 valid teacher/prompt joins
- frozen seed 20260824
- 128 train records and 32 disjoint target-loss records
- 16 protected next-token blocks from the repository README
- LoRA rank 8, alpha 16, dropout 0, all linear modules
- 5,411,328 trainable parameters
- 24 AdamW steps at learning rate 2e-4, batch size 1
- BF16, maximum sequence length 384, no model quantization

## Results

| Gate | Measurement | Result |
|---|---:|---|
| Finite losses | all finite | `PASS` |
| Nonzero trainable gradient | observed | `PASS` |
| Held-out target loss | 0.659118 -> 0.404571 (-38.62%) | `PASS` vs >=1% improvement |
| Protected-text loss | 2.985595 -> 3.001408 (+0.53%) | `PASS` vs <=15% regression |
| Clean adapter reload | 0.404571; 0.00% delta | `PASS` vs <=0.5% delta |
| Peak allocated VRAM | 4.876 GiB | `PASS` vs <23 GiB |
| Timed train/eval/save/reload section | 16.39 s | descriptive |

The first and last optimizer-step losses are not a learning curve because each
step uses a different example. The preregistered disjoint aggregate target loss
is the learning endpoint.

## Operational restoration

Only `llm-inference.service` was stopped. The embedding service stayed active.
After the run, both services were active, GPU use returned to approximately
20.9 GiB, and the no-thinking Fable canary returned exactly
`adapt00-baseline-restored-ok` with `finish_reason=stop`. Fan Control and MSI
Afterburner were untouched.

## Scope

This proves training mechanics on the 0.8B base and frozen corpus. It does not
show generated-answer accuracy or concision, does not select LoRA over another
geometry, and does not imply transfer to the 27B production merge. Those claims
belong to later gated stages.
