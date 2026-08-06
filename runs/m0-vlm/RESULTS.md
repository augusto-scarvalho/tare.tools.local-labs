# M0 VLM accept — raw evidence (2026-08-06)

Fixtures (known ground truth): `error_dialog.png`, `ui_mockup.png` (regenerate with
`gen_test_images.ps1`). Probe: repo-root `vlm_probe.py`. Full write-up: `../../M_A_VLM.md`.

Served each via `python lmctl.py serve <profile>` (:8092), then:
`python vlm_probe.py <img> "<prompt>" --port 8092`.

## error_dialog.png — "Transcribe every line…"  (ground truth: Application Error / Unhandled
## exception: NullReferenceException / at PaymentService.Charge(order=4471) line 132. / The
## transaction was not completed. Retry? / Retry / Cancel)
- **qwen3-vl-8b** (0.6s): exact, all lines + buttons. PASS.
- **qwen3-vl-30b** (0.7s): exact (icons X/! omitted). PASS.
- **gemma-4-12b** (6.3s, max_tokens=1024): exact. PASS. (At max_tokens=512 content came back
  EMPTY — thinking budget consumed; see the gotcha in M_A_VLM.md.)

## ui_mockup.png — "list every field label, placeholder, link, button top to bottom"
- **qwen3-vl-8b** (2.9s): all elements + correctly split "Don't have an account?" (text) vs
  "Sign up" (link). Most detailed. PASS.
- **qwen3-vl-30b** (0.8s): all elements, concise. PASS.
- **gemma-4-12b** (9.0s): fields/links/button correct but MISSED the "Sign in to Acme" heading +
  subheading. PASS-with-omission.

## VRAM (nvidia-smi, @-ngl 99, 8k ctx)
qwen3-vl-8b 8886 MiB · gemma-4-12b 10470 MiB · qwen3-vl-30b 20550 MiB (3773 free — under the 4 GB
reserve; the one config that is envelope-tight).
