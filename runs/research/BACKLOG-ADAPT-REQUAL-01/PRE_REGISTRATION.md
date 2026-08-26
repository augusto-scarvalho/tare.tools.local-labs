# BACKLOG-ADAPT-REQUAL-01 preregistration

Task: Requalify saved ADAPT-01A through ADAPT-05 artifacts
Evidence class: `artifact_requalification`

## Hypothesis

The 13 saved adapter artifacts from ADAPT-01A through ADAPT-05 can be loaded into an isolated offline evaluation runtime on the unadapted base model `Qwen/Qwen3.5-0.8B-Base` (revision `dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68`), reproducing deterministic behavioral outputs across the frozen 32-sample math panel (GSM8K) and 16-sample protected ordinary-QA panel, enabling independent qualification without establishing training reproducibility.

## Frozen inputs

Source directories and artifact ledgers:
- `runs/research/ADAPT-01A-LOKR-SCALE-2026-08-25`:
  - `raw/lokr_1ep/adapter/adapter_config.json` (823 bytes, SHA-256: `382fc9c7d5bd6397ecf7fcd7c0eb1168a5f55d368e687b7b37573d7de4d36d34`)
  - `raw/lokr_1ep/adapter/adapter_model.safetensors` (1505752 bytes, SHA-256: `d8dce041cb77246fc39af719d444d9958851109262dea8554eca79d031d466b1`)
  - `raw/lokr_3ep/adapter/adapter_config.json` (823 bytes, SHA-256: `8abe9a3c7a9e35898718ec8fd76ab8b4b5f264fe6ed0d73aafc3774e7707062e`)
  - `raw/lokr_3ep/adapter/adapter_model.safetensors` (1505752 bytes, SHA-256: `ca51b85cdecbf4163378f4cdc9d31ce216d7fc8c1d00f3074d9cb3945806df7d`)
  - `raw/lokr_3ep_lr1e4/adapter/adapter_config.json` (823 bytes, SHA-256: `256f52486955d168a6c0eccf0a9c5f6e690e632c1831ce21d1f2802f3842d33e`)
  - `raw/lokr_3ep_lr1e4/adapter/adapter_model.safetensors` (1505752 bytes, SHA-256: `ec1a719936fc09f070eeb02a364f5b02f15c0035cd1176467863dbede3e66c53`)
  - `raw/lokr_5ep/adapter/adapter_config.json` (823 bytes, SHA-256: `d3067f6a9832d8701a9af6cc93d80256e4b5303f0904dc6cc93c64bee94047f1`)
  - `raw/lokr_5ep/adapter/adapter_model.safetensors` (1505752 bytes, SHA-256: `fcbf25ed7f74814e82c5a733fa540652a0b4377ffe1210164e7662a8058b06ed`)
- `runs/research/ADAPT-02-MODULE-TARGETING-2026-08-25`:
  - `raw/target_all_linear/adapter/adapter_config.json` (737 bytes, SHA-256: `726d06e5f34f82a12b9d83464b9aaf9d08c7085f4acb435673945af9200ab944`)
  - `raw/target_all_linear/adapter/adapter_model.safetensors` (931712 bytes, SHA-256: `7e7aad095dd884888aff915d51e105f23ac6b73b559007fc3be1f0588111d61c`)
  - `raw/target_attn_only/adapter/adapter_config.json` (688 bytes, SHA-256: `8516576a6d6a79f13bddd2388483d0638804d3367f661041be9c1b99aa0008fd`)
  - `raw/target_attn_only/adapter/adapter_model.safetensors` (168568 bytes, SHA-256: `839d777b848ec202349266fc271aaacd4eff3078bb68e8a46f341dbc9b3194eb`)
  - `raw/target_mlp_only/adapter/adapter_config.json` (681 bytes, SHA-256: `45067f22d87e53ba56114cd0126c20d0591cefc5c9261a1de6c83b705f56e784`)
  - `raw/target_mlp_only/adapter/adapter_model.safetensors` (763088 bytes, SHA-256: `3fda4d2bae7c6388e97fc69c3c2e4de5d85a614e99f436d8c04373ced3b38966`)
  - `raw/target_qv_gate/adapter/adapter_config.json` (677 bytes, SHA-256: `e8c3984f473e26ed59ab8419533f364f181ac9984980cc6b0ac4cb160f2e1726`)
  - `raw/target_qv_gate/adapter/adapter_model.safetensors` (350992 bytes, SHA-256: `6ab6fedf5761d68c52c52fbf5f72d139d5efe4e683d33d1cc29becb2ec7a248c`)
- `runs/research/ADAPT-03-SOFT-PROMPTS-2026-08-25`:
  - `raw/adapter/adapter_config.json` (636 bytes, SHA-256: `651a139860958d3a7c6afaecab886ee68c9affec14f0668d0bf90d06b078fc73`)
  - `raw/adapter/adapter_model.safetensors` (32888 bytes, SHA-256: `be28a069469a287d38c8ec170e67564d9c6af7fa9231532c4442bfe90901c939`)
- `runs/research/ADAPT-04-PRIOR-PRESERVATION-2026-08-25`:
  - `raw/lokr_prior_lambda02/adapter/adapter_config.json` (823 bytes, SHA-256: `d79537207343f18aeb4908d8a9f16ca848ec47a3b72e51496bdc046ba25a7b2f`)
  - `raw/lokr_prior_lambda02/adapter/adapter_model.safetensors` (1505752 bytes, SHA-256: `4cd386da0ef2a96bfb3e728ee55ba5aa05482b1f3413a181ba99c3c91c1ed12e`)
  - `raw/lokr_prior_lambda05/adapter/adapter_config.json` (823 bytes, SHA-256: `d79537207343f18aeb4908d8a9f16ca848ec47a3b72e51496bdc046ba25a7b2f`)
  - `raw/lokr_prior_lambda05/adapter/adapter_model.safetensors` (1505752 bytes, SHA-256: `3b4edfd4be27c4ac9ac4ce6f58b7185d5c04207288b837035d823930f5d5dfc7`)
  - `raw/lokr_unreg_5ep/adapter/adapter_config.json` (823 bytes, SHA-256: `d79537207343f18aeb4908d8a9f16ca848ec47a3b72e51496bdc046ba25a7b2f`)
  - `raw/lokr_unreg_5ep/adapter/adapter_model.safetensors` (1505752 bytes, SHA-256: `57e90c620fd773b79a4dfbc6dd3a08324bd61cd291d5f05e1e36119f39ca34d9`)
- `runs/research/ADAPT-05-MODULAR-MERGING-2026-08-25`:
  - `raw/disjoint_composite/adapter_config.json` (738 bytes, SHA-256: `2942f4f378d5182a86a1712a930169e1a88bbc6a1b4f7048023a098cb3008c9c`)
  - `raw/disjoint_composite/adapter_model.safetensors` (931680 bytes, SHA-256: `30c5ae07536169fed67260c77243fca816b167eae616078e41e1acfa935286d0`)

Base model & datasets:
- Base model weights: `/home/augus/models/adapt00/qwen3.5-0.8b-base-dc7cdfe/model.safetensors-00001-of-00001.safetensors` (1746942600 bytes, SHA-256: `c2b1e5a17d9c1e27685d92ed9b382911ebb99955ecd89052d1721241adfbab6c`)
- Base config: `/home/augus/models/adapt00/qwen3.5-0.8b-base-dc7cdfe/config.json` (2907 bytes, SHA-256: `b90b86f35c8e6925ef74ee04d0e758f0a845c83a42089ad82bbaa948de9b4204`)
- Base tokenizer: `/home/augus/models/adapt00/qwen3.5-0.8b-base-dc7cdfe/tokenizer.json` (12807196 bytes, SHA-256: `fe000e3ed39ed12b8d2481d527d44f93c65d37e87645d2dcc80d1bf9d50d2927`)
- Math panel: `workloads/gsm8k.jsonl` (389701 bytes, SHA-256: `68a72276898a45dacb893154477621ff3d05ae7e043a10820644b691d8b63d77`)
- Protected QA panel: `runs/requalification/QWEN38-HAUHAUCS-NORMAL-QA-2026-08-23/tasks.jsonl` (11016 bytes, SHA-256: `56434ebd53ff9f5adb477bd902566e28644fe9ea88619834f81417a06c66b84f`)

## Command

```powershell
python tools/research/run_adapter_requalification.py --outdir runs/research/BACKLOG-ADAPT-REQUAL-01
```

## Factors

- Arms: 1 unadapted base control (`base`) + 13 saved adapter arms = 14 evaluated arms total.
- Math panel: 32 disjoint GSM8K task IDs (`gsm8k/392`, `gsm8k/1226`, `gsm8k/541`, `gsm8k/44`, `gsm8k/489`, `gsm8k/1298`, `gsm8k/663`, `gsm8k/1217`, `gsm8k/1186`, `gsm8k/225`, `gsm8k/110`, `gsm8k/174`, `gsm8k/986`, `gsm8k/173`, `gsm8k/317`, `gsm8k/529`, `gsm8k/236`, `gsm8k/831`, `gsm8k/86`, `gsm8k/19`, `gsm8k/967`, `gsm8k/724`, `gsm8k/1001`, `gsm8k/1212`, `gsm8k/1264`, `gsm8k/662`, `gsm8k/34`, `gsm8k/1294`, `gsm8k/551`, `gsm8k/175`, `gsm8k/430`, `gsm8k/386`).
- Protected QA panel: 16 tasks (`f01`, `f02`, `f03`, `m01`, `m02`, `m03`, `r01`, `r02`, `r03`, `i01`, `i02`, `i03`, `c01`, `c02`, `s01`, `s02`).
- Decoding contract: Greedy decoding (`temperature=0.0`, `do_sample=False`), seed=20260824. Max new tokens: 192 for math, 128 for QA.
- Hardware / Runtime: NVIDIA GeForce RTX 3090, WSL2 Ubuntu-24.04, Python virtual environment `/home/augus/.venvs/adapt00-20260824` (PyTorch 2.5.1+cu124, Transformers 5.15.1, PEFT 0.20.0).

## Acceptance gates

- `artifact_identity`: `hashed_artifacts eq 13`
- `frozen_math_panel`: `scored_math_samples_per_arm ge 32`
- `frozen_qa_panel`: `scored_qa_samples_per_arm ge 16`
- `base_control`: `base_control_present eq True`
- `independent_score`: `independent_scorer_match eq True`

## Abort conditions

- Missing or corrupted adapter weights or configs.
- Base model weights hash mismatch.
- Dataset hash mismatch on math or protected panels.
- Out of memory (OOM) or CUDA hardware execution faults.
- Incomplete provenance or failure of independent scorer verification.

## Allowed claims

- `ARTIFACT_REQUALIFIED`
- `ARTIFACT_REJECTED`

Claims outside these codes are forbidden even if a metric looks favorable.
