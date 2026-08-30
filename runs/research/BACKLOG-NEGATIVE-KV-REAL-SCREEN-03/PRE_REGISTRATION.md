# BACKLOG-NEGATIVE-KV-REAL-SCREEN-03 preregistration

Task: Repeat the five-candidate real Qwen KV screen with decisive tensor retention
Evidence class: `artifact_requalification`

## Hypothesis

Independent replay from retained physical Qwen tensors will reproduce the 78
candidate-cell metrics and confirm that none of the five frozen negative
decisions reverses. A candidate is a false negative only if every one of its
original conjunctive thresholds passes after replay.

## Frozen inputs

- Admission: `a9f11108e986e7ccdb1c7fa134d67214e13cf84cfb0e08e9e59dd7eb39548234`.
- R2 receipt and independent HOLD review:
  `18fac16ec34a64258fec6e83f36aed713803eb0766fb4f4e382bacc1fc57fc4e`,
  `fe967a16c32a16a49d59bc511ebfcf1d961b374ad96a4cbe35eb1bf897080252`.
- Frozen R2 wrapper, host and worker:
  `579e8225464fd0b8f9d21968c08fe9d7f6d12b86445b641ccb11b37d2c5d46b7`,
  `5295afff2c3f8e4fe0ce9f6c85c9409ffb1ae8285f54085d25a95beabdfec97a`,
  `a3d059da1f80592d4a0a3c35c6a8f36e3b9deb5118b3200dd0402505b08534c4`.
- GSM8K source: `68a72276898a45dacb893154477621ff3d05ae7e043a10820644b691d8b63d77`.
- Model identity remains the locally frozen Qwen3.5-0.8B base verified by the
  existing host preflight.

## Command

```powershell
python tools/research/run_negative_kv_real_screen_r3.py --outdir runs/research/BACKLOG-NEGATIVE-KV-REAL-SCREEN-03
```

## Factors

- Repeat the exact three deterministic 4,096-token contexts, six full-attention
  layers and five frozen mechanisms used by R2.
- Retain 18 K-projection activation tensors, three entropy vectors, 12 Q-weight
  slices and six input matrices: 39 digest-bound tensor files in total.
- Retain and harness-seal all 78 candidate cells. Randomness remains restricted
  to RSH-04 seeds 20260824, 20260825 and 20260826.
- Stop the 8080 inference service for VRAM, leave 8081 healthy, then restore and
  verify the exact service baseline before sealing.

## Acceptance gates

Evidence gates require the bound R2 review, 18 activation cells, 12 weight
matrices, five candidates, 78 retained cells, 39 verified tensor files and full
service restoration. Scientific thresholds are unchanged from R2:

- RSH-01: MSE ratio <=0.70, SQNR gain >=2.5 dB and cosine >=0.995.
- REP-03: MSE reduction >=0.50 and attention cosine >=0.99.
- RSH-03: recovery >=0.50, output cosine >=0.998 and overhead <=0.01.
- RSH-04: top-block recall >=0.90 and retained fraction <=0.30.
- REP-06: average bits <=7.0, attention cosine >=0.992 and beats static INT4 in every cell.

## Abort conditions

Abort without scientific verdict on source/model drift, incomplete hooks,
missing tensor file, hash mismatch, fewer than 78 cells, worker failure, harness
failure, or incomplete restoration. Never infer a candidate reversal from a
partial gate.

## Allowed claims

Only `NEGATIVE_KV_REAL_SCREEN_VERIFIED_R3` or the candidate-specific R3 false
negative code whose full conjunction passes may be reviewed. No packed-byte,
VRAM, native-kernel, throughput, deployment or out-of-panel claim is allowed.
