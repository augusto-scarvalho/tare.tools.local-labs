# Von CPU diagnostic — frozen before inference

Scope: 15 existing cases unchanged: six strategy comparisons and six boundary cases
from LitJEV, plus three Laya demand cases. Twelve routing rows represent six unique
scenarios in two orders. No prompt tuning, repeats, retries or automatic promotion.
No new held-out or real accepted-delivery quality claims.

Gate for further shadow evaluation: all 15 correct, six routing groups order-stable,
four boundary abstentions correct, valid complete distributions, no truncation;
warm p95 <= 5 seconds and measured process peak RSS <= 4 GiB on acer.
Even a pass does not qualify autonomous routing or calibrated probabilities.

Device CPU, FP32, two threads, existing isolated assessment environment. No GPU,
service changes, network inference, training, package upgrades or Jev charges.
Wall bound: 300 seconds per process. Minimum free disk 10 GiB plus 2 GiB growth;
minimum available RAM 6 GiB before loading. Reuse existing Torch CPU installation.
Record hashes, model and source revisions, dependency versions, cold load, per-call
latency, actual packed tokens, peak working set and zero generated output tokens.
Abort/refuse artifact drift, marker injection, >2048 packed tokens or unsupported
architecture. No silent truncation; missing data stays a failure.

Model revision: aa2fdc9630ecdadef32c56073553b3a69bed38bf.
Source revision: 2656a69be2ef0ebf2f3058302e52791f99cbd8de.
Use complete option_marker.pt trained weights with weights_only=True and strict
state loading; construct encoder from local config to avoid downloading/loading an
unnecessary second 1.5 GB encoder. Preserve upstream pack_sequence and scoring head.
Use marker_calibration.json (actual temperature 2.2), not inconsistent README numbers.
CPU attention uses SDPA; disable compilation. Limit to checkpoint config 2048 tokens,
not SDK's unqualified automatic extension to 8192.

Reproduce: assessment-venv/Scripts/python.exe -B tools/analysis/qualify_von.py
--kernel-root C:/projects/tare.tools.kernel --manifest <checkpoint.json>
--protocol docs/research/evidence/von-20260921/PROTOCOL.json --output <new-result.json>.
External supervisor samples memory, imposes wall deadline and verifies process exit.
