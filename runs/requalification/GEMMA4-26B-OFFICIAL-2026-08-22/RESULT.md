# Gemma 4 26B A4B official QAT compact qualification result

Decision: **HOLD_CACHE**  
Date: 2026-08-22

The official 14,439,363,584-byte artifact matched its upstream SHA-256 and left 8,050 MiB free at
32,768 context. The long-ID agent suite passed the 7/8 threshold with no blind retry; the miss was the
status inspection after an unknown transfer. Cache correctness then failed 0/4 despite nonzero reuse in
all four cells, including 25,034 cached tokens in the long cell. GSM and MBPP were stopped.

Agent SHA-256: `0a317dccd045f2eeb0fc8931e80d32a28dfab7a483267f7065008bfaf6d0866f`.
Cache SHA-256: `94992f491e979eee7dab7bc6d6395b69af567be84d6f66f559484d39ad1690c5`.
