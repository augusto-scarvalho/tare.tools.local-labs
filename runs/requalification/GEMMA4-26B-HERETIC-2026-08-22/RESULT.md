# Gemma 4 26B A4B Heretic compact qualification result

Decision: **HOLD_CACHE**  
Date: 2026-08-22

The exact 16,796,015,904-byte artifact matched its upstream SHA-256. The existing 16,384 profile left
5,482 MiB free and the long-ID agent suite passed 8/8 with zero blind retries. Cache correctness then
passed only 1/4: partial-removal and cancel/reuse produced cold/warm divergence and incorrect content;
the long fixture also exceeded 16,384. These substantive failures stop expansion, so GSM and MBPP were
not run.

Agent SHA-256: `b3d8a76fa1f7894368e9f77ddfa2ebd61a9e5c775933b65218c2e104fdd87b80`.
Cache SHA-256: `219694a8f789f707f4cacfca7528d0bb69879c9d0adada5f3a5cf5f721bbdcd4`.
