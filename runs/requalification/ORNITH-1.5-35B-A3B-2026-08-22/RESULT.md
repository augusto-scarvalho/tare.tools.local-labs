# Ornith 1.5 35B-A3B IQ4_XS compact qualification result

Decision: **HOLD_CACHE**  
Date: 2026-08-22

The revision-pinned 19,278,554,784-byte artifact matched SHA-256
`d6aef57fa948e9bba3ca4959b3c237ed898c605471f48c73a32cedbd24aabe70`. At 32,768 context with the
embedding endpoint resident, the warmed server left 4,352 MiB free, passing the frozen 4 GiB reserve
by 256 MiB.

The long-tool-ID agent suite passed 8/8 with no blind irreversible retry. Cache correctness then passed
3/4: shared-prefix divergent suffix, partial removal, and cancel/reuse passed. Long-context reuse found
the correct `MAGNOLIA` oracle in both cold and warm outputs and reported `cache_n=24541`, but the full
cold/warm text differed in the reasoning block, failing the frozen exact-equality rule. GSM8K and MBPP
were therefore not opened.

This is a promising agent-capability result, not a deployable role qualification. The canonical Qwen
service was restored unchanged.

- Agent evidence SHA-256: `00e13d230a3e175aa77e1941a630e683a411ba31590ee31fae9197110b7f635c`
- Cache evidence SHA-256: `1318268a9a4f7f38c0c2520abfb89b72517dff32bdc3400762d2f99bf4174791`
- Frozen packet SHA-256: `3acbcfbaeb316db475c54ceb7d96aae329c1677dd07a028be95e61b553a77c92`

