# Locale contract selection — frozen before test

Selected: `contract_v2.txt`

- SHA-256: `b66b8c553061be0fd8afb11a2096ead161bd1eb61e511c1b8e1ae9daa6bac143`
- 370 decoded characters; 383 UTF-8 bytes including final newline.
- Dev result: 48/48, with every category perfect and 48/48 answered.
- Stored system-prompt identity exactly matches the UTF-8 contract file.

`contract_v1.txt` also achieved 48/48 after the declared clarification-grader
correction, but has 513 decoded characters. The frozen tie-break therefore
selects the shorter v2 contract. The v1 raw summary retains the superseded 43/48
pre-correction score; `regrade__...__grader-r1.json` is its immutable-source
derivative receipt.

No generation from `locale_test_48.jsonl` had occurred when this selection was
written. The test task SHA-256 is
`75a0f623f5d859fc3952f6776e3db8a75c7b42bf36ce7a73049de67c7b09632f`.
