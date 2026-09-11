# Lossless Qwen tool arguments

The original vanilla Qwen3.8 template renders JSON null and the string `"null"`
as identical tagged parameter text. The tagged parser then selects its string
branch for a string/null union. This prevents the native agent from creating a
file using the required null source hash.

`qwen38-json-arguments.jinja` retains the chat, reasoning, tool-call and function
markers and renders the complete argument object with `tojson`. The tool-use
instruction matches that representation. The existing slop.cpp auto-parser
selects `TAG_WITH_JSON`; no C++ engine patch or Kernel coercion is used.

- Original embedded template SHA-256: `12827f24b742ea4e80cdc12dbcf9622227056b9f797252a3149263d4f9aaadce`.
- Corrected template SHA-256: `bf6686a32ad4e999feafb1031e8f84191541953be0ca6363657831e227966efe`.
- Qualified deployed build: `b10165-71676e46c`.
- Fleet card: `qwen38`, plus `serve_qwen38_vanilla_32k.conf` for the single-model fallback.
- Model weights, context size, KV types and MTP settings are unchanged.

The CPU probe in `tare.tools.os/scripts/diagnostics/native_template_roundtrip.cpp`
renders with the real Jinja engine, detects the parser, generates the grammar and
replays complete output and every byte prefix. The corrected template passes all
23 cases; the old one fails the two null cases. Live rendering distinguishes null,
literal null and a different control. A native canary creates, reads, verifies
and proposes delivery in five generations, with thinking enabled/low and a 2048
reasoning budget. Terminal resume in a new process adds no HTTP or write.

Detailed measured results and preparation failures are owned by OS in
`docs/operations/NATIVE_JSON_ARGUMENTS_2026-09-11.md` and its JSON receipt.
The central evidence lot is `native-json-arguments-20260911`. This qualification
does not establish long-context continuation or equivalent behavior for other
fleet models. Historical benchmark launchers retain their original protocols.

Deploy the template before selecting its absolute path in the fleet card.
Restart an idle `llm-inference.service`, verify `/props` template hash and fleet
health, then adopt OS 0.1.7 with matching template pins. For rollback, restore
the old card selection and OS pins together, restart and verify the old hash.
Changing only one side must not be disguised as a valid counted request.
