# Post-hoc language mitigation arm

Declared after both frozen primary arms completed and before this arm runs.

The primary A/B is immutable: vanilla scored 43/48 and HauhauCS scored 38/48.
Review showed all five exclusive candidate failures were semantically correct
answers emitted in English to Portuguese prompts. This arm tests one operational
mitigation, not a new unbiased primary comparison.

Exactly one full 48-task replay will be run for HauhauCS with the original
runtime/sampling and this fixed system instruction:

> Responda sempre no mesmo idioma da mensagem do usuário. Se o usuário escrever
> em português, responda em português do Brasil, preservando exatamente as
> restrições de formato solicitadas.

No prompt, grader, expected answer, task order, token limit or model setting is
changed. Outcome is reported separately as post-hoc evidence.

## Transport correction

The first replay (`pt-system`) exposed U+FFFD replacement characters in its
stored system-prompt identity because PowerShell passed the accented text through
the command line incorrectly. Its generations are retained, but it is exploratory
and not the exact-prompt mitigation result. The sole corrective replay
(`pt-system-utf8`) reads the unchanged instruction from `system_prompt_pt.txt`
as UTF-8. No wording or task was tuned from the first replay's outcomes.

Because a system instruction can change more than language, the same exact
UTF-8 arm is also run once on vanilla Qwen3.8. Only the matched
vanilla-with-system versus HauhauCS-with-system comparison may isolate model
differences under this mitigation. Candidate-with-system versus vanilla-without-
system is operationally interesting but not a controlled model A/B.

## Port-collision invalidation and clean rerun

The first two nominally UTF-8 response files (`pt-system-utf8`) are **INVALID**:
Fable-TC had already been restored on port 8080, and the old harness accepted
that unrelated endpoint's health response before its requested model finished
binding. Both files therefore contain Fable-TC responses, not their filename's
model. Their suspicious byte-identical outcome exposed the contamination.

The runner now fails if port 8080 is occupied and verifies `/props.model_path`
against the requested registry path after health. The cleanup adapter was also
narrowed from a host-wide `pkill -f llama-server` to the exact model path so an
experimental cleanup cannot kill the independent embedding service. Corrective
matched arms use tag `pt-system-utf8-clean`; invalid files remain unchanged as
receipts and are excluded from every conclusion.

## Clean matched result

The two verified `pt-system-utf8-clean` arms completed with the exact prompt
decoded from `system_prompt_pt.txt`:

- vanilla: 44/48;
- HauhauCS: 43/48;
- paired outcomes: 41 both pass, 2 both fail, 3 vanilla-only and 2
  HauhauCS-only; two-sided exact McNemar/binomial p=1.0.

This meets the pre-registered `NO_MEASURABLE_LOSS` boundary: the candidate trails
by one overall and by at most one in any category. It does not erase the frozen
38/48 versus 43/48 no-system result. The fixed language prompt mitigates most,
but not all, of the candidate's Portuguese-language drift.

## Subsequent Fable-TC control

Fable-TC was subsequently run on the same frozen tasks as an additional control,
not as part of the original HauhauCS-versus-vanilla pre-registration. Its clean
matched-prompt score was 45/48. Against HauhauCS's 43/48, paired outcomes were 41
both pass, 1 both fail, 4 Fable-only and 2 HauhauCS-only (two-sided exact p=0.6875).
The exact prompt stored in the Fable summary matches `system_prompt_pt.txt` after
UTF-8 decoding.
