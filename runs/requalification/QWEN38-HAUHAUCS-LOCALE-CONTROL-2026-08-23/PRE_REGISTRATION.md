# HauhauCS locale-control experiment pre-registration

## Goal

Recover Portuguese-language and exact-format adherence without erasing the
candidate's measured coding and low-refusal advantages.

## Data boundary

- `locale_dev_48.jsonl` may be used to select one generic system contract.
- `locale_test_48.jsonl` is frozen before any dev generation and must be run only
  after the contract is selected.
- Neither panel contains prompts from the original 48-task normal-QA benchmark.
- No held-out test response may change a prompt, grader, contract or threshold.

## Prompt-stage decision

Candidate contracts are tested on dev only. Select the shortest contract that:

1. scores at least 47/48 overall;
2. scores 12/12 binary, 8/8 clarification and 8/8 reading;
3. produces no empty or malformed output.

If none qualifies, retain the highest score with category priority in the order
binary, clarification, reading, format, lexical, summary. Exact ties prefer the
shorter contract.

### Dev-grader correction receipt

The first v1 dev score exposed five false negatives: every answer was a valid,
short Portuguese clarification question, but the original `contains_all` grader
required the model to repeat an arbitrary noun from the request. Before any v2
generation or any test generation, the original task files were preserved with
suffix `.v0-grader`, and clarification grading was replaced by a structural
`pt_question` rule: exactly one terminal question mark, at least one Portuguese
interrogative signal, no declared English phrase, and at most 14 words. Prompts,
answers and all other graders are unchanged. The v1 raw response file remains
immutable; its corrected score is stored as a derivative regrade receipt.

## Frozen test comparison

Run the selected contract once on HauhauCS and Fable-TC. Locale control is
operationally sufficient only if HauhauCS trails Fable by no more than one task,
has no category deficit greater than one, and has no empty/malformed answer.

If it fails, a weight intervention may proceed, but the test panel remains
untouched. A new independently frozen validation panel is required for any
trained adapter.

## Weight-stage safety gates

Any future adapter must also retain at least 55/60 HumanEval+, at least 42/44
benign comply, native-MTP functionality, and English-language behavior on a new
balanced panel. Fable-TC remains the serving baseline throughout experiments.
