# Agent instructions

- Write code, comments, prompts and documentation in English; agents still reply in the user's language.
  The Portuguese version for Brazilian readers lives only in README.pt-BR.md: update it with its English source.
- Run the checks CI runs before pushing: `python -m compileall -q src tools tests benchmark_harness_qa.py`,
  `python -m pytest -q` and `python tests/benchmark_harness/benchmark_harness_selftest.py`.
- Material changes need a new bullet under `## Unreleased` in `CHANGELOG.md`; never edit or move existing bullets
  (see `docs/CHANGELOG_POLICY.md`).
- Raw run records, the research backlog and other private material are ignored by `.gitignore`; never force-add them.
