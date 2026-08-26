from tools.research.run_spec01_live_hybrid import prompts
def test_frozen_prompt_generator():
 rows=prompts();assert len(rows)==30 and len({r["case"] for r in rows})==30
