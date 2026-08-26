from __future__ import annotations

from tools.research.run_ctrl01_real_token import runtime_binding_evidence


def test_research_mentions_do_not_count_as_production_binding() -> None:
    evidence = runtime_binding_evidence()
    assert evidence["production_code_matches"] == []
    assert evidence["logit_mask_runtime_integrated"] is False
