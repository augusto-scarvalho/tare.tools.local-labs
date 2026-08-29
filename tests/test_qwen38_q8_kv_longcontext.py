from tools.research import run_qwen38_q8_kv_longcontext as subject
from tools.research import run_qwen38_q8_kv_longcontext_r2 as r2


def test_block_design_is_balanced_crossover():
    assert [block["arm"] for block in subject.BLOCKS] == ["f16", "q8", "q8", "f16"]
    assert [block["pair"] for block in subject.BLOCKS] == [0, 0, 1, 1]
    assert all(len(subject.cases_for(block)) == 12 for block in subject.BLOCKS)


def test_unique_cases_are_matched_within_pair_and_disjoint_between_pairs():
    pair_zero = {case["case_id"] for case in subject.cases_for(subject.BLOCKS[0])}
    pair_zero_q8 = {case["case_id"] for case in subject.cases_for(subject.BLOCKS[1])}
    pair_one = {case["case_id"] for case in subject.cases_for(subject.BLOCKS[2])}
    pair_one_f16 = {case["case_id"] for case in subject.cases_for(subject.BLOCKS[3])}
    assert pair_zero == pair_zero_q8
    assert pair_one == pair_one_f16
    assert pair_zero.isdisjoint(pair_one)
    assert len(pair_zero | pair_one) == 24


def test_prompt_has_exact_target_and_31_numbered_decoys():
    prompt, code = subject.make_prompt(8000, "middle", 0)
    assert prompt.count("SECURE ACCESS RECORD") == 32
    assert prompt.count("[ORION-DELTA]") >= 2
    assert prompt.count(f"The access code is {code}.") == 1
    assert len(prompt) > 20_000


def test_exact_scorer_rejects_incidental_substrings():
    code = "NX-Q8-08000-M0"
    assert subject.normalize(f"  `{code}`.\n") == code
    assert subject.normalize(f"The code is {code}") != code
    assert subject.normalize(f"{code}-D01") != code


def test_paired_bootstrap_clear_noninferiority_and_missing_pairs():
    rows = [{"q8_correct": True, "f16_correct": True} for _ in range(24)]
    result = subject.paired_bootstrap(rows)
    assert result["paired_cases"] == 24
    assert result["point"] == 0.0
    assert result["lower_95"] == 0.0
    assert result["upper_95"] == 0.0


def test_r2_chat_contract_disables_thinking_and_uses_usage_prompt_tokens(monkeypatch):
    captured = {}

    def fake_http(url, payload, timeout):
        captured.update({"url": url, "payload": payload, "timeout": timeout})
        return 200, {
            "choices": [{"message": {"content": "NX-Q8-08000-M0", "reasoning_content": ""}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 8120, "completion_tokens": 8},
            "timings": {"predicted_n": 8, "predicted_ms": 100.0},
        }

    monkeypatch.setattr(r2.r1.infra, "http_json", fake_http)
    result = r2.chat_completion("frozen prompt")
    assert captured["url"].endswith("/v1/chat/completions")
    assert captured["payload"]["messages"] == [{"role": "user", "content": "frozen prompt"}]
    assert captured["payload"]["chat_template_kwargs"] == {"enable_thinking": False}
    assert result["content"] == "NX-Q8-08000-M0"
    assert result["prompt_n"] == 8120
    assert result["predicted_n"] == 8
    assert result["throughput_tps"] == 80.0
