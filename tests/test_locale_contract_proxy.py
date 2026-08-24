from __future__ import annotations

import importlib.util
import json
import pathlib

import pytest


ROOT = pathlib.Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "ops" / "qwen38-bringup" / "locale_contract_proxy.py"
SPEC = importlib.util.spec_from_file_location("locale_contract_proxy", MODULE_PATH)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def apply(body: dict, contract: str = "Responda em português.") -> dict:
    encoded = json.dumps(body, ensure_ascii=False).encode("utf-8")
    return json.loads(MODULE.inject_contract(encoded, contract).decode("utf-8"))


def test_injects_contract_before_user_message() -> None:
    result = apply({"messages": [{"role": "user", "content": "Olá"}]})
    assert result["messages"] == [
        {"role": "system", "content": "Responda em português."},
        {"role": "user", "content": "Olá"},
    ]


def test_preserves_existing_system_message_after_contract() -> None:
    result = apply({"messages": [
        {"role": "system", "content": "Use JSON."},
        {"role": "user", "content": "Liste."},
    ]})
    assert result["messages"][0]["content"] == "Responda em português.\n\nUse JSON."
    assert result["messages"][1] == {"role": "user", "content": "Liste."}


def test_contract_injection_is_idempotent() -> None:
    once = apply({"messages": [{"role": "user", "content": "Olá"}]})
    twice = apply(once)
    assert twice == once


def test_rejects_non_chat_payload() -> None:
    with pytest.raises(ValueError, match="messages list"):
        apply({"prompt": "Olá"})


def test_rejects_non_object_message() -> None:
    with pytest.raises(ValueError, match="message must be an object"):
        apply({"messages": ["Olá"]})
