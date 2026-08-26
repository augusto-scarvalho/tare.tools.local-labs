#!/usr/bin/env python3
"""Discover and use the role-qualified local model fleet on port 8080."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any
from urllib import error as urllib_error
from urllib import request as urllib_request

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from model_lifecycle.qualified_fleet import (  # noqa: E402
    DEFAULT_REGISTRY,
    FleetConfigError,
    load_registry,
    public_card,
    recommend,
    resolve_model,
)


def _base_url(config: dict[str, Any]) -> str:
    return str(config["fleet"]["api_base"]).rstrip("/")


def _request_json(url: str, *, payload: dict[str, Any] | None = None,
                  timeout: float = 600) -> dict[str, Any]:
    body = None if payload is None else json.dumps(payload).encode("utf-8")
    request = urllib_request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"} if body else {},
    )
    try:
        with urllib_request.urlopen(request, timeout=timeout) as response:
            return json.load(response)
    except urllib_error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code}: {detail}") from exc


def cmd_list(args: argparse.Namespace, config: dict[str, Any]) -> int:
    cards = [public_card(model_id, card) for model_id, card in sorted(config["models"].items())]
    if args.json:
        print(json.dumps({"models": cards, "aliases": config["aliases"]}, indent=2))
        return 0
    print("QUALIFIED LOCAL MODELS (:8080, one resident at a time)\n")
    print(f"{'MODEL':14} {'STATUS':15} {'MODALITY':10} USE FOR")
    for card in cards:
        print(f"{card['id']:14} {card['qualification']:15} "
              f"{','.join(card['modalities']):10} {', '.join(card['qualified_for'])}")
    print("\nAliases: " + ", ".join(f"{key}={value}" for key, value in sorted(config["aliases"].items())))
    print("Use: python tools/agents/modelctl.py show MODEL")
    print("Call: set the OpenAI JSON field model=MODEL against http://127.0.0.1:8080/v1")
    return 0


def cmd_show(args: argparse.Namespace, config: dict[str, Any]) -> int:
    model_id, card = resolve_model(config, args.model)
    payload = public_card(model_id, card)
    payload["requested_as"] = args.model
    payload["artifact_sha256"] = card["artifact"]["sha256"]
    payload["artifact_path"] = card["artifact"]["path"]
    payload["api_base"] = _base_url(config)
    if args.json:
        print(json.dumps(payload, indent=2))
        return 0
    print(f"{model_id} — {card['display_name']}")
    print(f"status:       {card['qualification']}")
    print(f"modalities:   {', '.join(card['modalities'])}")
    print(f"use for:      {', '.join(card['qualified_for'])}")
    print(f"do not use:   {', '.join(card['not_for'])}")
    print(f"summary:      {card['summary']}")
    print(f"limits:       {card['limits']}")
    print("evidence:")
    for evidence in card["evidence"]:
        print(f"  - {evidence}")
    print(f"\nAPI: {_base_url(config)}/chat/completions with model={args.model!r}")
    return 0


def cmd_recommend(args: argparse.Namespace, config: dict[str, Any]) -> int:
    model_id, card = recommend(config, args.role)
    payload = {
        "role": args.role,
        "model": model_id,
        "reason": card["summary"],
        "limits": card["limits"],
        "api_base": _base_url(config),
    }
    if args.json:
        print(json.dumps(payload, indent=2))
    else:
        print(model_id)
        print(f"  {card['summary']}")
        print(f"  limit: {card['limits']}")
    return 0


def cmd_example(args: argparse.Namespace, config: dict[str, Any]) -> int:
    model_id, card = resolve_model(config, args.model)
    payload: dict[str, Any] = {
        "model": args.model,
        "messages": [{"role": "user", "content": "Responda apenas: OK"}],
        "temperature": 0,
        "max_tokens": 32,
    }
    if "image" in card["modalities"]:
        payload["messages"] = [{
            "role": "user",
            "content": [
                {"type": "text", "text": "Descreva a imagem."},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}},
            ],
        }]
    payload.update(card.get("example_overrides", {}))
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    print(f"\nPOST {_base_url(config)}/chat/completions")
    print(f"# canonical route: {model_id}; the first call after a model change includes load time")
    return 0


def cmd_status(args: argparse.Namespace, config: dict[str, Any]) -> int:
    gateway = _base_url(config).removesuffix("/v1")
    try:
        status = _request_json(f"{gateway}/fleet/status", timeout=args.timeout)
    except (OSError, RuntimeError) as exc:
        print(f"gateway unavailable: {exc}", file=sys.stderr)
        return 2
    if args.json:
        print(json.dumps(status, indent=2))
    else:
        print(f"gateway={status.get('status')} current={status.get('current_model') or 'none'} "
              f"backend_healthy={status.get('backend_healthy')}")
        print(f"available={', '.join(status.get('available_models', []))}")
        if status.get("last_switch_seconds") is not None:
            print(f"last_switch={status['last_switch_seconds']:.2f}s")
    return 0


def cmd_chat(args: argparse.Namespace, config: dict[str, Any]) -> int:
    model_id, _ = resolve_model(config, args.model)
    messages: list[dict[str, str]] = []
    if args.system:
        messages.append({"role": "system", "content": args.system})
    messages.append({"role": "user", "content": args.prompt})
    payload = {
        "model": args.model,
        "messages": messages,
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
        "stream": False,
    }
    if args.no_thinking:
        payload["chat_template_kwargs"] = {"enable_thinking": False}
    try:
        response = _request_json(
            f"{_base_url(config)}/chat/completions", payload=payload, timeout=args.timeout
        )
        content = response["choices"][0]["message"].get("content", "")
    except (OSError, RuntimeError, KeyError, IndexError, TypeError) as exc:
        print(f"request failed: {exc}", file=sys.stderr)
        return 2
    if args.json:
        print(json.dumps(response, ensure_ascii=False, indent=2))
    else:
        print(content)
        print(f"\n[model={model_id}]", file=sys.stderr)
    return 0


def cmd_validate(_args: argparse.Namespace, config: dict[str, Any]) -> int:
    print(f"registry OK: {len(config['models'])} qualified routes, "
          f"{len(config['aliases'])} aliases, max_resident_models=1")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", default=str(DEFAULT_REGISTRY))
    sub = parser.add_subparsers(dest="command", required=True)

    listing = sub.add_parser("list", help="list qualified models and roles")
    listing.add_argument("--json", action="store_true")
    listing.set_defaults(func=cmd_list)

    show = sub.add_parser("show", help="show qualification evidence and limits")
    show.add_argument("model")
    show.add_argument("--json", action="store_true")
    show.set_defaults(func=cmd_show)

    rec = sub.add_parser("recommend", help="recommend a qualified model for an exact role")
    rec.add_argument("role")
    rec.add_argument("--json", action="store_true")
    rec.set_defaults(func=cmd_recommend)

    example = sub.add_parser("example", help="print an OpenAI-compatible request example")
    example.add_argument("model")
    example.set_defaults(func=cmd_example)

    status = sub.add_parser("status", help="inspect the live gateway and resident model")
    status.add_argument("--json", action="store_true")
    status.add_argument("--timeout", type=float, default=5)
    status.set_defaults(func=cmd_status)

    chat = sub.add_parser("chat", help="send one text request through the qualified gateway")
    chat.add_argument("model")
    chat.add_argument("prompt")
    chat.add_argument("--system")
    chat.add_argument("--temperature", type=float, default=0)
    chat.add_argument("--max-tokens", type=int, default=256)
    chat.add_argument("--timeout", type=float, default=900)
    chat.add_argument("--no-thinking", action="store_true",
                      help="pass chat_template_kwargs.enable_thinking=false")
    chat.add_argument("--json", action="store_true")
    chat.set_defaults(func=cmd_chat)

    sub.add_parser("validate", help="validate the fail-closed registry").set_defaults(func=cmd_validate)

    args = parser.parse_args(argv)
    try:
        config = load_registry(args.registry)
        return args.func(args, config)
    except (FleetConfigError, KeyError) as exc:
        print(f"model registry error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
