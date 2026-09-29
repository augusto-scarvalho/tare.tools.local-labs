"""Validated registry helpers for the role-qualified single-GPU model fleet."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REGISTRY = REPO_ROOT / "config" / "qualified_model_fleet.json"
ROUTABLE_QUALIFICATIONS = {"promoted", "qualified_role"}
BACKEND_BINARIES = {"llama": "/llama-server", "ninfer": "/ninfer-serve", "strata": "/python"}
MODEL_STORES = ("/home/augus/models/", "/mnt/wsl/models/")
# NInfer 0.6.1 rejects these inside chat_template_kwargs but reads them at top level.
NINFER_TOP_LEVEL_TEMPLATE_KWARGS = ("enable_thinking", "reasoning_effort", "preserve_thinking")


def backend_kind(card: dict[str, Any]) -> str:
    return card["runtime"].get("kind", "llama")


def ninfer_chat_request(payload: dict[str, Any]) -> dict[str, Any]:
    """Move thinking controls from chat_template_kwargs to the top level NInfer accepts."""
    kwargs = payload.get("chat_template_kwargs")
    if not isinstance(kwargs, dict):
        return payload
    result, rest = dict(payload), dict(kwargs)
    for key in NINFER_TOP_LEVEL_TEMPLATE_KWARGS:
        if key in rest:
            value = rest.pop(key)
            result.setdefault(key, value)
    if rest:
        result["chat_template_kwargs"] = rest
    else:
        result.pop("chat_template_kwargs")
    return result


class FleetConfigError(ValueError):
    """The fleet registry is unsafe or internally inconsistent."""


def load_registry(path: str | Path = DEFAULT_REGISTRY) -> dict[str, Any]:
    config_path = Path(path)
    try:
        data = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise FleetConfigError(f"cannot read fleet registry {config_path}: {exc}") from exc
    validate_registry(data, repo_root=config_path.resolve().parents[1])
    return data


def validate_registry(data: dict[str, Any], *, repo_root: Path = REPO_ROOT) -> None:
    if data.get("schema_version") != 1:
        raise FleetConfigError("schema_version must be 1")
    fleet = data.get("fleet")
    models = data.get("models")
    aliases = data.get("aliases")
    if not isinstance(fleet, dict) or not isinstance(models, dict) or not models:
        raise FleetConfigError("fleet and at least one model are required")
    if not isinstance(aliases, dict):
        raise FleetConfigError("aliases must be an object")
    if fleet.get("max_resident_models") != 1:
        raise FleetConfigError("this RTX 3090 fleet must fail closed at max_resident_models=1")
    if fleet.get("default_model") not in models:
        raise FleetConfigError("fleet.default_model must name a registered model")

    modes = fleet.get("host_modes")
    if modes is not None and not (
        isinstance(modes, dict) and modes.get("shared") in models and modes.get("solo") in models
        and str(modes.get("status_path", "")).endswith(".json")
        and type(modes.get("max_status_age_seconds")) is int and modes["max_status_age_seconds"] > 0
        and type(modes.get("solo_host_ram_gb")) in (int, float) and modes["solo_host_ram_gb"] > 0
        and type(modes.get("ram_margin_gb")) in (int, float) and modes["ram_margin_gb"] >= 0
    ):
        raise FleetConfigError("fleet.host_modes needs shared/solo models, status_path, max_status_age_seconds, "
                               "solo_host_ram_gb, ram_margin_gb")

    for alias, target in aliases.items():
        if not alias or target not in models:
            raise FleetConfigError(f"alias {alias!r} points to unknown model {target!r}")

    for model_id, card in models.items():
        if not model_id or not isinstance(card, dict):
            raise FleetConfigError("model ids must be non-empty objects")
        if card.get("qualification") not in ROUTABLE_QUALIFICATIONS:
            raise FleetConfigError(
                f"{model_id}: only promoted/qualified_role artifacts may enter the fleet"
            )
        for key in ("qualified_for", "not_for", "modalities", "evidence"):
            if not isinstance(card.get(key), list) or not card[key]:
                raise FleetConfigError(f"{model_id}: non-empty {key} is required")
        artifact = card.get("artifact", {})
        runtime = card.get("runtime", {})
        digest = artifact.get("sha256", "")
        if not isinstance(digest, str) or len(digest) != 64:
            raise FleetConfigError(f"{model_id}: a 64-character artifact sha256 is required")
        if not str(artifact.get("path", "")).startswith(MODEL_STORES):
            raise FleetConfigError(f"{model_id}: artifact path is outside the model store")
        kind = runtime.get("kind", "llama")
        if kind not in BACKEND_BINARIES:
            raise FleetConfigError(f"{model_id}: runtime.kind must be one of {sorted(BACKEND_BINARIES)}")
        if not str(runtime.get("binary", "")).endswith(BACKEND_BINARIES[kind]):
            raise FleetConfigError(f"{model_id}: runtime binary must be {BACKEND_BINARIES[kind].lstrip('/')}")
        if not isinstance(runtime.get("args"), list):
            raise FleetConfigError(f"{model_id}: runtime.args must be a list")
        if "example_overrides" in card and not isinstance(card["example_overrides"], dict):
            raise FleetConfigError(f"{model_id}: example_overrides must be an object")
        if any(token in runtime["args"] for token in ("--host", "--port", "--alias", "-m", "--model", "--model-id",
                                                      "--config", "--engine")):
            raise FleetConfigError(f"{model_id}: gateway-owned flags found in runtime.args")
        if kind == "strata" and not (str(runtime.get("server", "")).endswith("/serve/server.py")
                                     and str(runtime.get("config", "")).endswith(".json")
                                     and isinstance(runtime.get("model_name"), str) and runtime["model_name"]):
            raise FleetConfigError(f"{model_id}: strata needs runtime.server, runtime.config and runtime.model_name")
        idle = runtime.get("idle_unload_seconds")
        if idle is not None and (type(idle) is not int or idle < 60):
            raise FleetConfigError(f"{model_id}: runtime.idle_unload_seconds must be an integer >= 60")
        for evidence in card["evidence"]:
            evidence_path = repo_root / evidence
            if not evidence_path.is_file():
                raise FleetConfigError(f"{model_id}: missing evidence {evidence}")


def resolve_model(data: dict[str, Any], requested: str | None) -> tuple[str, dict[str, Any]]:
    name = requested or data["fleet"]["default_model"]
    seen: set[str] = set()
    while name in data.get("aliases", {}):
        if name in seen:
            raise FleetConfigError(f"alias cycle at {name}")
        seen.add(name)
        name = data["aliases"][name]
    try:
        return name, data["models"][name]
    except KeyError as exc:
        raise KeyError(requested) from exc


def recommend(data: dict[str, Any], role: str) -> tuple[str, dict[str, Any]]:
    normalized = role.strip().lower()
    if normalized in data.get("aliases", {}):
        return resolve_model(data, normalized)
    candidates = [
        (model_id, card)
        for model_id, card in data["models"].items()
        if normalized in {item.lower() for item in card["qualified_for"]}
    ]
    if not candidates:
        raise KeyError(role)
    default_model = data["fleet"]["default_model"]
    candidates.sort(key=lambda pair: (
        pair[1]["qualification"] != "promoted",
        pair[0] != default_model,
        pair[0],
    ))
    return candidates[0]


def host_mode(data: dict[str, Any], status: dict[str, Any] | None, now: float,
              resident: str | None = None) -> dict[str, Any]:
    """Pick the solo model when it fits what the host has free right now; otherwise the shared one.

    Solo is "nobody else needs this memory": the solo model's host RAM plus a margin must be free on
    Windows. A resident solo model already holds its share, so it keeps fitting. Missing or stale host
    status does not fit: the shared model is the safe side. The GPU itself is the shared lease's job.
    """
    modes = data["fleet"]["host_modes"]
    needed = modes["solo_host_ram_gb"] + modes["ram_margin_gb"]
    free = status.get("windows_free_gb") if isinstance(status, dict) else None
    if not isinstance(status, dict) or not isinstance(status.get("ts"), (int, float)):
        reason = "host_status_missing"
    elif now - status["ts"] > modes["max_status_age_seconds"]:
        reason = "host_status_stale"
    elif not isinstance(free, (int, float)):
        reason = "host_status_missing"
    elif free + (modes["solo_host_ram_gb"] if resident == modes["solo"] else 0) >= needed:
        reason = "solo_fits"
    else:
        reason = "host_ram_short"
    mode = "solo" if reason == "solo_fits" else "shared"
    return {"mode": mode, "model": modes[mode], "reason": reason, "needed_free_gb": needed,
            "resident": resident, "host": status if isinstance(status, dict) else None}


def build_backend_command(
    model_id: str,
    card: dict[str, Any],
    *,
    host: str,
    port: int,
) -> list[str]:
    runtime = card["runtime"]
    if backend_kind(card) == "ninfer":
        return [runtime["binary"], card["artifact"]["path"], "--host", host, "--port", str(port),
                "--model-id", model_id, *[str(token) for token in runtime["args"]]]
    if backend_kind(card) == "strata":
        # The Strata config (engine args, pack, shards) is written by its setup; the gateway owns the address.
        return [runtime["binary"], runtime["server"], "--engine", "strata", "--config", runtime["config"],
                "--host", host, "--port", str(port), *[str(token) for token in runtime["args"]]]
    return [
        card["runtime"]["binary"],
        "-m",
        card["artifact"]["path"],
        "--alias",
        model_id,
        "--host",
        host,
        "--port",
        str(port),
        *[str(token) for token in card["runtime"]["args"]],
    ]


def public_card(model_id: str, card: dict[str, Any]) -> dict[str, Any]:
    result = {
        "id": model_id,
        "display_name": card["display_name"],
        "qualification": card["qualification"],
        "qualified_for": card["qualified_for"],
        "not_for": card["not_for"],
        "modalities": card["modalities"],
        "summary": card["summary"],
        "limits": card["limits"],
        "quant": card["artifact"]["quant"],
        "evidence": card["evidence"],
    }
    if card.get("example_overrides"):
        result["example_overrides"] = card["example_overrides"]
    return result
