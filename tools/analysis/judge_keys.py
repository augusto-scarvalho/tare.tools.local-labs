#!/usr/bin/env python3
"""Small terminal TUI to store the A2 judge-quorum API keys in the OS keyring (Windows
Credential Manager on this box) -- so keys never touch env vars, shell history, or disk in
plaintext. Interactive:

    python judge_keys.py          # (in Claude Code, run as:  ! python judge_keys.py )

The judge harness retrieves a key with:  from judge_keys import get_key; get_key('GEMINI_API_KEY')
Retrieval order: keyring first, then environment variable of the same name (fallback for CI).
"""
from __future__ import annotations

import os
import sys

try:
    import keyring
except ImportError:
    sys.exit("keyring nao instalado. Rode:  python -m pip install keyring")

if os.name == "nt":
    os.system("")  # enable ANSI/VT processing in the Windows console
_TTY = sys.stdout.isatty()


def _c(code: str, s: str) -> str:
    """Colorize only on a real terminal, so piped/captured output stays clean."""
    return f"\033[{code}m{s}\033[0m" if _TTY else s


SERVICE = "a2-judge"

# provider key-name -> (human label, connectivity-test endpoint, header-builder)
# NOTE (2026-08-05): the Gate 3 quorum needs ONLY [1] Gemini + [2] NVIDIA. The Claude judge now runs
# as a Claude Code WORKER (subagent, model=sonnet) -- NO Anthropic key needed. [3]/[4] are optional.
PROVIDERS = {
    "1": ("GEMINI_API_KEY",    "Gemini (Google AI Studio, free tier)   <- NEEDED"),
    "2": ("NVIDIA_API_KEY",    "NVIDIA Build (build.nvidia.com, free)   <- NEEDED"),
    "3": ("ANTHROPIC_API_KEY", "Anthropic  (NOT needed -- Claude = worker subagent)"),
    "4": ("OPENAI_API_KEY",    "OpenAI-compativel (optional, unused)"),
}

# lightweight validation endpoints (GET a models list; 200 => key works)
_TEST = {
    # OpenAI-compat surface (same one the judge harness uses); the native ?key= surface also works
    # but we test the compat path so a passing t1 proves the exact call the quorum will make.
    "GEMINI_API_KEY":    lambda k: ("https://generativelanguage.googleapis.com/v1beta/openai/models", {"Authorization": "Bearer " + k}),
    "NVIDIA_API_KEY":    lambda k: ("https://integrate.api.nvidia.com/v1/models", {"Authorization": "Bearer " + k}),
    "ANTHROPIC_API_KEY": lambda k: ("https://api.anthropic.com/v1/models", {"x-api-key": k, "anthropic-version": "2023-06-01"}),
    "OPENAI_API_KEY":    lambda k: ("https://api.openai.com/v1/models", {"Authorization": "Bearer " + k}),
}


def get_key(name: str) -> str | None:
    """Keyring first, env-var fallback. This is the function the judge harness imports."""
    return keyring.get_password(SERVICE, name) or os.environ.get(name)


def _mask(v: str | None) -> str:
    if not v:
        return _c("90", "unset")
    return f"{_c('92', 'SET')} (...{v[-4:]}, {len(v)} chars)"


def _status() -> None:
    print(f"\n  Chaves no keyring  (servico '{SERVICE}', backend {keyring.get_keyring().__class__.__name__}):")
    for k, (name, desc) in PROVIDERS.items():
        print(f"    [{k}] {name:<18} {_mask(keyring.get_password(SERVICE, name)):<28}  {desc}")


def _set(name: str) -> None:
    import getpass
    val = getpass.getpass(f"  cole a chave de {name} (entrada oculta, Enter p/ cancelar): ").strip()
    if not val:
        print("  (vazio, ignorado)")
        return
    keyring.set_password(SERVICE, name, val)
    print(f"  {_c('92', 'salvo')}: {name}")


def _delete(name: str) -> None:
    try:
        keyring.delete_password(SERVICE, name)
        print(f"  removido: {name}")
    except keyring.errors.PasswordDeleteError:
        print(f"  (nada salvo em {name})")


def _test(name: str) -> None:
    import urllib.error
    import urllib.request
    k = keyring.get_password(SERVICE, name)
    if not k:
        print(f"  {name}: sem chave salva")
        return
    url, headers = _TEST[name](k)
    req = urllib.request.Request(url, headers=headers, method="GET")
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            print(f"  {name}: {_c('92', 'OK')} (HTTP {resp.status})")
    except urllib.error.HTTPError as e:
        tag = _c("91", "CHAVE INVALIDA") if e.code in (401, 403) else f"HTTP {e.code}"
        print(f"  {name}: {tag}")
    except Exception as e:
        print(f"  {name}: erro de rede ({e.__class__.__name__})")


def main() -> int:
    print("=== Judge keys — cofre do OS (keyring) ===")
    while True:
        _status()
        print("\n  Acoes:  <numero>=salvar   d<numero>=apagar   t<numero>=testar conexao   ta=testar todas   q=sair")
        try:
            choice = input("  > ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if choice in ("q", "quit", "exit"):
            break
        elif choice == "ta":
            for name, _ in (p for p in PROVIDERS.values()):
                _test(name)
        elif choice.startswith("d") and choice[1:] in PROVIDERS:
            _delete(PROVIDERS[choice[1:]][0])
        elif choice.startswith("t") and choice[1:] in PROVIDERS:
            _test(PROVIDERS[choice[1:]][0])
        elif choice in PROVIDERS:
            _set(PROVIDERS[choice][0])
        else:
            print("  opcao invalida")
    print("  ok, ate mais.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
