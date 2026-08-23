"""One-shot refusal test: run ONE probe prompt at multiple max_tokens in a single server
session, to test the starvation hypothesis (a reasoning model never reaches an answer at 1024,
but does at 4096). Controlled: same server, same model, same prompt -- only max_tokens varies."""
import argparse
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent / "src"))

from model_lifecycle.collectors.request import chat_stream, count_tokens  # noqa: E402
from model_lifecycle.models import MODELS                                 # noqa: E402
from model_lifecycle.servers.llama_cpp import (                           # noqa: E402
    LlamaCppAdapter, ServerProfile)
from a2_refusal_probe import PROMPTS, is_refusal                          # noqa: E402

LOCAL_BIN = "/home/augus/src/slop.cpp-main/build/bin/llama-server"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=sorted(MODELS))
    ap.add_argument("--idx", type=int, required=True)
    ap.add_argument("--budgets", default="1024,4096")
    args = ap.parse_args()

    prompt = PROMPTS[args.idx]
    budgets = [int(x) for x in args.budgets.split(",")]
    print(f"MODEL={args.model}  IDX={args.idx}")
    print(f"PROMPT: {prompt}\n")

    adapter = LlamaCppAdapter(server_bin=LOCAL_BIN, env={"GGML_CUDA_REGISTER_HOST": "1"})
    profile = ServerProfile(model_path=MODELS[args.model].path, port=8080, n_cpu_moe=0,
                            ctx_size=8192, extra_args=("--jinja", "--reasoning-format", "deepseek"))
    h = adapter.start(profile)
    try:
        if not adapter.wait_until_healthy(h, timeout_s=1800):
            print("SERVER NEVER HEALTHY")
            for ln in h.stderr_tail[-10:]:
                print("  | " + ln)
            return 1
        for mt in budgets:
            r = chat_stream(h.base_url, prompt, max_tokens=mt, temperature=0.0, cache_prompt=False)
            rt = count_tokens(h.base_url, r.reasoning_text)
            at = count_tokens(h.base_url, r.text)
            print(f"===== max_tokens={mt} =====")
            print(f"  answered={r.answered}  predicted_n={r.predicted_n}  finish={r.error}")
            print(f"  reasoning_tokens={rt}  answer_tokens={at}  is_refusal(answer)={is_refusal(r.text)}")
            print(f"  --- answer (first 400 chars) ---\n    {(r.text or '(VAZIO)')[:400]!r}")
            print()
    finally:
        adapter.stop(h)
        adapter.force_stop(h)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
