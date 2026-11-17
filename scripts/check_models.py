"""Real-call budget priority 0 (DESIGN §19.1): one tiny call per model to confirm the model IDs exist on the
account and to see the response shape (where streaming usage appears, which rate-limit headers come back).
2 hosted calls. Goes through the governor like every hosted call.

    LLM_PROVIDER=groq uv run --env-file .env python scripts/check_models.py
"""
from __future__ import annotations

import json

from relay.config import app_config
from relay.governor.governor import Governor
from relay.llm.client import LLMError, from_env
from relay.llm.hosted import HostedLLM


def main() -> None:
    cfg = app_config()
    client, governor = from_env(cfg["timeouts"]), Governor.from_config()
    print(f"provider source: {client.source}  base: {client.base_url}")
    for key in ("primary", "fallback"):
        model = dict(cfg["models"][key], max_tokens=16)  # reasoning models spend some tokens before any text
        llm = HostedLLM(client, governor, model, output_reserve=16,
                        request_overhead=cfg["reservation"]["request_overhead_tokens"])
        try:
            text = llm([{"role": "user", "content": "Reply with the single word OK."}])
            print(f"{model['id']}: ok, text={text!r}")
        except LLMError as e:
            print(f"{model['id']}: {e.kind} HTTP {e.call.status} {e.call.error}")
        print(json.dumps(llm.calls[-1], indent=2))


if __name__ == "__main__":
    main()
