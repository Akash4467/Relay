"""Turn recorded real answers into stand-in Mode A fixtures (DESIGN §16.2): fixtures/perf/{question_id}.json.

Each fixture keeps the real reply text and its real completion and reasoning token counts; when the real run made
a redraft, it is kept as drafts[1] so the stand-in can replay it. Reads only results/raw (never edits it).

    uv run python scripts/record_fixtures.py
"""
from __future__ import annotations

import json
from pathlib import Path

REAL = Path("results/raw/real")
E2E = Path("results/raw/e2e")
OUT = Path("fixtures/perf")


def build(calls: list[dict], runs: list[dict]) -> dict[str, dict]:
    """Match each e2e run to its hosted calls in order (a run made 1 or 2 calls; model checks have no run)."""
    answer_calls = [c for c in calls if c["request"]["max_completion_tokens"] > 16]  # skip the 16-token model checks
    fixtures, i = {}, 0
    for run in runs:
        n = len(run["llm_calls"])
        mine = answer_calls[i:i + n]
        i += n
        if not run["question_id"] or not mine:
            continue
        fixtures[run["question_id"]] = {
            "question_id": run["question_id"], "model": mine[0]["request"]["model"],
            "prompt_version": "p1", "recorded_at": mine[0]["started_at"],
            "drafts": [{"content": c["text"], "completion_tokens": c["usage"]["completion_tokens"],
                        "reasoning_tokens": c["usage"]["completion_tokens_details"]["reasoning_tokens"]}
                       for c in mine]}
    return fixtures


def main() -> dict[str, dict]:
    calls = [json.loads(l) for f in sorted(REAL.glob("*.jsonl")) for l in f.read_text(encoding="utf-8").splitlines()]
    runs = [json.loads(l) for f in sorted(E2E.glob("*.jsonl")) for l in f.read_text(encoding="utf-8").splitlines()]
    fixtures = build(calls, runs)
    OUT.mkdir(parents=True, exist_ok=True)
    for qid, fx in fixtures.items():
        (OUT / f"{qid}.json").write_text(json.dumps(fx, indent=2, ensure_ascii=False), encoding="utf-8")
    return fixtures


if __name__ == "__main__":
    for qid, fx in main().items():
        print(qid, [(d["completion_tokens"], d["reasoning_tokens"], d["content"][:50]) for d in fx["drafts"]])
