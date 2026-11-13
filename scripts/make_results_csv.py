"""Build results/results.csv: one row per request (DESIGN §18.3). Reads results/raw only, never edits it.

Sources, joined by request_id:
- results/raw/loadgen/*.jsonl: the user's view (arrival = scheduled time, first content, total, gave up);
- results/raw/traces/*.jsonl: the server's view (queue, retrieval, governor, provider first byte / first token,
  generation, NLI verification, tokens, retries);
- results/raw/e2e/*.jsonl and results/raw/real/*.jsonl: the real Groq runs (scenario real_smoke) and the two
  one-token model checks (scenario probe_model_check).
The brief's twelve columns come first, in order; extra columns follow. answer_grade: refusals are graded by rule
(an answerable question refused = refused_wrongly, an unanswerable one = refused_correctly); answered rows need
a manual grade file (eval/grades/*.csv: request_id,answer_grade) and stand-in rows are never graded (their
replies are recorded fixtures, CLAUDE.md rule 10).

    uv run python scripts/make_results_csv.py
"""
from __future__ import annotations

import csv
import json
import re
from pathlib import Path

RAW = Path("results/raw")
OUT = Path("results/results.csv")
SKIP_SCENARIOS = {"smoke"}  # pipeline smoke tests, not a load shape

REQUIRED = ["request_id", "scenario", "source", "start_time", "first_token_ms", "total_ms", "status", "retries",
            "tokens_in", "tokens_out", "retrieval_setting", "answer_grade"]
EXTRA = ["failure_reason", "model", "answer_tier", "degradation_level", "cache_hit", "question_id",
         "question_category", "admitted", "gave_up", "patience_ms", "scenario_duration_s", "queue_length_at_arrival",
         "queue_wait_ms", "retrieval_ms", "governor_wait_ms", "first_byte_ms", "model_ttft_ms", "generation_ms",
         "verification_ms", "context_tokens", "llm_calls", "initial_attempts", "redraft_attempts",
         "fallback_generations", "reasoning_tokens", "reservation_overrun_tokens", "provider_429s",
         "ratelimit_limit_requests", "ratelimit_remaining_requests", "ratelimit_reset_requests_s",
         "ratelimit_limit_tokens", "ratelimit_remaining_tokens", "ratelimit_reset_tokens_s", "retry_after_s"]


def jsonl(pattern: str) -> list[dict]:
    return [json.loads(l) for f in sorted(RAW.glob(pattern)) for l in f.read_text(encoding="utf-8").splitlines() if l]


def seconds(groq: str | None) -> float | None:
    """'1m26.4s' -> 86.4, '705ms' -> 0.705, '17.48s' -> 17.48."""
    if not groq:
        return None
    m = re.fullmatch(r"(?:(\d+)m)?(?:([\d.]+)s)?|([\d.]+)ms", groq)
    if not m:
        return None
    if m.group(3):
        return float(m.group(3)) / 1000
    return int(m.group(1) or 0) * 60 + float(m.group(2) or 0)


def headers_cols(calls: list[dict]) -> dict:
    """Rate-limit headers of the last call that returned some, plus 429 counts."""
    out: dict = {"provider_429s": sum(c.get("http") == 429 or c.get("status") == 429 for c in calls)}
    for c in calls:
        h = c.get("rate_headers") or c.get("headers") or {}
        if h:
            out.update(ratelimit_limit_requests=h.get("x-ratelimit-limit-requests"),
                       ratelimit_remaining_requests=h.get("x-ratelimit-remaining-requests"),
                       ratelimit_reset_requests_s=seconds(h.get("x-ratelimit-reset-requests")),
                       ratelimit_limit_tokens=h.get("x-ratelimit-limit-tokens"),
                       ratelimit_remaining_tokens=h.get("x-ratelimit-remaining-tokens"),
                       ratelimit_reset_tokens_s=seconds(h.get("x-ratelimit-reset-tokens")),
                       retry_after_s=h.get("retry-after"))
    return out


def grade(row: dict, questions: dict, manual: dict) -> str:
    if row["request_id"] in manual:
        return manual[row["request_id"]]
    q = questions.get(row.get("question_id"))
    if row["source"] != "real" or not q or row["status"] != "ok":
        return ""
    if row.get("answer_tier") == "refused":
        return "refused_wrongly" if q["answerable"] else "refused_correctly"
    return ""  # an answered real row needs a manual grade


def load_rows() -> list[dict]:
    questions = {json.loads(l)["question_id"]: json.loads(l)
                 for l in Path("eval/questions.jsonl").read_text(encoding="utf-8").splitlines()}
    manual = {}
    for f in sorted(Path("eval/grades").glob("*.csv")) if Path("eval/grades").exists() else []:
        manual |= {r["request_id"]: r["answer_grade"] for r in csv.DictReader(f.open(encoding="utf-8"))}
    traces = {t["request_id"]: t for t in jsonl("traces/*.jsonl")}
    durations = {}
    for f in sorted(RAW.glob("loadgen/*.summary.json")):
        s = json.loads(f.read_text(encoding="utf-8"))
        durations[s["scenario"]] = s["duration_s"]
    rows = []
    for r in jsonl("loadgen/*.jsonl"):
        if r["scenario"] in SKIP_SCENARIOS:
            continue
        t = traces.get(r["request_id"], {})
        if r["gave_up"]:
            status, reason = "timeout", "gave_up"
        elif r.get("error"):
            status, reason = "error", r["error"]
        else:
            status, reason = r["done_status"] or "error", r.get("done_reason") or None
        row = {
            "request_id": r["request_id"], "scenario": r["scenario"], "source": r["source"],
            "start_time": r["start_time"], "first_token_ms": r.get("first_content_ms"), "total_ms": r["total_ms"],
            "status": status, "retries": t.get("retries", 0), "tokens_in": t.get("tokens_in", 0),
            "tokens_out": t.get("tokens_out", 0), "retrieval_setting": t.get("retrieval_setting", ""),
            "failure_reason": reason if status != "ok" else None, "model": t.get("model"),
            # a user who gave up received nothing, even if the server finished the work afterwards
            "answer_tier": None if r["gave_up"] else (r.get("done_tier") or t.get("answer_tier")),
            "degradation_level": t.get("degradation_level"),
            "cache_hit": t.get("cache_hit", False), "question_id": r["question_id"],
            "question_category": questions.get(r["question_id"], {}).get("category"),
            "admitted": r.get("admitted"), "gave_up": r["gave_up"], "patience_ms": r["patience_ms"],
            "scenario_duration_s": durations.get(r["scenario"]),
            **{k: t.get(k) for k in ("queue_length_at_arrival", "queue_wait_ms", "retrieval_ms", "governor_wait_ms",
                                     "first_byte_ms", "model_ttft_ms", "generation_ms", "verification_ms",
                                     "context_tokens", "llm_calls", "initial_attempts", "redraft_attempts",
                                     "fallback_generations", "reasoning_tokens", "reservation_overrun_tokens")},
            **headers_cols(t.get("calls", [])),
        }
        rows.append(row)

    for e in jsonl("e2e/*.jsonl"):  # real requests through scripts/ask.py
        calls = e["llm_calls"]
        ok = [c for c in calls if c.get("status") == "ok"]
        refused = e["status"] == "refused"
        rows.append({
            "request_id": f"real-{e['question_id']}-{e['at']}", "scenario": "real_smoke", "source": "real",
            "start_time": e["at"], "first_token_ms": e["timing_ms"]["total"],  # verified before release: whole answer at once
            "total_ms": e["timing_ms"]["total"], "status": "ok" if e["status"] in ("answered", "refused") else "error",
            "retries": 0, "tokens_in": sum(c["tokens_in"] for c in ok), "tokens_out": sum(c["tokens_out"] for c in ok),
            "retrieval_setting": e["setting"], "failure_reason": None, "model": ok[0]["model"] if ok else None,
            "answer_tier": "refused" if refused else "normal", "degradation_level": "L0", "cache_hit": False,
            "question_id": e["question_id"], "question_category": questions.get(e["question_id"], {}).get("category"),
            "admitted": True, "gave_up": False, "retrieval_ms": e["timing_ms"]["retrieval"],
            "governor_wait_ms": round(sum(c["governor_wait_ms"] for c in calls), 1),
            "first_byte_ms": ok[0]["first_byte_ms"] if ok else None, "model_ttft_ms": ok[0]["model_ttft_ms"] if ok else None,
            "generation_ms": e["timing_ms"]["llm"], "verification_ms": e["timing_ms"]["verification_and_rest"],
            "context_tokens": e["context_tokens"], "llm_calls": len(calls), "initial_attempts": 1,
            "redraft_attempts": max(0, len(calls) - 1), "fallback_generations": 0,
            "reasoning_tokens": sum((c["usage"].get("completion_tokens_details") or {}).get("reasoning_tokens", 0) for c in ok),
            "reservation_overrun_tokens": sum(c["reservation_overrun_tokens"] for c in ok), **headers_cols(calls)})

    for c in jsonl("real/*.jsonl"):  # the one-token model checks (priority-0 probes)
        if c["request"]["max_completion_tokens"] > 16:
            continue
        u = c.get("usage") or {}
        rows.append({
            "request_id": f"probe-{c['model'].split('/')[-1]}-{c['started_at']}", "scenario": "probe_model_check",
            "source": "real", "start_time": c["started_at"], "first_token_ms": c.get("first_content_ms"),
            "total_ms": c["total_ms"], "status": "ok" if c["status"] == 200 else "error", "retries": 0,
            "tokens_in": u.get("prompt_tokens", 0), "tokens_out": u.get("completion_tokens", 0), "retrieval_setting": "",
            "model": c["model"], "first_byte_ms": c.get("first_byte_ms"), "model_ttft_ms": c.get("first_content_ms"),
            "reasoning_tokens": (u.get("completion_tokens_details") or {}).get("reasoning_tokens"), "llm_calls": 1,
            "initial_attempts": 1, **headers_cols([c | {"http": c["status"]}])})

    for row in rows:
        row["answer_grade"] = grade(row, questions, manual)
    return rows


def main() -> list[dict]:
    rows = load_rows()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=REQUIRED + EXTRA, extrasaction="ignore")
        w.writeheader()
        for row in rows:
            w.writerow({k: ("" if row.get(k) is None else row.get(k)) for k in REQUIRED + EXTRA})
    return rows


if __name__ == "__main__":
    rows = main()
    by = {}
    for r in rows:
        by[r["scenario"]] = by.get(r["scenario"], 0) + 1
    print(f"{OUT}: {len(rows)} rows", by)
