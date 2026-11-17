"""Phase 10, live Groq tests (approved by Akash 2026-11-14, with the current key: "use the current key, I accept the
risk", although the machine-security check and key rotation are not done).

    uv run --env-file .env python experiments/phase10/live_groq.py connectivity
    LLM_PROVIDER=groq ALLOW_REAL=1 uv run --env-file .env python experiments/phase10/live_groq.py pipeline

Hard caps (cumulative across runs, from the ledger; no override):
  connectivity: 5 provider requests, 1,000 tokens   pipeline: 16 provider calls, 55,000 tokens
The key is read from the environment only, never printed or logged (the client logs requests without headers).
Everything is written under experiments/phase10/out/live/; nothing under results/.
"""
from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, ".")
import httpx  # noqa: E402

from relay.config import app_config  # noqa: E402
from relay.llm.client import ChatClient, LLMError  # noqa: E402

OUT = Path("experiments/phase10/out/live")
LEDGER = OUT / "ledger.json"
BASE = "https://api.groq.com/openai/v1"
CAPS = {"connectivity": {"calls": 5, "tokens": 1000}, "pipeline": {"calls": 16, "tokens": 55000}}
EST_PIPELINE_CALL = 4000  # worst-case tokens per pipeline call, used to refuse a call that could cross the cap
RISK_NOTE = ("Owner accepted the risk 2026-11-14: run with the current key although a GoDro trojan executed at 01:16 "
             "and the key file was last changed at 01:08; no full scan, key not rotated.")
STAMP = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ")


def ledger() -> dict:
    return json.loads(LEDGER.read_text(encoding="utf-8")) if LEDGER.exists() else \
        {k: {"calls": 0, "tokens": 0} for k in CAPS} | {"note": RISK_NOTE}


def charge(group: str, tokens: int) -> None:
    led = ledger()
    led[group]["calls"] += 1
    led[group]["tokens"] += tokens
    LEDGER.write_text(json.dumps(led, indent=1), encoding="utf-8")


class CapExceeded(Exception):
    pass


class BudgetedClient(ChatClient):
    """Refuses any call that could cross the group's cap; charges every attempt (failed ones included)."""

    def __init__(self, *a, group: str, est: int, **k):
        super().__init__(*a, **k)
        self.group, self.est = group, est

    def complete(self, *a, **k):
        led, cap = ledger()[self.group], CAPS[self.group]
        if led["calls"] + 1 > cap["calls"] or led["tokens"] + self.est > cap["tokens"]:
            raise CapExceeded(f"{self.group} cap reached: {led} of {cap}")
        call = None
        try:
            call = super().complete(*a, **k)
            return call
        except LLMError as e:
            call = e.call
            raise
        finally:
            used = ((call.usage or {}).get("total_tokens") if call else None)
            charge(self.group, used if used is not None else (self.est if call and call.status == 200 else 0))


def key() -> str:
    k = os.environ.get("GROQ_API_KEY")
    if not k:
        raise SystemExit("GROQ_API_KEY not set: run with `uv run --env-file .env ...`")
    return k


def summary(call) -> dict:
    return {"http": call.status, "error": (call.error or "")[:300] or None, "finish_reason": call.finish_reason,
            "text": call.text[:200], "usage": call.usage, "first_byte_ms": call.ms(call.t_first_byte_ns),
            "first_content_ms": call.ms(call.t_first_content_ns), "total_ms": call.ms(call.t_last_ns),
            "rate_headers": call.headers}


# ---- connectivity
def connectivity() -> dict:
    t = app_config()["timeouts"]
    raw = OUT / "provider_raw"
    model = app_config()["models"]["primary"]
    msgs = [{"role": "user", "content": "Reply with exactly one word: pong"}]
    out = {}

    def counted(name, fn):
        led, cap = ledger()["connectivity"], CAPS["connectivity"]
        if led["calls"] >= cap["calls"] or led["tokens"] >= cap["tokens"]:
            out[name] = {"skipped": f"cap reached {led}"}
            return
        t0 = time.perf_counter()
        try:
            out[name] = fn()
        except LLMError as e:
            out[name] = {"llm_error": e.kind} | summary(e.call)
        out[name]["client_ms"] = round((time.perf_counter() - t0) * 1000, 1)
        print(name, json.dumps({k: v for k, v in out[name].items() if k != "rate_headers"}, ensure_ascii=False)[:400])

    def models():
        with httpx.Client(timeout=15) as h:  # headers are never logged
            r = h.get(f"{BASE}/models", headers={"Authorization": f"Bearer {key()}"})
        ids = [m["id"] for m in r.json().get("data", [])] if r.status_code == 200 else []
        charge("connectivity", 0)
        return {"http": r.status_code, "n_models": len(ids), "primary_listed": model["id"] in ids,
                "fallback_listed": "openai/gpt-oss-20b" in ids, "gpt_oss_models": sorted(i for i in ids if "gpt-oss" in i)}

    def plain(name_key, model_id, client_key, timeouts=t, max_tokens=128):
        c = ChatClient(BASE, client_key, timeouts, "real", raw)
        try:
            call = c.complete(model_id, msgs, max_tokens, 0.0, "low")
        except LLMError as e:
            charge("connectivity", (e.call.usage or {}).get("total_tokens") or 0)
            raise
        charge("connectivity", (call.usage or {}).get("total_tokens") or 0)
        return summary(call)

    counted("1_auth_and_model_list", models)
    counted("2_basic_completion", lambda: plain("basic", model["id"], key()))
    counted("3_invalid_model", lambda: plain("bad_model", "openai/this-model-does-not-exist", key()))
    counted("4_invalid_key", lambda: plain("bad_key", model["id"], "gsk_invalid_test_key_not_real"))
    counted("5_timeout_1ms", lambda: plain("timeout", model["id"], key(),
                                           t | {"first_byte_ms": 1, "connect_ms": 3000}, max_tokens=32))
    return out


# ---- pipeline
QUESTIONS = [  # (id, category); question text and expected answer come from the eval files
    ("q02", "answerable_factual"), ("q09", "multi_passage"), ("q12", "table_evidence"),
    ("q11", "numerical_comparison"), ("q18", "insufficient_evidence"), ("q20", "not_in_papers"),
    ("q30", "misleading_premise"), ("t10", "table_marks"),
]


def load_questions() -> dict:
    qs = {}
    for l in open("eval/questions.jsonl", encoding="utf-8"):
        q = json.loads(l)
        qs[q["question_id"]] = {"question": q["question"], "expected": q.get("expected_answer"),
                                "answerable": q.get("answerable")}
    d = json.load(open("eval/phase7/table_question_set.json", encoding="utf-8"))
    for q in d["questions"] if isinstance(d, dict) else d:
        qid = q.get("id") or q.get("question_id")
        qs[qid] = {"question": q["question"], "expected": q.get("expected_answer") or q.get("expected") or q.get("gold"),
                   "answerable": q.get("answerable")}
    return qs


def pipeline() -> dict:
    if os.environ.get("LLM_PROVIDER") != "groq" or os.environ.get("ALLOW_REAL") != "1":
        raise SystemExit("set LLM_PROVIDER=groq and ALLOW_REAL=1 for the live pipeline run")
    from fastapi.testclient import TestClient

    from relay import main
    from relay.llm.hosted import RetryBudget
    from relay.pipeline import answer_from_evidence as real_answer

    main.TRACE_DIR = OUT / "traces"
    s = main.State()
    s.client = BudgetedClient(BASE, key(), s.cfg["timeouts"], "real", OUT / "provider_raw",
                              group="pipeline", est=EST_PIPELINE_CALL)
    s.budget = RetryBudget.from_config(s.cfg["retry"])
    main.STATE = s
    captured = {}

    def capture(question, ev, llm, verifier, corpus_ids, *a, **k):
        r = real_answer(question, ev, llm, verifier, corpus_ids, *a, **k)
        captured[question] = r
        return r
    main.answer_from_evidence = capture
    http = TestClient(main.app)
    qs = load_questions()
    out = []
    only = os.environ.get("P10_IDS")  # optional comma-separated subset, e.g. a re-run after fixes
    for qid, cat in [qc for qc in QUESTIONS if not only or qc[0] in only.split(",")]:
        led, cap = ledger()["pipeline"], CAPS["pipeline"]
        if led["calls"] + 1 > cap["calls"] or led["tokens"] + EST_PIPELINE_CALL > cap["tokens"]:
            out.append({"id": qid, "category": cat, "skipped": f"cap {led} of {cap}"})
            print(qid, "SKIPPED: cap")
            continue
        q = qs[qid]
        t0 = time.perf_counter()
        events, name = [], None
        with http.stream("POST", "/ask", json={"question": q["question"], "question_id": qid, "request_id": f"p10-live-{qid}",
                                               "scenario": "phase10_live", "deadline_ms": 120000}) as resp:
            for line in resp.iter_lines():
                if line.startswith("event: "):
                    name = line[7:]
                elif line.startswith("data: "):
                    events.append((name, json.loads(line[6:])))
        ms = round((time.perf_counter() - t0) * 1000, 1)
        tr = [json.loads(l) for l in sorted((OUT / "traces").glob("*.jsonl"))[-1].read_text(encoding="utf-8").splitlines()][-1]
        r = captured.get(q["question"])
        done = next((d for n, d in events if n == "done"), {})
        rec = {"id": qid, "category": cat, "question": q["question"], "expected": q["expected"],
               "answerable": q["answerable"], "final_text": next((d["text"] for n, d in events if n == "content"), None),
               "done": done, "client_ms": ms,
               "evidence_ids": [p["chunk_id"] for p in r.evidence.passages] if r and r.evidence else None,
               "drafts": r.drafts if r else None,
               "verification": [{"ok": rep.ok, "cited": rep.cited(), "failures": rep.failures()} for rep in r.reports]
               if r else None,
               "trace": {k: tr.get(k) for k in ("status", "answer_tier", "reply_reason", "llm_calls", "retries",
                                                 "redraft_attempts", "tokens_in", "tokens_out", "reasoning_tokens",
                                                 "retrieval_ms", "governor_wait_ms", "first_byte_ms", "model_ttft_ms",
                                                 "generation_ms", "verification_ms", "total_ms", "context_tokens")},
               "calls": [{k: c.get(k) for k in ("attempt", "status", "http", "tokens_in", "tokens_out", "reasoning_tokens",
                                                "finish_reason", "first_byte_ms", "total_ms", "governor_wait_ms",
                                                "error", "backoff_s")} for c in tr.get("calls", [])]}
        out.append(rec)
        print(qid, cat, "|", done.get("status"), done.get("tier"), done.get("reason"), "| calls", tr.get("llm_calls"),
              "| tokens", tr.get("tokens_in"), "+", tr.get("tokens_out"), "| total", tr.get("total_ms"), "ms")
    return {"questions": out, "ids": os.environ.get("P10_IDS"),
            "versions": {k: s.cfg[k] for k in ("verification_policy_version", "prompt_version", "corpus_version")}}


if __name__ == "__main__":
    group = sys.argv[1] if len(sys.argv) > 1 else ""
    if group not in CAPS:
        raise SystemExit("usage: live_groq.py connectivity|pipeline")
    OUT.mkdir(parents=True, exist_ok=True)
    if not LEDGER.exists():
        LEDGER.write_text(json.dumps(ledger(), indent=1), encoding="utf-8")
    result = connectivity() if group == "connectivity" else pipeline()
    path = OUT / f"{group}_{STAMP}.json"
    with path.open("x", encoding="utf-8") as f:
        json.dump({"group": group, "stamp": STAMP, "note": RISK_NOTE, "caps": CAPS[group],
                   "ledger_after": ledger()[group], **result}, f, indent=1, ensure_ascii=False)
    print(path.as_posix(), "| ledger", ledger()[group])
