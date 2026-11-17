"""Phase 10, group A (offline, no quota): the real FastAPI app, real c3 indexes, real cross-encoder and real NLI
(verifier v4, threshold 0.9 from config), with the provider replaced by an httpx MockTransport that plays each
failure. Never calls Groq: the key is a dummy, LLM_PROVIDER=standin, and the ChatClient points at a fake host.

Each scenario sends one or more POST /ask requests through TestClient and records the SSE events, every provider
request the app made, and the trace line. Safety rule checked for every reply: an answer (tier "normal") must be
a draft the mock sent and the verifier passed; anything else must be one of the fixed refusal / busy / unavailable /
internal-error texts. Nothing is written under results/: provider logs and traces go to experiments/phase10/out/.

    uv run python experiments/phase10/offline_integration.py
"""
from __future__ import annotations

import json
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

os.environ["LLM_PROVIDER"] = "standin"
os.environ["GROQ_API_KEY"] = "dummy-offline-key"  # never a real key; the fake transport ignores it
os.environ.pop("ALLOW_REAL", None)
sys.path.insert(0, ".")

import httpx  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from relay import main  # noqa: E402
from relay.governor.governor import Governor  # noqa: E402
from relay.llm import prompts  # noqa: E402
from relay.llm.client import ChatClient  # noqa: E402
from relay.llm.hosted import RetryBudget  # noqa: E402
from relay.retrieval.search import Evidence  # noqa: E402

OUT = Path("experiments/phase10/out")
STAMP = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ")
SAFE_PREFIXES = ("I couldn't find this in the passages", "The assistant is at capacity",
                 "The research assistant has used today's", "The assistant hit an internal error")
USAGE = {"choices": [], "x_groq": {"usage": {"prompt_tokens": 1500, "completion_tokens": 120, "total_tokens": 1620,
                                             "completion_tokens_details": {"reasoning_tokens": 60}}}}
RATE = {"x-ratelimit-remaining-requests": "999", "x-ratelimit-remaining-tokens": "7000"}


# ---- fake provider
def sse(*chunks: dict) -> bytes:
    return b"".join(f"data: {json.dumps(c)}\n\n".encode() for c in chunks) + b"data: [DONE]\n\n"


def delta(text: str, finish: str | None = None) -> dict:
    return {"choices": [{"delta": {"content": text}, "finish_reason": finish}]}


def reply(text: str, finish: str = "stop") -> httpx.Response:
    return httpx.Response(200, headers=RATE, content=sse(delta(text), delta("", finish), USAGE))


def passages_in(req: httpx.Request) -> list[tuple[str, str]]:
    """(chunk_id, text) of every passage in the prompt, in order."""
    user = json.loads(req.content)["messages"][1]["content"]
    blocks = re.split(r"\n\n(?=\[[A-Z]+:[^\]]+\] \()", user.split("Passages:\n", 1)[1])
    out = []
    for b in blocks:
        m = re.match(r"\[([^\]]+)\] \([^\n]*\)\n(.*)", b, re.S)
        if m:
            out.append((m.group(1), m.group(2)))
    return out


def echo_answer(req: httpx.Request) -> str:
    """A faithful draft: the first prose sentence of the first prose passage, cited to it."""
    for cid, text in passages_in(req):
        if not re.match(r"(Table \d+:|Columns:|Configuration )", text):
            sent = re.split(r"(?<=[.!?])\s", text.strip())[0].rstrip(".")
            return f"{sent} [{cid}]."
    cid, text = passages_in(req)[0]
    return f"{text.splitlines()[0].rstrip('.')} [{cid}]."


class Provider:
    """Plays a list of behaviours, one per provider request; records every request."""

    def __init__(self, plan: list):
        self.plan, self.requests, self.sent = list(plan), [], []

    def __call__(self, req: httpx.Request) -> httpx.Response:
        self.requests.append(json.loads(req.content))
        step = self.plan.pop(0) if self.plan else "unexpected"
        if step == "unexpected":
            return httpx.Response(500, content=b'{"error":"test: more provider calls than planned"}')
        if callable(step):
            r = step(req)
            if isinstance(r, str):
                self.sent.append(r)
                return reply(r)
            return r
        if isinstance(step, httpx.Response):
            return step
        raise AssertionError(step)


def err(status: int, message: str, retry_after: str | None = None) -> httpx.Response:
    h = dict(RATE) | ({"retry-after": retry_after} if retry_after else {})
    return httpx.Response(status, headers=h, content=json.dumps({"error": {"message": message}}).encode())


def raise_(exc_type, msg="test"):
    def f(req):
        raise exc_type(msg, request=req)
    return f


def drop_mid_stream(req):
    def body():
        yield f"data: {json.dumps(delta('The Transformer uses '))}\n\n".encode()
        raise httpx.ReadError("connection reset mid-stream", request=req)
    return httpx.Response(200, headers=RATE, content=body())


def malformed(req):
    return httpx.Response(200, headers=RATE, content=b"data: {\"choices\": [{\"delta\": {\"content\": \"Hal\n\ndata: [DONE]\n\n")


def truncated(req):
    text = echo_answer(req)[: 40]  # cut mid-sentence, no citation: what a max_completion_tokens stop looks like
    return httpx.Response(200, headers=RATE, content=sse(delta(text), delta("", "length"), USAGE))


# ---- running the app
def fresh_state(s, provider: Provider) -> None:
    s.client = ChatClient("https://fake-provider.invalid/v1", "dummy-offline-key", s.cfg["timeouts"], "standin",
                          OUT / "provider_raw", httpx.MockTransport(provider))
    s.governor, s.budget = Governor.from_config(), RetryBudget.from_config(s.cfg["retry"])  # as main.State does


def ask(http: TestClient, body: dict) -> tuple[list, float]:
    t0 = time.perf_counter()
    events, name = [], None
    with http.stream("POST", "/ask", json=body) as r:
        status = r.status_code
        for line in r.iter_lines():
            if line.startswith("event: "):
                name = line[7:]
            elif line.startswith("data: "):
                events.append((name, json.loads(line[6:])))
            elif status == 400 and line.strip():
                events.append(("bad_request", json.loads(line)))
    return [("http", status)] + events, round((time.perf_counter() - t0) * 1000, 1)


def last_trace() -> dict:
    files = sorted((OUT / "traces").glob("*.jsonl"))
    lines = files[-1].read_text(encoding="utf-8").strip().splitlines() if files else []
    return json.loads(lines[-1]) if lines else {}


def safe(events: list, provider: Provider) -> tuple[bool, str]:
    if events[0] == ("http", 400):  # rejected before any work: no model text can be in the reply
        return not provider.requests, "HTTP 400 before any provider call"
    content =next((d["text"] for n, d in events if n == "content"), None)
    done = next((d for n, d in events if n == "done"), {})
    if content is None:
        return False, "no content event"
    if done.get("tier") == "normal":
        from relay.verify.citations import normalize_brackets  # v4.1 shows 【ID】 as [ID]; nothing else may differ
        ok = content in {normalize_brackets(d) for d in provider.sent} and bool(done.get("citations"))
        return ok, "verified draft" if ok else "ANSWER NOT FROM A VERIFIED DRAFT"
    ok = content.startswith(SAFE_PREFIXES)
    return ok, "fixed non-answer text" if ok else "UNKNOWN NON-ANSWER TEXT"


Q = "How many attention heads does the Transformer base model use?"
SCENARIOS = [
    # name, request body, provider plan, expectation (checked loosely: tier/status/reason and provider calls)
    ("normal_answer", {"question": Q}, [echo_answer], {"tier": "normal", "calls": 1}),
    ("lenticular_citation", {"question": Q}, [lambda r: echo_answer(r).replace(" [", " 【").replace("].", "】.")],
     {"tier": "normal", "calls": 1, "note": "L1: gpt-oss citation style"}),
    ("lenticular_unretrieved", {"question": Q}, [lambda r: "The base model uses 8 heads 【HNET:9.9:c001】."] * 2,
     {"tier": "refused", "calls": 2, "note": "L1: still strict"}),
    ("empty_question", {"question": ""}, [], {"http": 400, "calls": 0, "note": "F1"}),
    ("whitespace_question", {"question": "   \n  "}, [], {"http": 400, "calls": 0, "note": "F1"}),
    ("very_long_question", {"question": ("Explain the attention mechanism in detail. " * 500).strip()},
     [], {"http": 400, "calls": 0, "note": "F1: ~21k chars"}),
    ("unknown_retrieval_setting", {"question": Q, "retrieval_setting": "Z"}, [], {"http": 400, "calls": 0, "note": "F6"}),
    ("empty_retrieval", {"question": Q}, [], {"tier": "refused", "calls": 0, "patch": "empty_search"}),
    ("not_in_papers", {"question": "What is GPT-4's MMLU score?"}, [lambda r: "NOT_IN_PAPERS"],
     {"tier": "refused", "calls": 1}),
    ("not_in_papers_near_miss", {"question": "What is GPT-4's MMLU score?"},
     [lambda r: "NOT_IN_PAPERS.", lambda r: "NOT_IN_PAPERS."], {"tier": "refused", "calls": 1, "note": "F7"}),
    ("invalid_citation", {"question": Q}, [lambda r: "The base model uses 8 heads [AIAYN:9.9:c999]."] * 2,
     {"tier": "refused", "calls": 2}),
    ("uncited_answer", {"question": Q}, [lambda r: "The base model uses 8 attention heads."] * 2,
     {"tier": "refused", "calls": 2}),
    ("unsupported_claim", {"question": Q},
     [lambda r: f"The Transformer was trained on one trillion tokens of web text [{passages_in(r)[0][0]}]."] * 2,
     {"tier": "refused", "calls": 2}),
    ("contradicts_evidence", {"question": Q},
     [lambda r: f"The Transformer base model uses 64 attention heads [{passages_in(r)[0][0]}]."] * 2,
     {"tier": "refused", "calls": 2}),
    ("bad_then_good_redraft", {"question": Q}, [lambda r: "The base model uses 8 heads.", echo_answer],
     {"tier": "normal", "calls": 2}),
    ("truncated_length_finish", {"question": Q}, [truncated, truncated],
     {"tier": "refused", "calls": 2, "reason": "model output truncated after redraft", "note": "F2"}),
    ("truncated_then_good", {"question": Q}, [truncated, echo_answer], {"tier": "normal", "calls": 2, "note": "F2"}),
    ("malformed_stream_chunk", {"question": Q}, [malformed, malformed, malformed],
     {"status": "error", "calls": 3, "note": "F3: retried"}),
    ("malformed_then_good", {"question": Q}, [malformed, echo_answer], {"tier": "normal", "calls": 2, "note": "F3"}),
    ("drop_mid_stream", {"question": Q}, [drop_mid_stream, drop_mid_stream, drop_mid_stream],
     {"status": "error", "calls": 3}),
    ("drop_then_recover", {"question": Q}, [drop_mid_stream, echo_answer], {"tier": "normal", "calls": 2}),
    ("timeout_first_byte", {"question": Q}, [raise_(httpx.ReadTimeout)] * 3, {"status": "timeout", "calls": 3}),
    ("connect_error", {"question": Q}, [raise_(httpx.ConnectError)] * 3, {"status": "error", "calls": 3}),
    ("http_500_then_ok", {"question": Q}, [lambda r: err(500, "internal"), echo_answer], {"tier": "normal", "calls": 2}),
    ("http_429_minute_then_ok", {"question": Q},
     [lambda r: err(429, "Rate limit reached ... on tokens per minute (TPM)", "1"), echo_answer],
     {"tier": "normal", "calls": 2}),
    ("http_429_per_day", {"question": Q},
     [lambda r: err(429, "Rate limit reached for model `openai/gpt-oss-120b` on tokens per day (TPD): Limit 200000", "3600")],
     {"status": "rate_limited", "calls": 1, "then_immediate": True}),
    ("http_429_day_unrecognised_wording", {"question": Q},
     [lambda r: err(429, "Daily token quota (TPD) exhausted", "3600")] * 3,
     {"status": "rate_limited", "reason": "daily_budget_exhausted", "calls": 1, "note": "F4"}),
    ("http_429_rpd_header_zero", {"question": Q},
     [lambda r: httpx.Response(429, headers={"x-ratelimit-remaining-requests": "0", "retry-after": "600"},
                               content=b'{"error":{"message":"Rate limit exceeded"}}')],
     {"status": "rate_limited", "reason": "daily_budget_exhausted", "calls": 1, "note": "F4"}),
    ("http_401_bad_key", {"question": Q}, [lambda r: err(401, "Invalid API Key")], {"status": "error", "calls": 1}),
    ("http_404_bad_model", {"question": Q}, [lambda r: err(404, "The model `x` does not exist")],
     {"status": "error", "calls": 1}),
    ("http_400_context_too_long", {"question": Q}, [lambda r: err(400, "Please reduce the length of the messages")],
     {"status": "error", "calls": 1}),
]


def main_() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    main.TRACE_DIR = OUT / "traces"
    t0 = time.perf_counter()
    s = main.State()  # real config, indexes, reranker, NLI; client replaced per scenario
    main.STATE = s
    load_s = round(time.perf_counter() - t0, 1)
    http = TestClient(main.app)
    results = []
    real_search = main.search
    for name, body, plan, expect in SCENARIOS:
        provider = Provider(plan)
        fresh_state(s, provider)
        main.search = (lambda *a, **k: Evidence([], 0.0, None, 0)) if expect.get("patch") == "empty_search" else real_search
        body = body | {"request_id": f"p10-{name}", "scenario": "phase10_offline"}
        events, ms = ask(http, body)
        trace = last_trace()
        extra = None
        if expect.get("then_immediate"):
            extra_events, extra_ms = ask(http, {"question": Q, "request_id": f"p10-{name}-2", "scenario": "phase10_offline"})
            extra = {"events": extra_events, "ms": extra_ms, "provider_calls_after": len(provider.requests)}
        ok, why = safe(events, provider)
        done = next((d for n, d in events if n == "done"), {})
        checks = {"safe_reply": ok}
        if "tier" in expect:
            checks["tier"] = done.get("tier") == expect["tier"]
        if "status" in expect:
            checks["status"] = done.get("status") == expect["status"]
        if "http" in expect:
            checks["http"] = events[0] == ("http", expect["http"])
        if "reason" in expect:
            checks["reason"] = done.get("reason") == expect["reason"]
        if "calls" in expect:
            checks["provider_calls"] = len(provider.requests) == expect["calls"]
        results.append({"scenario": name, "question_chars": len(body["question"]), "expect": expect,
                        "events": events, "client_ms": ms, "provider_calls": len(provider.requests),
                        "drafts_sent": provider.sent, "safety": why, "checks": checks, "passed": all(checks.values()),
                        "trace": {k: trace.get(k) for k in ("status", "answer_tier", "reply_reason", "failure_reason",
                                                            "llm_calls", "retries", "redraft_attempts", "tokens_in",
                                                            "tokens_out", "retrieval_ms", "verification_ms",
                                                            "generation_ms", "total_ms", "context_tokens")},
                        "then": extra})
        print(f"{'PASS' if results[-1]['passed'] else 'FAIL'} {name:34} calls={len(provider.requests)} "
              f"{done.get('status')}/{done.get('tier')}/{done.get('reason')} {why} {ms}ms")
    main.search = real_search
    out = OUT / f"offline_integration_{STAMP}.json"
    with out.open("x", encoding="utf-8") as f:
        json.dump({"stamp": STAMP, "model_load_s": load_s, "scenarios": results}, f, indent=1, ensure_ascii=False)
    print(out.as_posix(), "| passed", sum(r["passed"] for r in results), "of", len(results))
    return 0


if __name__ == "__main__":
    raise SystemExit(main_())
