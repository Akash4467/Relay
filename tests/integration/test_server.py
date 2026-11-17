"""The /ask front door with fakes: fake index/embedder/cross-encoder/NLI and a fake OpenAI-compatible provider.
No model downloads, never real Groq."""
import asyncio
import json
import threading
from concurrent.futures import ThreadPoolExecutor

import httpx
import pytest
from fastapi.testclient import TestClient

from relay import main
from relay.config import app_config, retrieval_settings
from relay.governor.governor import Governor
from relay.llm.client import ChatClient
from relay.llm.hosted import RetryBudget
from relay.pipeline import Verifier
from relay.retrieval import index_build as ib
from relay.retrieval.search import Index
from tests.unit.test_client import TIMEOUTS, delta, sse
from tests.unit.test_index_build import CHUNKS, FakeModel
from tests.unit.test_rerank_context import FakeCrossEncoder
from tests.unit.test_verifier import FakeNLI

GOOD = "The model employs 8 parallel attention heads [AIAYN:3.2.2:c001]."
USAGE = {"choices": [], "usage": {"prompt_tokens": 300, "completion_tokens": 40, "total_tokens": 340,
                                  "completion_tokens_details": {"reasoning_tokens": 12}}}
LIMITS = {"openai/gpt-oss-120b": {"rpm": 30, "rpd": 1000, "tpm": 8000, "tpd": 200000}}


def ok_reply():
    return httpx.Response(200, headers={"x-ratelimit-remaining-tokens": "7600"}, content=sse(delta(GOOD, "stop"), USAGE))


def make_state(tmp_path, replies, monkeypatch):
    proc, idx = tmp_path / "proc", tmp_path / "index"
    proc.mkdir()
    (proc / "chunks_256.jsonl").write_text("".join(json.dumps(c) + "\n" for c in CHUNKS), encoding="utf-8")
    (proc / "chunks_512.jsonl").write_text("".join(json.dumps(c) + "\n" for c in CHUNKS), encoding="utf-8")
    ib.build(256, FakeModel(), proc, idx)
    ib.build(512, FakeModel(), proc, idx)
    queue = list(replies)
    transport = httpx.MockTransport(lambda req: queue.pop(0))
    monkeypatch.setattr(main, "TRACE_DIR", tmp_path / "traces")

    s = object.__new__(main.State)
    s.cfg = app_config()
    s.settings, s.default_setting = retrieval_settings(), "B"
    s.client = ChatClient("https://x/v1", "k", TIMEOUTS, "standin", tmp_path / "raw", transport)
    s.indexes = {size: Index(size, FakeModel(), idx, proc) for size in (256, 512)}
    s.reranker = FakeCrossEncoder()
    s.verifier = Verifier(FakeNLI({"The model employs 8 parallel attention heads.": 0.95}), 0.9, 0.9)
    s.governor = Governor(LIMITS)
    s.budget = RetryBudget(0.10)
    s.workers, s.max_queue = 2, 3
    s.executor = ThreadPoolExecutor(max_workers=2)
    s.slots = asyncio.Semaphore(2)
    s.waiting = s.busy = s.peak_waiting = 0
    s.service_ema_s = 1.0
    s.primary = s.cfg["models"]["primary"]
    s.output_reserve, s.typical_reserve = 770, 3270
    s.trace_lock = threading.Lock()
    monkeypatch.setattr(main, "STATE", s)
    return s


def ask(body):
    events = []
    with TestClient(main.app) as _:
        pass  # (lifespan is not used: STATE is set by the test)
    client = TestClient(main.app)
    with client.stream("POST", "/ask", json=body) as r:
        name = None
        for line in r.iter_lines():
            if line.startswith("event: "):
                name = line[7:]
            elif line.startswith("data: "):
                events.append((name, json.loads(line[6:])))
    return events


def traces(tmp_path):
    return [json.loads(l) for f in (tmp_path / "traces").glob("*.jsonl") for l in f.read_text(encoding="utf-8").splitlines()]


@pytest.fixture(autouse=True)
def no_lifespan(monkeypatch):
    """TestClient as a context manager would run the real lifespan (model loading); replace it."""
    from contextlib import asynccontextmanager

    @asynccontextmanager
    async def noop(_app):
        yield
    monkeypatch.setattr(main.app.router, "lifespan_context", noop)


BODY = {"question": "how many attention heads", "question_id": "q99", "request_id": "r1", "scenario": "test",
        "deadline_ms": 20000}


def test_answered_request_streams_and_traces_every_timing(tmp_path, monkeypatch):
    make_state(tmp_path, [ok_reply()], monkeypatch)
    events = ask(BODY)
    assert [e for e, _ in events] == ["meta", "content", "done"]
    assert events[0][1]["admitted"] is True and events[1][1]["text"] == GOOD
    assert events[2][1]["status"] == "ok" and events[2][1]["tier"] == "normal"
    (t,) = traces(tmp_path)
    assert t["request_id"] == "r1" and t["status"] == "ok" and t["source"] == "stand-in"
    for k in ("retrieval_ms", "queue_wait_ms", "governor_wait_ms", "first_byte_ms", "model_ttft_ms", "generation_ms",
              "verification_ms", "first_token_ms", "total_ms"):
        assert isinstance(t[k], (int, float)), k
    assert (t["tokens_in"], t["tokens_out"], t["reasoning_tokens"], t["retries"]) == (300, 40, 12, 0)
    assert t["first_token_ms"] <= t["total_ms"]


def test_full_queue_is_rejected_immediately(tmp_path, monkeypatch):
    s = make_state(tmp_path, [], monkeypatch)
    s.waiting = s.max_queue
    events = ask(BODY)
    assert events[0][1]["admitted"] is False
    assert events[-1][1] == {"status": "rate_limited", "tier": "unavailable", "reason": "queue_rejected"}
    assert "at capacity" in events[1][1]["text"]
    assert traces(tmp_path)[0]["failure_reason"] == "queue_rejected"


def test_known_daily_exhaustion_answers_without_queue_or_call(tmp_path, monkeypatch):
    s = make_state(tmp_path, [], monkeypatch)
    s.governor.on_daily_429(s.primary["id"], 600)
    events = ask(BODY)
    assert events[-1][1]["reason"] == "daily_budget_exhausted"
    assert "today's model capacity" in events[1][1]["text"] and "10 minutes" in events[1][1]["text"]


def test_provider_per_day_429_becomes_unavailable_and_is_not_retried(tmp_path, monkeypatch):
    body = {"error": {"message": "Rate limit reached for model `x` on requests per day (RPD): Limit 1000, Used 1000",
                      "type": "requests", "code": "rate_limit_exceeded"}}
    s = make_state(tmp_path, [httpx.Response(429, headers={"retry-after": "87"}, json=body)], monkeypatch)
    events = ask(BODY)
    assert events[-1][1]["reason"] == "daily_budget_exhausted"
    t = traces(tmp_path)[0]
    assert t["retries"] == 0 and t["llm_calls"] == 1
    assert s.governor.daily_exhausted_for_s(s.primary["id"]) > 80
    assert ask(BODY | {"request_id": "r2"})[0][1]["admitted"] is False  # next request: no call at all


def test_upstream_500_is_retried_within_policy(tmp_path, monkeypatch):
    make_state(tmp_path, [httpx.Response(500, json={"error": {"message": "boom"}}), ok_reply()], monkeypatch)
    events = ask(BODY)
    assert events[-1][1]["status"] == "ok"
    t = traces(tmp_path)[0]
    assert (t["retries"], t["llm_calls"], t["initial_attempts"]) == (1, 2, 1)
