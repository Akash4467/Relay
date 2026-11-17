import json

import httpx
import pytest

from relay.governor.governor import Governor
from relay.llm.client import ChatClient, LLMError
from relay.llm.hosted import GovernorReject, HostedLLM

TIMEOUTS = {"connect_ms": 1000, "first_byte_ms": 1000, "stream_idle_ms": 1000, "total_call_ms": 5000}
RATE = {"x-ratelimit-remaining-requests": "999", "x-ratelimit-remaining-tokens": "7000"}


def sse(*chunks) -> bytes:
    return b"".join(f"data: {json.dumps(c)}\n\n".encode() for c in chunks) + b"data: [DONE]\n\n"


def delta(text, finish=None):
    return {"choices": [{"delta": {"content": text}, "finish_reason": finish}]}


def server(responses, seen=None):
    """MockTransport serving the given httpx.Response objects in order, recording requests."""
    queue = list(responses)

    def handler(req: httpx.Request) -> httpx.Response:
        if seen is not None:
            seen.append(req)
        return queue.pop(0)
    return httpx.MockTransport(handler)


def client(responses, tmp_path, source="real", seen=None):
    return ChatClient("https://api.example/v1", "secret-key-123", TIMEOUTS, source, tmp_path, server(responses, seen))


OK = httpx.Response(200, headers=RATE, content=sse(
    {"choices": [{"delta": {"role": "assistant"}}]}, delta("Mimi "), delta("runs at 12.5Hz.", "stop"),
    {"choices": [], "x_groq": {"usage": {"prompt_tokens": 50, "completion_tokens": 20, "total_tokens": 70}}}))


def test_stream_text_timing_usage_headers_and_request_body(tmp_path):
    seen = []
    call = client([OK], tmp_path, seen=seen).complete("m", [{"role": "user", "content": "q"}], 900, 0.2, "low")
    assert call.text == "Mimi runs at 12.5Hz." and call.finish_reason == "stop"
    assert call.usage == {"prompt_tokens": 50, "completion_tokens": 20, "total_tokens": 70}
    assert call.headers == RATE
    assert call.t_send_ns <= call.t_first_byte_ns <= call.t_first_content_ns <= call.t_last_ns
    body = json.loads(seen[0].content)
    assert body["max_completion_tokens"] == 900 and body["include_reasoning"] is False
    assert body["reasoning_effort"] == "low" and body["stream"] is True and "max_tokens" not in body
    assert "X-Relay-Question-Id" not in seen[0].headers  # only the stand-in gets the question id


def test_top_level_usage_also_read(tmp_path):
    resp = httpx.Response(200, content=sse(delta("ok", "stop"), {"choices": [], "usage": {"total_tokens": 9}}))
    assert client([resp], tmp_path).complete("m", [], 5, 0, "low").usage == {"total_tokens": 9}


def test_raw_log_appends_and_never_contains_the_key(tmp_path):
    c = client([OK, httpx.Response(429, headers={"retry-after": "3"}, json={"error": {"message": "slow down"}})], tmp_path)
    c.complete("m", [{"role": "user", "content": "q"}], 900, 0.2, "low")
    with pytest.raises(LLMError):
        c.complete("m", [{"role": "user", "content": "q"}], 900, 0.2, "low")
    files = list((tmp_path / "real").glob("*.jsonl"))
    lines = files[0].read_text(encoding="utf-8").splitlines()
    assert len(files) == 1 and len(lines) == 2
    assert "secret-key-123" not in files[0].read_text(encoding="utf-8")
    first = json.loads(lines[0])
    assert first["text"] == "Mimi runs at 12.5Hz." and first["first_content_ms"] is not None
    assert json.loads(lines[1])["status"] == 429


@pytest.mark.parametrize("status,kind", [(429, "rate_limited"), (500, "upstream_error"), (400, "bad_request")])
def test_http_errors_raise_with_kind_and_body(tmp_path, status, kind):
    resp = httpx.Response(status, headers={"retry-after": "2"}, json={"error": {"message": "nope"}})
    with pytest.raises(LLMError) as e:
        client([resp], tmp_path).complete("m", [], 5, 0, "low")
    assert e.value.kind == kind and "nope" in e.value.call.error
    assert e.value.call.retry_after_s == 2.0


def test_standin_gets_question_id_header_and_logs_separately(tmp_path):
    seen = []
    client([OK], tmp_path, source="standin", seen=seen).complete("m", [], 5, 0, "low", question_id="q05")
    assert seen[0].headers["X-Relay-Question-Id"] == "q05"
    assert list((tmp_path / "standin").glob("*.jsonl"))


def test_connection_failure_is_upstream_error(tmp_path):
    def boom(req):
        raise httpx.ConnectError("refused")
    c = ChatClient("https://x/v1", "k", TIMEOUTS, "real", tmp_path, httpx.MockTransport(boom))
    with pytest.raises(LLMError) as e:
        c.complete("m", [], 5, 0, "low")
    assert e.value.kind == "upstream_error"


MODEL = {"id": "m", "max_tokens": 900, "temperature": 0.2, "reasoning_effort": "low"}
LIMITS = {"m": {"rpm": 30, "rpd": 1000, "tpm": 8000, "tpd": 200000}}


def test_hosted_llm_reserves_calls_settles_and_records(tmp_path):
    gov = Governor(LIMITS, clock=lambda: 0.0)
    llm = HostedLLM(client([OK], tmp_path), gov, MODEL, output_reserve=770)
    assert llm([{"role": "user", "content": "q"}]) == "Mimi runs at 12.5Hz."
    rec = llm.calls[0]
    assert rec["status"] == "ok" and rec["tokens_in"] == 50 and rec["tokens_out"] == 20 and not rec["usage_estimated"]
    # settled to the 70 tokens actually used, then resynced down to the header's 7000 remaining this minute
    assert gov.buckets["m"]["tpm"].available(0) == pytest.approx(7000)
    assert gov.buckets["m"]["rpd"].available(0) == 999


def test_hosted_llm_adds_request_overhead_and_reports_estimate_error(tmp_path):
    gov = Governor(LIMITS, clock=lambda: 0.0)
    llm = HostedLLM(client([OK], tmp_path), gov, MODEL, output_reserve=770, request_overhead=40)
    llm([{"role": "user", "content": "q"}])
    rec = llm.calls[0]
    assert rec["tokens_in_local"] == 40 + 1 + 4  # overhead + "q" + per-message format tokens
    assert rec["reserved"] == rec["tokens_in_local"] + 770
    assert rec["prompt_tokens_local_error"] == 50 - rec["tokens_in_local"]
    assert rec["usage_source"] == "x_groq.usage"


def _err(status, retry_after=None, msg="x"):
    return httpx.Response(status, headers={"retry-after": retry_after} if retry_after else {}, json={"error": {"message": msg}})


def test_retry_after_500_waits_at_least_retry_after(tmp_path):
    from relay.llm.hosted import RetryBudget
    slept = []
    llm = HostedLLM(client([_err(500, "3"), OK], tmp_path), Governor(LIMITS, clock=lambda: 0.0), MODEL, 770,
                    max_retries=2, budget=RetryBudget(), sleep=slept.append)
    assert llm([{"role": "user", "content": "q"}]) == "Mimi runs at 12.5Hz."
    assert llm.retries == 1 and slept[-1] >= 3.0 and llm.calls[0]["backoff_s"] >= 3.0


def test_per_day_429_is_never_retried_and_freezes_daily_buckets(tmp_path):
    from relay.llm.hosted import DailyExhausted
    gov = Governor(LIMITS, clock=lambda: 0.0)
    llm = HostedLLM(client([_err(429, "90", "Rate limit reached on requests per day (RPD)"), OK], tmp_path), gov,
                    MODEL, 770, max_retries=2, sleep=lambda s: None)
    with pytest.raises(DailyExhausted):
        llm([{"role": "user", "content": "q"}])
    assert llm.retries == 0 and gov.daily_exhausted_for_s("m") == pytest.approx(90)
    with pytest.raises(DailyExhausted):  # the next call does not even reach the provider
        llm([{"role": "user", "content": "q"}])
    assert len(llm.calls) == 1


def test_no_retry_past_the_deadline(tmp_path):
    import time as _t
    llm = HostedLLM(client([_err(500, "30"), OK], tmp_path), Governor(LIMITS, clock=lambda: 0.0), MODEL, 770,
                    max_retries=2, deadline=_t.monotonic() + 5, sleep=lambda s: None)
    with pytest.raises(LLMError):
        llm([{"role": "user", "content": "q"}])
    assert llm.retries == 0  # a 30 s retry-after cannot fit in the 5 s left


def test_retry_budget_is_ten_percent_of_first_attempts_but_at_least_one():
    from relay.llm.hosted import RetryBudget
    t = {"now": 0.0}
    b = RetryBudget(0.10, clock=lambda: t["now"])
    for _ in range(5):
        b.first_attempt()
    assert b.try_retry() and not b.try_retry()  # 5 first attempts: the floor of one retry
    for _ in range(25):
        b.first_attempt()
    assert b.try_retry() and b.try_retry() and not b.try_retry()  # 30 first attempts: 3 retries in total
    t["now"] = 61.0
    b.first_attempt()
    assert b.try_retry()  # the window moved on


def test_hosted_llm_429_freezes_model_and_reraises(tmp_path):
    gov = Governor(LIMITS, clock=lambda: 0.0)
    llm = HostedLLM(client([httpx.Response(429, headers={"retry-after": "5"}, json={})], tmp_path), gov, MODEL, 770)
    with pytest.raises(LLMError):
        llm([{"role": "user", "content": "q"}])
    assert gov.wait_estimate("m", 10) == pytest.approx(5)
    assert llm.calls[0]["status"] == "rate_limited"


def test_hosted_llm_refuses_when_wait_is_too_long(tmp_path):
    gov = Governor({"m": {"rpm": 30, "rpd": 1000, "tpm": 1000, "tpd": 200000}}, clock=lambda: 0.0)
    gov.try_reserve("m", 1000)
    seen = []
    llm = HostedLLM(client([OK], tmp_path, seen=seen), gov, MODEL, output_reserve=500, max_wait_s=5)
    with pytest.raises(GovernorReject):
        llm([{"role": "user", "content": "q"}])
    assert seen == []  # never sent


def test_hosted_llm_waits_then_calls(tmp_path):
    t = {"now": 0.0}
    gov = Governor({"m": {"rpm": 30, "rpd": 1000, "tpm": 1200, "tpd": 200000}}, clock=lambda: t["now"])
    gov.try_reserve("m", 1000)
    slept = []

    def sleep(s):
        slept.append(s)
        t["now"] += s
    llm = HostedLLM(client([OK], tmp_path), gov, MODEL, output_reserve=500, max_wait_s=60, sleep=sleep)
    assert llm([{"role": "user", "content": "q"}])
    assert slept and 0 < slept[0] <= 60
