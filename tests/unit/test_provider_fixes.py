"""Phase 10B regression tests (fake transports only, never real Groq): truncated drafts (F2), malformed stream chunks
(F3), per-day 429 recognition (F4), stream idle timeout decision (F5), NOT_IN_PAPERS normalisation (F7) and the retry
budget floor (F9). Each test reproduced its bug before the fix."""
import httpx
import pytest

from relay.config import app_config
from relay.governor.governor import Governor
from relay.llm import prompts
from relay.llm.client import ChatClient, LLMError, is_daily_429
from relay.llm.hosted import DailyExhausted, HostedLLM, RetryBudget, TruncatedOutput
from relay.pipeline import answer_from_evidence
from tests.unit.test_client import LIMITS, MODEL, OK, RATE, TIMEOUTS, _err, client, delta, sse
from tests.unit.test_pipeline import CORPUS, GOOD, SUPPORTED, FakeLLM, ev, verifier
from tests.unit.test_verifier import P

Q = [{"role": "user", "content": "q"}]
USAGE = {"choices": [], "usage": {"prompt_tokens": 50, "completion_tokens": 900, "total_tokens": 950}}


def truncated_reply():
    return httpx.Response(200, headers=RATE, content=sse(delta("Mimi uses 8 quan"), delta("", "length"), USAGE))


# ---- F2: a draft cut off at max_completion_tokens is detected, never verified or returned
def test_length_finish_raises_truncated_and_keeps_accounting(tmp_path):
    gov = Governor(LIMITS, clock=lambda: 0.0)
    llm = HostedLLM(client([truncated_reply()], tmp_path), gov, MODEL, 770)
    with pytest.raises(TruncatedOutput) as e:
        llm(Q)
    assert e.value.text == "Mimi uses 8 quan"
    rec = llm.calls[0]
    assert rec["status"] == "ok" and rec["finish_reason"] == "length" and rec["tokens_out"] == 900  # still counted


class TruncatingLLM(FakeLLM):
    def __call__(self, messages):
        self.calls.append(messages)
        r = self.replies.pop(0)
        if isinstance(r, Exception):
            raise r
        return r


def test_truncated_draft_uses_the_one_redraft_and_is_never_returned():
    llm = TruncatingLLM(TruncatedOutput("Mimi uses 8 quan"), GOOD)
    r = answer_from_evidence("How many quantizers?", ev(P), llm, verifier(SUPPORTED), CORPUS)
    assert (r.status, r.text, r.llm_calls) == ("answered", GOOD, 2)
    assert "cut off" in llm.calls[1][-1]["content"]


def test_two_truncated_drafts_refuse_with_a_truncation_reason():
    llm = TruncatingLLM(TruncatedOutput("Mimi uses 8 quan"), TruncatedOutput("Mimi"))
    r = answer_from_evidence("How many quantizers?", ev(P), llm, verifier(SUPPORTED), CORPUS)
    assert r.status == "refused" and r.llm_calls == 2 and r.refusal_reason == "model output truncated after redraft"
    assert "Mimi uses 8 quan" not in r.text and r.text.startswith("I couldn't find this")


# ---- F3: a malformed chunk is a provider error: recorded, settled and retried
def test_malformed_chunk_is_an_upstream_error(tmp_path):
    bad = httpx.Response(200, headers=RATE, content=b'data: {"choices": [{"delta": {"content": "Hal\n\ndata: [DONE]\n\n')
    with pytest.raises(LLMError) as e:
        client([bad], tmp_path).complete("m", Q, 5, 0, "low")
    assert e.value.kind == "upstream_error" and "malformed" in e.value.call.error


def test_malformed_chunk_is_retried_and_accounted(tmp_path):
    bad = httpx.Response(200, headers=RATE, content=b"data: not json\n\ndata: [DONE]\n\n")
    llm = HostedLLM(client([bad, OK], tmp_path), Governor(LIMITS, clock=lambda: 0.0), MODEL, 770, max_retries=2,
                    budget=RetryBudget(min_retries=2), sleep=lambda s: None)
    assert llm(Q) == "Mimi runs at 12.5Hz."
    assert llm.retries == 1 and [c["status"] for c in llm.calls] == ["upstream_error", "ok"]


# ---- F4: per-day exhaustion recognised from the message wording or from the RPD header
@pytest.mark.parametrize("msg,headers,daily", [
    ("Rate limit reached for model `x` on tokens per day (TPD): Limit 200000", {}, True),
    ("Rate limit reached on requests per day (RPD)", {}, True),
    ("Daily token quota (TPD) exhausted", {}, True),
    ("Rate limit exceeded", {"x-ratelimit-remaining-requests": "0"}, True),  # Groq: the requests header is per day
    ("Rate limit reached on tokens per minute (TPM): Limit 8000", {"x-ratelimit-remaining-requests": "12"}, False),
    ("Rate limit reached on requests per minute (RPM)", {}, False),
])
def test_daily_429_recognition(tmp_path, msg, headers, daily):
    resp = httpx.Response(429, headers=headers | {"retry-after": "30"}, json={"error": {"message": msg}})
    with pytest.raises(LLMError) as e:
        client([resp], tmp_path).complete("m", Q, 5, 0, "low")
    assert is_daily_429(e.value) is daily


def test_unusually_worded_daily_429_becomes_daily_exhausted(tmp_path):
    gov = Governor(LIMITS, clock=lambda: 0.0)
    llm = HostedLLM(client([_err(429, "3600", "Daily token quota (TPD) exhausted"), OK], tmp_path), gov, MODEL, 770,
                    max_retries=2, sleep=lambda s: None)
    with pytest.raises(DailyExhausted):
        llm(Q)
    assert llm.retries == 0 and gov.daily_exhausted_for_s("m") > 0


# ---- F5: stream_idle_ms is documented as bounded by the read timeout, not enforced separately
def test_stream_idle_is_within_the_read_timeout_bound():
    t = app_config()["timeouts"]
    c = ChatClient("https://x/v1", "k", t, "standin", None)
    assert c.http.timeout.read == t["first_byte_ms"] / 1000 and t["stream_idle_ms"] <= t["first_byte_ms"]


# ---- F7: the refusal marker with harmless punctuation or markup is still the marker
@pytest.mark.parametrize("text,marker", [
    ("NOT_IN_PAPERS", True), ("NOT_IN_PAPERS.", True), ("**NOT_IN_PAPERS**", True), ("`NOT_IN_PAPERS`", True),
    ("  not_in_papers \n", True), ('"NOT_IN_PAPERS"', True),
    ("NOT_IN_PAPERS, but Mimi uses 8 quantizers [MOSHI:3.3.1:c030].", False),
    ("The answer is NOT_IN_PAPERS", False), ("", False),
])
def test_not_in_papers_marker(text, marker):
    assert prompts.is_not_in_papers(text) is marker


def test_marker_with_period_refuses_after_one_call():
    llm = FakeLLM("NOT_IN_PAPERS.")
    r = answer_from_evidence("What GPU count trained Moshi?", ev(P), llm, verifier({}), CORPUS)
    assert (r.status, r.refusal_reason, r.llm_calls) == ("refused", "model: not in papers", 1)


# ---- F9: the retry budget floor equals the configured retries per call
def test_quiet_system_allows_every_configured_retry(tmp_path):
    retry = app_config()["retry"]
    budget = RetryBudget.from_config(retry)
    fails = [_err(500)] * retry["max_transport_retries_per_call"]
    llm = HostedLLM(client(fails + [OK], tmp_path), Governor(LIMITS, clock=lambda: 0.0), MODEL, 770,
                    max_retries=retry["max_transport_retries_per_call"], budget=budget, sleep=lambda s: None)
    assert llm(Q) == "Mimi runs at 12.5Hz."
    assert llm.retries == retry["max_transport_retries_per_call"] == 2


def test_busy_system_still_caps_retries_at_ten_percent():
    t = {"now": 0.0}
    b = RetryBudget(0.10, min_retries=2, clock=lambda: t["now"])
    for _ in range(100):
        b.first_attempt()
    assert sum(b.try_retry() for _ in range(20)) == 10
