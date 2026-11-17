"""Phase 10B: /ask input validation (F1 question, F6 retrieval_setting). Bad requests get HTTP 400 with a clear error,
never reach retrieval or the provider, and still write one trace line. Fakes only, never real Groq."""
import json

import pytest
from fastapi.testclient import TestClient

from relay import main
from relay.config import app_config
from tests.integration.test_server import BODY, make_state, no_lifespan, ok_reply, traces  # noqa: F401

LIMIT = app_config()["input"]["max_question_chars"]


def post(body):
    return TestClient(main.app).post("/ask", json=body)


@pytest.mark.parametrize("question,why", [("", "empty"), ("   \n\t ", "empty"), ("x" * (LIMIT + 1), "too long")])
def test_bad_question_is_rejected_before_any_work(tmp_path, monkeypatch, question, why):
    s = make_state(tmp_path, [], monkeypatch)  # no provider replies queued: any call would fail the test
    calls = []
    monkeypatch.setattr(main, "search", lambda *a, **k: calls.append(1))
    r = post(BODY | {"question": question})
    assert r.status_code == 400 and r.json()["error"] == "invalid_question" and why in r.json()["detail"]
    assert calls == [] and s.governor.buckets[s.primary["id"]]["rpd"].available(0) == 1000
    (t,) = traces(tmp_path)
    assert t["status"] == "bad_request" and t["failure_reason"] == "invalid_question" and t["admitted"] is False
    assert t["question_chars"] == len(question)


def test_unknown_retrieval_setting_is_a_bad_request(tmp_path, monkeypatch):
    make_state(tmp_path, [], monkeypatch)
    r = post(BODY | {"retrieval_setting": "Z"})
    assert r.status_code == 400 and r.json()["error"] == "invalid_retrieval_setting"
    assert "Z" in r.json()["detail"] and "A" in r.json()["detail"]
    assert traces(tmp_path)[0]["failure_reason"] == "invalid_retrieval_setting"


def test_question_at_the_limit_is_still_answered(tmp_path, monkeypatch):
    make_state(tmp_path, [ok_reply()], monkeypatch)
    with TestClient(main.app).stream("POST", "/ask", json=BODY | {"question": "attention heads " + "x" * (LIMIT - 16)}) as r:
        assert r.status_code == 200
        events = [json.loads(l[6:]) for l in r.iter_lines() if l.startswith("data: ")]
    assert events[-1]["status"] == "ok"


def test_trace_records_the_expansion_and_keeps_the_original_question(tmp_path, monkeypatch):
    """Phase 11B: the original question is what the model is asked; the expanded retrieval query is traced apart."""
    import httpx
    s = make_state(tmp_path, [], monkeypatch)
    sent = []

    def provider(req):
        sent.append(json.loads(req.content))
        return ok_reply()
    s.client.http = httpx.Client(transport=httpx.MockTransport(provider))
    q = "what does the model with a single attention head reach"
    with TestClient(main.app).stream("POST", "/ask", json=BODY | {"question": q}) as r:
        assert r.status_code == 200
        list(r.iter_lines())
    (t,) = traces(tmp_path)
    assert t["query_expansion"] == ["h = 1"] and t["retrieval_query"] == q + " [h = 1]"
    assert t["caption_sharing"] is True and t["evidence_ids"] and all(":" in c for c in t["evidence_ids"])
    user = sent[0]["messages"][1]["content"]  # the model is asked the ORIGINAL question, never the expanded query
    assert user.startswith(f"Question: {q}\n") and "[h = 1]" not in user
