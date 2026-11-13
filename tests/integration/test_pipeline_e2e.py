"""Phase 2 done-check on a tiny corpus: question -> search -> structural sufficiency -> draft (fake LLM) ->
verify (fake NLI) -> answer / caught unsupported claim / refusal. No model downloads, never real Groq."""
import json

import pytest

from relay.config import retrieval_settings
from relay.pipeline import Verifier, run
from relay.retrieval import index_build as ib
from relay.retrieval.search import Index
from tests.unit.test_index_build import CHUNKS, FakeModel
from tests.unit.test_pipeline import FakeLLM
from tests.unit.test_rerank_context import FakeCrossEncoder
from tests.unit.test_verifier import FakeNLI


@pytest.fixture
def index(tmp_path):
    proc, idx = tmp_path / "proc", tmp_path / "index"
    proc.mkdir()
    (proc / "chunks_256.jsonl").write_text("".join(json.dumps(c) + "\n" for c in CHUNKS), encoding="utf-8")
    ib.build(256, FakeModel(), proc, idx)
    return Index(256, FakeModel(), idx, proc)


def verifier(index, entail):
    return Verifier(FakeNLI(entail), 0.5, 0.8)


B = retrieval_settings()["B"]


def test_answerable_question_is_answered_with_citations(index):
    llm = FakeLLM("The model employs 8 parallel attention heads [AIAYN:3.2.2:c001].")
    r = run("how many attention heads", B, 2500, index, llm,
            verifier(index, {"The model employs 8 parallel attention heads.": 0.9}), FakeCrossEncoder())
    assert r.status == "answered" and r.citations == ["AIAYN:3.2.2:c001"]
    # the prompt carried the retrieved passage and its ID
    assert "[AIAYN:3.2.2:c001] (AIAYN §3.2.2 Multi-Head Attention)" in llm.calls[0][1]["content"]


def test_unsupported_claim_is_caught_and_never_reaches_the_user(index):
    bad = "The model employs 16 parallel attention heads [AIAYN:3.2.2:c001]."
    r = run("how many attention heads", B, 2500, index, FakeLLM(bad, bad), verifier(index, {}), FakeCrossEncoder())
    assert r.status == "refused" and r.llm_calls == 2
    assert "16" not in r.text
    assert "number 16 not in the cited passage" in r.reports[0].failures()[0][1]


def test_unanswerable_question_is_refused(index):
    r = run("what GPU trained the mimi codec", B, 2500, index, FakeLLM("NOT_IN_PAPERS"), verifier(index, {}),
            FakeCrossEncoder())
    assert r.status == "refused" and r.text.startswith("I couldn't find this in the passages I retrieved from the three papers")


def test_full_request_through_http_client_and_governor(index, tmp_path):
    """question -> search -> structural check -> HostedLLM (governor + streaming client, fake HTTP server) ->
    verify -> redraft -> grounded answer; both calls logged raw, governor settled after each."""
    import httpx

    from relay.governor.governor import Governor
    from relay.llm.client import ChatClient
    from relay.llm.hosted import HostedLLM
    from tests.unit.test_client import TIMEOUTS, delta, sse

    bad, good = "It uses 16 parallel attention heads [AIAYN:3.2.2:c001].", "It uses 8 parallel attention heads [AIAYN:3.2.2:c001]."
    usage = {"choices": [], "x_groq": {"usage": {"prompt_tokens": 300, "completion_tokens": 40, "total_tokens": 340}}}
    replies = [httpx.Response(200, content=sse(delta(bad, "stop"), usage)),
               httpx.Response(200, content=sse(delta(good, "stop"), usage))]
    transport = httpx.MockTransport(lambda req: replies.pop(0))
    gov = Governor({"m": {"rpm": 30, "rpd": 1000, "tpm": 8000, "tpd": 200000}}, clock=lambda: 0.0)
    llm = HostedLLM(ChatClient("https://x/v1", "k", TIMEOUTS, "real", tmp_path, transport), gov,
                    {"id": "m", "max_tokens": 900, "temperature": 0.2, "reasoning_effort": "low"}, output_reserve=770)

    r = run("how many attention heads", B, 2500, index, llm,
            verifier(index, {"It uses 8 parallel attention heads.": 0.95}), FakeCrossEncoder())

    assert (r.status, r.text, r.llm_calls) == ("answered", good, 2)
    assert [c["status"] for c in llm.calls] == ["ok", "ok"]
    assert gov.buckets["m"]["tpd"].available(0) == pytest.approx(200000 - 2 * 340)  # settled to actual usage
    assert len((tmp_path / "real").glob("*.jsonl").__next__().read_text(encoding="utf-8").splitlines()) == 2


def test_cited_passage_from_another_request_is_rejected(index):
    # the model cites a real chunk that was not retrieved for this question (context cap 0 keeps nothing)
    llm = FakeLLM("Mimi is a codec [MOSHI:3.3:c002].")
    r = run("what is mimi", B, 0, index, llm, verifier(index, {}), FakeCrossEncoder())
    assert r.status == "refused" and llm.calls == []  # empty context: refused structurally, model never called
