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
    assert r.status == "refused" and r.text.startswith("The three papers don't state this. I searched:")


def test_cited_passage_from_another_request_is_rejected(index):
    # the model cites a real chunk that was not retrieved for this question (context cap 0 keeps nothing)
    llm = FakeLLM("Mimi is a codec [MOSHI:3.3:c002].")
    r = run("what is mimi", B, 0, index, llm, verifier(index, {}), FakeCrossEncoder())
    assert r.status == "refused" and llm.calls == []  # empty context: refused structurally, model never called
