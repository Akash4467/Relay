import json

import pytest

from relay.config import RetrievalSetting
from relay.retrieval import index_build as ib
from relay.llm.tokenizer import count
from relay.retrieval.context import passage_block
from relay.retrieval.search import Index, reserve, search
from tests.unit.test_index_build import CHUNKS, FakeModel
from tests.unit.test_rerank_context import FakeCrossEncoder


@pytest.fixture
def index(tmp_path):
    proc, idx = tmp_path / "proc", tmp_path / "index"
    proc.mkdir()
    (proc / "chunks_256.jsonl").write_text("".join(json.dumps(c) + "\n" for c in CHUNKS), encoding="utf-8")
    ib.build(256, FakeModel(), proc, idx)
    return Index(256, FakeModel(), idx, proc)


A = RetrievalSetting(piece_tokens=256, candidates=20, rerank=False, final_passages=2, context_budget=1600,
                     expansion=False)
B = A.model_copy(update={"rerank": True})


def test_search_without_rerank(index):
    ev = search("how many attention heads", A, 2500, index)
    assert ev.passages[0]["chunk_id"] == "AIAYN:3.2.2:c001"
    assert len(ev.passages) == 2  # final_passages
    assert ev.top_ce is None and 0 < ev.top_dense_cos <= 1.0001
    assert all("score_rrf" in c and "dense_cos" in c for c in ev.candidates)
    assert ev.tokens > 0


def test_search_with_rerank_uses_cross_encoder_order(index):
    ev = search("routing module boundary cosine", B, 2500, index, FakeCrossEncoder())
    assert ev.passages[0]["chunk_id"] == "HNET:2.2.1:c003"
    assert ev.top_ce == ev.passages[0]["score_ce"] > 0


def test_rerank_setting_without_model_fails_loudly(index):
    with pytest.raises(ValueError):
        search("q", B, 2500, index)


def test_context_cap_limits_passages(index):
    assert search("attention", A, 0, index).passages == []


def test_tighter_of_pressure_cap_and_setting_budget_applies(index):
    tiny = A.model_copy(update={"context_budget": 0, "final_passages": 3})
    assert search("attention", tiny, 2500, index).passages == []  # budget binds
    full = search("attention", A.model_copy(update={"final_passages": 3}), 2500, index)
    one = count(passage_block(full.passages[0])) + 2
    capped = search("attention", A.model_copy(update={"final_passages": 3}), one, index)
    assert len(capped.passages) == 1 and capped.tokens <= one  # pressure cap binds


def test_ordinary_question_keeps_global_policy(index):
    ev = search("what are attention heads in the original Transformer", A, 2500, index)
    assert ev.policy == "global" and ev.named_papers == ["AIAYN"]


def test_multi_paper_question_reserves_each_named_paper(index):
    # globally, both passages would be AIAYN-heavy; the policy must still bring in the H-Net passage
    q = "compare attention heads attention heads in the original Transformer with H-Net"
    one = A.model_copy(update={"final_passages": 2})
    ev = search(q, one, 2500, index, reserve_per_paper=1)
    assert ev.policy == "per_paper" and ev.named_papers == ["AIAYN", "HNET"]
    assert {p["paper"] for p in ev.passages} == {"AIAYN", "HNET"}
    assert {c["paper"] for c in ev.candidates} <= {"AIAYN", "HNET"}  # unnamed papers are not candidates


def test_reserve_round_robin_then_rank_order():
    c = [{"id": i, "paper": p} for i, p in enumerate(["A", "A", "A", "B", "B"])]
    out = reserve(c, ["A", "B"], 2)
    assert [x["id"] for x in out] == [0, 3, 1, 4, 2]
    assert [x["id"] for x in reserve(c, ["A", "B"], 1)] == [0, 3, 1, 2, 4]
    assert [x["id"] for x in reserve(c, ["A", "C"], 1)] == [0, 1, 2, 3, 4]  # paper with no candidates: no-op


def test_stale_index_is_rejected(tmp_path):
    proc, idx = tmp_path / "proc", tmp_path / "index"
    proc.mkdir()
    (proc / "chunks_256.jsonl").write_text("".join(json.dumps(c) + "\n" for c in CHUNKS), encoding="utf-8")
    ib.build(256, FakeModel(), proc, idx)
    (proc / "chunks_256.jsonl").write_text(json.dumps(CHUNKS[0]) + "\n", encoding="utf-8")  # rebuilt corpus
    with pytest.raises(ValueError, match="stale"):
        Index(256, FakeModel(), idx, proc)
