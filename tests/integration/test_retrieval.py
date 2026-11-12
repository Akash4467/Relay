"""Retrieval wired together: chunks -> index_build -> BM25 + dense -> RRF -> rerank -> context cap -> Evidence,
plus the dev-set done-check on a tiny corpus. Fake embedder and cross-encoder: no model downloads."""
import json
from pathlib import Path

import pytest

from relay.config import retrieval_settings
from relay.eval import retrieval_recall as rr
from relay.retrieval import index_build as ib
from relay.retrieval.search import Index, search
from tests.unit.test_index_build import CHUNKS, FakeModel
from tests.unit.test_rerank_context import FakeCrossEncoder


@pytest.fixture
def index(tmp_path):
    proc, idx = tmp_path / "proc", tmp_path / "index"
    proc.mkdir()
    (proc / "chunks_256.jsonl").write_text("".join(json.dumps(c) + "\n" for c in CHUNKS), encoding="utf-8")
    ib.build(256, FakeModel(), proc, idx)
    return Index(256, FakeModel(), idx, proc)


def test_each_paper_question_gets_its_own_paper(index):
    setting = retrieval_settings()["B"]
    for question, paper in [("attention heads", "AIAYN"), ("mimi codec", "MOSHI"), ("chunking boundary", "HNET")]:
        ev = search(question, setting, 2500, index, FakeCrossEncoder())
        assert ev.passages[0]["paper"] == paper


def test_done_check_reports_gold_coverage(index):
    questions = [
        {"question_id": "t1", "answerable": True, "question": "how many attention heads",
         "evidence": [{"section": "AIAYN §3.2.2", "quote": "parallel attention heads"}]},
        {"question_id": "t2", "answerable": False, "question": "what GPU trained moshi", "evidence": []},
    ]
    report = rr.evaluate(questions, "B", index, 2500, FakeCrossEncoder())
    assert report["answerable"] == 1 and report["any_gold_retrieved_pct"] == 100.0
    assert report["rows"][0]["first_gold_rank"] == 1
    assert [q for q, *_ in report["unanswerable_top_scores"]] == ["t2"]


REAL = Path("data/index/faiss_256.index")


@pytest.mark.skipif(not REAL.exists(), reason="real indexes not built")
def test_real_indexes_load_and_match_chunks():
    """Index files must match the current chunk files (catches a corpus rebuilt without re-indexing)."""
    for size in (256, 512):
        Index(size, embedder=None)
