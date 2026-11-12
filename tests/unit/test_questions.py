"""The eval set is data the grade depends on: check its shape and that every gold quote exists in the corpus."""
import collections
import json
import re
from pathlib import Path

import pytest

QUESTIONS = [json.loads(l) for l in Path("eval/questions.jsonl").read_text(encoding="utf-8").splitlines()]
CATEGORIES = {"direct": 6, "multi_section": 4, "numerical_table": 3, "cross_paper": 2, "unanswerable": 15}


def test_ids_fields_and_distribution():
    assert [q["question_id"] for q in QUESTIONS] == [f"q{i:02d}" for i in range(1, 31)]
    for q in QUESTIONS:
        assert {"question", "expected_answer", "paper_id", "evidence", "answerable", "split", "category"} <= set(q)
        assert set(q["paper_id"].split("|")) <= {"AIAYN", "MOSHI", "HNET"}
        assert q["answerable"] == (q["category"] != "unanswerable")
        assert bool(q["evidence"]) == q["answerable"]
        if not q["answerable"]:
            assert q["refusal_type"] in {"near_miss", "misleading", "out_of_scope"}
    assert collections.Counter(q["category"] for q in QUESTIONS) == CATEGORIES


def test_dev_test_split_is_15_15_and_stratified():
    split = collections.Counter((q["category"], q["split"]) for q in QUESTIONS)
    assert sum(v for (c, s), v in split.items() if s == "dev") == 15
    for cat, n in CATEGORIES.items():
        dev = split[(cat, "dev")]
        assert abs(dev - (n - dev)) <= 1, cat


def _chunks(size: int) -> list[tuple[str, str]]:
    """(paper, whitespace-normalized text) per chunk."""
    path = Path(f"data/processed/chunks_{size}.jsonl")
    if not path.exists():
        pytest.skip("corpus not built")
    rows = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines()]
    return [(r["paper"], " ".join(r["text"].split())) for r in rows]


@pytest.mark.parametrize("size", [256, 512])
def test_every_gold_quote_is_in_a_chunk_of_its_paper(size):
    chunks = _chunks(size)
    for q in QUESTIONS:
        for ev in q["evidence"]:
            quote, paper = " ".join(ev["quote"].split()), ev["section"].split()[0]
            assert any(p == paper and quote in t for p, t in chunks), (q["question_id"], quote)


@pytest.mark.parametrize("size", [256, 512])
def test_near_miss_terms_really_absent_from_the_target_paper(size):
    chunks = _chunks(size)
    for q in QUESTIONS:
        corpus = "\n".join(t for p, t in chunks if p in q["paper_id"].split("|"))
        for term in q.get("must_not_appear", []):
            assert not re.search(rf"\b{re.escape(term)}\b", corpus, re.I), (q["question_id"], term)
