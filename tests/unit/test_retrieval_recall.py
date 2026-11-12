import pytest

from relay.eval import retrieval_recall as rr

CHUNKS = {
    "AIAYN:1:c001": {"paper": "AIAYN", "text": "We use eight\nheads here."},
    "AIAYN:2:c002": {"paper": "AIAYN", "text": "Unrelated."},
    "MOSHI:1:c003": {"paper": "MOSHI", "text": "We use eight heads here."},
}


def test_gold_chunks_resolve_quotes_within_the_named_paper_and_ignore_line_breaks():
    q = {"question_id": "q", "evidence": [{"section": "AIAYN §1", "quote": "We use eight heads"}]}
    assert rr.gold_chunks(q, CHUNKS) == [{"AIAYN:1:c001"}]


def test_missing_quote_is_an_error():
    q = {"question_id": "q", "evidence": [{"section": "HNET §1", "quote": "We use eight heads"}]}
    with pytest.raises(ValueError):
        rr.gold_chunks(q, CHUNKS)


def test_frozen_test_report_is_never_overwritten(tmp_path, monkeypatch):
    monkeypatch.setattr(rr, "REPORTS", tmp_path)
    (tmp_path / "retrieval_test_frozen.json").write_text("{}")
    with pytest.raises(FileExistsError):
        rr.main("test", out="retrieval_test_frozen.json")


def test_snapshot_records_code_config_and_questions():
    meta = rr.snapshot("dev", ("A", "C"))
    assert len(meta["git_commit"]) == 40 and isinstance(meta["git_dirty"], bool)
    assert set(meta["settings"]) == {"A", "C"} and meta["settings"]["C"]["piece_tokens"] == 512
    assert len(meta["questions_sha256"]) == 64 and meta["retrieval_policy"]["cross_paper_reserve_per_paper"] == 2


def test_load_questions_filters_split():
    assert {q["split"] for q in rr.load_questions("dev")} == {"dev"}
    assert len(rr.load_questions(None)) == 30
