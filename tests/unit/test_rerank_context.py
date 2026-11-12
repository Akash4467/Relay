from relay.llm.tokenizer import count
from relay.retrieval import context, rerank


class FakeCrossEncoder:
    """Scores a pair by how many question words appear in the passage."""

    def predict(self, pairs):
        return [float(sum(w in p.lower() for w in q.lower().split())) for q, p in pairs]


def p(cid, text, paper="AIAYN", sid="1", title="Intro"):
    return {"chunk_id": cid, "paper": paper, "section_id": sid, "section_title": title, "text": text}


def test_rerank_sorts_by_cross_encoder_and_records_score():
    passages = [p("a", "nothing here"), p("b", "attention heads"), p("c", "attention")]
    out = rerank.rerank("attention heads", passages, FakeCrossEncoder())
    assert [x["chunk_id"] for x in out] == ["b", "c", "a"]
    assert [x["score_ce"] for x in out] == [2.0, 1.0, 0.0]
    assert rerank.rerank("q", [], FakeCrossEncoder()) == []


def test_passage_block_matches_prompt_format():
    assert context.passage_block(p("MOSHI:3.3:c010", "Mimi.", "MOSHI", "3.3", "Audio Tokenization")) == \
        "[MOSHI:3.3:c010] (MOSHI §3.3 Audio Tokenization)\nMimi."


def test_fill_keeps_rank_order_skips_what_does_not_fit_never_cuts():
    small, big = p("s1", "short text"), p("big", "word " * 300)
    cap = count(context.passage_block(small)) * 2 + 10
    kept, used = context.fill([small, big, p("s2", "short text")], cap)
    assert [k["chunk_id"] for k in kept] == ["s1", "s2"]  # big skipped, smaller lower-ranked one fills the gap
    assert used <= cap
    assert kept[0]["text"] == "short text"


def test_fill_zero_cap_keeps_nothing():
    assert context.fill([p("a", "x")], 0) == ([], 0)
