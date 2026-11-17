from relay.candidate_r3 import search as r3

Q14 = "How is the input segmented in the original Transformer's translation experiments compared with H-Net?"


def test_terms_are_scoped_to_the_paper_that_owns_them():
    assert r3.scoped_terms(Q14, "AIAYN") == ["byte-pair encoding", "word-piece vocabulary"]
    assert r3.scoped_terms(Q14, "HNET") == []
    assert r3.scoped_terms("Which NVIDIA GPUs were used to train the original Transformer and Moshi?", "AIAYN") == []


class FakeIndex:
    """Two papers; records the query each slot was retrieved with."""
    def __init__(self):
        self.queries = {}
        self.chunks = {f"{p}:1:c00{i}": {"chunk_id": f"{p}:1:c00{i}", "paper": p, "section_id": "1", "section_title": "s",
                                         "kind": "text", "text": f"{p} text {i}", "n_tokens": 50}
                       for p in ("AIAYN", "HNET") for i in range(3)}

    def embed_query(self, q):
        return q

    def fused(self, q, qv, k, paper):
        self.queries[paper] = q
        ids = [c for c in self.chunks if c.startswith(paper)][:k]
        return [dict(self.chunks[c], score_rrf=1.0 / (i + 1), dense_cos=0.5) for i, c in enumerate(ids)]


class Setting:
    candidates, rerank, final_passages, context_budget = 6, False, 4, 1600


def test_each_slot_gets_only_its_own_terms_and_best_fused_first():
    idx = FakeIndex()
    ev = r3.search(Q14, Setting(), 2500, idx)
    assert idx.queries["AIAYN"].endswith("byte-pair encoding word-piece vocabulary")
    assert idx.queries["HNET"] == Q14  # the H-Net slot sees the original question only
    assert [p["chunk_id"] for p in ev.passages][0] == "AIAYN:1:c000"  # only the expanded slot is protected
    assert ev.term_expansion_by_paper == {"AIAYN": ["byte-pair encoding", "word-piece vocabulary"], "HNET": []}


def test_single_paper_question_is_r2():
    idx = FakeIndex()
    assert r3.scoped_terms("What limitation of tokenization motivates the Dynamic Chunking paper?", "HNET") == []


def test_cross_paper_question_without_expansion_is_not_protected():
    idx = FakeIndex()
    ev = r3.search("Which GPUs trained the original Transformer and H-Net?", Setting(), 2500, idx)
    assert ev.term_expansion == [] and idx.queries["AIAYN"] == idx.queries["HNET"]
