"""Phase 10C, REAL-PAPER OFFLINE q11 retrieval regressions (real indexes, embedder and cross-encoder; no API).
Records the diagnosed failure so a future retrieval change is measured against it."""
import json
import sys

import pytest

sys.path.insert(0, "experiments/phase10")
import q11_retrieval as q  # noqa: E402
from relay.config import retrieval_settings  # noqa: E402
from relay.retrieval import search as S  # noqa: E402

GOLD = list(q.GOLD)  # AIAYN:T3:r01 (base), AIAYN:T3:r02 ((A) rows incl. h = 1)


@pytest.fixture(scope="module")
def env():
    from relay.retrieval.index_build import load_model
    from relay.retrieval.rerank import load_reranker
    model, reranker = load_model(), load_reranker()
    idx = {c: S.Index(256, model, *q.CORPORA[c]) for c in ("c3", "c6")}
    return idx, reranker, retrieval_settings()


def final(ev):
    return [p["chunk_id"] for p in ev.passages]


@pytest.mark.parametrize("corpus", ["c3", "c6"])
def test_original_q11_failure_reproduces_under_B(env, corpus):
    idx, rr, st = env
    s = q.stages(q.Q11["question"], st["B"], idx[corpus], rr)
    assert not any(s["gold"][g]["in_context"] for g in GOLD)
    for g in GOLD:  # both are candidates; each is lost after fusion: reranked below the 8 final passages, or
        # (c6, r02 at rank 8) kept in the top 8 but dropped by the 1,600-token context budget
        assert s["gold"][g]["in_fused_top"] is not None
        assert s["gold"][g]["after_rerank"] > st["B"].final_passages or g in s["dropped_by_budget"]
    assert s["table3_in_context"]  # other Table 3 rows were substituted (wrong variations)


def test_without_the_reranker_c3_retrieves_both_rows(env):
    idx, rr, st = env
    assert set(GOLD) <= set(final(S.search(q.Q11["question"], st["A"], q.CAP, idx["c3"])))


def test_the_exact_rows_hold_the_comparison_with_labels(env):
    idx, _, _ = env
    rows = {r["configuration"] + ":" + next(v["value"] for v in r["values"] if v["metric"] == "h"):
            {v["column"]: v["value"] for v in r["values"]} for g in GOLD for r in idx["c6"].chunks[g]["rows"]}
    assert rows["base:8"]["BLEU / (dev)"] == "25.8" and rows["base:8"]["PPL / (dev)"] == "4.92"
    assert rows["(A):1"]["BLEU / (dev)"] == "24.9" and rows["(A):1"]["PPL / (dev)"] == "5.29"
    assert "(A):16" in rows and rows["(A):16"]["BLEU / (dev)"] != "24.9"  # sibling (A) rows are not the h = 1 row


def test_table_reference_candidates_do_not_change_q11(env):
    """The experimental rule adds every Table 3 chunk before reranking: no effect, because the gold rows were
    already candidates; the reranker is what drops them."""
    idx, rr, st = env
    ev, meta = q.search_tableref(q.Q11["question"], st["B"], idx["c6"], rr)
    assert meta["added"] and not set(GOLD) & set(final(ev))
    assert final(ev)[:3] == final(S.search(q.Q11["question"], st["B"], q.CAP, idx["c6"], rr))[:3]


def test_questions_without_a_table_reference_are_identical(env):
    idx, rr, st = env
    for l in list(open("eval/questions.jsonl", encoding="utf-8"))[:4]:
        question = json.loads(l)["question"]
        assert not q.TABLE_REF.search(question)
        assert final(q.search_tableref(question, st["B"], idx["c6"], rr)[0]) == \
            final(S.search(question, st["B"], q.CAP, idx["c6"], rr))
