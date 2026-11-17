"""Phase 11B REAL-PAPER OFFLINE end-to-end checks of Configuration G on the PRODUCTION corpus c3 (data/processed,
data/index): real embedder, cross-encoder and NLI (verifier v4.2, threshold 0.9 from config); the model is a
scripted fake (no API). Retrieval uses exactly the production call: search(..., **policy_flags(app_config()))."""
import copy
import json
import sys

import pytest

sys.path.insert(0, ".")
from relay.config import app_config, retrieval_settings  # noqa: E402
from relay.eval.retrieval_recall import gold_chunks, load_questions  # noqa: E402
from relay.pipeline import Verifier, answer_from_evidence  # noqa: E402
from relay.retrieval.search import Index, policy_flags, search  # noqa: E402
from relay.verify.verifier import verify  # noqa: E402
from tests.unit.test_pipeline import FakeLLM  # noqa: E402

CFG = app_config()
B = retrieval_settings()["B"]
CAP = CFG["pressure"]["L0"]["context_cap"]
Q = {q["question_id"]: q for q in load_questions(None)}
NEW = {q["question_id"]: q for q in json.load(open("experiments/phase11/new_cases.json", encoding="utf-8"))["questions"]}


@pytest.fixture(scope="module")
def env():
    from relay.eval.verifier_v2_check import CachedNLI
    from relay.retrieval.index_build import load_model
    from relay.retrieval.rerank import load_reranker
    from relay.verify import nli
    index = Index(256, load_model())
    v = CFG["verification"]
    return index, load_reranker(), Verifier(CachedNLI(nli.load_nli(v["nli_model"])), v["entail_min"], v["entail_min_flagged"])


def run(env, question, g=True):
    index, rr, _ = env
    return search(question, B, CAP, index, rr, **(policy_flags(CFG) if g else {}))


def ids(ev):
    return [p["chunk_id"] for p in ev.passages]


def test_production_config_has_g_on():
    assert policy_flags(CFG) == {"expand": True, "share_captions": True}


def test_q11_h1_row_is_retrieved_and_base_row_stays_unresolved(env):
    ev = run(env, Q["q11"]["question"])
    assert ev.expansion["terms"] == ["h = 1"] and ev.expansion["original"] == Q["q11"]["question"]
    assert "AIAYN:T3:r02" in ids(ev)          # (A) rows incl. h = 1: BLEU 24.9, PPL 5.29
    assert "AIAYN:T3:r01" not in ids(ev)      # base row (BLEU 25.8) NOT retrieved: q11 is NOT fully fixed
    r02 = next(p for p in ev.passages if p["chunk_id"] == "AIAYN:T3:r02")
    assert "Configuration (A): h = 1; d_k = 512; d_v = 512; PPL / (dev) = 5.29; BLEU / (dev) = 24.9" in r02["text"]
    # a draft that cites the base row cannot pass: that chunk is not in the evidence
    _, _, ver = env
    rep = verify("The base configuration reaches 25.8 BLEU [AIAYN:T3:r01].", ev.passages, {"AIAYN:T3:r01"} | set(ids(ev)),
                 ver.nli_model, ver.entail_min, ver.entail_min_flagged)
    assert not rep.ok and "not retrieved" in rep.failures()[0][1]


@pytest.mark.parametrize("qid", ["q02", "q09", "q12"])
def test_gold_evidence_is_kept_or_improved(env, qid):
    index = env[0]
    gold = gold_chunks(Q[qid], index.chunks)
    b, g = set(ids(run(env, Q[qid]["question"], g=False))), set(ids(run(env, Q[qid]["question"])))
    assert sum(bool(s & g) for s in gold) >= sum(bool(s & b) for s in gold)


def test_q12_keeps_both_table8_rows(env):
    got = ids(run(env, Q["q12"]["question"]))
    assert {"MOSHI:T8:r01", "MOSHI:T8:r02"} <= set(got)


def test_q09_unit_bearing_evidence_in_c3(env):
    """c3 lost the siunitx units ("the 230 average", fixed only in c6); the plain-text units that c3 does have
    (abstract: "160ms, 200ms in practice") must still reach the context."""
    ev = run(env, Q["q09"]["question"])
    assert any("200ms" in p["text"] for p in ev.passages)


@pytest.mark.parametrize("qid,draft", [
    ("q20", "The Transformer scores 86.4 on MMLU, below GPT-4 [{cid}]."),
    ("n11", "The variant with 64 attention heads reaches 25.8 development BLEU [{cid}]."),
    ("n12", "The variant with 12 identical layers reaches 27.1 test BLEU [{cid}]."),
])
def test_unanswerable_questions_stay_unanswerable(env, qid, draft):
    question = (Q.get(qid) or NEW[qid])["question"]
    ev = run(env, question)
    _, _, ver = env
    for cid in ids(ev):  # whichever retrieved chunk the fabricated claim cites, it is rejected
        rep = verify(draft.format(cid=cid), ev.passages, set(env[0].chunks), ver.nli_model, ver.entail_min,
                     ver.entail_min_flagged)
        assert not rep.ok, (qid, cid)


def test_invalid_citation_and_false_absence_still_rejected(env):
    ev = run(env, Q["q12"]["question"])
    _, _, ver = env
    corpus = set(env[0].chunks)
    bad = verify("Moshi reaches 26.6 on Web Questions [MOSHI:T8:r99].", ev.passages, corpus, ver.nli_model, 0.9, 0.9)
    assert not bad.ok
    absent = verify("Moshi without Inner Monologue gets 9.2 on Web Questions [MOSHI:T8:r02]. The retrieved passages do "
                    "not state the spoken question answering accuracy of Moshi with Inner Monologue.",
                    ev.passages, corpus, ver.nli_model, 0.9, 0.9)
    assert not absent.ok and any("absence contradicted" in w for _, w in absent.failures())


def test_redraft_and_verification_use_the_original_complete_evidence(env):
    ev = run(env, Q["q11"]["question"])
    before = copy.deepcopy(ev.passages)
    seen = []

    def spy(draft, passages, *a, **k):
        seen.append(copy.deepcopy(passages))
        return verify(draft, passages, *a, **k)
    llm = FakeLLM("Single-head attention reaches 24.9 BLEU [AIAYN:9.9:c999].", "NOT_IN_PAPERS")
    r = answer_from_evidence(Q["q11"]["question"], ev, llm, env[2], set(env[0].chunks), verify_fn=spy)
    assert r.status == "refused" and len(seen) == 1 and seen[0] == before and ev.passages == before
    for p in seen[0]:  # every table chunk the verifier saw still starts with its caption
        if p.get("kind") == "table":
            assert p["text"].startswith("Table ")
    shown = llm.calls[0][1]["content"]
    assert Q["q11"]["question"] in shown and "[h = 1]" not in shown  # the model is asked the original question
