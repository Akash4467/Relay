"""Verifier v4.2 absence claims (phase 10C). Two kinds of test, labelled:
  SYNTHETIC  hand-written passages (tests/unit/test_verifier.py fixtures);
  REAL-PAPER the c3 chunks the live q12 run actually retrieved (data/processed/chunks_256.jsonl, read only).
No model is loaded: absence checks are deterministic; the one cited fact uses the FakeNLI table."""
import json
from pathlib import Path

import pytest

from relay.pipeline import answer_from_evidence
from relay.verify import absence
from relay.verify.verifier import verify
from tests.unit.test_pipeline import FakeLLM, ev, verifier
from tests.unit.test_verifier import CORPUS, P, FakeNLI

CHUNKS = {c["chunk_id"]: c for c in map(json.loads, Path("data/processed/chunks_256.jsonl").open(encoding="utf-8"))}
Q12_IDS = ["MOSHI:T8:r02", "MOSHI:5.5:c090", "MOSHI:5.4:c089", "MOSHI:5.5:c091", "MOSHI:5.5:c092", "MOSHI:T8:r01",
           "MOSHI:4.3:c061", "MOSHI:1:c008"]  # retrieved for q12 in the phase 10 live run
Q12_EV = [CHUNKS[i] for i in Q12_IDS]
IDS = set(CHUNKS)
FACT = "Mimi uses 8 quantizers [MOSHI:3.3.1:c030]."
NLI = FakeNLI({"Mimi uses 8 quantizers.": 0.95})


def check(answer, passages):
    return verify(answer, passages, IDS | CORPUS, NLI, 0.9, 0.9)


def reasons(report, i=0):
    return " | ".join(report.sentences[i].reasons)


# ---- SYNTHETIC: recognising absence claims and their scope
@pytest.mark.parametrize("claim,scope", [
    ("The retrieved passages do not state the bitrate.", "retrieved"),
    ("The retrieved passages do not provide the bitrate.", "retrieved"),
    ("The passages don't mention the bitrate.", "retrieved"),
    ("No information is available in the retrieved passages about the bitrate.", "retrieved"),
    ("The bitrate is not reported in the provided passages.", "retrieved"),
    ("The paper does not specify the bitrate.", "paper"),
    ("The authors don't mention the bitrate.", "paper"),
    ("The value is not reported.", "unscoped"),
    ("No information is available.", "unscoped"),
    ("Mimi runs at 1.1 kbps.", None),
])
def test_scope_classification(claim, scope):
    assert absence.classify(claim, cited=False) == scope


def test_cited_negative_fact_without_scope_is_an_ordinary_claim():
    assert absence.classify("There is no recurrence in the Transformer.", cited=True) is None
    assert absence.classify("The bitrate is not reported.", cited=True) is None  # NLI decides, as before


@pytest.mark.parametrize("claim", ["The paper does not specify the optimizer.", "The value is not reported.",
                                   "The authors do not provide the training cost."])
def test_whole_paper_or_unscoped_absence_is_rejected(claim):
    r = check(claim, P)
    assert not r.ok and "whole-paper absence cannot be established" in reasons(r)


def test_missing_part_of_paraphrases():
    assert absence.missing_part("The retrieved passages do not provide the exact accuracies of Moshi.") == \
        "the exact accuracies of Moshi"
    assert absence.missing_part("The bitrate of Mimi is not reported in the retrieved passages.") == "The bitrate of Mimi"


def test_genuinely_absent_value_is_accepted():  # SYNTHETIC: nothing about GPUs in the passage
    for wording in ("The retrieved passages do not state the number of training GPUs.",
                    "The retrieved passages do not provide the number of training GPUs.",
                    "The passages don't mention the training GPU count."):
        r = check(FACT + " " + wording, P)
        assert r.ok, (wording, reasons(r, 1))


def test_value_present_in_another_retrieved_passage_is_rejected():  # SYNTHETIC
    for wording in ("The retrieved passages do not state the codebook size of the quantizers.",
                    "The retrieved passages do not provide the codebook size of the quantizers.",
                    "The passages don't mention the quantizers' codebook size."):
        r = check(FACT + " " + wording, P)
        assert not r.ok and "absence contradicted" in reasons(r, 1), wording


@pytest.mark.parametrize("wording,ok", [
    ("The retrieved passages do not state this.", False),           # nothing named: cannot be checked
    ("The retrieved passages do not state the codebook.", False),   # one term the passage mentions
    ("The retrieved passages do not state the optimizer.", True),   # one term no passage mentions
])
def test_vague_absence_fails_closed(wording, ok):
    assert check(FACT + " " + wording, P).ok is ok


# ---- REAL-PAPER: the q12 evidence from the live run (c3 chunks)
Q12_RUN1 = "The retrieved passages do not state the spoken question answering accuracy of Moshi with Inner Monologue."
Q12_RUN2 = "The retrieved passages do not state the accuracies of Moshi with Inner Monologue."
Q12_REDRAFT = "The retrieved passages do not provide the exact accuracies of Moshi with Inner Monologue."


def test_q12_run1_false_absence_is_now_rejected_by_the_table():
    r = check(Q12_RUN1, Q12_EV)
    assert not r.ok and "MOSHI:T8 row 'Moshi'" in reasons(r) and "26.6" in reasons(r) and "62.3" in reasons(r)


@pytest.mark.parametrize("claim", [Q12_RUN2, Q12_REDRAFT, "The passages don't mention the accuracy of Moshi with Inner Monologue.",
                                   "No information about the accuracy of Moshi with Inner Monologue is given in the retrieved passages."])
def test_q12_paraphrases_are_absence_claims_and_rejected(claim):
    r = check(claim, Q12_EV)
    assert r.sentences[0].absence and not r.ok and "absence contradicted" in reasons(r), claim


def test_q12_value_in_the_paper_but_not_retrieved():
    """Without Table 8's rows and the 5.5 prose, a scoped claim is true of the retrieved passages (accepted);
    the same claim about the paper is never accepted from a limited retrieval."""
    partial = [CHUNKS[i] for i in ("MOSHI:4.3:c061", "MOSHI:1:c008")]
    assert check(FACT + " " + Q12_REDRAFT, P + partial).ok  # (an answer of absence sentences alone is a refusal)
    r = check("The paper does not provide the accuracy of Moshi with Inner Monologue.", partial)
    assert not r.ok and "whole-paper absence" in reasons(r)


def test_mixed_answer_supported_fact_and_false_absence():
    passages = P + Q12_EV
    r = check(FACT + " " + Q12_RUN1, passages)
    assert not r.ok and r.sentences[0].ok and "absence contradicted" in reasons(r, 1)


def test_redraft_that_changes_the_wording_is_still_caught():
    """Pipeline: draft 1 uses the canonical phrase, the redraft paraphrases it; both are false absence claims."""
    llm = FakeLLM(FACT + " " + Q12_RUN2, FACT + " " + Q12_REDRAFT)
    r = answer_from_evidence("Spoken QA accuracy of Moshi with and without Inner Monologue?", ev(P + Q12_EV), llm,
                             verifier({"Mimi uses 8 quantizers.": 0.95}), IDS | CORPUS)
    assert r.status == "refused" and r.llm_calls == 2
    assert all(any("absence contradicted" in why for _, why in rep.failures()) for rep in r.reports)


def test_redraft_that_answers_instead_is_accepted():
    llm = FakeLLM(FACT + " " + Q12_RUN2, FACT)
    r = answer_from_evidence("How many quantizers?", ev(P + Q12_EV), llm, verifier({"Mimi uses 8 quantizers.": 0.95}),
                             IDS | CORPUS)
    assert (r.status, r.text, r.llm_calls) == ("answered", FACT, 2)
