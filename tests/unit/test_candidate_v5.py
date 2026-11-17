"""Candidate verifier v5 (P1 full-width citations, P6 table-backed absence). v4 is untouched; no models loaded."""
import pytest

from relay.candidate_v5.verifier import normalize_citations, table_absence_contradicted, verify
from relay.verify.citations import citations_in
from relay.verify.verifier import verify as verify_v4
from tests.unit.test_verifier import CORPUS, P, FakeNLI

SUPPORTED = FakeNLI({"Mimi uses 8 quantizers.": 0.95})
ABSENT = "The retrieved passages do not state "


def run(answer, nli=SUPPORTED, **kw):
    return verify(answer, P, CORPUS, nli, 0.5, 0.8, **kw)


# ---------------------------------------------------------------- P1
@pytest.mark.parametrize("text, ids", [
    ("Mimi uses 8 quantizers 【MOSHI:3.3.1:c030】.", ["MOSHI:3.3.1:c030"]),                      # valid
    ("X 【MOSHI:3.3.1:c030; AIAYN:3.2.2:c014】.", ["MOSHI:3.3.1:c030", "AIAYN:3.2.2:c014"]),     # multiple in one
    ("X 【MOSHI:3.3.1:c030】【AIAYN:3.2.2:c014】.", ["MOSHI:3.3.1:c030", "AIAYN:3.2.2:c014"]),     # multiple
    ("X [MOSHI:3.3.1:c030]【AIAYN:3.2.2:c014】.", ["MOSHI:3.3.1:c030", "AIAYN:3.2.2:c014"]),      # mixed
    ("X 【source†L1-L3】.", []),                                                                 # not an ID
    ("X 【】.", []),                                                                            # empty
    ("X 【MOSHI:3.3.1】.", []),                                                                  # malformed ID
    ("X 【MOSHI:3.3.1:c030, see text】.", []),                                                   # ID plus prose
])
def test_p1_reads_only_full_width_brackets_made_of_chunk_ids(text, ids):
    assert citations_in(normalize_citations(text)) == ids


def test_p1_valid_citation_still_goes_through_every_v4_check():
    assert run("Mimi uses 8 quantizers 【MOSHI:3.3.1:c030】.").ok
    # phase 10 (L1): the base verifier is v4.1, which reads 【…】 itself, so P1 no longer changes this outcome
    assert run("Mimi uses 8 quantizers 【MOSHI:3.3.1:c030】.", p1=False).ok
    unsupported = run("Mimi uses 8 quantizers 【MOSHI:3.3.1:c030】.", nli=FakeNLI({}))
    assert not unsupported.ok and "not supported" in unsupported.sentences[0].reasons[0]  # NLI still decides
    unretrieved = run("Heads are 8 【AIAYN:3.2.2:c014】.")                               # in the corpus, not retrieved
    assert not unretrieved.ok and "not retrieved" in unretrieved.sentences[0].reasons[0]
    for bad in ("Mimi uses 8 quantizers 【】.", "Mimi uses 8 quantizers 【source†L1】.",
                "Mimi uses 8 quantizers 【MOSHI:3.3.1】."):
        assert not run(bad).ok and "no citation" in run(bad).sentences[0].reasons


def test_both_switches_off_is_exactly_v4():
    for text in ("Mimi uses 8 quantizers [MOSHI:3.3.1:c030].", "Mimi uses 8 quantizers 【MOSHI:3.3.1:c030】."):
        a, b = run(text, p1=False, p6=False), verify_v4(text, P, CORPUS, SUPPORTED, 0.5, 0.8)
        assert [(s.ok, s.reasons, s.entailment) for s in a.sentences] == [(s.ok, s.reasons, s.entailment) for s in b.sentences]


# ---------------------------------------------------------------- P6
def table(paper, tid, caption, rows):
    return {"chunk_id": f"{paper}:{tid}:r01", "paper": paper, "table_id": tid, "kind": "table",
            "section_id": tid, "section_title": "t", "text": f"Table {tid[1:]}: {caption}\nColumns: ...",
            "rows": [{"configuration": c, "values": [{"metric": m, "split": None, "column": m, "value": v}
                                                     for m, v in vals.items()]} for c, vals in rows]}


T8 = table("MOSHI", "T8", "Evaluation of spoken question answering.", [
    ("Moshi", {"Web Q.": "26.6", "LlaMA Q.": "62.3", "Audio Trivia QA": "22.8"}),
    ("Moshi (no Inner Monologue)", {"Web Q.": "9.2", "LlaMA Q.": "21.0", "Audio Trivia QA": "7.3"}),
    ("AudioLM", {"Web Q.": "2.3", "LlaMA Q.": "7.0", "Audio Trivia QA": "-"}),
    ("TWIST (7B)", {"Web Q.": "1.1", "LlaMA Q.": "0.5", "Audio Trivia QA": "3.0"}),
    ("Spectron", {"Web Q.": "6.1", "LlaMA Q.": "22.9"})])  # no Audio Trivia QA cell at all
HNET_T2 = table("HNET", "T2", "Zero-shot performance comparison across multiple benchmarks, all data-/FLOP-matched.", [
    ("H-Net (2-stage)", {"Average": "55.5"})])
Q12 = ABSENT + "the spoken question answering accuracy of Moshi with Inner Monologue."


def test_p6_catches_q12_exactly():
    hits = table_absence_contradicted(Q12, [T8])
    assert len(hits) == 1 and "row 'Moshi':" in hits[0]  # the with-Inner-Monologue row only, not its sibling
    assert all(v in hits[0] for v in ("26.6", "62.3", "22.8"))  # every requested cell shown


def test_p6_tells_the_negated_variant_apart():
    hits = table_absence_contradicted(ABSENT + "the spoken question answering accuracy of Moshi without Inner Monologue.", [T8])
    assert len(hits) == 1 and "row 'Moshi (no Inner Monologue)'" in hits[0] and "9.2" in hits[0]


@pytest.mark.parametrize("sentence", [
    ABSENT + "the spoken question answering accuracy of AudioLM.",           # no column named, one cell is "-"
    ABSENT + "the spoken question answering accuracy of Spectron.",          # no column named, one cell missing
    ABSENT + "the spoken question answering accuracy of Spectron on Audio Trivia QA.",  # requested cell missing
    ABSENT + "the spoken question answering accuracy of TWIST.",             # row qualifier (7B) not named
])
def test_p6_needs_the_exact_requested_cells(sentence):
    assert table_absence_contradicted(sentence, [T8]) == []


@pytest.mark.parametrize("sentence, passages", [
    (ABSENT + "the number of H100 GPUs used to train Moshi.", [T8]),                          # really missing
    (ABSENT + "the spoken question answering accuracy of GPT-4.", [T8]),                     # caption, no row
    (ABSENT + "the spoken question answering accuracy of AudioLM on Audio Trivia QA.", [T8]),  # row, value "-"
    (ABSENT + "the spoken question answering accuracy of Moshi on Natural Questions.", [T8]),  # ambiguous
    (Q12, [HNET_T2]),                                                                        # unrelated table
    (ABSENT + "the zero-shot accuracy of Moshi.", [T8, HNET_T2]),                             # unrelated topic
])
def test_p6_keeps_legitimate_and_ambiguous_absence_claims(sentence, passages):
    assert table_absence_contradicted(sentence, passages) == []


def test_p6_rejects_when_the_requested_value_is_in_the_table():
    hits = table_absence_contradicted(ABSENT + "the spoken question answering accuracy of AudioLM on Web Q.", [T8])
    assert hits and "Web Q. = 2.3" in hits[0]


def test_p6_only_rejects_and_only_absence_sentences():
    answer = "Mimi uses 8 quantizers [MOSHI:3.3.1:c030]. " + Q12
    passages = P + [T8]
    ids = CORPUS | {T8["chunk_id"]}
    v4 = verify_v4(answer, passages, ids, SUPPORTED, 0.5, 0.8)
    v5 = verify(answer, passages, ids, SUPPORTED, 0.5, 0.8)
    # phase 10C: the base verifier is v4.2, which includes P6 itself, so it rejects q12's pattern too (v4 accepted it)
    assert not v4.ok and not v5.ok
    assert v5.sentences[0].ok and not v5.sentences[1].ok              # the cited claim is untouched
    assert v4.sentences[0].ok and not v4.sentences[1].ok
    assert not verify(answer, passages, ids, SUPPORTED, 0.5, 0.8, p6=False).ok
