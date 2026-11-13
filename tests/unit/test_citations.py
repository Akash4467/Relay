import pytest

from relay.verify import citations as cit

CORPUS = {"MOSHI:3.3.1:c030", "MOSHI:T4:r01", "AIAYN:3.2.2:c014", "HNET:2.2.1:c023"}
RETRIEVED = {"MOSHI:3.3.1:c030", "MOSHI:T4:r01"}


def reasons(answer):
    return [(c.sentence, c.reasons) for c in cit.check(answer, RETRIEVED, CORPUS)]


def test_citations_in_parses_single_multi_and_ignores_non_citation_brackets():
    s = "Delays [0, 1] give 160ms [MOSHI:3.3.1:c030; MOSHI:T4:r01] and [MOSHI:T4:r01]."
    assert cit.citations_in(s) == ["MOSHI:3.3.1:c030", "MOSHI:T4:r01"]
    assert cit.strip_citations(s) == "Delays [0, 1] give 160ms and."


def test_valid_answer_passes():
    out = cit.check("Mimi runs at 12.5Hz [MOSHI:3.3.1:c030]. Its bitrate is 1.1kbps [MOSHI:T4:r01].",
                    RETRIEVED, CORPUS)
    assert [c.ok for c in out] == [True, True]
    assert out[1].citations == ["MOSHI:T4:r01"]


def test_missing_citation_fails():
    assert reasons("Mimi runs at 12.5Hz.") == [("Mimi runs at 12.5Hz.", ["no citation"])]


def test_citation_not_retrieved_or_not_existing_fails():
    out = cit.check("It has 8 heads [AIAYN:3.2.2:c014]. It is fast [MOSHI:9:c999].", RETRIEVED, CORPUS)
    assert out[0].reasons == ["cites AIAYN:3.2.2:c014, which was not retrieved for this question"]
    assert out[1].reasons == ["cites MOSHI:9:c999, which does not exist"]


def test_scoped_absence_sentence_is_allowed_without_citation():
    out = cit.check("Mimi runs at 12.5Hz [MOSHI:3.3.1:c030]. The retrieved passages do not state the GPU count.",
                    RETRIEVED, CORPUS)
    assert [c.ok for c in out] == [True, True] and out[1].absence and not out[0].absence


@pytest.mark.parametrize("sentence", [
    # q11 regression: the model claimed the papers omit a value they do contain (the row was just not retrieved)
    "The papers do not state the development BLEU of the base configuration.",
    "The paper does not report the GPU count.",
    "The three papers do not mention VoiceBench.",
    "The authors do not specify the learning rate.",
])
def test_unscoped_claims_about_the_papers_are_rejected(sentence):
    (c,) = cit.check(sentence, RETRIEVED, CORPUS)
    assert not c.ok and not c.absence and "retrieved passages" in c.reasons[0]


def test_absence_sentence_cannot_cite_or_carry_a_value():
    (cited,) = cit.check("The retrieved passages do not state the GPU count [MOSHI:T4:r01].", RETRIEVED, CORPUS)
    assert not cited.ok and "must not cite" in cited.reasons[0]
    (valued,) = cit.check("The retrieved passages do not state the base BLEU, which is 25.8.", RETRIEVED, CORPUS)
    assert not valued.ok and "must not contain a value" in valued.reasons[0]


def test_citation_after_the_period_still_counts():
    out = cit.check("Mimi runs at 12.5Hz. [MOSHI:3.3.1:c030] Its bitrate is 1.1kbps. [MOSHI:T4:r01]",
                    RETRIEVED, CORPUS)
    assert [c.ok for c in out] == [True, True]
