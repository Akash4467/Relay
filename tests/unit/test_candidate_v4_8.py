import json
from pathlib import Path

import pytest

from relay.candidate_v4_8 import absence_labels, marks
from relay.verify.citations import SentenceCheck, strip_citations
from relay.verify.verifier import Report

C6 = Path("experiments/phase10/corpus/c6/processed/chunks_256.jsonl")
C3 = Path("data/processed/chunks_256.jsonl")
T = {q["question_id"]: q["question"] for q in
     json.loads(Path("eval/phase7/table_question_set.json").read_text(encoding="utf-8"))["questions"]}


@pytest.fixture(scope="module")
def C():
    if not C6.exists():
        pytest.skip("c6 not built")
    return {json.loads(l)["chunk_id"]: json.loads(l) for l in C6.open(encoding="utf-8")}


def rep(sentence, cites, absence=False):
    return Report([SentenceCheck(sentence, cites, absence=absence)])


def fix1(question, sentence, cid, chunks):
    r = rep(sentence, [cid])
    marks.check(question, r, [chunks[cid]], strip_citations)
    return r.ok


def row(chunks, cid, label):
    return next(r for r in chunks[cid]["rows"] if r["configuration"] == label)


# ---------------------------------------------------------------- fix 1: parsing
def test_conditions_are_parsed_conservatively():
    assert marks.condition(T["t29"]) == {"all": False}
    assert marks.condition(T["t05"]) == {"all": True}
    assert marks.condition(T["t13"])["cols"] == [("RQ-Transformer", True)]
    assert marks.condition(T["t14"])["cols"] == [("RQ-Transformer", False)]
    assert marks.condition(T["t30"]) == {"cols": [("Transformer in decoder", True)], "only": True}
    assert marks.condition(T["t15"]) is None and marks.condition(T["t23"]) is None  # comparisons: not enforced
    assert marks.condition(T["t01"]) is None  # asks ABOUT a mark, does not select a row by it
    assert marks.condition("How many attention heads does the base Transformer use?") is None


# ---------------------------------------------------------------- fix 1: row conditions on real c6 rows
def test_no_marks_condition(C):
    assert fix1(T["t29"], "Row 1 reports ABX (↓) = 31.3% [MOSHI:T17:r01].", "MOSHI:T17:r01", C)          # all blank
    assert not fix1(T["t29"], "Configuration Row 5 reports ABX (↓) = 29.1% [MOSHI:T17:r03].", "MOSHI:T17:r03", C)
    assert not fix1(T["t29"], "The configuration with no marks reports an ABX of 31.4% [MOSHI:T17:r01].",
                    "MOSHI:T17:r01", C)  # Row 2 (encoder checked): the condition dropped or substituted


def test_all_checked_condition(C):
    assert fix1(T["t05"], "The fully checked configuration has a MUSHRA of 64.0±1.7 [MOSHI:T3:r03].", "MOSHI:T3:r03", C)
    assert not fix1(T["t05"], "It has a MUSHRA of 68.3±1.7 [MOSHI:T3:r03].", "MOSHI:T3:r03", C)  # Row 5: Quantization blank


def test_named_column_conditions(C):
    assert fix1(T["t13"], "The row reports a perplexity of 40.3 [MOSHI:T5:r01].", "MOSHI:T5:r01", C)       # RQ ✓
    assert not fix1(T["t13"], "The row reports a perplexity of 42.2 [MOSHI:T5:r01].", "MOSHI:T5:r01", C)   # RQ blank
    assert fix1(T["t14"], "The row reports a perplexity of 42.2 [MOSHI:T5:r01].", "MOSHI:T5:r01", C)
    assert not fix1(T["t14"], "The row reports a perplexity of 40.3 [MOSHI:T5:r01].", "MOSHI:T5:r01", C)


def test_only_and_combination_conditions_on_cells(C):
    r1, r2 = row(C, "MOSHI:T17:r01", "Row 1"), row(C, "MOSHI:T17:r01", "Row 2")  # Row 2: only encoder checked
    only_enc = {"cols": [("Transformer in encoder", True)], "only": True}
    assert marks._satisfies(r2, only_enc) is True and marks._satisfies(r1, only_enc) is False
    combo = {"cols": [("Transformer in encoder", True), ("WavLM distillation and Split quantizer", False)], "only": False}
    assert marks._satisfies(r2, combo) is True and marks._satisfies(r1, combo) is False
    assert marks._satisfies(r2, {"cols": [("no such column", True)], "only": False}) is None  # unknown: not judged


def test_c3_rows_without_marks_cannot_satisfy_a_mark_condition():
    if not C3.exists():
        pytest.skip("c3 not built")
    c3 = {json.loads(l)["chunk_id"]: json.loads(l) for l in C3.open(encoding="utf-8")}
    assert not fix1(T["t05"], "It obtains a MUSHRA score of 65.9±1.7 [MOSHI:T3:r01].", "MOSHI:T3:r01", c3)


def test_sentence_stating_no_row_value_and_absence_sentences_are_not_judged(C):
    assert fix1(T["t29"], "Table 17 is an ablation of the Mimi codec [MOSHI:T17:r03].", "MOSHI:T17:r03", C)
    r = rep("The retrieved passages do not state that value.", [], absence=True)
    marks.check(T["t29"], r, [C["MOSHI:T17:r03"]], strip_citations)
    assert r.sentences[0].ok  # (an all-absence Report is a refusal by design, so check the sentence)


# ---------------------------------------------------------------- fix 2
def fix2(sentence, cids, chunks):
    r = rep(sentence, [], absence=True)
    absence_labels.check(r, [chunks[c] for c in cids])
    return r.sentences[0].ok  # the sentence's own verdict (an all-absence Report is a refusal by design)


def test_q13_false_absence_is_rejected(C):
    assert not fix2("The retrieved passages do not state the average zero-shot accuracy of the BPE Transformer.",
                    ["HNET:T2:r10", "HNET:T2:r01", "HNET:T2:r11"], C)


def test_legitimate_absence_claims_pass(C):
    assert fix2("The retrieved passages do not state the frame rate.", ["MOSHI:T17:r03", "MOSHI:T3:r03"], C)
    # base row NOT retrieved: the claim is legitimate
    assert fix2("The retrieved passages do not state the development BLEU for the base configuration.",
                ["AIAYN:T3:r02"], C)
    # only the "no Inner Monologue" row retrieved; the claim is about the WITH variant
    assert fix2("The retrieved passages do not state the accuracy of Moshi with Inner Monologue.", ["MOSHI:T8:r02"], C)
    # a label that is only a generic word + number
    assert fix2("The retrieved passages do not state the ABX error of Row 3.", ["MOSHI:T17:r03"], C)
    # subject matches but no metric of that row is named
    assert fix2("The retrieved passages do not state how the Transformer was trained.", ["HNET:T2:r01"], C)


def test_base_row_in_context_makes_the_absence_false(C):
    assert not fix2("The retrieved passages do not state the development BLEU for the base configuration.",
                    ["AIAYN:T3:r01", "AIAYN:T3:r02"], C)


def test_sibling_qualifier_is_respected(C):
    # both Moshi rows retrieved: a claim about "with Inner Monologue" is NOT matched to the plain "Moshi" row
    # (conservative: a known false negative; v4.2's own absence checks still apply)
    assert fix2("The retrieved passages do not state the accuracy of Moshi with Inner Monologue.",
                ["MOSHI:T8:r01", "MOSHI:T8:r02"], C)
    # the negated variant itself is matched when the claim carries the negation
    assert not fix2("The retrieved passages do not state the Web Q. accuracy of Moshi without Inner Monologue (no Inner Monologue).",
                    ["MOSHI:T8:r02"], C)
