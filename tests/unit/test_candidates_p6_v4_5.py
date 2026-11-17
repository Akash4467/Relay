import relay.pipeline as pipeline
from relay.candidate_p5r import prompts as p5r
from relay.candidate_p6 import prompts as p6
from relay.candidate_v4_5 import verifier as v45
from relay.llm import prompts as p4
from relay.verify import verifier as v42
from tests.unit.test_pipeline import CORPUS, GOOD, P, SUPPORTED, FakeLLM, ev, verifier

ROW_NO_MARKS = {"chunk_id": "MOSHI:T3:r01", "paper": "MOSHI", "section_id": "5.1", "section_title": "Text Language Modeling",
                "kind": "table", "text": "Table 3: Ablation of Mimi.\nConfiguration : MUSHRA = 65.9±1.7"}
ROW_MARKS = dict(ROW_NO_MARKS, text="Table 3: Ablation of Mimi.\nConfiguration ✓ ✓ ✓ ✓ ✓: MUSHRA = 64.0±1.7")
MARK_CLAIM = "The configuration with a check mark in all five columns obtains a MUSHRA of 65.9 [MOSHI:T3:r01]."


class AlwaysEntails:
    def predict(self, pairs):
        return [[0.0, 0.0, 9.0] for _ in pairs]

    class config:
        id2label = {0: "contradiction", 1: "neutral", 2: "entailment"}


def test_p6_system_prompt_is_p4_plus_rules_and_redraft_is_p5r():
    assert p6.SYSTEM.startswith(p4.SYSTEM) and "8. Write each sentence close to the wording" in p6.SYSTEM
    assert "9. For a value from a table row" in p6.SYSTEM and "10. A sentence that compares" in p6.SYSTEM
    m = p6.messages("q", P)
    assert m[0]["content"] == p6.SYSTEM and m[1] == p4.messages("q", P)[1]  # same user message and passages
    r = p6.redraft_messages("q", P, "x", [("x", "bad")])
    assert p5r.REDRAFT_INSTRUCTION in r[-1]["content"] and r[0]["content"] == p6.SYSTEM


def test_p6_bound_in_the_pipeline(monkeypatch):
    monkeypatch.setattr(pipeline, "prompts", p6.as_prompts_module())
    llm = FakeLLM(GOOD)
    r = pipeline.answer_from_evidence("How many quantizers does Mimi use?", ev(P), llm, verifier(SUPPORTED), CORPUS)
    assert r.status == "answered" and llm.calls[0][0]["content"] == p6.SYSTEM


def test_v45_rejects_a_mark_claim_on_rows_without_marks():
    ids = {ROW_NO_MARKS["chunk_id"]}
    assert v42.verify(MARK_CLAIM, [ROW_NO_MARKS], ids, AlwaysEntails(), 0.9, 0.9).ok  # the shown wrong answer
    rep = v45.verify(MARK_CLAIM, [ROW_NO_MARKS], ids, AlwaysEntails(), 0.9, 0.9)
    assert not rep.ok and "carry no marks" in rep.sentences[0].reasons[0]


def test_v45_leaves_rows_with_marks_and_non_mark_claims_to_v42():
    ids = {ROW_NO_MARKS["chunk_id"]}
    assert v45.verify(MARK_CLAIM.replace("65.9", "64.0"), [ROW_MARKS], ids, AlwaysEntails(), 0.9, 0.9).ok
    plain = "In Table 3, the configuration has MUSHRA = 65.9±1.7 [MOSHI:T3:r01]."
    assert v45.verify(plain, [ROW_NO_MARKS], ids, AlwaysEntails(), 0.9, 0.9).ok == \
        v42.verify(plain, [ROW_NO_MARKS], ids, AlwaysEntails(), 0.9, 0.9).ok


def test_production_bindings_unchanged():
    assert pipeline.prompts is p4 and p4.PROMPT_VERSION == "p4" and pipeline.verify is v42.verify
