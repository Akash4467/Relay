import relay.pipeline as pipeline
from relay.candidate_p5r import prompts as p5r
from relay.llm import prompts as p4
from tests.unit.test_pipeline import CORPUS, GOOD, P, SUPPORTED, FakeLLM, ev, verifier

BAD = "Mimi runs at 2.4 kbps [MOSHI:3.3.1:c030]."


def test_first_draft_messages_are_identical_to_p4():
    m = p5r.as_prompts_module()
    assert m.messages("q", P) == p4.messages("q", P) and m.SYSTEM == p4.SYSTEM


def test_redraft_differs_from_p4_only_in_the_instruction():
    fails = [("Mimi runs at 2.4 kbps.", "not supported")]
    a, b = p4.redraft_messages("q", P, BAD, fails), p5r.redraft_messages("q", P, BAD, fails)
    assert a[:-1] == b[:-1] and a[-1]["role"] == b[-1]["role"] == "user"
    assert a[-1]["content"].split("\n")[:2] == b[-1]["content"].split("\n")[:2]  # same header and failure lines
    assert p5r.REDRAFT_INSTRUCTION in b[-1]["content"] and "Never repeat a listed sentence unchanged" in b[-1]["content"]


def test_pipeline_uses_the_candidate_redraft_when_bound(monkeypatch):
    monkeypatch.setattr(pipeline, "prompts", p5r.as_prompts_module())
    llm = FakeLLM(BAD, GOOD)
    r = pipeline.answer_from_evidence("How many quantizers does Mimi use?", ev(P), llm, verifier(SUPPORTED), CORPUS)
    assert r.status == "answered" and r.text == GOOD and r.llm_calls == 2
    assert p5r.REDRAFT_INSTRUCTION in llm.calls[1][-1]["content"]


def test_production_binding_is_p4():
    assert pipeline.prompts is p4 and p4.PROMPT_VERSION == "p4"
