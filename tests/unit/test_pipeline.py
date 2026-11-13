"""The draft -> verify -> redraft -> refuse path with a fake LLM that records its calls."""
from relay.llm import prompts
from relay.pipeline import Verifier, answer_from_evidence
from relay.retrieval.search import Evidence
from tests.unit.test_verifier import P, FakeNLI

CORPUS = {"MOSHI:3.3.1:c030", "AIAYN:3.2.2:c014"}
AIAYN = {"chunk_id": "AIAYN:3.2.2:c014", "paper": "AIAYN", "section_id": "3.2.2", "section_title": "Multi-Head Attention",
         "text": "In this work we employ h=8 parallel attention layers, or heads."}
GOOD = "Mimi uses 8 quantizers [MOSHI:3.3.1:c030]."
BAD_NUMBER = "Mimi runs at 2.4 kbps [MOSHI:3.3.1:c030]."


class FakeLLM:
    def __init__(self, *replies):
        self.replies, self.calls = list(replies), []

    def __call__(self, messages):
        self.calls.append(messages)
        return self.replies.pop(0)


def verifier(entail):
    return Verifier(FakeNLI(entail), 0.5, 0.8)


def ev(passages, question_papers=None):
    return Evidence(passages=passages, top_dense_cos=0.8, top_ce=None, tokens=100)


SUPPORTED = {"Mimi uses 8 quantizers.": 0.95}


def test_supported_answer_is_returned_after_one_call():
    llm = FakeLLM(GOOD)
    r = answer_from_evidence("How many quantizers does Mimi use?", ev(P), llm, verifier(SUPPORTED), CORPUS)
    assert (r.status, r.text, r.citations, r.llm_calls) == ("answered", GOOD, ["MOSHI:3.3.1:c030"], 1)
    assert llm.calls[0][0]["content"] == prompts.SYSTEM


def test_no_evidence_refuses_without_calling_the_model():
    llm = FakeLLM()
    r = answer_from_evidence("How many quantizers does Mimi use?", ev([]), llm, verifier(SUPPORTED), CORPUS)
    assert r.status == "refused" and r.llm_calls == 0 and llm.calls == []
    assert r.text.startswith("I couldn't find this in the passages I retrieved")


def test_named_paper_without_passage_refuses_without_calling_the_model():
    llm = FakeLLM()
    r = answer_from_evidence("Which GPUs trained the original Transformer and Moshi?", ev(P), llm,
                             verifier(SUPPORTED), CORPUS)
    assert r.status == "refused" and "AIAYN" in r.refusal_reason and llm.calls == []


def test_lexical_mismatch_never_refuses_before_the_model():
    # "segmented" is nowhere in the papers; the model is still asked and its supported answer accepted
    llm = FakeLLM(GOOD)
    r = answer_from_evidence("How is Mimi's latent segmented into quantizers?", ev(P), llm, verifier(SUPPORTED), CORPUS)
    assert r.status == "answered" and len(llm.calls) == 1


def test_model_not_in_papers_is_a_refusal():
    llm = FakeLLM("NOT_IN_PAPERS")
    r = answer_from_evidence("What GPU count trained Moshi?", ev(P), llm, verifier({}), CORPUS)
    assert (r.status, r.refusal_reason, r.llm_calls) == ("refused", "model: not in papers", 1)


def test_unsupported_claim_is_caught_then_fixed_by_the_redraft():
    llm = FakeLLM(BAD_NUMBER, GOOD)
    r = answer_from_evidence("What bitrate?", ev(P), llm, verifier(SUPPORTED), CORPUS)
    assert r.status == "answered" and r.llm_calls == 2 and r.text == GOOD
    redraft = llm.calls[1][-1]["content"]
    assert "Your previous answer had unsupported sentences" in redraft and "2.4 kbps" in redraft
    assert not r.reports[0].ok and r.reports[1].ok


def test_unanswerable_with_similar_passages_and_a_made_up_answer_is_refused():
    # passages look relevant (Mimi), the model invents a figure twice: nothing unsupported reaches the user
    llm = FakeLLM(BAD_NUMBER, BAD_NUMBER)
    r = answer_from_evidence("What bitrate does Mimi reach on Romanian speech?", ev(P), llm, verifier(SUPPORTED), CORPUS)
    assert (r.status, r.refusal_reason, r.llm_calls) == ("refused", "verification failed after redraft", 2)
    assert "2.4" not in r.text


def test_no_redraft_when_not_allowed():
    llm = FakeLLM(BAD_NUMBER)
    r = answer_from_evidence("What bitrate?", ev(P), llm, verifier(SUPPORTED), CORPUS, allow_redraft=False)
    assert (r.status, r.refusal_reason, r.llm_calls) == ("refused", "verification failed", 1)


def test_cross_paper_answer_citing_both_papers_passes():
    claim1, claim2 = "Mimi uses 8 quantizers.", "The Transformer uses 8 heads."
    llm = FakeLLM("Mimi uses 8 quantizers [MOSHI:3.3.1:c030]. The Transformer uses 8 heads [AIAYN:3.2.2:c014].")
    r = answer_from_evidence("Compare the original Transformer and Moshi", ev(P + [AIAYN]), llm,
                             verifier({claim1: 0.9, claim2: 0.9}), CORPUS)
    assert r.status == "answered" and r.citations == ["MOSHI:3.3.1:c030", "AIAYN:3.2.2:c014"]
