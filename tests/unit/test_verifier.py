"""Verifier decisions with a controllable fake NLI model (the real model is tested in integration)."""
import pytest

from relay.verify import nli
from relay.verify.verifier import verify


class FakeNLI:
    """Entailment probability per hypothesis from a table (default: unsupported)."""

    class config:
        id2label = {0: "contradiction", 1: "entailment", 2: "neutral"}

    def __init__(self, entail: dict[str, float]):
        self.entail = entail

    def predict(self, pairs):
        import math
        out = []
        for _premise, hyp in pairs:
            e = min(max(self.entail.get(hyp, 0.02), 1e-6), 1 - 1e-6)
            rest = (1 - e) / 2
            out.append([math.log(rest), math.log(e), math.log(rest)])
        return out


P = [{"chunk_id": "MOSHI:3.3.1:c030", "paper": "MOSHI", "section_id": "3.3.1", "section_title": "Architecture",
      "text": "We use Q = 8 quantizers, each with a codebook size of 2048. At 12.5Hz, this represents 1.1kbps."}]
CORPUS = {"MOSHI:3.3.1:c030", "AIAYN:3.2.2:c014"}


def run(answer, entail):
    return verify(answer, P, CORPUS, FakeNLI(entail), entail_min=0.5, entail_min_flagged=0.8)


def test_nli_scores_softmax_and_label_order():
    (e, c), = nli.scores(FakeNLI({"h": 0.9}), [("p", "h")])
    assert e == pytest.approx(0.9) and c == pytest.approx(0.05)
    assert nli.scores(FakeNLI({}), []) == []


def test_common_terminology_supported_answer_passes():
    r = run("Mimi uses 8 quantizers [MOSHI:3.3.1:c030].", {"Mimi uses 8 quantizers.": 0.95})
    assert r.ok and r.cited() == ["MOSHI:3.3.1:c030"]


def test_paraphrase_absent_from_the_paper_is_accepted_when_supported():
    claim = "Its audio is described by eight separate quantization stages"
    r = run(f"{claim} [MOSHI:3.3.1:c030].", {claim + ".": 0.9})
    assert r.ok and r.sentences[0].flags == []


def test_correct_technical_claim_in_other_terminology_passes_even_if_flagged():
    claim = "The codec's residual stack has 8 quantizers"
    r = run(f"{claim} [MOSHI:3.3.1:c030].", {claim + ".": 0.92})
    assert r.ok


def test_invented_model_name_needs_strong_support_and_fails_without_it():
    claim = "The RQ-Transformer uses 8 quantizers"
    r = run(f"{claim} [MOSHI:3.3.1:c030].", {claim + ".": 0.6})  # >= 0.5 but flagged, so 0.8 is needed
    assert not r.ok
    assert "rq-transformer" in r.failures()[0][1]


def test_unsupported_number_is_rejected():
    claim = "Mimi runs at 2.4 kbps"
    r = run(f"{claim} [MOSHI:3.3.1:c030].", {claim + ".": 0.3})
    assert not r.ok and "number 2.4 kbps" in r.failures()[0][1]


def test_familiar_wording_does_not_save_an_unsupported_claim():
    claim = "Mimi uses 8 quantizers with a codebook"  # every word is in the passage, NLI says unsupported
    r = run(f"{claim} [MOSHI:3.3.1:c030].", {claim + ".": 0.1})
    assert not r.ok and r.sentences[0].flags == []


def test_level1_failure_rejects_without_nli():
    r = run("Mimi uses 8 quantizers [AIAYN:3.2.2:c014].", {})
    assert not r.ok and "not retrieved" in r.failures()[0][1]
    assert r.sentences[0].entailment is None


def test_partial_answer_with_not_stated_sentence_passes_but_not_stated_alone_does_not():
    claim = "Mimi uses 8 quantizers"
    ok = run(f"{claim} [MOSHI:3.3.1:c030]. The papers do not state the GPU count.", {claim + ".": 0.9})
    assert ok.ok
    assert not run("The papers do not state the GPU count.", {}).ok  # nothing answered: treated as refusal
