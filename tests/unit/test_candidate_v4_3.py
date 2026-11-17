import json
from pathlib import Path

import pytest

from relay.candidate_v4_3 import verifier as cand
from relay.verify import verifier as v42


def chunks(path: str) -> dict:
    return {json.loads(l)["chunk_id"]: json.loads(l) for l in open(path, encoding="utf-8")}


@pytest.mark.parametrize("path", ["data/processed/chunks_256.jsonl",
                                  "experiments/phase10/corpus/c6/processed/chunks_256.jsonl"])
def test_self_reference_sentences_are_verbatim_in_the_paper(path):
    if not Path(path).exists():
        pytest.skip(f"{path} not built")
    c = chunks(path)
    for paper, (cid, sentence) in cand.SELF_REFERENCE.items():
        assert cid.startswith(paper + ":") and " ".join(sentence.split()) in " ".join(c[cid]["text"].split())


TEXT = {"chunk_id": "AIAYN:5.2:c026", "paper": "AIAYN", "section_id": "5.2", "section_title": "Hardware and Schedule",
        "kind": "text", "text": "We trained our models on one machine with 8 NVIDIA P100 GPUs."}
TABLE = {"chunk_id": "AIAYN:T3:r01", "paper": "AIAYN", "section_id": "6.2", "section_title": "Model Variations",
         "kind": "table", "text": "Table 3: Variations\nbase: BLEU = 25.8",
         "rows": [{"configuration": "base", "values": [{"metric": "BLEU", "split": "dev", "value": "25.8"}]}]}


def test_text_premise_starts_with_the_papers_self_reference_and_keeps_the_passage():
    (p,) = cand.premises([TEXT], "claim")
    assert cand.SELF_REFERENCE["AIAYN"][1] in p and TEXT["text"] in p
    assert v42.premises([TEXT], "claim") == [p.replace(cand.SELF_REFERENCE["AIAYN"][1] + "\n", "")]


def test_table_premises_and_the_passage_itself_are_unchanged():
    assert cand.premises([TABLE], "base BLEU 25.8") == v42.premises([TABLE], "base BLEU 25.8")
    cand.premises([TEXT])
    assert TEXT["text"].startswith("We trained")  # the caller's chunk is never modified


class FakeNLI:
    """Entails only when the premise holds the self-reference; records every premise it saw."""
    def __init__(self):
        self.seen = []

    def predict(self, pairs):
        self.seen += [p for p, _ in pairs]
        return [[0.0, 0.0, 9.0] if "we propose the Transformer" in p else [9.0, 0.0, 0.0] for p, _ in pairs]

    class config:
        id2label = {0: "contradiction", 1: "neutral", 2: "entailment"}


def test_candidate_is_the_frozen_verify_with_only_premises_changed():
    ans = "The Transformer was trained on one machine with 8 NVIDIA P100 GPUs [AIAYN:5.2:c026]."
    corpus = {"AIAYN:5.2:c026"}
    assert cand.verify(ans, [TEXT], corpus, FakeNLI(), 0.9, 0.9).ok
    assert not v42.verify(ans, [TEXT], corpus, FakeNLI(), 0.9, 0.9).ok
    # level-1 checks still reject regardless of NLI: an unknown citation
    bad = "The Transformer was trained on 8 GPUs [AIAYN:9.9:c999]."
    assert not cand.verify(bad, [TEXT], corpus, FakeNLI(), 0.9, 0.9).ok
    assert v42.verify.__globals__["premises"] is v42.premises  # production binding untouched
