from relay.candidate_v4_4 import verifier as cand
from relay.verify import verifier as v42

TEXT = {"chunk_id": "AIAYN:3.2.2:c014", "paper": "AIAYN", "section_id": "3.2.2", "section_title": "Multi-Head Attention",
        "kind": "text", "text": "In this work we employ h=8 parallel attention layers, or heads. For each of these we use "
                                "d_k=d_v=d_model/h=64. Due to the reduced dimension of each head, the cost is similar."}
TABLE = {"chunk_id": "AIAYN:T3:r01", "paper": "AIAYN", "section_id": "6.2", "section_title": "Model Variations",
         "kind": "table", "text": "Table 3: Variations\nbase: BLEU = 25.8",
         "rows": [{"configuration": "base", "values": [{"metric": "BLEU", "split": "dev", "value": "25.8"}]}]}


def test_text_passage_adds_sentences_and_adjacent_pairs_after_the_v42_premises():
    out = cand.premises([TEXT], "c")
    base = v42.premises([TEXT], "c")
    assert out[:len(base)] == base
    extra = out[len(base):]
    assert len(extra) == 3 + 2  # three sentences, two adjacent pairs
    assert all(e.startswith("Attention Is All You Need (Transformer) | 3.2.2 Multi-Head Attention\n") for e in extra)
    assert any(e.endswith("or heads. For each of these we use d_k=d_v=d_model/h=64.") for e in extra)


def test_tables_are_unchanged():
    assert cand.premises([TABLE], "base BLEU 25.8") == v42.premises([TABLE], "base BLEU 25.8")


class FakeNLI:
    """Entails only a premise that is exactly the header plus one short sentence (a whole chunk never entails)."""
    def predict(self, pairs):
        return [[0.0, 0.0, 9.0] if "For each of these" in p and "Due to" not in p else [9.0, 0.0, 0.0] for p, _ in pairs]

    class config:
        id2label = {0: "contradiction", 1: "neutral", 2: "entailment"}


def test_candidate_is_the_frozen_verify_with_only_premises_changed():
    ans = "For each of these we use d_k=d_v=d_model/h=64 [AIAYN:3.2.2:c014]."
    assert cand.verify(ans, [TEXT], {TEXT["chunk_id"]}, FakeNLI(), 0.9, 0.9).ok
    assert not v42.verify(ans, [TEXT], {TEXT["chunk_id"]}, FakeNLI(), 0.9, 0.9).ok
    assert not cand.verify("We use d_k=64 [AIAYN:9:c999].", [TEXT], {TEXT["chunk_id"]}, FakeNLI(), 0.9, 0.9).ok
    assert v42.verify.__globals__["premises"] is v42.premises
