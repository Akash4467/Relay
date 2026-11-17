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


def test_partial_answer_with_scoped_absence_passes_but_absence_alone_does_not():
    claim = "Mimi uses 8 quantizers"
    ok = run(f"{claim} [MOSHI:3.3.1:c030]. The retrieved passages do not state the GPU count.", {claim + ".": 0.9})
    assert ok.ok and ok.sentences[1].absence
    assert not run("The retrieved passages do not state the GPU count.", {}).ok  # answers nothing: a refusal
    # q11 regression: an unscoped "the papers do not state" sentence fails the answer instead of passing unchecked
    bad = run(f"{claim} [MOSHI:3.3.1:c030]. The papers do not state the GPU count.", {claim + ".": 0.9})
    assert not bad.ok and bad.failures()[0][0] == "The papers do not state the GPU count."


class PremiseNLI(FakeNLI):
    """Entailment from (substring the premise must contain, hypothesis) rules; anything else is unsupported."""

    def __init__(self, rules: list[tuple[str, str, float]]):
        super().__init__({})
        self.rules, self.seen = rules, []

    def predict(self, pairs):
        import math
        out = []
        for premise, hyp in pairs:
            self.seen.append((premise, hyp))
            e = max([p for need, h, p in self.rules if need in premise and h == hyp], default=0.02)
            rest = (1 - e) / 2
            out.append([math.log(rest), math.log(e), math.log(rest)])
        return out


Q8 = "Mimi uses 8 quantizers."   # verified first sentence
IT = "It has a codebook size of 2048."


def test_context_resolves_a_leading_reference_from_an_accepted_previous_sentence():
    # NLI can tie "It" to the codec only when the previous sentence (naming Mimi) is in the premise
    nli_model = PremiseNLI([("", Q8, 0.95), ("Mimi uses 8 quantizers.\n", IT, 0.93)])
    r = verify(f"Mimi uses 8 quantizers [MOSHI:3.3.1:c030]. It has a codebook size of 2048 [MOSHI:3.3.1:c030].",
               P, CORPUS, nli_model, 0.9, 0.9)
    assert r.ok and r.sentences[1].context == "Mimi uses 8 quantizers."
    assert r.sentences[0].context is None  # the first sentence has nothing to resolve


def test_context_never_adds_support_the_evidence_lacks():
    added = "It also doubles the training speed."
    nli_model = PremiseNLI([("", Q8, 0.95)])  # nothing supports the added claim, with or without context
    r = verify(f"Mimi uses 8 quantizers [MOSHI:3.3.1:c030]. {added[:-1]} [MOSHI:3.3.1:c030].",
               P, CORPUS, nli_model, 0.9, 0.9)
    assert not r.ok and r.sentences[1].context is not None and r.sentences[1].entailment < 0.9


def test_no_context_from_a_rejected_sentence_or_for_a_self_contained_claim():
    nli_model = PremiseNLI([("Mimi uses 8 quantizers.\n", IT, 0.93)])  # Q8 itself is unsupported here
    r = verify(f"Mimi uses 8 quantizers [MOSHI:3.3.1:c030]. It has a codebook size of 2048 [MOSHI:3.3.1:c030].",
               P, CORPUS, nli_model, 0.9, 0.9)
    assert not r.sentences[0].ok and r.sentences[1].context is None and not r.sentences[1].ok
    nli_model = PremiseNLI([("", Q8, 0.95), ("", "The codebook size is 2048.", 0.95)])
    r = verify("Mimi uses 8 quantizers [MOSHI:3.3.1:c030]. The codebook size is 2048 [MOSHI:3.3.1:c030].",
               P, CORPUS, nli_model, 0.9, 0.9)
    assert r.ok and r.sentences[1].context is None  # no reference word, no context
    assert all(not prem.startswith("Mimi uses 8 quantizers.\n") for prem, _ in nli_model.seen)


TABLE = {"chunk_id": "AIAYN:T3:r02", "paper": "AIAYN", "section_id": "6.2", "section_title": "Model Variations",
         "kind": "table", "table_id": "T3",
         "text": "Table 3: Variations on the Transformer architecture.\nColumns: col1 | h | BLEU / (dev)\n"
                 "Configuration (A): h = 1; BLEU / (dev) = 24.9\nConfiguration (A): h = 4; BLEU / (dev) = 25.5",
         "rows": [{"configuration": "(A)", "values": [
                       {"metric": "h", "split": None, "column": "h", "value": "1"},
                       {"metric": "BLEU", "split": "dev", "column": "BLEU / (dev)", "value": "24.9"}]},
                  {"configuration": "(A)", "values": [
                       {"metric": "h", "split": None, "column": "h", "value": "4"},
                       {"metric": "BLEU", "split": "dev", "column": "BLEU / (dev)", "value": "25.5"}]}]}


def test_table_premises_isolate_each_row_and_add_only_the_rows_a_claim_is_about():
    from relay.verify.verifier import premises
    prem = premises([TABLE])  # no claim: each row alone, never the chunk's rows side by side (v3)
    assert prem == [("Attention Is All You Need (Transformer), Table 3: Variations on the Transformer architecture.\n"
                     "The configuration (A) has h = 1, BLEU on the dev set = 24.9."),
                    ("Attention Is All You Need (Transformer), Table 3: Variations on the Transformer architecture.\n"
                     "The configuration (A) has h = 4, BLEU on the dev set = 25.5.")]
    assert len(premises([TABLE], "With one head the dev BLEU is 24.9.")) == 2  # one relevant row: no extra block
    assert premises([TABLE], "BLEU rises from 24.9 to 25.5.")[-1].count("The configuration (A) has") == 2  # the two quoted


def test_single_configuration_value_is_supported_by_its_own_row():
    claim = "With a single head, the dev BLEU is 24.9."
    nli_model = PremiseNLI([("BLEU on the dev set = 24.9.", claim, 0.95)])
    r = verify(f"{claim[:-1]} [AIAYN:T3:r02].", [TABLE], {"AIAYN:T3:r02"}, nli_model, 0.9, 0.9)
    assert r.ok


def test_comparison_across_two_configurations_can_use_the_all_rows_premise():
    claim = "Using 4 heads gives a higher dev BLEU than using 1 head."
    both = "The configuration (A) has h = 1, BLEU on the dev set = 24.9.\nThe configuration (A) has h = 4, BLEU on the dev set = 25.5."
    nli_model = PremiseNLI([(both, claim, 0.93)])
    r = verify(f"{claim[:-1]} [AIAYN:T3:r02].", [TABLE], {"AIAYN:T3:r02"}, nli_model, 0.9, 0.9)
    assert r.ok


def test_wrong_value_for_a_configuration_is_still_rejected():
    claim = "With a single head, the dev BLEU is 25.5."  # that is the h = 4 row's value
    r = verify(f"{claim[:-1]} [AIAYN:T3:r02].", [TABLE], {"AIAYN:T3:r02"}, PremiseNLI([]), 0.9, 0.9)
    assert not r.ok


# --- policy v3 -------------------------------------------------------------------------------------------------

def test_a_supported_clause_cannot_carry_an_unsupported_one():
    whole = "Mimi uses 8 quantizers, and it was trained on 10 GPUs."
    # the fake model is fooled by the whole sentence, but the second clause on its own is unsupported
    r = run(f"{whole[:-1]} [MOSHI:3.3.1:c030].", {whole: 0.95, "Mimi uses 8 quantizers.": 0.95})
    assert not r.ok and 'clause: "It was trained on 10 GPUs."' in r.sentences[0].reasons[0]
    ok = run(f"{whole[:-1]} [MOSHI:3.3.1:c030].",
             {whole: 0.95, "Mimi uses 8 quantizers.": 0.95, "It was trained on 10 GPUs.": 0.95})
    assert ok.ok


FORMULA = [{"chunk_id": "HNET:2.2.1:c023", "paper": "HNET", "section_id": "2.2.1", "section_title": "Chunking Layer",
            "kind": "text", "text": r"It computes [EQ] p_t = \frac{1}{2} \left(1 - c_t\right), where c_t is a similarity."}]


def test_latex_spelling_does_not_change_what_the_model_is_asked():
    seen = PremiseNLI([])
    ids = {"HNET:2.2.1:c023"}
    verify(r"The boundary probability is \(p_t = \tfrac12\bigl(1 - c_t\bigr)\) [HNET:2.2.1:c023].", FORMULA, ids,
           seen, 0.9, 0.9)
    verify(r"The boundary probability is \(p_t = \frac{1}{2}\!\left(1 - c_t\right)\) [HNET:2.2.1:c023].", FORMULA,
           ids, seen, 0.9, 0.9)
    hyps = {h for _, h in seen.seen}
    assert hyps == {"The boundary probability is p_t = 1 / 2 (1 - c_t)."}


T3_BASE = {"chunk_id": "AIAYN:T3:r01", "paper": "AIAYN", "section_id": "6.2", "section_title": "Model Variations",
           "kind": "table", "table_id": "T3", "text": "Table 3: Variations on the Transformer architecture.",
           "rows": [{"configuration": "base", "values": [
               {"metric": "h", "split": None, "column": "h", "value": "8"},
               {"metric": "BLEU", "split": "dev", "column": "BLEU / (dev)", "value": "25.8"}]}]}


def test_q11_base_and_single_head_bleu_with_both_rows_retrieved():
    ids = {"AIAYN:T3:r01", "AIAYN:T3:r02"}
    claim = "The base configuration reaches 25.8 dev BLEU, while the single-head row reaches 24.9."
    base_vs_a = "The configuration base has h = 8, BLEU on the dev set = 25.8."
    rules = [(base_vs_a, claim, 0.95), (base_vs_a, "The base configuration reaches 25.8 dev BLEU.", 0.95),
             ("BLEU on the dev set = 24.9", "The single-head row reaches 24.9.", 0.95)]
    r = verify(f"{claim[:-1]} [AIAYN:T3:r01, AIAYN:T3:r02].", [T3_BASE, TABLE], ids, PremiseNLI(rules), 0.9, 0.9)
    assert r.ok
    # with the base row retrieved, saying its BLEU is missing is rejected, whatever NLI says
    absent = verify("The single-head row reaches 24.9 [AIAYN:T3:r02]. The retrieved passages do not state the dev "
                    "BLEU of the base configuration.", [T3_BASE, TABLE], ids, PremiseNLI(rules), 0.9, 0.9)
    assert not absent.ok and "absence contradicted" in absent.sentences[1].reasons[0]


# --- policy v4: rejection-only structure checks ------------------------------------------------------------------

def test_a_formula_must_appear_in_the_cited_passage_up_to_notation():
    ids, yes = {"HNET:2.2.1:c023"}, PremiseNLI([("", "The probability is p_t = 1 / 2 (1 - c_t).", 0.97),
                                               ("", "The probability is p_t = 1 / 2 (c_t - 1).", 0.97)])
    ok = verify(r"The probability is \(p_t = \tfrac{1}{2}(1 - c_t)\) [HNET:2.2.1:c023].", FORMULA, ids, yes, 0.9, 0.9)
    assert ok.ok
    # NLI is fooled (0.97), the swapped operands are not in the passage: rejected before NLI
    bad = verify(r"The probability is \(p_t = \tfrac{1}{2}(c_t - 1)\) [HNET:2.2.1:c023].", FORMULA, ids, yes, 0.9, 0.9)
    assert not bad.ok and "is not written this way" in bad.sentences[0].reasons[0]


def test_an_absence_sentence_contradicted_by_retrieved_text_is_rejected():
    text = [{"chunk_id": "AIAYN:5.3:c027", "paper": "AIAYN", "section_id": "5.3", "section_title": "Optimizer",
             "kind": "text", "text": "We used the Adam optimizer. We used warmup_steps=4000."}]
    ids, nli_ok = {"AIAYN:5.3:c027"}, PremiseNLI([("", "We used Adam.", 0.95)])
    lead = "We used Adam [AIAYN:5.3:c027]. "
    bad = verify(lead + "The retrieved passages do not state the number of warmup steps.", text, ids, nli_ok, 0.9, 0.9)
    assert not bad.ok and bad.sentences[1].reasons[0].startswith("absence contradicted: AIAYN:5.3:c027 says")
    fine = verify(lead + "The retrieved passages do not state the batch size.", text, ids, nli_ok, 0.9, 0.9)
    assert fine.ok  # the passage never talks about batch size: the scoped absence stands
