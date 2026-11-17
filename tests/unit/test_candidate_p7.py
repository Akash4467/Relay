import relay.pipeline as pipeline
from relay.candidate_p6 import prompts as p6
from relay.candidate_p7 import prompts as p7
from relay.llm import prompts as p4

T8 = {"chunk_id": "MOSHI:T8:r01", "paper": "MOSHI", "section_id": "5.5", "section_title": "Spoken QA", "kind": "table",
      "text": "Table 8: Evaluation of spoken question answering.\nConfiguration SpeechGPT (7B): Web Q. = 6.5\n"
              "Configuration Moshi: Web Q. = 26.6; LlaMA Q. = 62.3",
      "rows": [{"configuration": "SpeechGPT (7B)", "values": [{"metric": "Web Q.", "column": "Web Q.", "split": None, "value": "6.5"}]},
               {"configuration": "Moshi", "values": [{"metric": "Web Q.", "column": "Web Q.", "split": None, "value": "26.6"},
                                                      {"metric": "LlaMA Q.", "column": "LlaMA Q.", "split": None, "value": "62.3"}]}]}
TXT = {"chunk_id": "AIAYN:5.2:c026", "paper": "AIAYN", "section_id": "5.2", "section_title": "Hardware", "kind": "text",
       "text": "We trained our models on one machine with 8 NVIDIA P100 GPUs. Each training step took about 0.4 seconds."}


def test_table_hint_shows_the_row_holding_the_sentences_values_exactly():
    h = p7.hints("Moshi achieves 26.6 % accuracy on Web Questions [MOSHI:T8:r01].", [T8])
    assert 'Row in [MOSHI:T8:r01]: "Moshi: Web Q. = 26.6; LlaMA Q. = 62.3"' in h
    assert not any("SpeechGPT" in x for x in h)  # only the row whose values the sentence states
    assert any("questions" in x for x in h if x.startswith("Words"))  # 'Questions' is not in the row


def test_text_hint_shows_closest_sentence_and_missing_words():
    h = p7.hints("The original Transformer was trained on eight NVIDIA P100 GPUs [AIAYN:5.2:c026].", [TXT])
    assert 'Closest sentence in [AIAYN:5.2:c026]: "We trained our models on one machine with 8 NVIDIA P100 GPUs."' in h
    missing = next(x for x in h if x.startswith("Words"))
    assert "original" in missing and "transformer" in missing and "nvidia" not in missing
    h2 = p7.hints("The authors report 8 NVIDIA P100 GPUs [AIAYN:5.2:c026].", [TXT])
    assert not any(x.startswith("Words") for x in h2)  # rule-8 vocabulary is never flagged


def test_uncited_or_unknown_citation_gets_no_hints():
    assert p7.hints("No citation here.", [TXT]) == []
    assert p7.hints("Something [AIAYN:9:c999].", [TXT]) == []


def test_redraft_contains_hints_and_first_draft_is_p6():
    r = p7.redraft_messages("q", [TXT], "d", [("The original Transformer used 8 GPUs [AIAYN:5.2:c026].", "not supported")])
    assert "Closest sentence in [AIAYN:5.2:c026]" in r[-1]["content"] and "reuse its exact labels" in r[-1]["content"]
    assert r[0]["content"] == p6.SYSTEM and p7.messages("q", [TXT]) == p6.messages("q", [TXT])


def test_production_binding_is_p4():
    assert pipeline.prompts is p4
