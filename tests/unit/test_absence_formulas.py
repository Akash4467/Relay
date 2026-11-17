"""The v4 rejection-only checks on their own."""
from relay.verify import absence, formulas

P = [{"chunk_id": "MOSHI:4.2:c057", "kind": "text",
      "text": "We use transcription obtained with the package, along with the medium Whisper model. "
              "The pairs of participants were recorded with separate channels per speaker."}]


def test_absence_words_must_all_meet_in_one_sentence():
    say = "The retrieved passages do not state "
    assert absence.contradicted(say + "which Whisper model was used for transcription.", P)  # used ~ use
    assert absence.contradicted(say + "how many speakers took part.", P) == []  # "took" never appears
    assert absence.contradicted(say + "the Whisper model.", P)  # both words in one sentence: rejected (lexical limit)
    assert absence.contradicted(say + "the optimizer.", P) == []  # one word is a mention, not a report


def test_tables_are_left_to_the_table_check():
    table = [{"chunk_id": "T", "kind": "table", "text": "Configuration Mimi: bitrate = 1.1kbps"}]
    assert absence.contradicted("The retrieved passages do not state the Mimi bitrate.", table) == []


def test_formulas_only_with_a_relation_and_up_to_notation():
    assert formulas.formulas(r"the output \(\hat{x}_t\) and \(p_t \ge 0.5\)") == [r"p_t \ge 0.5"]
    cited = [{"chunk_id": "c", "paper": "AIAYN", "section_id": "3.3", "section_title": "FFN",
              "text": r"[EQ] \mathrm{FFN}(x)=\max(0, xW_1 + b_1) W_2 + b_2"}]
    assert formulas.unsupported(r"\(\mathrm{FFN}(x) = \max(0,\, xW_1 + b_1)\,W_2 + b_2\)", cited) == []
    assert formulas.unsupported(r"\(\mathrm{FFN}(x) = \max(0, xW_2 + b_2) W_1 + b_1\)", cited)
