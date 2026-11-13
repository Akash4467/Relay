from relay.verify import evidence as ev


def chunk(text, paper="MOSHI", sid="3.3.1", title="Architecture"):
    return {"chunk_id": "X:1:c001", "paper": paper, "section_id": sid, "section_title": title, "text": text}


MIMI = chunk("We use Q = 8 quantizers, each with a codebook size of 2048. At 12.5Hz, this represents 1.1kbps.")


def test_normalize_and_tokens():
    assert ev.tokens("At 12.5Hz, 1,024 eight heads – fine") == ["at", "12.5", "hz", "1024", "8", "heads", "fine"]


def test_one_is_not_a_number_but_other_number_words_are():
    # calibration pair s20: "one for itself and one for the user" was flagged as the number 1
    assert ev.tokens("one for itself and one for the user") == ["one", "for", "itself", "and", "one", "for", "the", "user"]
    assert ev.tokens("two streams") == ["2", "streams"]


def test_technical_terms_detect_names_not_sentence_starts():
    assert ev.technical_terms("Moshi uses Mimi and the RQ-Transformer on H100 GPUs.") == \
        ["mimi", "rq-transformer", "h100", "gpus"]


def test_supported_numbers_and_units_raise_no_flags():
    assert ev.signals("The codec Mimi uses 8 quantizers and a codebook of 2048 at 12.5 Hz", [MIMI]) == \
        ["term 'mimi' not in the cited passage"]  # the passage never names Mimi: a warning, not a rejection


def test_unsupported_number_is_flagged():
    assert ev.signals("It runs at 2.4 kbps", [MIMI]) == ["number 2.4 kbps not in the cited passage"]


def test_number_with_unit_must_sit_near_the_unit():
    c = chunk("There are 8 layers in the encoder and the decoder uses many heads in total.")
    assert ev.signals("It uses 8 heads", [c]) == ["number 8 heads not in the cited passage"]


def test_invented_technical_name_is_flagged():
    assert ev.signals("The RQ-Transformer uses 8 quantizers", [MIMI]) == ["term 'rq-transformer' not in the cited passage"]


def test_ordinary_words_are_never_flagged_however_rare():
    # calibration pairs s04/s06/u19: "parallelized", "lets", "generalize" were flagged for being rare
    assert ev.signals("each of the 8 quantizers lets it be parallelized and generalize", [MIMI]) == []


def test_delatex_drops_notation_but_keeps_symbols_and_numbers():
    assert ev.delatex(r"\(\hat{x}_t\) and \(k_{t-1}\)").split() == ["x_t", "and", "k_t-1"]
    assert "0.5" in ev.delatex(r"\(b_t = \mathds{1}_{\{p_t \ge 0.5\}}\)")


def test_latex_macros_are_not_flagged_as_technical_terms():
    # q05 regression: "\tfrac12" was flagged as the term 'tfrac12'
    c = chunk(r"p_t = \frac{1}{2} \left(1 - \frac{q_t^\top k_{t-1}}{...}\right) and b_t = \mathds{1}_{\{p_t \geq 0.5\}}",
              "HNET", "2.2.1", "Chunking Layer")
    claim = r"It computes \(p_t = \tfrac12\bigl(1 - \frac{q_t^\top k_{t-1}}{\|q_t\|\|k_{t-1}\|}\bigr)\) and sets \(b_t\) at \(p_t \ge 0.5\)"
    assert ev.signals(claim, [c]) == []


def test_numbers_inside_latex_are_still_checked():
    c = chunk("b_t = 1 when p_t >= 0.5", "HNET", "2.2.1", "Chunking Layer")
    assert ev.signals(r"a boundary is set when \(p_t \ge 0.7\)", [c]) == ["number 0.7 not in the cited passage"]


def test_named_paper_must_be_cited():
    flags = ev.signals("According to the original Transformer paper, 8 quantizers are used", [MIMI])
    assert flags == ["names AIAYN but cites no AIAYN passage"]
    assert ev.signals("Moshi uses 8 quantizers", [MIMI]) == []  # the passage header names Moshi
