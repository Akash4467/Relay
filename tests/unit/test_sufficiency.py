from relay.verify.sufficiency import check


def p(paper, text="some evidence"):
    return {"paper": paper, "text": text}


def test_no_passages_or_only_empty_ones_refuse():
    assert not check("What is Mimi?", []).ok
    assert not check("What is Mimi?", [p("MOSHI", "  ")]).ok


def test_answerable_question_with_common_terminology_passes():
    assert check("How many heads does it use?", [p("AIAYN")]).ok


def test_paraphrase_absent_from_the_paper_is_not_refused_before_the_model():
    # "segmented" never occurs in the corpus; the structural check must not care (q14's failure mode)
    assert check("How is the input segmented in the original Transformer?", [p("AIAYN")]).ok


def test_unanswerable_with_similar_passages_is_left_to_model_and_verifier():
    # q17-like: BLEU passages retrieve strongly; structurally there is evidence, so the model gets to try
    assert check("What BLEU did the Transformer get on English-to-Romanian?", [p("AIAYN", "BLEU 28.4")]).ok


def test_two_named_papers_need_a_passage_from_each():
    q = "Which GPUs trained the original Transformer and Moshi?"
    res = check(q, [p("AIAYN"), p("AIAYN")])
    assert not res.ok and "MOSHI" in res.reason
    assert check(q, [p("AIAYN"), p("MOSHI")]).ok


def test_single_named_paper_is_not_constrained():
    assert check("According to Vaswani et al., what is MMLU?", [p("MOSHI")]).ok
