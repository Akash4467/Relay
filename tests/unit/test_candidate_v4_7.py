from relay.candidate_v4_7 import verifier as v47
from relay.verify.citations import SentenceCheck
from relay.verify.verifier import Report

Q14 = "How is the input segmented in the original Transformer's translation experiments compared with H-Net?"
Q15 = "Which NVIDIA GPUs were used to train the original Transformer and Moshi?"


def rep(*sents):
    return Report([SentenceCheck(s, c, absence=a) for s, c, a in sents])


def test_answer_citing_only_one_of_two_named_papers_is_uncovered():
    r = rep(("The Transformer uses the GPT-2 tokenizer [HNET:3.1:c053].", ["HNET:3.1:c053"], False),
            ("H-Net takes raw byte inputs [HNET:abstract:c001].", ["HNET:abstract:c001"], False))
    assert v47.uncovered(Q14, r) == ["AIAYN"]


def test_both_papers_cited_or_honest_absence_is_covered():
    both = rep(("The authors trained their models on 8 NVIDIA P100 GPUs [AIAYN:5.2:c026].", ["AIAYN:5.2:c026"], False),
               ("All models are trained on H100 GPUs [MOSHI:4.4:c062].", ["MOSHI:4.4:c062"], False))
    assert v47.uncovered(Q15, both) == []
    honest = rep(("H-Net takes raw byte inputs [HNET:abstract:c001].", ["HNET:abstract:c001"], False),
                 ("The retrieved passages do not state how the original Transformer segmented its input.", [], True))
    assert v47.uncovered(Q14, honest) == []


def test_question_naming_no_paper_is_unaffected():
    r = rep(("Something [HNET:3.1:c053].", ["HNET:3.1:c053"], False))
    assert v47.uncovered("How many attention heads does the base Transformer use?", r) == []


def test_a_rejected_sentence_does_not_count_as_coverage():
    r = Report([SentenceCheck("x [AIAYN:5.2:c026].", ["AIAYN:5.2:c026"], reasons=["not supported"]),
                SentenceCheck("y [MOSHI:4.4:c062].", ["MOSHI:4.4:c062"])])
    assert v47.uncovered(Q15, r) == ["AIAYN"]
