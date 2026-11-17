import pytest

from relay.verify.sentences import split_sentences


def test_basic_split():
    assert split_sentences("One thing. Two things! Three? Four") == ["One thing.", "Two things!", "Three?", "Four"]


@pytest.mark.parametrize("text", [
    "Models, e.g. Transformers, are used here.",
    "Models, i.e. Transformers, are used here.",
    "As shown by Vaswani et al. The model works.",
    "See Fig. 2 and Eq. 3 and Sec. 4 for details.",
    "Compare A vs. B for this result.",
    "We use dropout of 0.1 and version 2.3.1 here.",
    "Described in §3. Then more text follows.",
])
def test_protected_periods_do_not_split(text):
    assert split_sentences(text) == [text]


def test_newline_is_a_boundary_for_lists_and_equations():
    assert split_sentences("We compute:\n[EQ] a = b\n- item one\n- item two") == \
        ["We compute:", "[EQ] a = b", "- item one", "- item two"]


def test_trailing_citation_stays_with_its_sentence():
    text = "Moshi uses Mimi [MOSHI:3.3:c010]. It runs at 12.5 Hz [MOSHI:3.3:c011; MOSHI:T4:r01]."
    assert split_sentences(text) == ["Moshi uses Mimi [MOSHI:3.3:c010].",
                                     "It runs at 12.5 Hz [MOSHI:3.3:c011; MOSHI:T4:r01]."]


def test_citation_after_period_moves_back_to_previous_sentence():
    text = "Moshi uses Mimi. [MOSHI:3.3:c010] It runs at 12.5 Hz. [MOSHI:3.3:c011]"
    assert split_sentences(text) == ["Moshi uses Mimi. [MOSHI:3.3:c010]", "It runs at 12.5 Hz. [MOSHI:3.3:c011]"]


def test_empty_and_whitespace():
    assert split_sentences("") == []
    assert split_sentences("  \n\n ") == []
