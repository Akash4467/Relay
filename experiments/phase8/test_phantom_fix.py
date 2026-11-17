"""Tests for the phase-8 phantom candidate (experiments only). Run: uv run pytest -q experiments/phase8"""
import sys

import pytest

sys.path.insert(0, "experiments/phase8")
from phantom_fix import e3, strip_phantoms  # noqa: E402

from relay.retrieval import latex_parse as lp


def text(tex):
    with e3():
        return lp.to_text(tex, {})


@pytest.mark.parametrize("tex, out", [
    (r"$\phantom{1}42.2$", "42.2"),          # MOSHI Table 5
    (r"42.2", "42.2"),                       # a plain number is untouched
    (r"1\phantom{0}", "1"),                  # the visible 1 survives
    (r"$\phantom{00}1.0$", "1.0"),           # MOSHI Table 6 weight
    (r"\hphantom{0}9.2", "9.2"),             # already correct before, still correct
    (r"\vphantom{X}7", "7"),
    (r"\phantom{\phantom{1}2}3", "3"),       # nested: no visible character duplicated or leaked
    (r"100.00\hphantom{00}", "100.00"),
])
def test_invisible_spacing_is_removed_and_visible_text_kept(tex, out):
    assert text(tex) == out


def test_old_behaviour_is_the_defect():
    assert lp.to_text(r"$\phantom{1}42.2$", {}) == "142.2"  # e1/e2 unchanged outside the e3 block


@pytest.mark.parametrize("tex", [r"$a = b + c$", r"$x^2 + y_i$", r"$d_k = d_{model}/h = 64$", r"We use 8 heads and 51.2\% of data."])
def test_ordinary_math_and_prose_are_unchanged(tex):
    assert text(tex) == lp.to_text(tex, {})


def test_unbalanced_braces_are_left_alone():
    assert strip_phantoms(r"\phantom{1 42") == r"\phantom{1 42"


def test_the_e3_block_restores_the_converter():
    original = lp._L2T
    with e3():
        pass
    assert lp._L2T is original
