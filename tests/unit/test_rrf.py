import pytest

from relay.retrieval.rrf import rrf


def test_fuses_by_reciprocal_rank():
    out = dict(rrf([["a", "b", "c"], ["b", "a", "d"]]))
    assert out["a"] == pytest.approx(1 / 61 + 1 / 62)
    assert out["b"] == pytest.approx(1 / 62 + 1 / 61)
    assert out["d"] == pytest.approx(1 / 63)


def test_agreement_beats_a_single_top_rank():
    ranked = [d for d, _ in rrf([["x", "both"], ["y", "both"]])]
    assert ranked[0] == "both"


def test_ties_keep_first_seen_order_and_empty_inputs():
    assert [d for d, _ in rrf([["a"], ["b"]])] == ["a", "b"]
    assert rrf([]) == [] and rrf([[], []]) == []
