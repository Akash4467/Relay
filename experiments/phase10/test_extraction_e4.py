"""Phase 10C extraction e4: unit tests (synthetic LaTeX) and real-paper regressions (MOSHI/HNET sources on disk)."""
import sys

import pytest

sys.path.insert(0, "experiments/phase10")
import extraction_e4 as e4  # noqa: E402
from relay.retrieval import latex_parse as lp  # noqa: E402


def text(tex: str) -> str:
    tok = lp.EXTRACTION.set("e2")
    try:
        with e4.e4():
            return lp.to_text(tex, {})
    finally:
        lp.EXTRACTION.reset(tok)


# ---- unit tests (synthetic LaTeX)
@pytest.mark.parametrize("tex,want", [
    (r"\SI{160}{\ms}", "160 ms"), (r"\SI{230}{\ms}", "230 ms"), (r"\SI {12} {\second}", "12 s"),
    (r"\SI{5}{\minute}", "5 min"), (r"\SI[round-mode=places]{1.1}{kbps}", "1.1 kbps"),
    (r"\SI{24}{\kilo\hertz}", "24 kHz"), (r"\SI{12.5}{\hertz}", "12.5 Hz"), (r"\SI{3}{\milli\second}", "3 ms"),
    (r"\SI{10}{\kilo\byte\per\second}", "10 kB/s"), (r"\SI{50}{\percent}", "50 %"), (r"\si{\ms}", "ms"),
    (r"\num{1000}", "1000"), (r"\SIrange{10}{20}{\ms}", "10–20 ms"), (r"\qty{7}{\ms}", "7 ms"),
    ("\\SI{160}\n{\\ms}", "160 ms"),
])
def test_siunitx_forms(tex, want):
    assert text(tex) == want


def test_sentence_keeps_both_units_and_the_dash():
    assert text(r"a latency of \SI{160}{\ms}---lower than the \SI{230}{\ms} average") == \
        "a latency of 160 ms—lower than the 230 ms average"


def test_unknown_unit_macro_is_kept_by_name_and_reported():
    e4.UNKNOWN.clear()
    assert text(r"\SI{3}{\furlong}") == "3 furlong" and e4.UNKNOWN == ["furlong"]


def test_malformed_and_unrelated_macros_are_left_alone():
    assert e4.expand_siunitx(r"\SI{160") == r"\SI{160"
    assert e4.expand_siunitx(r"\Sigma \simeq \numbered{x}") == r"\Sigma \simeq \numbered{x}"
    assert e4.expand_siunitx("no macros") == "no macros"


def test_e3_phantom_fix_is_kept():
    assert text(r"$\phantom{1}42.2$") == "42.2"


def test_context_restores_the_production_converter():
    original = lp._L2T
    with e4.e4():
        assert lp._L2T is not original
    assert lp._L2T is original and lp.to_text(r"\SI{160}{\ms}", {}) == "160"  # production e1 unchanged


# ---- real-paper regressions (sources on disk; no network)
@pytest.fixture(scope="module")
def moshi():
    return e4.parse_paper_e4("MOSHI")


def paragraphs_with(parsed, needle):
    return [p["text"] for p in parsed.paragraphs if needle in p["text"]]


def test_moshi_latency_units_survive(moshi):
    hits = paragraphs_with(moshi, "160 ms—lower than the 230 ms average")
    assert len(hits) == 1


def test_moshi_all_four_siunitx_values_have_units(moshi):
    alltext = " ".join(p["text"] for p in moshi.paragraphs)
    for s in ("160 ms", "230 ms", "5 min", "12 s"):
        assert s in alltext, s


def table(parsed, n):
    return next(t for t in parsed.tables if t["table_id"] == f"T{n}")


def test_moshi_table4_causal_marks_rows_and_columns(moshi):
    t = table(moshi, 4)
    assert "causal" in t["headers"]
    by = {r["row_header"]: r["cells"] for r in t["rows"]}
    for row in ("Mimi, adv. loss only", "Same, downsampled at 16kHz", "Mimi, non adv. only"):
        assert by[row]["causal"] == "✓", row
    assert by["RVQGAN"]["causal"] == "(blank)"  # \xmark is defined as empty: an explicit blank, not ✓/✗
    assert by["RVQGAN"]["MUSHRA (↑)"] == "31.3±1.3"  # value stays with its row and column


def test_moshi_table3_check_and_blank_cells(moshi):
    t = table(moshi, 3)
    row2 = t["rows"][1]["cells"]
    assert row2["WavLM / distillation"] == "✓" and row2["Split / quantizer"] == "(blank)" and row2["ABX (↓)"] == "6.5%"


def test_hnet_table6_cross_marks_and_multirow():
    t = table(e4.parse_paper_e4("HNET"), 6)
    by = {r["cells"]["Example Architectures"]: r for r in t["rows"]}
    assert by["ByT5"]["cells"]["Autoregressive"] == "✗" and by["MambaByte"]["cells"]["Autoregressive"] == "✓"
    assert by["SpaceByte"]["cells"]["Autoregressive"] == "✓"  # \multirow carried down
    assert by["MANTa"]["cells"]["Multi-stage / Hierarchy"] == "✗" and by["H-Net"]["row_header"] == "Hierarchical (dynamic)"
