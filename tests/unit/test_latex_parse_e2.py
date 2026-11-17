"""Extraction e2 (corpus c4): table symbols, blank marks, multirow spans, empty cells. e1 stays the default."""
import pytest

from relay.retrieval import latex_parse as lp
from relay.retrieval.chunker import structured_row


def e2(fn, *args, **kw):
    token = lp.EXTRACTION.set("e2")
    try:
        return fn(*args, **kw)
    finally:
        lp.EXTRACTION.reset(token)


@pytest.mark.parametrize("tex, out", [
    (r"\ding{51}", "✓"), (r"\ding{52}", "✓"), (r"\ding{55}", "✗"), (r"\ding{56}", "✗"), (r"\checkmark", "✓"),
    (r"\ding { 51 }", "✓"),
])
def test_every_supported_macro(tex, out):
    assert e2(lp.to_text, tex, {}) == out


def test_unknown_ding_never_leaks_its_number():
    assert e2(lp.to_text, r"\ding{99}", {}) == "[symbol]"


def test_default_extraction_is_unchanged():
    assert lp.EXTRACTION.get() == "e1"
    assert lp.to_text(r"\ding{51}", {}) == "51" and lp.to_text(r"\checkmark", {}) == ""  # the c1-c3 behaviour


def test_ordinary_numbers_and_prose_without_symbols_are_untouched():
    for tex in ("Accuracy rose from 51.2 to 55 points on 52 tasks.", r"We use $d_k = 64$ and 8 heads \cite{x}."):
        assert e2(lp.to_text, tex, {}) == lp.to_text(tex, {})


def test_mixed_text_and_symbols():
    assert e2(lp.to_text, r"\checkmark{} (2 stages)", {}) == "✓ (2 stages)"
    assert e2(lp.to_text, r"yes \ding{55} no", {}) == "yes ✗ no"


TABLE = r"""{lll}
\toprule
Name & Flag & Score \\
\midrule
alpha & \multirow{2}{*}{\ding{51}} & 1.0 \\
beta & & 2.0 \\
gamma & & 3.0 \\
delta & \ding{55} & \\
"""


def test_multirow_spans_its_rows_only_and_empty_cells_keep_their_place():
    headers, rows = e2(lp.parse_tabular_e2, TABLE, {})
    assert headers == ["Name", "Flag", "Score"]
    assert [(r["row_header"], r["cells"]) for r in rows] == [
        ("alpha", {"Flag": "✓", "Score": "1.0"}),
        ("beta", {"Flag": "✓", "Score": "2.0"}),      # inside the \multirow{2}
        ("gamma", {"Flag": "", "Score": "3.0"}),      # after the span: empty, not carried
        ("delta", {"Flag": "✗", "Score": ""}),        # an empty cell keeps its column
    ]
    assert lp.row_text(rows[3]) == "delta: Flag = ✗"  # empty cells are not spelled out
    assert [v["column"] for v in structured_row(rows[3])["values"]] == ["Flag"]


def test_first_column_of_marks_stays_a_column():
    body = (r"{lll} \toprule Rate & Split & ABX \\ \midrule \ding{51} & \ding{51} & 23.3 \\ "
            + lp._BLANK + r" & \ding{51} & 6.5 \\")
    headers, rows = e2(lp.parse_tabular_e2, body, {})
    assert [(r["row_header"], r["cells"]) for r in rows] == [
        ("Row 1", {"Rate": "✓", "Split": "✓", "ABX": "23.3"}),
        ("Row 2", {"Rate": lp.BLANK_MARK, "Split": "✓", "ABX": "6.5"}),  # a blank mark is not filled down
    ]


def test_plain_tables_parse_the_same_in_e1_and_e2():
    body = r"{lll} \toprule Model & BLEU & PPL \\ \midrule base & 25.8 & 4.92 \\ big & 26.4 & 4.33 \\"
    assert e2(lp.parse_tabular_e2, body, {}) == lp.parse_tabular(body, {})


def test_paper_macros_end_to_end(tmp_path):
    """A paper defining \\cmark via \\ding and \\xmark as empty: ticks become ✓, blanks stay blank marks in tables
    and disappear from prose, exactly as in the PDF."""
    src = tmp_path / "TEST" / "src"
    src.mkdir(parents=True)
    (src / "main.tex").write_text(r"""\documentclass{article}
\newcommand{\cmark}{\ding{51}}
\newcommand{\xmark}{}
\begin{document}
\section{Results}
Our model is marked \cmark{} here and \xmark{} there; it scores 51 points.

\begin{table}
\caption{Ablation.}
\begin{tabular}{lll}
\toprule
Encoder & Decoder & Score \\
\midrule
\cmark & \xmark & 2.91 \\
\xmark & \cmark & 2.22 \\
\bottomrule
\end{tabular}
\end{table}
\end{document}
""", encoding="utf-8")
    old = lp.parse_paper("TEST", tmp_path)
    new = lp.parse_paper("TEST", tmp_path, "e2")
    assert old.tables[0]["rows"][0]["row_header"] == "51"  # the defect, as built into c1-c3
    t = new.tables[0]
    assert [(r["row_header"], r["cells"]) for r in t["rows"]] == [
        ("Row 1", {"Encoder": "✓", "Decoder": lp.BLANK_MARK, "Score": "2.91"}),
        ("Row 2", {"Encoder": lp.BLANK_MARK, "Decoder": "✓", "Score": "2.22"})]
    prose = new.paragraphs[0]["text"]
    assert "✓" in prose and lp._BLANK not in prose and "51 points" in prose
    assert lp.EXTRACTION.get() == "e1"  # parse_paper restores the default


def test_unknown_extraction_version_is_refused(tmp_path):
    with pytest.raises(ValueError, match="unknown extraction"):
        lp.parse_paper("X", tmp_path, "e9")


def test_upward_multirow_fills_the_rows_above_and_leaves_no_artifact():
    """HNET Table 1 writes \\multirow{-2}{*}{Byte}: the value spans this row and the one above."""
    body = (r"{lll} \toprule Model & Input & Size \\ \midrule a & & 1 \\ b & & 2 \\ "
            r"c & \multirow{-2}{*}{Byte} & 3 \\ d & Token & 4 \\")
    _, rows = e2(lp.parse_tabular_e2, body, {})
    assert [r["cells"]["Input"] for r in rows] == ["", "Byte", "Byte", "Token"]
    assert not any("-2" in v for r in rows for v in r["cells"].values())
