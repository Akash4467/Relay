import pytest

from relay.retrieval import latex_parse as lp


# ---------- helpers ----------

def test_group_balanced_and_escaped_braces():
    s = r"{a {b} \} c}rest"
    content, end = lp.group(s, 0)
    assert content == r"a {b} \} c"
    assert s[end:] == "rest"


def test_group_line_break_before_brace_and_trailing_backslash():
    assert lp.group(r"{a \\}b", 0) == (r"a \\", 6)  # \\ is a line break, so the brace closes
    assert lp.group("{ll} x \\\\", 0) == ("ll", 4)  # a trailing backslash must not escape s[0]


@pytest.mark.parametrize("s", ["no brace", "{unclosed"])
def test_group_errors(s):
    with pytest.raises(ValueError):
        lp.group(s, 0)


def test_strip_comments_keeps_escaped_percent_and_eats_newline():
    assert lp.strip_comments("50\\% done % note\nnext") == "50\\% done next"
    assert lp.strip_comments("a%\n   b") == "ab"
    assert lp.strip_comments("x\n% whole line\n\ny") == "x\n\ny"


def test_load_tex_resolves_inputs_relative_to_main(tmp_path):
    (tmp_path / "sections").mkdir()
    (tmp_path / "main.tex").write_text("A \\input{sections/one} D % \\input{sections/missing}")
    (tmp_path / "sections" / "one.tex").write_text("B \\input{sections/two}")
    (tmp_path / "sections" / "two.tex").write_text("C")
    assert " ".join(lp.load_tex(tmp_path / "main.tex").split()) == "A B C D"


def test_load_tex_drops_missing_input(tmp_path):
    (tmp_path / "main.tex").write_text("A \\input{nope} B")
    assert lp.load_tex(tmp_path / "main.tex") == "A  B"


def test_find_main(tmp_path):
    (tmp_path / "a.tex").write_text("no doc")
    (tmp_path / "ms.tex").write_text("\\documentclass{x}\\begin{document}\\end{document}")
    assert lp.find_main(tmp_path).name == "ms.tex"
    with pytest.raises(FileNotFoundError):
        lp.find_main(tmp_path / "a.tex")


# ---------- macros ----------

def test_collect_and_expand_macros():
    tex = (r"\newcommand{\model}{H-Net}\newcommand{\best}[1]{\textbf{#1}}"
           r"\def\ours{Moshi}\newcommand{\pair}[2]{#2-#1}"
           r"\newcommand{\ens}[1]{\ifmmode #1 \else $#1$\fi}"
           r"Body \model{} \best{x} \ours, \pair{a}{b} $\ens{Q}$ \modelx")
    macros, rest = lp.collect_macros(tex)
    assert set(macros) == {"model", "best", "ours", "pair", "ens"}
    assert rest.startswith("Body")
    out = lp.expand_macros(rest, macros)
    assert out == r"Body H-Net{} \textbf{x} Moshi, b-a $Q$ \modelx"


def test_expand_nested_macros():
    macros = {"a": (0, r"\b"), "b": (0, "deep")}
    assert lp.expand_macros(r"\a", macros) == "deep"


# ---------- text ----------

def test_to_text_cites_refs_math_and_unicode():
    labels = {"tab:x": "Table 3", "sec:y": "Section 4.2", "eq:z": "Equation (2)"}
    tex = (r"We use $d_{\text{model}}=512$~\citep{vaswani} (see \ref{tab:x}, \cref{sec:y}, \eqref{eq:z})."
           r"\looseness=-1 \href{http://x}{link} $\mathsf{Chunk}$ \color{red}{red}\label{foo}")
    assert lp.to_text(tex, labels) == "We use d_model=512 (see 3, Section 4.2, (2)). link Chunk red"


def test_to_text_unknown_label_marked():
    assert lp.to_text(r"Section \ref{nope}", {}) == "Section ??"


def test_to_text_keeps_display_equation_as_tagged_tex():
    out = lp.to_text("Before\n\\begin{equation}\\label{e}\n  a = b \\nonumber\n\\end{equation}\nafter", {})
    assert "[EQ] a = b" in out
    assert out.startswith("Before") and out.endswith("after")


def test_to_text_items():
    out = lp.to_text(r"\begin{itemize}\item one \item[(b)] two\end{itemize}", {})
    assert "- one" in out and "- (b) two" in out


# ---------- tables ----------

TABULAR = r"""{l|cc}
\toprule
\multirow{2}{*}{Model} & \multicolumn{2}{c}{BLEU} \\
\cmidrule[0.05pt](lr){2-3}
 & EN-DE & EN-FR \\
\midrule
\multicolumn{3}{l}{Base models} \\
ByteNet & 23.75 & \\
\multirow{2}{*}{Transformer} & 27.3 & 38.1 \\[-0.15em]
 & 28.4 & 41.8 \\
\bottomrule
"""


def test_parse_tabular_headers_rows_group_and_multirow():
    headers, rows = lp.parse_tabular(TABULAR, {})
    assert headers == ["Model", "BLEU / EN-DE", "BLEU / EN-FR"]
    assert [lp.row_text(r) for r in rows] == [
        "ByteNet (Base models): BLEU / EN-DE = 23.75",
        "Transformer (Base models): BLEU / EN-DE = 27.3; BLEU / EN-FR = 38.1",
        "Transformer (Base models): BLEU / EN-DE = 28.4; BLEU / EN-FR = 41.8",
    ]


def test_parse_tabular_bare_star_multirow_and_rule_struts():
    tab = r"{ll}\toprule \rule{0pt}{2.0ex} & h \\ \midrule \multirow{2}*{(A)} & 1 \\ & 2 \\"
    headers, rows = lp.parse_tabular(tab, {})
    assert [lp.row_text(r) for r in rows] == ["(A): h = 1", "(A): h = 2"]
    assert "0pt" not in headers[0]


def test_parse_tabular_without_rules_uses_first_row_as_header():
    headers, rows = lp.parse_tabular(r"{ll} a & b \\ x & 1 \\ y & 2", {})
    assert headers == ["a", "b"]
    assert [lp.row_text(r) for r in rows] == ["x: b = 1", "y: b = 2"]


def test_caption_skips_starred_subcaption():
    assert lp.caption_of(r"\caption*{(a) sub}\captionof{figure}{Main one}", {}) == "Main one"


def test_float_parts_splits_minipages_with_own_captions():
    inner = (r"\begin{minipage}{x}\begin{tabular}{l}a\end{tabular}\captionof{table}{T}\end{minipage}"
             r"\begin{minipage}{x}\captionof{figure}{F}\end{minipage}")
    assert [k for k, _ in lp.float_parts("figure", inner)] == ["table", "figure"]
    assert [k for k, _ in lp.float_parts("table", r"\caption{only}")] == ["table"]


# ---------- structure ----------

DOC = r"""
\begin{abstract}Short abstract.\end{abstract}
\section{Intro}\label{sec:intro}
Hello \ref{sec:method}.
\section{Method \label{sec:method}}
\subsection{Part}
Body with \eqref{eq:one}.
\begin{equation}\label{eq:one} x = 1 \end{equation}
\begin{align} a &= b \nonumber \\ c &= d \label{eq:two} \end{align}
\subsubsection*{Aside}
Aside text.
\begin{table}\section{Moved}\caption{Cap}\label{tab:a}\begin{tabular}{ll} h & v \\ \midrule r & 1 \\ \end{tabular}\end{table}
\appendix
\section{Extra}
\subsection{More}
\section*{Thanks}
"""


def test_split_sections_numbering_appendix_and_starred():
    secs = lp.split_sections(lp.hoist_headings(DOC))
    assert [(s.section_id, s.parent) for s in secs] == [
        ("abstract", None), ("1", None), ("2", None), ("2.1", "2"), ("3", None),
        ("A", None), ("A.1", "A"), ("Thanks", None),
    ]
    assert "\\paragraphtitle{Aside}" in secs[3].body  # unnumbered subsubsection folded into its parent


def test_label_map_sections_tables_equations():
    labels = lp._label_map(lp.split_sections(lp.hoist_headings(DOC)))
    assert labels["sec:intro"] == "Section 1"
    assert labels["sec:method"] == "Section 2"
    assert labels["tab:a"] == "Table 1"
    assert labels["eq:one"] == "Equation (1)"
    assert labels["eq:two"] == "Equation (2)"  # the \nonumber line is not counted


def test_display_equation_and_where_clause_join_their_paragraph(tmp_path):
    src = tmp_path / "X" / "src"
    src.mkdir(parents=True)
    (src / "main.tex").write_text(
        "\\documentclass{a}\\begin{document}\\section{S}\nWe compute:\n\n"
        "\\begin{equation} y = f(x) \\end{equation}\n\nwhere f is a map.\n\nNew paragraph.\n\\end{document}")
    paras = lp.parse_paper("X", tmp_path).paragraphs
    assert [p["text"] for p in paras] == ["We compute:\n[EQ] y = f(x)\nwhere f is a map.", "New paragraph."]
    assert [p["para_idx"] for p in paras] == [0, 1]
    assert paras[0]["has_equation"] and not paras[1]["has_equation"]


def test_inline_paragraph_titles_run_into_text():
    out = lp._inline_paragraph_titles("\\paragraphtitle{Setup.}\n\nWe train.")
    assert out == "\n\nSetup. We train."
