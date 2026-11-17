"""Limited-scope LaTeX parser for the three papers (DESIGN §6.2).

data/raw/{CODE}/src -> data/processed/{sections,paragraphs,tables,figures}.jsonl
and data/processed/bibliography_{CODE}.txt (kept out of retrieval).
Only the constructs these papers use are handled; anything else falls through to pylatexenc.
"""
from __future__ import annotations

import json
import re
import unicodedata
from contextvars import ContextVar
from dataclasses import dataclass, field
from pathlib import Path

from pylatexenc.latex2text import LatexNodes2Text

from relay.retrieval.fetch_papers import PAPERS, RAW_DIR

PROCESSED_DIR = Path("data/processed")
_L2T = LatexNodes2Text(math_mode="text", strict_latex_spaces=False)

_COMMENT = re.compile(r"(?<!\\)%[^\n]*(?:\n[ \t]*)?")
_INPUT = re.compile(r"\\(?:input|include)\s*\{([^}]+)\}")
_HEADING = re.compile(r"\\(section|subsection|subsubsection)(\*?)\s*(?:\[[^\]]*\])?\s*(?=\{)")
_DISPLAY_MATH = re.compile(
    r"\\begin\{(equation|align|gather|multline)(\*?)\}(.*?)\\end\{\1\2\}|\\\[(.*?)\\\]|\$\$(.*?)\$\$", re.S)
_FLOAT = re.compile(r"\\begin\{(table|figure)(\*?)\}(.*?)\\end\{\1\2\}", re.S)
_TABULAR = re.compile(r"\\begin\{tabular\*?\}(.*?)\\end\{tabular\*?\}", re.S)
_CITE = re.compile(r"~?\\(?:cite[a-z]*|parencite|textcite)\*?(?:\[[^\]]*\])*\{[^}]*\}")
_REF = re.compile(r"\\(ref|eqref|cref|Cref|autoref)\*?\{([^}]*)\}")
_LABEL = re.compile(r"\\label\{([^}]*)\}")
_HREF = re.compile(r"\\href\{[^}]*\}\{([^}]*)\}")
_URL = re.compile(r"\\url\{([^}]*)\}")
_HYPERREF = re.compile(r"\\hyperref\[[^\]]*\]")
# layout-only commands whose leftovers would otherwise leak as text ("=-1", "Gray7!20", "[-0.15em]")
_LAYOUT = re.compile(
    r"\\[A-Za-z]+\s*=\s*-?[\d.]+[a-z]*"
    r"|\\(?:cellcolor|color|rowcolor|columncolor)\s*(?:\[[^\]]*\])?\s*\{[^}]*\}"
    r"|\\(?:vspace|hspace)\*?\s*\{[^}]*\}|\\[vh]skip\s*-?[\d.]+[a-z]*|\\rule\s*\{[^}]*\}\s*\{[^}]*\}"
    r"|\\begin\{minipage\}\s*(?:\[[^\]]*\])*\s*\{[^}]*\}"
    r"|^\s*\[-?[\d.]+(?:em|pt|ex)\]", re.M)
# \caption* is an unnumbered sub-caption, so it is not matched and never counts as a float
_CAPTION = re.compile(r"\\caption(?:of\s*\{(table|figure)\})?(?!\s*\*)\s*(?:\[[^\]]*\])?\s*(?=\{)")
_RULES = re.compile(r"\\(?:hline|toprule|midrule|bottomrule|addlinespace|specialrule(?:\s*\{[^}]*\}){3}|cline\{[^}]*\}|cmidrule(?:\[[^\]]*\])?(?:\([^)]*\))?\{[^}]*\}|rowcolor(?:\[[^\]]*\])?\{[^}]*\})")
_EQ_SLOT = "\x00EQ{}\x00"

# Extraction versions. "e1" (default) is the behaviour that built corpora c1-c3, unchanged. "e2" (corpus c4, opt-in):
# dingbat ticks/crosses and \checkmark become ✓/✗ (pylatexenc renders \ding{51} as "51" and \checkmark as nothing);
# a table cell made of a paper macro defined as empty (MOSHI's \xmark) is kept as an explicit blank mark; \multirow
# values carry down for their exact span in every column; empty cells keep their column in the table record; a first
# column made only of marks stays a column instead of becoming the row label.
EXTRACTION: ContextVar[str] = ContextVar("extraction", default="e1")
_BLANK = "\x00BLANK\x00"  # stands for a paper macro whose definition is empty, until the cell or prose is rendered
BLANK_MARK = "(blank)"
_MARKS = {"51": "✓", "52": "✓", "55": "✗", "56": "✗"}  # pifont \ding: 51/52 check marks, 55/56 crosses


def map_symbols(tex: str) -> str:
    """e2: \\ding ticks and crosses as ✓/✗, \\checkmark as ✓; any other \\ding{N} as [symbol], never the bare N."""
    tex = re.sub(r"\\ding\s*\{\s*(\d+)\s*\}", lambda m: _MARKS.get(m.group(1), "[symbol]"), tex)
    return re.sub(r"\\checkmark(?![A-Za-z])", "✓", tex)


# ---------- low-level helpers ----------

def group(s: str, i: int) -> tuple[str, int]:
    """Balanced {...} starting at s[i] == '{'. Returns (content, index after closing brace)."""
    if i >= len(s) or s[i] != "{":
        raise ValueError(f"expected '{{' at {i}")
    depth, j = 0, i
    while j < len(s):
        c = s[j]
        if c == "\\":  # skip the escaped character: \{ \} and \\ never count
            j += 2
            continue
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return s[i + 1:j], j + 1
        j += 1
    raise ValueError("unbalanced braces")


def strip_comments(tex: str) -> str:
    # like TeX, a comment also eats its newline and the next line's leading spaces
    return _COMMENT.sub("", tex.replace("\r\n", "\n"))


def load_tex(path: Path, root: Path | None = None, _depth: int = 0) -> str:
    """Read a .tex file with comments removed and \\input/\\include resolved recursively.

    \\input paths are relative to the main file's directory (root), not the including file's.
    """
    if _depth > 10:
        raise RecursionError(f"\\input nesting too deep at {path}")
    root = root or path.parent
    text = strip_comments(path.read_text(encoding="utf-8", errors="replace"))

    def sub(m: re.Match) -> str:
        cand = root / m.group(1).strip()
        for c in (cand, cand.with_name(cand.name + ".tex")):
            if c.is_file():
                return load_tex(c, root, _depth + 1)
        return ""  # missing file (e.g. a generated figure): drop the \input
    return _INPUT.sub(sub, text)


def find_main(src: Path) -> Path:
    for p in sorted(src.rglob("*.tex")):
        t = p.read_text(encoding="utf-8", errors="replace")
        if "\\documentclass" in t and "\\begin{document}" in t:
            return p
    raise FileNotFoundError(f"no main .tex in {src}")


# ---------- macros ----------

_DEF = re.compile(r"\\(?:re)?newcommand\*?\s*\{?\\([A-Za-z]+)\}?\s*(?:\[(\d)\])?\s*(?=\{)|\\def\\([A-Za-z]+)\s*(?=\{)")


def collect_macros(tex: str) -> tuple[dict[str, tuple[int, str]], str]:
    """Paper-defined \\newcommand/\\def macros with ≤ 2 args. Returns (macros, tex without the definitions)."""
    macros: dict[str, tuple[int, str]] = {}
    out, pos = [], 0
    for m in _DEF.finditer(tex):
        if m.start() < pos:
            continue
        name = m.group(1) or m.group(3)
        nargs = int(m.group(2) or 0)
        try:
            body, end = group(tex, m.end())
        except ValueError:
            continue
        if nargs <= 2:
            # \ifmmode A \else $A$ \fi (MOSHI's \ensuremathmode): pylatexenc would print both branches
            body = re.sub(r"\\ifmmode(.*?)\\else.*?\\fi", lambda m: m.group(1).strip(), body, flags=re.S)
            macros[name] = (nargs, body)
        out.append(tex[pos:m.start()])
        pos = end
    out.append(tex[pos:])
    return macros, "".join(out)


def expand_macros(tex: str, macros: dict[str, tuple[int, str]], passes: int = 3) -> str:
    if not macros:
        return tex
    pat = re.compile(r"\\(" + "|".join(sorted(map(re.escape, macros), key=len, reverse=True)) + r")(?![A-Za-z])")
    for _ in range(passes):
        out, pos, changed = [], 0, False
        for m in pat.finditer(tex):
            if m.start() < pos:
                continue
            nargs, body = macros[m.group(1)]
            end, args = m.end(), []
            try:
                for _ in range(nargs):
                    while end < len(tex) and tex[end].isspace():
                        end += 1
                    a, end = group(tex, end)
                    args.append(a)
            except ValueError:
                continue
            for k, a in enumerate(args, 1):
                body = body.replace(f"#{k}", a)
            out.append(tex[pos:m.start()])
            out.append(body)
            pos, changed = end, True
        out.append(tex[pos:])
        tex = "".join(out)
        if not changed:
            break
    return tex


# ---------- text conversion ----------

def to_text(tex: str, labels: dict[str, str], keep_blank: bool = False) -> str:
    if EXTRACTION.get() == "e2":
        tex = map_symbols(tex)
    tex = _HYPERREF.sub("", _CITE.sub("", tex))
    tex = _HREF.sub(r"\1", _URL.sub(lambda m: m.group(1).replace("_", r"\_"), tex))  # pylatexenc crashes on \href
    tex = _REF.sub(lambda m: _ref_text(m.group(1), m.group(2), labels), tex)
    tex = _LABEL.sub("", _LAYOUT.sub("", tex))
    eqs: list[str] = []

    def keep_eq(m: re.Match) -> str:
        body = next(g for g in (m.group(3), m.group(4), m.group(5)) if g is not None)
        body = _LABEL.sub("", body).replace("\\nonumber", "")
        eqs.append("[EQ] " + " ".join(body.split()))
        return " " + _EQ_SLOT.format(len(eqs) - 1) + " "

    tex = _DISPLAY_MATH.sub(keep_eq, tex)
    tex = re.sub(r"\\item(?:\[([^\]]*)\])?", lambda m: f"\n- {m.group(1) + ' ' if m.group(1) else ''}", tex)
    text = _L2T.latex_to_text(tex)
    for k, eq in enumerate(eqs):
        text = text.replace(_EQ_SLOT.format(k), eq)
    # NFKC folds math alphabets (\ud835\udda2\ud835\uddc1\ud835\uddce\ud835\uddc7\ud835\uddc4 -> Chunk) so BM25 and the verifier see plain words
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"\s*\n(?!- |\[EQ\])\s*", " ", text)  # source line breaks are not meaningful; list items are
    if not keep_blank:
        text = text.replace(_BLANK, "")  # only e2 ever inserts it; prose and captions show the PDF's blank
    return re.sub(r"[ \t]+", " ", re.sub(r" *\n *", "\n", text)).strip()


def _ref_text(kind: str, keys: str, labels: dict[str, str]) -> str:
    parts = []
    for key in (k.strip() for k in keys.split(",")):
        target = labels.get(key, "??")
        if kind in ("cref", "Cref", "autoref"):
            parts.append(target)
        else:
            parts.append(target.split(" ", 1)[-1])
    out = ", ".join(parts)
    return out


# ---------- tables ----------

def parse_tabular(body: str, labels: dict[str, str]) -> tuple[list[str], list[dict]]:
    spec_end = group(body, body.index("{"))[1] if body.lstrip().startswith("{") else 0
    body = body[spec_end:]
    raw_rows = re.split(r"\\\\(?:\s*\[[^\]]*\])?", body)
    header_rows: list[list[str]] = []
    data_rows: list[list[str]] = []
    in_data = False
    for raw in raw_rows:
        # the first \midrule or \hline after some content separates header rows from data rows
        if header_rows and re.search(r"\\(?:midrule|hline)", raw):
            in_data = True
        cells = [c for part in re.split(r"(?<!\\)&", _RULES.sub("", raw)) for c in _cell(part, labels)]
        if not any(cells):
            continue
        (data_rows if in_data else header_rows).append(cells)
    if not data_rows and header_rows:  # no rule separated header from data: first row is the header
        header_rows, data_rows = header_rows[:1], header_rows[1:]
    ncols = max((len(r) for r in header_rows + data_rows), default=0)
    headers = []
    for i in range(ncols):
        parts = []
        for hr in header_rows:
            h = hr[i] if i < len(hr) else ""
            if h and (not parts or parts[-1] != h):
                parts.append(h)
        headers.append(" / ".join(parts) or f"col{i + 1}")
    rows, prev, group_label = [], [""] * ncols, ""
    for r in data_rows:
        r = r + [""] * (ncols - len(r))
        filled = [c for c in r if c]
        if ncols > 2 and len(set(filled)) == 1 and len(filled) >= 2:
            group_label = r[0]  # a \multicolumn spanning the whole row labels the rows below it
            continue
        if not r[0] and prev[0]:  # \multirow leaves the following cells empty: fill the row header down
            r[0] = prev[0]
        prev = r
        header = f"{r[0]} ({group_label})" if group_label else r[0]
        rows.append({"row_header": header, "cells": {headers[i]: r[i] for i in range(1, ncols) if r[i]}})
    return headers, rows


def _cell(tex: str, labels: dict[str, str]) -> list[str]:
    tex = _LAYOUT.sub("", tex).strip()  # a leading \rule strut would hide \multicolumn/\multirow
    m = re.match(r"\\multicolumn\s*\{(\d+)\}\s*\{[^}]*\}\s*(?=\{)", tex)
    if m:
        content, _ = group(tex, m.end())
        return [to_text(content, labels)] * int(m.group(1))
    # \multirow{n}{width}{text}; AIAYN writes the width as a bare * (\multirow{4}*{(A)})
    m = re.match(r"\\multirow\s*\{[^}]*\}\s*(?:\[[^\]]*\])?\s*(?:\{[^}]*\}|\*)\s*(?=\{)", tex)
    if m:
        tex = group(tex, m.end())[0]
    return [to_text(tex, labels)]


def _cell_spans(tex: str, labels: dict[str, str]) -> list[tuple[str, int]]:
    """e2: like _cell, with the rows each value spans: \\multirow{n} -> n (downward), \\multirow{-n} -> -n (this row and
    the n-1 rows above, as LaTeX draws it), otherwise 1."""
    tex = _LAYOUT.sub("", tex).strip()
    m = re.match(r"\\multicolumn\s*\{(\d+)\}\s*\{[^}]*\}\s*(?=\{)", tex)
    if m:
        content, _ = group(tex, m.end())
        return [(to_text(content, labels, keep_blank=True), 1)] * int(m.group(1))
    m = re.match(r"\\multirow\s*\{\s*(-?\d+)\s*\}\s*(?:\[[^\]]*\])?\s*(?:\{[^}]*\}|\*)\s*(?=\{)", tex)
    if m:
        return [(to_text(group(tex, m.end())[0], labels, keep_blank=True), int(m.group(1)) or 1)]
    return [(to_text(tex, labels, keep_blank=True), 1)]


_MARK_VALUES = {"✓", "✗", BLANK_MARK, ""}


def parse_tabular_e2(body: str, labels: dict[str, str]) -> tuple[list[str], list[dict]]:
    """e2 version of parse_tabular: same header/data split and headers; \\multirow values carry down for their span in
    every column; every column keeps its place in each row's cells ("" when empty; "(blank)" for an empty-defined
    mark macro); a first column of only marks stays a column and the row is labelled "Row k"."""
    spec_end = group(body, body.index("{"))[1] if body.lstrip().startswith("{") else 0
    body = body[spec_end:]
    header_rows: list[list[tuple[str, int]]] = []
    data_rows: list[list[tuple[str, int]]] = []
    in_data = False
    for raw in re.split(r"\\\\(?:\s*\[[^\]]*\])?", body):
        if header_rows and re.search(r"\\(?:midrule|hline)", raw):
            in_data = True
        cells = [c for part in re.split(r"(?<!\\)&", _RULES.sub("", raw)) for c in _cell_spans(part, labels)]
        if not any(t for t, _ in cells):
            continue
        (data_rows if in_data else header_rows).append(cells)
    if not data_rows and header_rows:
        header_rows, data_rows = header_rows[:1], header_rows[1:]
    ncols = max((len(r) for r in header_rows + data_rows), default=0)
    headers = []
    for i in range(ncols):
        parts = []
        for hr in header_rows:
            h = (hr[i][0] if i < len(hr) else "").replace(_BLANK, "")
            if h and (not parts or parts[-1] != h):
                parts.append(h)
        headers.append(" / ".join(parts) or f"col{i + 1}")
    grid = [[t for t, _ in r] + [""] * (ncols - len(r)) for r in data_rows]
    spans = [[n for _, n in r] + [1] * (ncols - len(r)) for r in data_rows]
    for j, row in enumerate(grid):  # a \multirow value fills the empty cells it spans, downward or upward
        for i, n in enumerate(spans[j]):
            if not row[i] or n == 1:
                continue
            for k in (range(j + 1, min(len(grid), j + n)) if n > 1 else range(max(0, j + n + 1), j)):
                if grid[k][i] == "":
                    grid[k][i] = row[i]
    col0 = [r[0] for r in grid]
    mark_first = bool(grid) and all(t in _MARK_VALUES or t == _BLANK for t in col0) and any(col0)
    rows, prev0, group_label = [], "", ""
    for j, texts in enumerate(grid):
        filled = [c for c in texts if c]
        if ncols > 2 and len(set(filled)) == 1 and len(filled) >= 2 and spans[j][0] == 1 and not mark_first:
            group_label = texts[0]  # a \multicolumn spanning the whole row labels the rows below it
            continue
        texts = list(texts)
        if not mark_first and not texts[0] and prev0:  # as in e1: an empty row header continues the one above
            texts[0] = prev0
        prev0 = texts[0]
        texts = [BLANK_MARK if t == _BLANK else t for t in texts]
        start = 0 if mark_first else 1
        label = f"Row {len(rows) + 1}" if mark_first else texts[0]
        header = f"{label} ({group_label})" if group_label else label
        rows.append({"row_header": header, "cells": {headers[i]: texts[i] for i in range(start, ncols)}})
    return headers, rows


def row_text(row: dict) -> str:
    # an empty cell keeps its place in the record but is not spelled out (e1 rows never hold one)
    cells = "; ".join(f"{h} = {v}" for h, v in row["cells"].items() if v != "")
    return f"{row['row_header']}: {cells}" if cells else row["row_header"]


def caption_of(tex: str, labels: dict[str, str]) -> str:
    m = _CAPTION.search(tex)
    return to_text(group(tex, m.end())[0], labels) if m else ""


# ---------- document walk ----------

@dataclass
class Section:
    section_id: str
    title: str
    parent: str | None
    order: int
    body: str = ""


@dataclass
class Parsed:
    sections: list[dict] = field(default_factory=list)
    paragraphs: list[dict] = field(default_factory=list)
    tables: list[dict] = field(default_factory=list)
    figures: list[dict] = field(default_factory=list)
    bibliography: str = ""


def _alnum(title: str) -> str:
    return re.sub(r"[^0-9A-Za-z]", "", title.title()) or "Unnumbered"


def split_sections(body: str) -> list[Section]:
    """Split the document body on \\section/\\subsection/\\subsubsection with PDF-style numbering."""
    sections: list[Section] = []
    m_abs = re.search(r"\\begin\{abstract\}(.*?)\\end\{abstract\}", body, re.S)
    if m_abs:
        sections.append(Section("abstract", "Abstract", None, 0, m_abs.group(1)))
        body = body[:m_abs.start()] + body[m_abs.end():]
    preface = body
    counters = [0, 0, 0]
    appendix_at = body.find("\\appendix")
    lettered = False
    heads = list(_HEADING.finditer(body))
    if heads:
        preface = body[:heads[0].start()]
    if preface.strip() and sections:
        sections[0].body += "\n\n" + preface  # text before the first heading (e.g. HNET's abstract file)
    for k, h in enumerate(heads):
        level = {"section": 0, "subsection": 1, "subsubsection": 2}[h.group(1)]
        title_tex, end = group(body, h.end())
        title = to_text(title_tex, {})
        nxt = heads[k + 1].start() if k + 1 < len(heads) else len(body)
        in_appendix = 0 <= appendix_at < h.start()
        if h.group(2) == "*" and level and sections:
            # unnumbered subsection: no own ID in the PDF, so it becomes a run-in title in its parent
            sections[-1].body += f"\n\n\\paragraphtitle{{{title_tex}}}" + body[end:nxt]
            continue
        if h.group(2) == "*":
            parent = None
            sid = _alnum(title)
        else:
            if in_appendix and not lettered:  # \appendix restarts numbering as A, B, ...
                counters, lettered = [0, 0, 0], True
            counters[level] += 1
            counters[level + 1:] = [0] * (2 - level)
            first = chr(ord("A") + counters[0] - 1) if in_appendix else str(counters[0])
            sid = ".".join([first] + [str(c) for c in counters[1:level + 1]])
            parent = sid.rsplit(".", 1)[0] if level else None
        title_labels = "".join(f"\\label{{{k}}}" for k in _LABEL.findall(title_tex))  # \section{X \label{..}}
        sections.append(Section(sid, title, parent, len(sections),
                                title_labels + body[end:nxt].replace("\\appendix", "")))
    return sections


def hoist_headings(body: str) -> str:
    """MOSHI puts \\section{...} inside \\begin{table}; move such headings in front of their float."""
    def fix(m: re.Match) -> str:
        inner, heads = m.group(0), []
        for h in reversed(list(_HEADING.finditer(inner))):
            end = group(inner, h.end())[1]
            heads.insert(0, inner[h.start():end])
            inner = inner[:h.start()] + inner[end:]
        return "\n\n".join(heads) + "\n\n" + inner if heads else inner
    return _FLOAT.sub(fix, body)


def float_parts(kind: str, inner: str) -> list[tuple[str, str]]:
    """One (kind, tex) per caption in a float. HNET puts \\captionof{table} in minipages inside figures."""
    if len(_CAPTION.findall(inner)) < 2:
        return [(kind, inner)]
    parts = []
    for seg in re.split(r"\\begin\{minipage\}", inner):
        m = _CAPTION.search(seg)
        if m:
            parts.append((m.group(1) or kind, seg))
    return parts


def _label_map(sections: list[Section]) -> dict[str, str]:
    labels: dict[str, str] = {}
    t = f = eq = 0
    for s in sections:
        # any label outside floats and equations names the section it sits in
        for key in _LABEL.findall(_DISPLAY_MATH.sub("", _FLOAT.sub("", s.body))):
            labels[key] = f"Section {s.section_id}"
        for m in _DISPLAY_MATH.finditer(s.body):
            if not m.group(1) or m.group(2):  # \[..\], $$..$$ and starred environments are unnumbered
                continue
            lines = re.split(r"\\\\", m.group(3)) if m.group(1) in ("align", "gather") else [m.group(3)]
            for line in lines:
                if re.search(r"\\(?:nonumber|notag)", line) or not line.strip():
                    continue
                eq += 1
                for key in _LABEL.findall(line):
                    labels[key] = f"Equation ({eq})"
        for fl in _FLOAT.finditer(s.body):
            for kind, seg in float_parts(fl.group(1), fl.group(3)):
                if kind == "table":
                    t += 1
                    name = f"Table {t}"
                else:
                    f += 1
                    name = f"Figure {f}"
                for key in _LABEL.findall(seg):
                    labels[key] = name
    return labels


def parse_paper(code: str, raw_dir: Path = RAW_DIR, extraction: str = "e1") -> Parsed:
    if extraction not in ("e1", "e2"):
        raise ValueError(f"unknown extraction version {extraction!r}")
    token = EXTRACTION.set(extraction)
    try:
        return _parse_paper(code, raw_dir)
    finally:
        EXTRACTION.reset(token)


def _parse_paper(code: str, raw_dir: Path) -> Parsed:
    src = raw_dir / code / "src"
    main = find_main(src)
    tex = load_tex(main)
    macros, tex = collect_macros(tex)
    if EXTRACTION.get() == "e2":  # an argument-less macro defined as empty (\newcommand{\xmark}{}) is a blank mark
        macros = {k: ((n, _BLANK) if n == 0 and not body.strip() else (n, body)) for k, (n, body) in macros.items()}
    m = re.search(r"\\begin\{document\}(.*)\\end\{document\}", tex, re.S)
    body = m.group(1) if m else tex

    out = Parsed()
    bib = re.search(r"\\begin\{thebibliography\}.*?\\end\{thebibliography\}", body, re.S)
    if bib:
        out.bibliography = bib.group(0)
        body = body.replace(bib.group(0), "")
    elif (src / "main.bbl").exists():
        out.bibliography = (src / "main.bbl").read_text(encoding="utf-8", errors="replace")
    body = re.sub(r"\\(?:printbibliography|bibliographystyle|bibliography)(?:\[[^\]]*\])?(?:\{[^}]*\})?", "", body)
    body = re.sub(r"\\(?:maketitle|tableofcontents|newpage|clearpage|FloatBarrier)\b", "", body)
    body = hoist_headings(expand_macros(body, macros).replace("\\xspace", ""))

    sections = split_sections(body)
    labels = _label_map(sections)
    t_no = f_no = 0
    for s in sections:
        out.sections.append({"paper": code, "section_id": s.section_id, "title": s.title,
                             "parent": s.parent, "order": s.order})
        prose = s.body
        for fl in _FLOAT.finditer(s.body):
            for kind, inner in float_parts(fl.group(1), fl.group(3)):
                cap = caption_of(inner, labels)
                label = next(iter(_LABEL.findall(inner)), None)
                if kind == "table":
                    t_no += 1
                    headers, rows = [], []
                    parse = parse_tabular_e2 if EXTRACTION.get() == "e2" else parse_tabular
                    for tab in _TABULAR.finditer(inner):
                        h, r = parse(tab.group(1), labels)
                        headers = headers or h
                        rows += r
                    out.tables.append({"paper": code, "table_id": f"T{t_no}", "label": label,
                                       "section_id": s.section_id, "caption": cap, "headers": headers,
                                       "rows": rows, "row_texts": [row_text(r) for r in rows]})
                else:
                    f_no += 1
                    out.figures.append({"paper": code, "figure_id": f"F{f_no}", "label": label,
                                        "section_id": s.section_id, "caption": cap, "type": "figure_caption"})
            prose = prose.replace(fl.group(0), "\n\n")
        prose = re.sub(r"\\paragraph\*?\s*(?=\{)", r"\\paragraphtitle", prose)
        prose = _inline_paragraph_titles(prose)
        mine: list[dict] = []
        for block in re.split(r"\n\s*\n", prose):
            text = to_text(block, labels)
            if not text:
                continue
            prev = mine[-1]["text"] if mine else ""
            # a display equation set off by blank lines, and the "where ..." text after it, belong to the
            # paragraph that introduces the equation
            if mine and (text.startswith("[EQ]") or (text[0].islower() and "[EQ]" in prev.rsplit("\n", 1)[-1])):
                mine[-1]["text"] = prev + "\n" + text
                mine[-1]["has_equation"] = True
                continue
            mine.append({"paper": code, "section_id": s.section_id, "para_idx": len(mine),
                         "text": text, "has_equation": "[EQ]" in text})
        out.paragraphs += mine
    return out


def _inline_paragraph_titles(tex: str) -> str:
    """\\paragraph{Title} is an unnumbered run-in heading: keep it as 'Title.' at the start of its paragraph."""
    out, pos = [], 0
    for m in re.finditer(r"\\paragraphtitle(?=\{)", tex):
        title, end = group(tex, m.end())
        out.append(tex[pos:m.start()])
        title = title.strip().rstrip(".")
        out.append(f"\n\n{title}. ")
        pos = end + len(re.match(r"\s*", tex[end:]).group(0))  # the title runs into its text, even after a blank line
    out.append(tex[pos:])
    return "".join(out)


def write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main(raw_dir: Path = RAW_DIR, out_dir: Path = PROCESSED_DIR, extraction: str = "e1") -> dict[str, Parsed]:
    parsed = {code: parse_paper(code, raw_dir, extraction) for code in PAPERS}
    for kind in ("sections", "paragraphs", "tables", "figures"):
        write_jsonl(out_dir / f"{kind}.jsonl", [r for p in parsed.values() for r in getattr(p, kind)])
    for code, p in parsed.items():
        (out_dir / f"bibliography_{code}.txt").write_text(p.bibliography, encoding="utf-8")
    return parsed


if __name__ == "__main__":
    for code, p in main().items():
        print(code, {k: len(getattr(p, k)) for k in ("sections", "paragraphs", "tables", "figures")})
