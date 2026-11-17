"""Phase 9A, investigation A: experimental extraction e3 = e2 + invisible-spacing removal.

pylatexenc prints the argument of \\phantom ("$\\phantom{1}42.2$" -> "142.2"), although LaTeX renders it as blank
space. e3 removes every \\phantom, \\hphantom and \\vphantom command together with its brace-matched argument before
conversion (nested phantoms go with their parent; unbalanced braces leave the text unchanged). It wraps
relay.retrieval.latex_parse in-process and restores it on exit; the production parser, e1 and e2 are untouched.

    uv run python experiments/phase9/extraction_e3.py      # validation -> experiments/phase9/out/e3_changes.json

Validation: e2 vs e3 on all three papers (every section, paragraph, figure caption, table caption, header, row
header and cell). Every changed cell is checked on its PDF page: the row's printed cells, in order, must appear as
consecutive lines of that table's region of the page with the e3 value, and must not with the e2 value."""
from __future__ import annotations

import json
import re
import sys
from contextlib import contextmanager
from pathlib import Path

sys.path.insert(0, ".")
from relay.retrieval import latex_parse as lp

_PHANTOM = re.compile(r"\\[hv]?phantom\s*(?=\{)")
OUT = Path("experiments/phase9/out")
EXPECTED = [("MOSHI", "T5", "Perplexity", "142.2", "42.2"), ("MOSHI", "T5", "Perplexity", "140.3", "40.3"),
            ("MOSHI", "T5", "Perplexity", "136.8", "36.8")] + \
           [("MOSHI", "T6", "Semantic Token / Weight", "001.0", "1.0")] * 3
# found in phase 9 (missed by phase 8, which compared figure but not table captions): MOSHI Table 18's caption has
# three \colorbox{..}{\phantom{x}} colour swatches; e2 prints them as "x", the PDF prints a coloured box (no text)
EXPECTED_CAPTIONS = ["T18"]


def words(s: str) -> list[str]:
    return re.findall(r"\w+|[^\w\s]", s)


def removed_words(old: str, new: str) -> list[tuple[str, str, str]]:
    """(left neighbour, removed word, right neighbour in the new text) for every word of old missing from new."""
    import difflib
    a, b = words(old), words(new)
    out = []
    for op, i1, i2, j1, j2 in difflib.SequenceMatcher(a=a, b=b, autojunk=False).get_opcodes():
        if op == "equal":
            continue
        if op != "delete" or i1 == 0 or j1 >= len(b):
            return []  # anything but a pure deletion inside the text is not the expected kind of change
        out += [(a[i1 - 1], w, b[j1]) for w in a[i1:i2]]
    return out


def strip_phantoms(tex: str) -> str:
    """Remove invisible-spacing commands and their brace-matched arguments; everything else is kept as is."""
    out, pos = [], 0
    for m in _PHANTOM.finditer(tex):
        if m.start() < pos:  # inside an argument already removed (nested phantom)
            continue
        try:
            _, end = lp.group(tex, m.end())
        except ValueError:  # unbalanced braces: leave the text alone
            continue
        out.append(tex[pos:m.start()])
        pos = end
    out.append(tex[pos:])
    return "".join(out)


class _PhantomFreeL2T:
    def __init__(self, inner):
        self.inner = inner

    def latex_to_text(self, tex, *a, **k):
        return self.inner.latex_to_text(strip_phantoms(tex), *a, **k)


@contextmanager
def e3():
    """Within this block latex_parse converts with phantoms removed (process-local, restored on exit)."""
    original = lp._L2T
    lp._L2T = _PhantomFreeL2T(original)
    try:
        yield
    finally:
        lp._L2T = original


def parse_paper_e3(code: str, raw_dir: Path = lp.RAW_DIR) -> lp.Parsed:
    with e3():
        return lp.parse_paper(code, raw_dir, extraction="e2")


def printed_cells(row: dict) -> list[str]:
    """The row as the PDF prints it: header, then every non-empty cell (a blank mark prints nothing)."""
    return [row["row_header"]] + [v for v in row["cells"].values() if v not in ("", "(blank)")]


def table_region(pages: list[str], number: int) -> tuple[int, list[str]] | None:
    """(1-based page, lines) from the caption 'Table N:' to the next 'Table M:' caption on that page."""
    for i, text in enumerate(pages):
        k = text.find(f"Table {number}:")
        if k >= 0:
            rest = text[k + 1:]
            nxt = re.search(r"\nTable \d+:", rest)
            region = rest[:nxt.start()] if nxt else rest
            return i + 1, [l.strip() for l in region.split("\n")]
    return None


def row_in_lines(cells: list[str], lines: list[str]) -> bool:
    n = len(cells)
    return any(lines[i:i + n] == cells for i in range(len(lines) - n + 1))


def diff(a: lp.Parsed, b: lp.Parsed) -> dict:
    """Every difference between two parses of one paper, by kind."""
    out = {"structure": [], "sections": [], "paragraphs": [], "figures": [], "tables": [], "cells": []}
    for kind in ("sections", "paragraphs", "figures", "tables"):
        if len(getattr(a, kind)) != len(getattr(b, kind)):
            out["structure"].append(f"{kind}: {len(getattr(a, kind))} vs {len(getattr(b, kind))}")
    for kind in ("sections", "paragraphs", "figures"):
        out[kind] = [{"e2": x, "e3": y} for x, y in zip(getattr(a, kind), getattr(b, kind)) if x != y]
    for ta, tb in zip(a.tables, b.tables):
        for key in ("table_id", "label", "section_id", "caption", "headers"):
            if ta[key] != tb[key]:
                out["tables"].append({"table": ta["table_id"], "field": key, "e2": ta[key], "e3": tb[key]})
        if len(ta["rows"]) != len(tb["rows"]):
            out["tables"].append({"table": ta["table_id"], "field": "row count"})
        for i, (ra, rb) in enumerate(zip(ta["rows"], tb["rows"])):
            if ra["row_header"] != rb["row_header"]:
                out["tables"].append({"table": ta["table_id"], "field": f"row {i} header", "e2": ra["row_header"], "e3": rb["row_header"]})
            if list(ra["cells"]) != list(rb["cells"]):
                out["tables"].append({"table": ta["table_id"], "field": f"row {i} columns"})
            for col in ra["cells"]:
                if ra["cells"][col] != rb["cells"].get(col):
                    out["cells"].append({"table": ta["table_id"], "row_index": i, "row": ra["row_header"], "column": col,
                                         "old": ra["cells"][col], "new": rb["cells"].get(col),
                                         "_rows": (ra, rb)})
    return out


def validate() -> dict:
    import pymupdf

    report = {"extraction": "e3", "parent": "e2", "papers": {}, "cells": [], "captions": [], "checks": {}}
    for code in lp.__dict__.get("PAPERS", None) or ("AIAYN", "MOSHI", "HNET"):
        a, b = lp.parse_paper(code, lp.RAW_DIR, extraction="e2"), parse_paper_e3(code)
        d = diff(a, b)
        pages = [p.get_text() for p in pymupdf.open(lp.RAW_DIR / code / "paper.pdf")]
        for c in d.pop("cells"):
            ra, rb = c.pop("_rows")
            reg = table_region(pages, int(c["table"][1:]))
            c |= {"paper": code, "pdf_page": reg[0] if reg else None,
                  "pdf_row_matches_new": bool(reg) and row_in_lines(printed_cells(rb), reg[1]),
                  "pdf_row_matches_old": bool(reg) and row_in_lines(printed_cells(ra), reg[1]),
                  "only_leading_characters_removed": c["old"].endswith(c["new"]) and c["new"] != ""}
            report["cells"].append(c)
        for t in [t for t in d["tables"] if t["field"] == "caption"]:
            d["tables"].remove(t)
            reg = table_region(pages, int(t["table"][1:]))
            text = " ".join(words(" ".join(reg[1]))) if reg else ""
            removed = removed_words(t["e2"], t["e3"])
            report["captions"].append(t | {"paper": code, "pdf_page": reg[0] if reg else None, "removed": removed,
                                           "pdf_matches_e3_not_e2": bool(removed) and all(
                                               f"{l} {w}" not in text and f"{l} {r}" in text for l, w, r in removed)})
        report["papers"][code] = d | {"n_tables": len(b.tables), "n_paragraphs": len(b.paragraphs)}
    got = sorted((c["paper"], c["table"], c["column"], c["old"], c["new"]) for c in report["cells"])
    report["checks"] = {
        "exactly_the_six_expected_changes": got == sorted(EXPECTED),
        "every_changed_row_matches_pdf_with_e3": all(c["pdf_row_matches_new"] for c in report["cells"]),
        "no_changed_row_matches_pdf_with_e2": not any(c["pdf_row_matches_old"] for c in report["cells"]),
        "only_leading_invisible_digits_removed": all(c["only_leading_characters_removed"] for c in report["cells"]),
        "caption_changes_are_the_expected_ones": sorted(c["table"] for c in report["captions"]) == sorted(EXPECTED_CAPTIONS),
        "every_caption_change_matches_pdf": all(c["pdf_matches_e3_not_e2"] for c in report["captions"]),
        "no_other_differences": all(not v for p in report["papers"].values()
                                    for k, v in p.items() if k in ("structure", "sections", "paragraphs", "figures", "tables")),
    }
    report["all_checks_passed"] = all(report["checks"].values())
    return report


if __name__ == "__main__":
    r = validate()
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "e3_changes.json").write_text(json.dumps(r, indent=1, ensure_ascii=False), encoding="utf-8")
    for c in r["cells"]:
        print(f"{c['paper']} {c['table']} p{c['pdf_page']} {c['row']} | {c['column']}: {c['old']} -> {c['new']} "
              f"| PDF row e3 {c['pdf_row_matches_new']} e2 {c['pdf_row_matches_old']}")
    print(json.dumps(r["checks"], indent=1), "\nALL CHECKS PASSED" if r["all_checks_passed"] else "\nCHECKS FAILED")
    sys.exit(0 if r["all_checks_passed"] else 1)
