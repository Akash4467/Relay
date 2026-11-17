"""Phase 8, investigation A (offline): e2 vs e3 on the three papers, in memory. Every changed table cell and prose
paragraph is listed; each changed cell value is checked against the rendered PDF's text (data/raw/{paper}/paper.pdf):
the e3 value should appear on the page and the e2 value should not. Writes experiments/phase8/out/phantom_validation.json."""
import json
import re
import sys
from pathlib import Path

import pymupdf

sys.path.insert(0, ".")
sys.path.insert(0, "experiments/phase8")
from phantom_fix import parse_paper_e3  # noqa: E402

from relay.retrieval import latex_parse as lp

report = {"cells": [], "prose_changes": [], "row_headers": [], "checks_failed": []}
for code in ("AIAYN", "MOSHI", "HNET"):
    a, b = lp.parse_paper(code, extraction="e2"), parse_paper_e3(code)
    pdf = " ".join(p.get_text() for p in pymupdf.open(lp.RAW_DIR / code / "paper.pdf"))
    tokens = set(re.findall(r"\d+(?:\.\d+)?", pdf))
    if [t["table_id"] for t in a.tables] != [t["table_id"] for t in b.tables] or len(a.paragraphs) != len(b.paragraphs):
        report["checks_failed"].append(f"{code}: table or paragraph structure changed")
    for ta, tb in zip(a.tables, b.tables):
        if ta["headers"] != tb["headers"] or len(ta["rows"]) != len(tb["rows"]):
            report["checks_failed"].append(f"{code}:{ta['table_id']}: headers or row count changed")
        for ra, rb in zip(ta["rows"], tb["rows"]):
            if ra["row_header"] != rb["row_header"]:
                report["row_headers"].append({"table": f"{code}:{ta['table_id']}", "e2": ra["row_header"], "e3": rb["row_header"]})
            for col in ra["cells"]:
                va, vb = ra["cells"][col], rb["cells"].get(col)
                if va != vb:
                    in_pdf_new = all(n in tokens for n in re.findall(r"\d+(?:\.\d+)?", vb or ""))
                    in_pdf_old = all(n in tokens for n in re.findall(r"\d+(?:\.\d+)?", va))
                    report["cells"].append({"table": f"{code}:{ta['table_id']}", "row": ra["row_header"], "column": col,
                                            "e2": va, "e3": vb, "e3_numbers_in_pdf_text": in_pdf_new,
                                            "e2_numbers_in_pdf_text": in_pdf_old})
                    if not in_pdf_new:
                        report["checks_failed"].append(f"{code}:{ta['table_id']} {col}: e3 value {vb!r} not in the PDF text")
    for pa, pb in zip(a.paragraphs, b.paragraphs):
        if pa["text"] != pb["text"]:
            report["prose_changes"].append({"paper": code, "section": pa["section_id"], "e2": pa["text"][:200], "e3": pb["text"][:200]})
    for fa, fb in zip(a.figures, b.figures):
        if fa["caption"] != fb["caption"]:
            report["prose_changes"].append({"paper": code, "figure": fa["figure_id"], "e2": fa["caption"][:200], "e3": fb["caption"][:200]})
out = Path("experiments/phase8/out/phantom_validation.json")
with out.open("x", encoding="utf-8") as f:
    json.dump(report, f, indent=1, ensure_ascii=False)
for c in report["cells"]:
    print(f"{c['table']:9} {c['row'][:28]:28} {c['column'][:26]:26} {c['e2']!r:>10} -> {c['e3']!r:<8} PDF: e3 {c['e3_numbers_in_pdf_text']}, e2 {c['e2_numbers_in_pdf_text']}")
print("row headers changed:", report["row_headers"])
print("prose/caption changes:", len(report["prose_changes"]))
for p in report["prose_changes"]:
    print("  ", p)
print("CHECKS:", "ALL PASSED" if not report["checks_failed"] else report["checks_failed"])
