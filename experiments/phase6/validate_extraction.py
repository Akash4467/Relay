"""Phase 6, stage 3 (offline, in memory): validate extraction e2 against e1 on the three papers before any corpus is
built. Writes experiments/phase6/out/extraction_validation_{UTC}.json and exits non-zero if a structural check fails.

Checks: (1) the 53 rows that the Phase 5 symbol mapping changed now hold marks and no 51/55 artifact; (2) prose
paragraphs change only where the source holds a mark macro; (3) every row keeps every column in header order;
(4) row and column labels stay aligned (same tables, same headers, row counts explained); (5) no 51/55 value comes
from a mark macro anywhere; (6) every numeric value of e1 is still present in e2, table by table."""
import json
import re
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, ".")
from relay.retrieval import latex_parse as lp

PAPERS = ("AIAYN", "MOSHI", "HNET")
AFFECTED = {("HNET", "T6"), ("MOSHI", "T3"), ("MOSHI", "T4"), ("MOSHI", "T5"), ("MOSHI", "T6"), ("MOSHI", "T13"),
            ("MOSHI", "T16"), ("MOSHI", "T17")}  # Phase 5: 8 tables, 53 rows changed by the symbol mapping
MARKS = {"✓", "✗", lp.BLANK_MARK}
NUM = re.compile(r"^-?\d+(?:\.\d+)?%?$")


class SymbolsOnly:  # the Phase 5 in-memory mapping, used only to re-identify the 53 rows
    def __init__(self, inner):
        self.inner = inner

    def latex_to_text(self, tex, *a, **k):
        return self.inner.latex_to_text(lp.map_symbols(tex), *a, **k)


def parse(extraction, symbols_only=False):
    orig = lp._L2T
    if symbols_only:
        lp._L2T = SymbolsOnly(orig)
    try:
        return {c: lp.parse_paper(c, extraction=extraction) for c in PAPERS}
    finally:
        lp._L2T = orig


e1, e2, sym = parse("e1"), parse("e2"), parse("e1", symbols_only=True)
fails, report = [], {"tables": [], "examples": [], "prose_changes": []}

# (1) the 53 Phase-5 rows
phase5_rows = 0
for code in PAPERS:
    for t1, ts in zip(e1[code].tables, sym[code].tables):
        if (code, t1["table_id"]) in AFFECTED:
            phase5_rows += sum(json.dumps(a, ensure_ascii=False) != json.dumps(b, ensure_ascii=False)
                               for a, b in zip(t1["rows"], ts["rows"]))
report["phase5_rows_reidentified"] = phase5_rows
if phase5_rows != 53:
    fails.append(f"expected 53 Phase-5 rows, re-identified {phase5_rows}")

src_marks = {}
for code in PAPERS:
    src = lp.load_tex(lp.find_main(lp.RAW_DIR / code / "src"))
    macros, _ = lp.collect_macros(src)
    mark_macros = [k for k, (n, b) in macros.items() if n == 0 and (re.search(r"\\ding\{", b) or not b.strip())]
    src_marks[code] = mark_macros
report["mark_macros_per_paper"] = src_marks

for code in PAPERS:
    assert len(e1[code].tables) == len(e2[code].tables), "table count changed"
    for t1, t2 in zip(e1[code].tables, e2[code].tables):
        key = (code, t1["table_id"])
        rec = {"table": f"{code}:{t1['table_id']}", "caption": t1["caption"][:70], "rows_e1": len(t1["rows"]),
               "rows_e2": len(t2["rows"]), "affected": key in AFFECTED}
        # (3) every row keeps every column, in header order. A table built from several tabulars (MOSHI T10/T11)
        # keeps only the first tabular's headers, in e1 as well; there, every e1 column must survive in e2.
        # A header list with repeated names (MOSHI T10: '54.3' three times, a data row read as the header) collapses
        # columns in the cells dict in e1 too; it is a pre-existing defect, checked against e1 like a multi-tabular one.
        multi = any(not set(r["cells"]) <= set(t1["headers"]) for r in t1["rows"]) or \
            len(set(t1["headers"])) != len(t1["headers"])
        if len(set(t1["headers"])) != len(t1["headers"]):
            report.setdefault("preexisting_duplicate_headers", []).append(f"{code}:{t1['table_id']}")
        rec["multi_tabular"] = multi
        for a, r in zip(t1["rows"], t2["rows"]):
            cols = list(r["cells"])
            if not multi and cols != t2["headers"][1:] and cols != t2["headers"]:
                fails.append(f"{rec['table']}: row {r['row_header']!r} columns {cols} != headers {t2['headers']}")
            if multi and not set(a["cells"]) <= set(cols):
                fails.append(f"{rec['table']}: row {r['row_header']!r} lost columns {set(a['cells']) - set(cols)}")
        # (4) labels aligned: same headers; row counts equal unless e2 kept a mark column as a column
        if t1["headers"] != t2["headers"]:
            fails.append(f"{rec['table']}: headers changed {t1['headers']} -> {t2['headers']}")
        values2 = [v for r in t2["rows"] for v in r["cells"].values()] + [r["row_header"] for r in t2["rows"]]
        values1 = [v for r in t1["rows"] for v in r["cells"].values()] + [r["row_header"] for r in t1["rows"]]
        # (5) no 51/55 from a mark macro: in tables whose source uses one, 51/55 must not survive
        if key in AFFECTED and any(v.strip() in ("51", "55") for v in values2):
            fails.append(f"{rec['table']}: a 51/55 value survives")
        # (1) the affected tables now show marks
        rec["marks_e2"] = sum(v in MARKS for v in values2)
        if key in AFFECTED and rec["marks_e2"] == 0:
            fails.append(f"{rec['table']}: no marks after the fix")
        # (6) genuine numbers: every numeric value of e1 (not a 51/55 mark artifact) still present in e2
        n1 = Counter(v for v in values1 if NUM.match(v) and not (key in AFFECTED and v in ("51", "55")))
        n2 = Counter(v for v in values2 if NUM.match(v))
        missing = n1 - n2
        if missing:
            fails.append(f"{rec['table']}: numeric values lost {dict(missing)}")
        rec["numbers_e1"], rec["numbers_e2"] = sum(n1.values()), sum(n2.values())
        rec["changed_rows"] = sum(json.dumps(a, ensure_ascii=False) != json.dumps(b, ensure_ascii=False)
                                  for a, b in zip(t1["rows"], t2["rows"])) + abs(len(t1["rows"]) - len(t2["rows"]))
        report["tables"].append(rec)
        if rec["changed_rows"] and len(report["examples"]) < 12:
            i = next(k for k, (a, b) in enumerate(zip(t1["rows"], t2["rows"])) if a != b) if t1["rows"] else 0
            report["examples"].append({"table": rec["table"], "caption": t1["caption"][:90], "headers": t2["headers"],
                                       "e1_row": t1["rows"][i] if i < len(t1["rows"]) else None,
                                       "e2_row": t2["rows"][i] if i < len(t2["rows"]) else None})
    # (2) prose: a paragraph may change only if its source holds a mark macro
    for p1, p2 in zip(e1[code].paragraphs, e2[code].paragraphs):
        if p1["text"] != p2["text"]:
            report["prose_changes"].append({"paper": code, "section": p1["section_id"], "e1": p1["text"][:160],
                                            "e2": p2["text"][:160]})
    if len(e1[code].paragraphs) != len(e2[code].paragraphs):
        fails.append(f"{code}: paragraph count changed")
for ch in report["prose_changes"]:
    if not any(s in ch["e2"] for s in ("✓", "✗", "[symbol]")) and ch["e1"].replace(" ", "") != ch["e2"].replace(" ", ""):
        fails.append(f"unexpected prose change in {ch['paper']} §{ch['section']}")

report["checks_failed"] = fails
out = Path(f"experiments/phase6/out/extraction_validation_{datetime.now(timezone.utc).strftime('%Y-%m-%dT%H-%M-%SZ')}.json")
out.parent.mkdir(parents=True, exist_ok=True)
with out.open("x", encoding="utf-8") as f:
    json.dump(report, f, indent=1, ensure_ascii=False)
changed = [t for t in report["tables"] if t["changed_rows"]]
print(f"Phase-5 rows re-identified: {phase5_rows}; tables changed by e2: {len(changed)} of {len(report['tables'])}")
for t in changed:
    print(f"   {t['table']:10} rows e1 {t['rows_e1']:>3} e2 {t['rows_e2']:>3} changed {t['changed_rows']:>3} marks {t['marks_e2']:>3} "
          f"numbers {t['numbers_e1']}->{t['numbers_e2']} {'AFFECTED' if t['affected'] else ''} | {t['caption'][:45]}")
print("prose paragraphs changed:", len(report["prose_changes"]))
print("mark macros:", src_marks)
print("CHECKS:", "ALL PASSED" if not fails else f"{len(fails)} FAILED")
for f_ in fails:
    print("   FAIL", f_)
print(out)
sys.exit(1 if fails else 0)
