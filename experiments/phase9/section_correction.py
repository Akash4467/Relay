"""Phase 9A, investigation B: experimental section correction for MOSHI Table 3 (section metadata version s1).

The parser assigns a table to the section in whose source text its float sits. MOSHI places the Table 3 float
(tab:mimi_ablations, sections/evaluation.tex) inside 5.1 "Text Language Modeling", just before the 5.2 "Audio
Tokenization" heading; LaTeX also prints it on page 22, among 5.1's pages. Its content is the Mimi codec ablation,
which 5.2 cites and discusses ("Results - Semantic tokens", "Results - Acoustic tokens"); 5.1 never cites it.
s1 assigns it to 5.2 by that rule (the evaluation subsection whose prose cites it). Only Table 3 is corrected (scope
of this phase); other tables with the same float placement are reported by validate(), not changed.

Applied after chunking, so chunk IDs, chunk order and chunk boundaries do not change: only the section_id and
section_title fields of Table 3's table chunks and of its full-caption chunk (kind "caption", ID MOSHI:5.1:c070 at
256; the ID keeps "5.1" so that citations stay stable). The production section parser is not changed.

    uv run python experiments/phase9/section_correction.py   # validation -> experiments/phase9/out/section_correction.json"""
from __future__ import annotations

import copy
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, ".")
VERSION = "s1"
CORRECTIONS = {("MOSHI", "T3"): {"label": "tab:mimi_ablations", "from": "5.1", "to": "5.2"}}
OUT = Path("experiments/phase9/out")


def correct_tables(tables: list[dict]) -> tuple[list[dict], list[dict]]:
    """Corrected copies of the table records (with the parsed section kept as section_id_parsed) and a record per fix."""
    out, records = [], []
    for t in tables:
        fix = CORRECTIONS.get((t["paper"], t["table_id"]))
        t = copy.deepcopy(t)
        if fix:
            if t["label"] != fix["label"] or t["section_id"] != fix["from"]:
                raise ValueError(f"{t['paper']}:{t['table_id']} is not the table the correction was written for")
            t["section_id_parsed"], t["section_id"] = t["section_id"], fix["to"]
            records.append({"paper": t["paper"], "table_id": t["table_id"], "label": t["label"],
                            "section_id_original": fix["from"], "section_id_corrected": fix["to"], "version": VERSION})
        out.append(t)
    return out, records


def correct_chunks(chunks: list[dict], sections: list[dict]) -> tuple[list[dict], list[str]]:
    """Copies of the chunks with the corrected tables' section fields changed (table chunks and the table's
    full-caption chunk); returns (chunks, changed chunk IDs). IDs, order, text and boundaries are unchanged."""
    titles = {(s["paper"], s["section_id"]): s["title"] for s in sections}
    out, changed = [], []
    for c in chunks:
        c = copy.deepcopy(c)
        for (paper, tid), fix in CORRECTIONS.items():
            mine = c["paper"] == paper and c["section_id"] == fix["from"] and (
                c.get("table_id") == tid or (c["kind"] == "caption" and c["text"].startswith(f"Table {tid[1:]}:")))
            if mine:
                c["section_id"], c["section_title"] = fix["to"], titles[(paper, fix["to"])]
                changed.append(c["chunk_id"])
        out.append(c)
    return out, changed


def citing_sections(paper: str, table_id: str, paragraphs: list[dict]) -> dict[str, int]:
    rx = re.compile(rf"\bTable {table_id[1:]}\b(?!\.\d)")
    refs: dict[str, int] = {}
    for p in paragraphs:
        if p["paper"] == paper and rx.search(p["text"]):
            refs[p["section_id"]] = refs.get(p["section_id"], 0) + 1
    return refs


def validate(processed: Path = Path("data/corpus/c4/processed")) -> dict:
    import pymupdf

    read = lambda k: [json.loads(l) for l in (processed / f"{k}.jsonl").read_text(encoding="utf-8").splitlines()]
    sections, paragraphs, tables = read("sections"), read("paragraphs"), read("tables")
    src = Path("data/raw/MOSHI/src/sections/evaluation.tex").read_text(encoding="utf-8").splitlines()
    label_line = next(i + 1 for i, l in enumerate(src) if r"\label{tab:mimi_ablations}" in l)
    headings = [(i + 1, l.strip()) for i, l in enumerate(src) if re.match(r"\s*\\(sub)*section\{", l)]
    before = [h for h in headings if h[0] < label_line][-1]
    after = next(h for h in headings if h[0] > label_line)
    pages = [p.get_text() for p in pymupdf.open("data/raw/MOSHI/paper.pdf")]
    page_of = lambda s: next((i + 1 for i, t in enumerate(pages) if s in t), None)

    fixed, records = correct_tables(tables)
    by_key = lambda ts: {(t["paper"], t["table_id"]): t for t in ts}
    old, new = by_key(tables), by_key(fixed)
    unchanged_other = all(old[k] == new[k] for k in old if k not in CORRECTIONS)
    t3_same_except_section = {k: v for k, v in new[("MOSHI", "T3")].items() if k not in ("section_id", "section_id_parsed")} == \
                             {k: v for k, v in old[("MOSHI", "T3")].items() if k != "section_id"}
    scan = []
    for t in tables:
        if t["paper"] == "MOSHI":
            refs = citing_sections("MOSHI", t["table_id"], paragraphs)
            same_top = {s for s in refs if s.split(".")[0] == t["section_id"].split(".")[0]}
            scan.append({"table_id": t["table_id"], "label": t["label"], "assigned": t["section_id"], "cited_in": refs,
                         "assigned_section_cites_it": t["section_id"] in refs,
                         "corrected_in_s1": (t["paper"], t["table_id"]) in CORRECTIONS,
                         "same_top_level_citing_sections": sorted(same_top)})
    report = {
        "version": VERSION, "corrections": records,
        "evidence": {
            "source": {"file": "sections/evaluation.tex", "label_line": label_line,
                       "heading_before_float": before, "heading_after_float": after},
            "pdf": {"table_3_caption_page": page_of("Table 3:"), "heading_5_1_page": page_of("5.1 Text Language Modeling"),
                    "heading_5_2_page": page_of("5.2 Audio Tokenization"), "table_17_caption_page": page_of("Table 17:")},
            "citations_of_table_3": citing_sections("MOSHI", "T3", paragraphs),
            "citations_of_table_17": citing_sections("MOSHI", "T17", paragraphs),
        },
        "checks": {"only_table_3_changed": unchanged_other and len(records) == 1,
                   "table_3_identity_caption_rows_unchanged": t3_same_except_section,
                   "table_17_unchanged": old[("MOSHI", "T17")] == new[("MOSHI", "T17")],
                   "target_section_cites_table_3": "5.2" in citing_sections("MOSHI", "T3", paragraphs),
                   "original_section_does_not_cite_table_3": "5.1" not in citing_sections("MOSHI", "T3", paragraphs)},
        "same_defect_not_corrected": [s for s in scan if not s["assigned_section_cites_it"] and not s["corrected_in_s1"]
                                      and s["same_top_level_citing_sections"]],
        "moshi_table_scan": scan,
    }
    report["all_checks_passed"] = all(report["checks"].values())
    return report


if __name__ == "__main__":
    r = validate()
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "section_correction.json").write_text(json.dumps(r, indent=1, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: r[k] for k in ("corrections", "evidence", "checks")}, indent=1, ensure_ascii=False))
    print("same defect, not corrected:", [(s["table_id"], s["assigned"], s["same_top_level_citing_sections"])
                                          for s in r["same_defect_not_corrected"]])
    print("ALL CHECKS PASSED" if r["all_checks_passed"] else "CHECKS FAILED")
    sys.exit(0 if r["all_checks_passed"] else 1)
