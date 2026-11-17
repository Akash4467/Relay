"""Phase 9B: build the independent-review package for the Phase 8 table-semantics cases (offline, read-only inputs).

For every development (31) and held-out (16) case it writes the claim, the cited table, caption, row, column, the c4
cell, the original LaTeX cell (with its source line and \\multirow origin), the PDF page and a page image, the current
label and why it was given, and empty reviewer columns. Labels are copied, never changed.

Writes eval/phase9b_human_review/{development_review.csv, heldout_review.csv, disagreement_log.csv, table_evidence.md,
pdf_pages/*.png}. Refuses to overwrite a filled-in review file.
"""
from __future__ import annotations

import csv
import io
import json
import re
import shutil
import sys
from pathlib import Path

sys.path.insert(0, ".")

ROOT = Path(".")
P8 = ROOT / "experiments/phase8"
CASE_FILES = {"development": P8 / "table_semantics_cases.json", "heldout": P8 / "table_semantics_heldout.json"}
TARGETS = json.loads((ROOT / "experiments/phase9b/review_targets.json").read_text(encoding="utf-8"))
C4 = ROOT / "data/corpus/c4/processed"
RAW = ROOT / "data/raw"
OUT = ROOT / "eval/phase9b_human_review"

REVIEW_COLUMNS = ["reviewer_label", "reviewer_evidence_basis", "reviewer_agrees_with_current_label",
                  "reviewer_needs_legend_or_prose", "reviewer_notes", "reviewer_id", "review_date"]
MARK_DEFS = {"HNET": "\\cmark = \\ding{51} (check, printed); \\xmark = \\ding{55} (cross, printed)  [HNET main.tex 94-95]",
             "MOSHI": "\\cmark = \\ding{51}, \\checkmark = check (printed); \\xmark = EMPTY (prints nothing)  [MOSHI main.tex 43-44]"}

RATIONALE = {
    "checkmark_correct": "The {col} cell of this row is {cell}; the claim says it is marked with a check.",
    "cross_correct": "The {col} cell of this row is {cell}; the claim says it is marked with a cross.",
    "checkmark_incorrect": "The {col} cell of this row is {cell}; the claim asserts a different mark.",
    "cross_incorrect": "The {col} cell of this row is {cell}; the claim asserts a different mark.",
    "wrong_row": "The claim gives this row a mark it does not hold: its {col} cell is {cell}.",
    "wrong_column": "The claim names column {col}, whose cell in this row is {cell}; the claimed mark belongs to another column or row.",
    "correct_symbol_wrong_configuration": "The claimed mark occurs elsewhere in the table, but this configuration's {col} cell is {cell}.",
    "blank_described_as_blank": "The {col} cell prints nothing ({cell}); the claim only describes it as blank.",
    "blank_described_as_unspecified": "The {col} cell prints nothing ({cell}); the claim only describes it as unspecified.",
    "empty_cell_described_as_blank": "The {col} cell prints nothing ({cell}); the claim only says it has no mark.",
    "blank_interpreted_as_false": "The {col} cell prints nothing ({cell}); the claim reads the blank as 'no/not'. The table has no legend saying so (rule: a blank is never read as false).",
    "blank_interpreted_as_zero": "The {col} cell prints nothing ({cell}); the claim reads it as the value 0, which the table does not state.",
    "blank_interpreted_as_true": "The {col} cell prints nothing ({cell}); the claim reads it as 'yes', which the table does not state.",
    "cross_interpreted_as_false": "The {col} cell is {cell}; the claim turns the cross into a property ('does not support'). No legend defines the cross (rule: a mark is never converted into a property).",
    "checkmark_interpreted": "The {col} cell is {cell}; the claim turns the check into a property ('is a causal codec'). No legend defines the check (rule: a mark is never converted into a property).",
    "unsupported_comparison": "The table gives marks or scores, not this comparison; the {col} cell of this row is {cell}.",
    "caption_supported_interpretation": "The caption states that H-Net is the first architecture that enables dynamic, multi-stage hierarchies.",
    "mark_claimed_on_text_cell": "The {col} cell holds the text {cell}, not a mark.",
}


def load_cases() -> dict[str, list[dict]]:
    return {s: json.loads(p.read_text(encoding="utf-8"))["cases"] for s, p in CASE_FILES.items()}


def tables() -> dict[tuple[str, str], dict]:
    return {(t["paper"], t["table_id"]): t for t in map(json.loads, open(C4 / "tables.jsonl", encoding="utf-8"))}


def cited_chunk(case: dict, chunks: list[dict]) -> dict:
    """The c4 256-token chunk whose row line contains every locator string (as in Phase 8)."""
    paper, tid = case["table"].split(":")
    hits = [c for c in chunks if c.get("rows") and c["paper"] == paper and c.get("table_id") == tid
            and any(all(s in l for s in case["locator"]) for l in c["text"].split("\n")[2:])]
    assert len(hits) == 1, (case["id"], len(hits))
    return hits[0]


def row_index(case: dict, chunk: dict, table: dict) -> int:
    """Index in the table record of the row whose chunk line holds every locator string."""
    lines = chunk["text"].split("\n")[2:]
    hit = [i for i, l in enumerate(lines) if all(s in l for s in case["locator"])]
    assert len(hit) == 1, (case["id"], hit)
    r = chunk["rows"][hit[0]]
    shown = {v["column"]: v["value"] for v in r["values"]}
    match = [i for i, t in enumerate(table["rows"]) if t["row_header"] == r["configuration"]
             and all(t["cells"].get(k) == v for k, v in shown.items())]
    assert len(match) == 1, (case["id"], match)
    return match[0]


# ---- LaTeX source of a table
def _strip_comment(line: str) -> str:
    return re.sub(r"(?<!\\)%.*", "", line)


def latex_block(paper: str, label: str) -> tuple[Path, int, list[str]]:
    """(file, first line number, lines) of the table environment holding \\label{label}."""
    for f in sorted((RAW / paper / "src").rglob("*.tex")):
        lines = f.read_text(encoding="utf-8", errors="replace").split("\n")
        at = [i for i, l in enumerate(lines) if f"\\label{{{label}}}" in l]
        if at:
            start = max(i for i in range(at[0] + 1) if "\\begin{table" in lines[i])
            end = min(i for i in range(at[0], len(lines)) if "\\end{table" in lines[i])
            return f, start + 1, lines[start:end + 1]
    raise LookupError(label)


def split_cells(line: str) -> list[str]:
    body = _strip_comment(line).strip()
    body = re.sub(r"\\\\\s*$", "", body)
    return [c.strip() for c in body.split("&")]


def data_rows(block: list[str], first: int) -> list[tuple[int, list[str]]]:
    """(source line number, cells) of each data row: lines with '&' ending in \\\\ after the first \\midrule."""
    out, started = [], False
    for k, line in enumerate(block):
        s = _strip_comment(line)
        if "\\midrule" in s:
            started = True
            continue
        if "\\bottomrule" in s:
            break
        if started and "&" in s and s.rstrip().endswith("\\\\"):
            out.append((first + k, split_cells(line)))
    return out


def latex_cell(rows: list[tuple[int, list[str]]], i: int, col: int) -> tuple[str, int, str]:
    """(cell LaTeX, source line, origin note) for data row i, column col; follows \\multirow{n} from rows above."""
    line, cells = rows[i]
    cell = cells[col] if col < len(cells) else ""
    if cell and not cell.startswith("\\multirow"):
        return cell, line, "own cell"
    if cell.startswith("\\multirow"):
        return cell, line, "starts a \\multirow"
    for j in range(i - 1, -1, -1):
        c = rows[j][1][col] if col < len(rows[j][1]) else ""
        m = re.match(r"\\multirow\{(\d+)\}", c)
        if m:
            return (c, rows[j][0], f"inherited from the \\multirow at line {rows[j][0]}") if j + int(m.group(1)) > i \
                else ("", line, "empty LaTeX cell")
        if c:
            break
    return "", line, "empty LaTeX cell"


# ---- PDF
def pdf_page(paper: str, number: int, pages: list[str]) -> int | None:
    for i, text in enumerate(pages):
        if re.search(rf"\bTable {number}:", text):  # the caption, not a text reference such as "Table 6."
            return i + 1
    return None


def rationale(case: dict, col: str, cell: str) -> str:
    shown = {"": "an empty cell", "(blank)": "a blank cell"}.get(cell, repr(cell))
    base = RATIONALE[case["category"]].format(col=col, cell=shown)
    note = TARGETS["cases"][case["id"]].get("note")
    return f"{base} {note}" if note else base


def build(write_images: bool = True) -> dict[str, list[dict]]:
    import pymupdf

    chunks = [json.loads(l) for l in open(C4 / "chunks_256.jsonl", encoding="utf-8")]
    tabs = tables()
    pdfs = {p: pymupdf.open(RAW / p / "paper.pdf") for p in ("HNET", "MOSHI")}
    texts = {p: [pg.get_text() for pg in d] for p, d in pdfs.items()}
    out: dict[str, list[dict]] = {}
    images: dict[tuple[str, int], str] = {}
    for split, cases in load_cases().items():
        rows_out = []
        for case in cases:
            paper, tid = case["table"].split(":")
            table = tabs[(paper, tid)]
            chunk = cited_chunk(case, chunks)
            i = row_index(case, chunk, table)
            row = table["rows"][i]
            col = TARGETS["cases"][case["id"]]["column"]
            assert col in table["headers"], (case["id"], col)
            f, first, block = latex_block(paper, table["label"])
            drows = data_rows(block, first)
            assert len(drows) == len(table["rows"]), (case["id"], len(drows), len(table["rows"]))
            ci = table["headers"].index(col)
            assert len(drows[i][1]) >= len(table["headers"]) - 1, (case["id"], drows[i])
            cell_tex, tex_line, origin = latex_cell(drows, i, ci)
            cell = row["cells"].get(col, row["row_header"] if ci == 0 else "")
            page = pdf_page(paper, int(tid[1:]), texts[paper])
            if page and (paper, page) not in images:
                images[(paper, page)] = f"pdf_pages/{paper}_p{page}.png"
                if write_images:
                    (OUT / "pdf_pages").mkdir(parents=True, exist_ok=True)
                    pdfs[paper][page - 1].get_pixmap(dpi=110).save(OUT / images[(paper, page)])
            rows_out.append({
                "case_id": case["id"], "set": split, "paper": paper, "table_id": tid, "latex_label": table["label"],
                "pdf_page": page, "pdf_image": images.get((paper, page), ""), "caption": table["caption"],
                "claim": case["claim"], "claim_category": case["category"], "cited_chunk": chunk["chunk_id"],
                "row_number": i + 1, "row_header": row["row_header"],
                "row_cells_c4": "; ".join(f"{k} = {v if v else '(empty)'}" for k, v in row["cells"].items()),
                "column": col, "cell_c4": cell if cell else "(empty)", "cell_latex": cell_tex or "(empty)",
                "cell_latex_source": f"{f.relative_to(ROOT).as_posix()}:{tex_line} ({origin})",
                "mark_macros": MARK_DEFS[paper], "table_note": TARGETS["table_notes"][case["table"]],
                "current_label": case["label"], "current_label_rationale": rationale(case, col, cell),
                "current_label_author": "assistant (Phase 8), not independent",
                "preparer_flags": " | ".join(TARGETS["cases"][case["id"]].get("flags", [])),
                **{k: "" for k in REVIEW_COLUMNS}})
        out[split] = rows_out
    return out


# ---- the blind, reviewer-facing view (two-level protocol)
# Shown to reviewers: evidence only. Hidden: claim category, current label, its rationale and author, preparer flags,
# and the preparer's table notes (which contain the preparer's reading of the prose).
REVIEWER_EVIDENCE = ["case_id", "set", "paper", "table_id", "latex_label", "pdf_page", "pdf_image", "caption", "claim",
                     "row_number", "row_header", "row_cells_c4", "column", "cell_c4", "cell_latex", "cell_latex_source",
                     "printed_on_page", "mark_macros", "source_notes"]
REVIEWER_INPUT = ["table_only_outcome", "whole_paper_outcome", "supporting_evidence", "source_page_or_section",
                  "legend_or_definition_needed", "confidence", "explanation", "reviewer_id", "review_date"]
HIDDEN = ["claim_category", "current_label", "current_label_rationale", "current_label_author", "preparer_flags",
          "table_note"]
REVIEWER_DIR = OUT / "reviewer"


def printed_on_page(paper: str, cell_latex: str, cell_c4: str) -> str:
    """What the PDF prints in the cell, from the paper's own macro definitions."""
    if cell_latex == "(empty)":
        return "nothing (blank): empty LaTeX cell"
    if re.search(r"\\xmark\b", cell_latex):
        return r"nothing (blank): \xmark is defined as empty in MOSHI" if paper == "MOSHI" else "a cross (✗)"
    if re.search(r"\\(?:cmark|checkmark)\b", cell_latex):
        return "a check (✓)"
    return f"text: {cell_c4}"


def reviewer_rows(rows: list[dict]) -> list[dict]:
    out = []
    for r in rows:
        v = {k: r[k] for k in REVIEWER_EVIDENCE if k in r}
        v["printed_on_page"] = printed_on_page(r["paper"], r["cell_latex"], r["cell_c4"])
        v["source_notes"] = TARGETS["reviewer_source_notes"][f"{r['paper']}:{r['table_id']}"]
        out.append({k: v[k] for k in REVIEWER_EVIDENCE} | {k: "" for k in REVIEWER_INPUT})
    return out


def csv_text(rows: list[dict]) -> str:
    buf = io.StringIO(newline="")
    w = csv.DictWriter(buf, fieldnames=list(rows[0]), lineterminator="\r\n")
    w.writeheader()
    w.writerows(rows)
    return buf.getvalue()


def write_new(path: Path, text: str) -> None:
    """Write a generated file; a file that already exists must be identical (never silently changed)."""
    if path.exists():
        if path.read_bytes() != text.encode("utf-8"):
            raise SystemExit(f"{path} exists and differs from what the builder generates; not overwriting it")
        return
    path.write_text(text, encoding="utf-8", newline="")


def table_evidence(rows: list[dict]) -> str:
    """Markdown with the full LaTeX block, the extraction and the PDF page of every cited table (no labels)."""
    tabs = tables()
    parts = ["# Table evidence for the Phase 9B review\n",
             "Full LaTeX source of each cited table, what the pipeline extracted from it (what the verifier sees), and "
             "its PDF page. The PDF page is the primary evidence. "
             "Generated by `experiments/phase9b/build_review_package.py`.\n"]
    for key in sorted({(r["paper"], r["table_id"]) for r in rows}):
        t = tabs[key]
        f, first, block = latex_block(key[0], t["label"])
        mine = [r for r in rows if (r["paper"], r["table_id"]) == key]
        parts += [f"\n## {key[0]} {key[1]} ({t['label']}), PDF page {mine[0]['pdf_page']}\n",
                  f"Page image: `{mine[0]['pdf_image']}`. Cases: {', '.join(r['case_id'] for r in mine)}.\n",
                  f"Source notes: {TARGETS['reviewer_source_notes'][':'.join(key)]}\n",
                  f"\nLaTeX `{f.relative_to(ROOT).as_posix()}` lines {first}-{first + len(block) - 1}:\n",
                  "```latex", *block, "```\n", "Extraction (row number: header | cells):\n", "```"]
        parts += [f"{n}: {r['row_header']} | " + "; ".join(f"{k} = {v if v else '(empty)'}" for k, v in r["cells"].items())
                  for n, r in enumerate(t["rows"], 1)]
        parts.append("```")
    return "\n".join(parts) + "\n"


def main() -> None:
    built = build(write_images=False)
    # the original package files are never rewritten: they must equal what the builder produces
    for split in ("development", "heldout"):
        original = OUT / f"{split}_review.csv"
        if original.read_bytes().decode("utf-8") != csv_text(built[split]):
            raise SystemExit(f"{original} differs from the builder output; the original cases must not change")
    REVIEWER_DIR.mkdir(exist_ok=True)
    (REVIEWER_DIR / "pdf_pages").mkdir(exist_ok=True)
    for png in sorted((OUT / "pdf_pages").glob("*.png")):
        target = REVIEWER_DIR / "pdf_pages" / png.name
        if not target.exists():
            shutil.copyfile(png, target)
        assert target.read_bytes() == png.read_bytes(), target
    for split in ("development", "heldout"):
        write_new(REVIEWER_DIR / f"template_{split}.csv", csv_text(reviewer_rows(built[split])))
    write_new(REVIEWER_DIR / "table_evidence.md", table_evidence(built["development"] + built["heldout"]))
    for s, rows in built.items():
        print(s, len(rows), "cases; originals unchanged; reviewer template written")


if __name__ == "__main__":
    main()
