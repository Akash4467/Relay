"""Phase 7, stage 2 (offline): validate eval/phase7/table_question_set.json against both corpora and the LaTeX source.
For each answerable question and gold row: the id strings match exactly one row line of that table in c4 (no
ambiguity); id + cells all appear on that line in c4 (evidence is satisfiable); the id strings also identify a row
line in c3. Every number in the expected answer must occur in the table's LaTeX source. Exit 1 on any failure."""
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, ".")
from relay.retrieval import latex_parse as lp

QS = json.loads(Path("eval/phase7/table_question_set.json").read_text(encoding="utf-8"))["questions"]


def row_lines(path: str, paper: str, table_id: str) -> list[str]:
    seen, out = set(), []
    for c in map(json.loads, open(path, encoding="utf-8")):
        if c.get("rows") and c["paper"] == paper and c.get("table_id") == table_id:
            for line in c["text"].split("\n")[2:]:
                if line not in seen:
                    seen.add(line)
                    out.append(line)
    return out


def table_source(paper: str, label: str) -> str:
    """The LaTeX float that carries this table's \label (the table body sits within a few thousand chars of it)."""
    src = lp.load_tex(lp.find_main(lp.RAW_DIR / paper / "src"))
    i = src.find("\\label{" + label + "}")
    if i < 0:
        return ""
    start = src.rfind("\\begin{table", 0, i)
    end = src.find("\\end{table", i)
    return src[start if start >= 0 else max(0, i - 6000): end if end >= 0 else i + 6000]


fails, n_rows = [], 0
labels = {}
for t in map(json.loads, open("data/corpus/c4/processed/tables.jsonl", encoding="utf-8")):
    labels[(t["paper"], t["table_id"])] = t["label"]
c3_unidentifiable = []
for q in QS:
    if not q["answerable"]:
        continue
    lines4 = row_lines("data/corpus/c4/processed/chunks_256.jsonl", q["paper"], q["table_id"])
    lines3 = row_lines("data/processed/chunks_256.jsonl", q["paper"], q["table_id"])
    for g in q["gold"]["rows"]:
        n_rows += 1
        m4 = [l for l in lines4 if all(s in l for s in g["id"])]
        if len(m4) > 1:  # rows that differ only in their marks: identifiable by id + marks (c4), not in c3
            m4 = [l for l in m4 if all(s in l for s in g["cells"])]
            c3_unidentifiable.append(q["question_id"])
        if len(m4) != 1:
            fails.append(f"{q['question_id']}: id {g['id']} (+cells) matches {len(m4)} c4 row lines (need exactly 1)")
            continue
        if not all(s in m4[0] for s in g["cells"]):
            fails.append(f"{q['question_id']}: cells {g['cells']} not on the c4 line: {m4[0][:150]}")
        if not any(all(s in l for s in g["id"]) for l in lines3):
            fails.append(f"{q['question_id']}: id {g['id']} identifies no c3 row line")
    # numbers of the expected answer must come from the table's source
    src = table_source(q["paper"], labels[(q["paper"], q["table_id"])])
    for num in re.findall(r"\d+\.\d+|\d{2,}", q["expected_answer"]):
        if num not in src:
            fails.append(f"{q['question_id']}: number {num} of the expected answer not found near the table in the LaTeX source")
print(f"{sum(q['answerable'] for q in QS)} answerable / {sum(not q['answerable'] for q in QS)} unanswerable questions; "
      f"{n_rows} gold rows checked")
print("rows identifiable only by their marks (ambiguous in c3):", sorted(set(c3_unidentifiable)))
print("ALL CHECKS PASSED" if not fails else f"{len(fails)} FAILED")
for f in fails:
    print("  FAIL", f)
sys.exit(1 if fails else 0)
