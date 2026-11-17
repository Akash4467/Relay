"""Post-hoc analysis of a retrieval_eval run (no models): refines 'table_structure_or_missing_row_cells' into
  extraction: no chunk of the corpus holds the row with its answering cells (e.g. c3 rows without check marks)
  row_context: some chunk holds it complete, but not one in the final context
and lists per-question failures. usage: retrieval_analysis.py <run dir> <processed dir>"""
import json
import sys
from pathlib import Path

sys.path.insert(0, "experiments/phase11f")
from reconcile import holds  # noqa: E402

run, proc = Path(sys.argv[1]), Path(sys.argv[2])
rows = [json.loads(l) for l in (run / "per_question.jsonl").open(encoding="utf-8")]
chunks = {s: {json.loads(l)["chunk_id"]: json.loads(l) for l in open(proc / f"chunks_{s}.jsonl", encoding="utf-8")} for s in (256, 512)}
for r in rows:
    if not r["answerable"]:
        continue
    ch = chunks[256 if r["setting"] in ("A", "B") else 512]
    for u in r["units"]:
        if u["failure_cause"] == "table_structure_or_missing_row_cells":
            unit = {"kind": "row", "row": {"id": u["unit"], "cells": u.get("gold_cells") or []},
                    "prefix": u["holders"][0].rsplit(":", 1)[0] + ":"}
            anywhere = any(holds(unit, cid, ch, True) for cid in ch if cid.startswith(unit["prefix"]))
            u["failure_cause"] = ("row_context (a chunk holds the row complete; not in context)" if anywhere else
                                  "extraction (no chunk holds the row's answering cells)")
out = {}
for s in ("A", "B", "C"):
    rs = [r for r in rows if r["setting"] == s and r["answerable"]]
    causes = {}
    for r in rs:
        for u in r["units"]:
            if u["failure_cause"]:
                causes.setdefault(u["failure_cause"], []).append(r["question_id"])
    out[s] = {"causes": {k: {"units": len(v), "questions": sorted(set(v))} for k, v in sorted(causes.items())},
              "questions_failing": sorted({r["question_id"] for r in rs if not r["all_gold_complete_in_context"]})}
(run / "failure_analysis.json").write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
with (run / "per_question_refined.jsonl").open("w", encoding="utf-8") as f:
    for r in rows:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
for s, v in out.items():
    print(s, len(v["questions_failing"]), "failing")
    for k, c in v["causes"].items():
        print("   ", c["units"], k, c["questions"])
