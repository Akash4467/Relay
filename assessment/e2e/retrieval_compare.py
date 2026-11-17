"""Per-question comparison of two retrieval_eval runs (same questions, same settings). Reports, per setting, every
question whose complete-evidence or any-gold status changes, so gains never hide losses.
usage: retrieval_compare.py <baseline run dir> <candidate run dir> <out.json>"""
import json
import sys
from pathlib import Path


def load(d):
    f = Path(d) / "per_question_refined.jsonl"
    return {(r["setting"], r["question_id"]): r for r in map(json.loads, f.open(encoding="utf-8"))}


a, b, out = load(sys.argv[1]), load(sys.argv[2]), {}
for s in ("A", "B", "C"):
    keys = [k for k in a if k[0] == s and a[k]["answerable"]]
    d = {"complete_gained": [], "complete_lost": [], "any_gold_gained": [], "any_gold_lost": [],
         "units_in_context_lost": [], "rr_drop_over_0.25": []}
    for k in keys:
        x, y = a[k], b[k]
        q = k[1]
        if not x["all_gold_complete_in_context"] and y["all_gold_complete_in_context"]: d["complete_gained"].append(q)
        if x["all_gold_complete_in_context"] and not y["all_gold_complete_in_context"]: d["complete_lost"].append(q)
        if not x["all_gold_in_context"] and y["all_gold_in_context"]: d["any_gold_gained"].append(q)
        if x["all_gold_in_context"] and not y["all_gold_in_context"]: d["any_gold_lost"].append(q)
        if sum(u["in_context"] for u in y["units"]) < sum(u["in_context"] for u in x["units"]): d["units_in_context_lost"].append(q)
        if x["rr"] - y["rr"] > 0.25: d["rr_drop_over_0.25"].append(f"{q} {x['rr']:.2f}->{y['rr']:.2f}")
    out[s] = d
    print(s, {k: v for k, v in d.items()})
Path(sys.argv[3]).write_text(json.dumps(out, indent=1), encoding="utf-8")
