"""Per-question changes of each configuration against B (and C+pack against C), from an ablation output.
usage: diff_vs_b.py <ablation json>"""
import json
import sys

d = json.load(open(sys.argv[1], encoding="utf-8"))
rows = d["rows"]
by = {(r["config"], r["dataset"], r["question_id"]): r for r in rows}
q11 = {r["config"]: (r.get("coverage"), r.get("all_gold"), r.get("gold_ranks")) for r in rows if r["question_id"] == "q11"}
print("q11 (coverage, all gold, gold ranks):", q11)
for cfg in d["configs"]:
    base = "C" if cfg.startswith("C") else "B"
    if cfg == base:
        continue
    gains, losses = [], []
    for (c, ds, qid), r in by.items():
        if c != cfg or "units" not in r:
            continue
        b = by[(base, ds, qid)]
        dc, dm = r["coverage"] - b["coverage"], r["mrr"] - b["mrr"]
        tag = f"{ds}:{qid}(cov {b['coverage']:.2f}->{r['coverage']:.2f}, mrr {b['mrr']:.2f}->{r['mrr']:.2f})"
        if dc > 0 or (dc == 0 and dm > 0.05):
            gains.append(tag)
        elif dc < 0 or (dc == 0 and dm < -0.05):
            losses.append(tag)
    print(f"\n{cfg} vs {base}: {len(gains)} gains, {len(losses)} losses")
    print("  +", gains)
    print("  -", losses)
    exp = [(r["question_id"], r["expansion"]) for (c, ds, qid), r in by.items() if c == cfg and r.get("expansion")]
    if exp:
        print("  expansions:", exp)
    un = [(qid, r["table_in_context"], by[(base, ds, qid)]["table_in_context"]) for (c, ds, qid), r in by.items()
          if c == cfg and "table_in_context" in r and r["table_in_context"] != by[(base, ds, qid)]["table_in_context"]]
    if un:
        print("  unanswerable table-in-context changes (cfg, base):", un)
