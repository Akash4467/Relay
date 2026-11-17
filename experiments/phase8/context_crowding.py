"""Phase 8, investigation D (offline): context crowding by one table's chunks.

Context-assembly policies, each applied to the same ranked candidates from the unchanged c4 index (base):
  baseline (production path), identity (policy hook with no change; must equal baseline), max2_per_table,
  max1_per_table, table_half_budget (table chunks may use at most 50% of the context budget).
Plus table-aware chunking (whole_table index) with baseline assembly.
Detail for q10 (settings A, B) and q11 (setting A). Writes experiments/phase8/out/context_crowding_{UTC}.json."""
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, ".")
sys.path.insert(0, "experiments/phase8")
from retrieval_common import CAP, MemIndex, passage_block, count, run, summarise, variant_chunks  # noqa: E402

from relay.retrieval.index_build import load_model
from relay.retrieval.rerank import load_reranker


def key(c):
    return (c["paper"], c.get("table_id")) if c.get("rows") else None


def per_table(limit):
    def policy(ordered, setting):
        seen, out = {}, []
        for c in ordered:
            k = key(c)
            if k is not None:
                if seen.get(k, 0) >= limit:
                    continue
                seen[k] = seen.get(k, 0) + 1
            out.append(c)
        return out
    return policy


def table_half_budget(ordered, setting):
    """In rank order, a table chunk is skipped once table tokens would pass 50% of the budget; others are kept."""
    limit, used, out = 0.5 * min(CAP, setting.context_budget), 0, []
    for c in ordered:
        if key(c) is not None:
            n = count(passage_block(c)) + 2
            if used + n > limit:
                continue
            used += n
        out.append(c)
    return out


POLICIES = {"identity": lambda ordered, setting: ordered, "max2_per_table": per_table(2),
            "max1_per_table": per_table(1), "table_half_budget": table_half_budget}
DETAIL = [("q10", "A"), ("q10", "B"), ("q11", "A")]

embedder, reranker = load_model(), load_reranker()
base = {s: MemIndex(variant_chunks("base", s), embedder) for s in (256, 512)}
whole = {s: MemIndex(variant_chunks("whole_table", s), embedder) for s in (256, 512)}
runs = {"baseline": run(base, reranker)}
for name, pol in POLICIES.items():
    runs[name] = run(base, reranker, pol)
runs["whole_table_index"] = run(whole, reranker)

same = all(a["context"] == b["context"] for a, b in zip(runs["baseline"], runs["identity"]))
if not same:
    raise SystemExit("identity policy differs from the production path: the policy hook is not controlled")

out = {"identity_equals_baseline": same, "summary": {}, "detail": {}, "changes_vs_baseline": {}}
for name, rows in runs.items():
    out["summary"][name] = summarise(rows)
    out["detail"][name] = {f"{q} {st}": next(r for r in rows if r["question_id"] == q and r["setting"] == st
                                            and r["dataset"] == "original") for q, st in DETAIL}
    if name != "baseline":  # per-question gains and losses against baseline (same order of rows)
        ch = []
        for a, b in zip(runs["baseline"], rows):
            for m in ("any_gold", "quotes_in", "table_in", "rows_in", "complete"):
                if m in a and a[m] != b[m]:
                    ch.append({"q": a["question_id"], "setting": a["setting"], "metric": m, "baseline": a[m], name: b[m]})
        out["changes_vs_baseline"][name] = ch

for name, s in out["summary"].items():
    print(f"\n### {name}")
    for k, v in s.items():
        print(f"   {k}: {v}")
    for k, r in out["detail"][name].items():
        print(f"   {k}: quotes {r.get('quotes_in')}/{r.get('quotes')} tokens {r['context_tokens']} dup {r['duplicate_table_chunks']} ctx {r['context']}")
    for c in out["changes_vs_baseline"].get(name, []):
        print(f"   change {c}")
p = Path(f"experiments/phase8/out/context_crowding_{datetime.now(timezone.utc).strftime('%Y-%m-%dT%H-%M-%SZ')}.json")
with p.open("x", encoding="utf-8") as f:
    json.dump(out, f, indent=1, ensure_ascii=False)
print(p)
