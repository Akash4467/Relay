"""Phase 8, investigation C follow-up: per-question gains and losses of each chunking variant against base, on the
original and the table sets (table_retrieval.py stored only aggregates). Writes experiments/phase8/out/variant_diff_{UTC}.json."""
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, ".")
sys.path.insert(0, "experiments/phase8")
from retrieval_common import MemIndex, run, variant_chunks  # noqa: E402

from relay.retrieval.index_build import load_model
from relay.retrieval.rerank import load_reranker

embedder, reranker = load_model(), load_reranker()
runs = {v: run({s: MemIndex(variant_chunks(v, s), embedder) for s in (256, 512)}, reranker)
        for v in ("base", "full_caption", "metadata", "whole_table")}
out = {}
for v in ("full_caption", "metadata", "whole_table"):
    ch = []
    for a, b in zip(runs["base"], runs[v]):
        assert (a["question_id"], a["setting"]) == (b["question_id"], b["setting"])
        for m in ("any_gold", "quotes_in", "table_in", "rows_in", "complete"):
            if m in a and a[m] != b[m]:
                ch.append({"q": a["question_id"], "setting": a["setting"], "metric": m, "base": a[m], v: b[m]})
    out[v] = ch
    print(f"\n### {v}")
    for c in ch:
        print("  ", c)
p = Path(f"experiments/phase8/out/variant_diff_{datetime.now(timezone.utc).strftime('%Y-%m-%dT%H-%M-%SZ')}.json")
with p.open("x", encoding="utf-8") as f:
    json.dump(out, f, indent=1)
print(p)
