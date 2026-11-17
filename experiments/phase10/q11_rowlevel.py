"""Phase 10C, REAL-PAPER OFFLINE probe: does the cross-encoder rank q11's rows correctly when each Table 3 row is
scored on its own (caption + columns + one row) instead of as part of a multi-row chunk? A diagnostic only: nothing
is indexed or changed. Writes experiments/phase10/out/q11_rowlevel_{UTC}.json."""
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, ".")
from relay.retrieval import latex_parse as lp  # noqa: E402
from relay.retrieval.rerank import load_reranker  # noqa: E402

Q = next(json.loads(l) for l in open("eval/questions.jsonl", encoding="utf-8") if json.loads(l)["question_id"] == "q11")["question"]
chunks = [json.loads(l) for l in open("experiments/phase10/corpus/c6/processed/chunks_256.jsonl", encoding="utf-8")]
t3 = [c for c in chunks if c["chunk_id"].startswith("AIAYN:T3:")]
table = next(t for t in map(json.loads, open("experiments/phase10/corpus/c6/processed/tables.jsonl", encoding="utf-8"))
             if t["paper"] == "AIAYN" and t["table_id"] == "T3")
head = "\n".join(t3[0]["text"].split("\n")[:2])
rows = [(i, r, head + "\nConfiguration " + lp.row_text(r)) for i, r in enumerate(table["rows"])]
ce = load_reranker()
chunk_scores = ce.predict([(Q, c["text"]) for c in t3])
row_scores = ce.predict([(Q, text) for _, _, text in rows])
by_chunk = sorted(((float(s), c["chunk_id"]) for s, c in zip(chunk_scores, t3)), reverse=True)
by_row = sorted(((float(s), i, r["row_header"], r["cells"].get("h", ""), r["cells"].get("BLEU (dev)", ""))
                 for s, (i, r, _) in zip(row_scores, rows)), reverse=True)
out = {"question": Q, "chunk_ranking": by_chunk, "row_ranking": by_row[:10]}
print("chunks:", [(round(s, 2), c[-3:]) for s, c in by_chunk])
print("rows  :", [(round(s, 2), f"#{i} {h} h={hh} BLEU={b}") for s, i, h, hh, b in by_row[:6]])
p = Path(f"experiments/phase10/out/q11_rowlevel_{datetime.now(timezone.utc).strftime('%Y-%m-%dT%H-%M-%SZ')}.json")
p.write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
print(p.as_posix())
