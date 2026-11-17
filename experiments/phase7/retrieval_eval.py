"""Phase 7, stage 3 (offline): retrieval-only comparison of corpus c3 and c4 on (1) the original 15 answerable
questions plus the 15 unanswerable ones (eval/questions.jsonl) and (2) the repaired-table set
(eval/phase7/table_question_set.json). Settings A/B/C, same embedder, reranker and context cap as production.

Original set: gold = evidence quotes (relay.eval.retrieval_recall.gold_chunks, per corpus); metrics as in Phase 6.
Table set: table level (a chunk of the gold table in the context; first candidate rank; MRR; Recall@k), row level (a
row line containing all `id` strings of a gold row in the context; its rank), evidence level (the same line also holds
the `cells` and `paper_cells`). Corrupted evidence: context lines holding a 51/55 mark artifact (HNET/MOSHI Table 6)
or a \\phantom-inflated value (MOSHI Table 5: 142.2, 140.3, 136.8).
Writes eval/phase7/retrieval_comparison.csv, eval/phase7/raw/retrieval_rows.jsonl (new files only)."""
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, ".")
from relay.config import app_config, retrieval_settings
from relay.eval.retrieval_recall import gold_chunks, load_questions
from relay.retrieval.index_build import load_model
from relay.retrieval.rerank import load_reranker
from relay.retrieval.search import Index, search

CORPORA = {"c3": (Path("data/index"), Path("data/processed")),
           "c4": (Path("data/corpus/c4/index"), Path("data/corpus/c4/processed"))}
TABLE_QS = json.loads(Path("eval/phase7/table_question_set.json").read_text(encoding="utf-8"))["questions"]
ORIG_QS = load_questions(None)
PHANTOM = ("Perplexity = 142.2", "Perplexity = 140.3", "Perplexity = 136.8")
OUT_CSV, OUT_RAW = Path("eval/phase7/retrieval_comparison.csv"), Path("eval/phase7/raw/retrieval_rows.jsonl")
cap = app_config()["pressure"]["L0"]["context_cap"]
settings = retrieval_settings()
embedder, reranker = load_model(), load_reranker()


def lines(chunk):
    return chunk["text"].split("\n")[2:] if chunk.get("rows") else []


def corrupted(chunk) -> int:
    n = 0
    for l in lines(chunk):
        if (chunk["paper"], chunk["table_id"]) in {("HNET", "T6"), ("MOSHI", "T6")} and any(
                f"= {a};" in l + ";" for a in ("51", "55")):
            n += 1
        if chunk["paper"] == "MOSHI" and chunk["table_id"] == "T5" and any(p in l for p in PHANTOM):
            n += 1
    return n


rows = []
for corpus, (idx_dir, proc_dir) in CORPORA.items():
    indexes = {}
    for name in "ABC":
        s = settings[name]
        index = indexes.setdefault(s.piece_tokens, Index(s.piece_tokens, embedder, idx_dir, proc_dir))
        for q in ORIG_QS + TABLE_QS:
            dataset = "table" if q in TABLE_QS else "original"
            ev = search(q["question"], s, cap, index, reranker if s.rerank else None)
            cand = [index.chunks[c["chunk_id"]] for c in ev.candidates]
            final = [index.chunks[p["chunk_id"]] for p in ev.passages]
            r = {"dataset": dataset, "corpus": corpus, "setting": name, "question_id": q["question_id"],
                 "answerable": q["answerable"], "context_tokens": ev.tokens, "context": [c["chunk_id"] for c in final],
                 "corrupted_lines_in_context": sum(corrupted(c) for c in final)}
            if dataset == "original" and q["answerable"]:
                gold = gold_chunks(q, index.chunks)
                allg = set().union(*gold)
                ids = [c["chunk_id"] for c in cand]
                first = min((ids.index(c) + 1 for c in allg if c in ids), default=None)
                fin = {c["chunk_id"] for c in final}
                r |= {"gold_quotes": len(gold), "quotes_in_context": sum(bool(g & fin) for g in gold),
                      "any_gold_in_context": any(g & fin for g in gold), "first_gold_rank": first,
                      "mrr": round(1 / first, 4) if first else 0.0,
                      **{f"recall@{k}": round(sum(bool(g & set(ids[:k])) for g in gold) / len(gold), 4) for k in (5, 10)},
                      "gold_chunks": sorted(allg)}
            if dataset == "table":
                key = (q["paper"], q["table_id"])
                in_table = lambda c: c.get("rows") and (c["paper"], c.get("table_id")) == key
                t_first = next((i + 1 for i, c in enumerate(cand) if in_table(c)), None)
                r |= {"table": f"{q['paper']}:{q['table_id']}", "type": q["type"],
                      "table_in_context": any(in_table(c) for c in final), "table_first_rank": t_first,
                      "table_mrr": round(1 / t_first, 4) if t_first else 0.0,
                      **{f"table_recall@{k}": bool(t_first and t_first <= k) for k in (1, 5, 10)}}
                if q["answerable"]:
                    gold_rows = q["gold"]["rows"]
                    ctx_lines = [l for c in final if in_table(c) for l in lines(c)]
                    def row_rank(g):
                        return next((i + 1 for i, c in enumerate(cand) if in_table(c)
                                     and any(all(x in l for x in g["id"]) for l in lines(c))), None)
                    found = [any(all(x in l for x in g["id"]) for l in ctx_lines) for g in gold_rows]
                    complete = [any(all(x in l for x in g["id"] + g["cells"] + g.get("paper_cells", [])) for l in ctx_lines)
                                for g in gold_rows]
                    ranks = [row_rank(g) for g in gold_rows]
                    r |= {"gold_rows": len(gold_rows), "rows_in_context": sum(found), "rows_complete_in_context": sum(complete),
                          "all_rows_in_context": all(found), "evidence_complete": all(complete),
                          "row_first_ranks": ranks, "row_mrr": round(sum(1 / x if x else 0 for x in ranks) / len(ranks), 4)}
            rows.append(r)

OUT_RAW.parent.mkdir(parents=True, exist_ok=True)
with OUT_RAW.open("x", encoding="utf-8") as f:
    for r in rows:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
by = {(r["dataset"], r["corpus"], r["setting"], r["question_id"]): r for r in rows}
cols = {"original": ["any_gold_in_context", "quotes_in_context", "first_gold_rank", "mrr", "recall@5", "recall@10"],
        "table": ["table_in_context", "table_first_rank", "rows_in_context", "rows_complete_in_context", "evidence_complete", "row_mrr"]}
with OUT_CSV.open("x", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["dataset", "setting", "question_id", "answerable", "metric", "c3", "c4", "change"])
    for (ds, corpus, st, qid), a in sorted(by.items()):
        if corpus != "c3":
            continue
        b = by[(ds, "c4", st, qid)]
        for m in cols[ds] + ["corrupted_lines_in_context", "context_tokens"]:
            if m in a or m in b:
                va, vb = a.get(m, ""), b.get(m, "")
                w.writerow([ds, st, qid, a["answerable"], m, va, vb, "" if va == vb else "changed"])


def agg(ds, corpus, st, split=None):
    sel = [r for r in rows if r["dataset"] == ds and r["corpus"] == corpus and r["setting"] == st]
    ans = [r for r in sel if r["answerable"]]
    out = {"corrupted_lines_in_context": sum(r["corrupted_lines_in_context"] for r in sel)}
    if ds == "original":
        out |= {"any_gold_pct": round(100 * sum(r["any_gold_in_context"] for r in ans) / len(ans), 1),
                "quote_recall_pct": round(100 * sum(r["quotes_in_context"] for r in ans) / sum(r["gold_quotes"] for r in ans), 1),
                "mrr": round(sum(r["mrr"] for r in ans) / len(ans), 3),
                "recall@5": round(sum(r["recall@5"] for r in ans) / len(ans), 3),
                "missing": [r["question_id"] for r in ans if not r["any_gold_in_context"]]}
    else:
        out |= {"table_in_context": f"{sum(r['table_in_context'] for r in ans)}/{len(ans)}",
                "table_mrr": round(sum(r["table_mrr"] for r in ans) / len(ans), 3),
                "table_recall@5": f"{sum(r['table_recall@5'] for r in ans)}/{len(ans)}",
                "rows_in_context": f"{sum(r['rows_in_context'] for r in ans)}/{sum(r['gold_rows'] for r in ans)}",
                "rows_complete": f"{sum(r['rows_complete_in_context'] for r in ans)}/{sum(r['gold_rows'] for r in ans)}",
                "questions_evidence_complete": f"{sum(r['evidence_complete'] for r in ans)}/{len(ans)}",
                "row_mrr": round(sum(r["row_mrr"] for r in ans) / len(ans), 3),
                "unanswerable_table_in_context": f"{sum(r['table_in_context'] for r in sel if not r['answerable'])}/{sum(not r['answerable'] for r in sel)}"}
    return out


summary = {f"{ds} {c} {s}": agg(ds, c, s) for ds in ("original", "table") for c in CORPORA for s in "ABC"}
Path("eval/phase7/raw/retrieval_summary.json").open("x", encoding="utf-8").write(json.dumps(summary, indent=1, ensure_ascii=False))
for k, v in summary.items():
    print(k, v)
for q in ("q06", "q10", "q11", "q14"):
    for st in "ABC":
        a, b = by[("original", "c3", st, q)], by[("original", "c4", st, q)]
        print(f"{q} {st}: rank {a['first_gold_rank']}->{b['first_gold_rank']} quotes {a['quotes_in_context']}->{b['quotes_in_context']} "
              f"| c3 ctx {a['context']} | c4 ctx {b['context']}")
