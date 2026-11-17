"""Phase 6, stage 5 (offline): controlled retrieval comparison, corpus c3 (data/processed, data/index) vs c4
(data/corpus/c4). Same 30 questions, settings A/B/C, embedder, reranker, context cap and gold evidence quotes.

Relevance: the only labels are the gold evidence quotes in eval/questions.jsonl. A chunk is "gold" for a quote when
it contains the quote (relay.eval.retrieval_recall.gold_chunks, per corpus). Recall@k and MRR are over the ranked
candidates (before the context cap); quote coverage is over the final context (the existing metric). Precision@5 and
nDCG@10 use a binary proxy relevance (the chunk contains any gold quote of the question); they are not graded labels.
Writes experiments/phase6/out/retrieval_comparison_{UTC}.json and eval/phase6_retrieval_comparison.csv (new)."""
import csv
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, ".")
from relay.config import app_config, retrieval_settings
from relay.eval.retrieval_recall import gold_chunks, load_questions
from relay.retrieval.index_build import load_model
from relay.retrieval.rerank import load_reranker
from relay.retrieval.search import Index, search

CORPORA = {"c3": (Path("data/index"), Path("data/processed")),
           "c4": (Path("data/corpus/c4/index"), Path("data/corpus/c4/processed"))}
ARTIFACT_TABLES = {("HNET", "T6"), ("MOSHI", "T6")}
CSV_OUT = Path("eval/phase6_retrieval_comparison.csv")
cap = app_config()["pressure"]["L0"]["context_cap"]
questions = load_questions(None)
embedder, reranker = load_model(), load_reranker()
settings = retrieval_settings()


def artifact(chunk: dict) -> bool:
    return bool(chunk.get("rows")) and (chunk["paper"], chunk.get("table_id")) in ARTIFACT_TABLES and any(
        v["value"] in ("51", "55") for r in chunk["rows"] for v in r["values"])


def dcg(rels):
    return sum(r / math.log2(i + 2) for i, r in enumerate(rels))


rows = []
for corpus, (idx_dir, proc_dir) in CORPORA.items():
    indexes = {}
    for name in ("A", "B", "C"):
        s = settings[name]
        index = indexes.setdefault(s.piece_tokens, Index(s.piece_tokens, embedder, idx_dir, proc_dir))
        for q in questions:
            ev = search(q["question"], s, cap, index, reranker if s.rerank else None)
            cand = [c["chunk_id"] for c in ev.candidates]
            final = [p["chunk_id"] for p in ev.passages]
            row = {"corpus": corpus, "setting": name, "question_id": q["question_id"], "split": q["split"],
                   "category": q["category"], "answerable": q["answerable"], "context_tokens": ev.tokens,
                   "final_passages": len(final),
                   "artifact_chunks_in_context": sum(artifact(index.chunks[c]) for c in final),
                   "table_chunks_in_context": sum(bool(index.chunks[c].get("rows")) for c in final)}
            if q["answerable"]:
                gold = gold_chunks(q, index.chunks)
                allg = set().union(*gold)
                first = min((cand.index(c) + 1 for c in allg if c in cand), default=None)
                rels = [1 if c in allg else 0 for c in cand]
                ideal = [1] * min(len(allg), 10)
                table_gold = [ids for e, ids in zip(q["evidence"], gold) if " Table " in f" {e['section']} "]
                row |= {"gold_quotes": len(gold), "quotes_in_context": sum(bool(ids & set(final)) for ids in gold),
                        "any_gold_in_context": any(ids & set(final) for ids in gold),
                        "all_gold_in_context": all(ids & set(final) for ids in gold),
                        "first_gold_rank": first, "mrr": round(1 / first, 4) if first else 0.0,
                        **{f"recall@{k}": round(sum(bool(ids & set(cand[:k])) for ids in gold) / len(gold), 4) for k in (1, 5, 10)},
                        "precision@5_proxy": round(sum(rels[:5]) / 5, 4),
                        "ndcg@10_proxy": round(dcg(rels[:10]) / dcg(ideal), 4) if ideal else 0.0,
                        "table_gold_quotes": len(table_gold),
                        "table_gold_in_context": sum(bool(ids & set(final)) for ids in table_gold)}
            rows.append(row)


def summary(sel):
    ans = [r for r in sel if r["answerable"]]
    mean = lambda k: round(sum(r[k] for r in ans) / len(ans), 4)
    tq = [r for r in ans if r["table_gold_quotes"]]
    return {"answerable": len(ans), "any_gold_pct": round(100 * sum(r["any_gold_in_context"] for r in ans) / len(ans), 1),
            "all_gold_pct": round(100 * sum(r["all_gold_in_context"] for r in ans) / len(ans), 1),
            "quote_recall_pct": round(100 * sum(r["quotes_in_context"] for r in ans) / sum(r["gold_quotes"] for r in ans), 1),
            "mrr": mean("mrr"), "recall@1": mean("recall@1"), "recall@5": mean("recall@5"), "recall@10": mean("recall@10"),
            "precision@5_proxy": mean("precision@5_proxy"), "ndcg@10_proxy": mean("ndcg@10_proxy"),
            "table_questions": len(tq), "table_quotes_in_context": sum(r["table_gold_in_context"] for r in tq),
            "table_quotes": sum(r["table_gold_quotes"] for r in tq),
            "missing_evidence": sorted(r["question_id"] for r in ans if not r["any_gold_in_context"]),
            "artifact_chunks_in_context_all_questions": sum(r["artifact_chunks_in_context"] for r in sel),
            "mean_context_tokens": round(sum(r["context_tokens"] for r in sel) / len(sel), 1)}


summ = {f"{c} {s} {split}": summary([r for r in rows if r["corpus"] == c and r["setting"] == s and (split == "all" or r["split"] == split)])
        for c in CORPORA for s in "ABC" for split in ("dev", "test", "all")}
out = Path(f"experiments/phase6/out/retrieval_comparison_{datetime.now(timezone.utc).strftime('%Y-%m-%dT%H-%M-%SZ')}.json")
with out.open("x", encoding="utf-8") as f:
    json.dump({"summary": summ, "rows": rows, "context_cap": cap}, f, indent=1, ensure_ascii=False)

# query-level comparison CSV: one row per (setting, question), c3 and c4 side by side
by = {(r["corpus"], r["setting"], r["question_id"]): r for r in rows}
fields = ["setting", "question_id", "split", "category", "answerable"]
metrics = ["any_gold_in_context", "quotes_in_context", "gold_quotes", "first_gold_rank", "mrr", "recall@5", "recall@10",
           "precision@5_proxy", "ndcg@10_proxy", "table_gold_in_context", "artifact_chunks_in_context", "context_tokens"]
with CSV_OUT.open("x", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(fields + [f"{m}_{c}" for m in metrics for c in ("c3", "c4")] + ["change"])
    for s in "ABC":
        for q in questions:
            a, b = by[("c3", s, q["question_id"])], by[("c4", s, q["question_id"])]
            change = "" if not q["answerable"] else (
                "gained gold" if b["any_gold_in_context"] and not a["any_gold_in_context"] else
                "lost gold" if a["any_gold_in_context"] and not b["any_gold_in_context"] else
                "rank up" if (b["first_gold_rank"] or 99) < (a["first_gold_rank"] or 99) else
                "rank down" if (b["first_gold_rank"] or 99) > (a["first_gold_rank"] or 99) else "same")
            w.writerow([s, q["question_id"], q["split"], q["category"], q["answerable"]] +
                       [r.get(m, "") for m in metrics for r in (a, b)] + [change])
for k, v in summ.items():
    if k.endswith(" all") or k.endswith(" dev") or k.endswith(" test"):
        print(f"{k:12} any {v['any_gold_pct']:>5} all {v['all_gold_pct']:>5} qrec {v['quote_recall_pct']:>5} MRR {v['mrr']:.3f} "
              f"R@5 {v['recall@5']:.3f} R@10 {v['recall@10']:.3f} P@5* {v['precision@5_proxy']:.3f} nDCG@10* {v['ndcg@10_proxy']:.3f} "
              f"tables {v['table_quotes_in_context']}/{v['table_quotes']} artifacts {v['artifact_chunks_in_context_all_questions']} "
              f"missing {v['missing_evidence']}")
print(out, CSV_OUT)
