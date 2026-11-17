"""Phase 10C, REAL-PAPER OFFLINE: why q11 misses AIAYN Table 3's base and h=1 rows, and an experimental table-reference
candidate rule. Production retrieval (relay/retrieval/search.py) is not modified; the variant wraps the same steps.

Variant "tableref": when the question names "Table N" (and possibly a paper), every row chunk of that table is
added to the candidate pool BEFORE the cross-encoder reranks it; the original query is used unchanged (no rewrite),
candidates/final_passages/context budget are unchanged. Settings without a reranker (A) are left as they are: the
added chunks would have no score to be ranked by.

Measured per corpus (c3, c6) and setting: stage ranks of the gold chunks (BM25, dense, RRF, CE), the final context,
budget drops, rerank time. Then the variant on every eval question that names a table (all others are identical by
construction, checked).
    uv run python experiments/phase10/q11_retrieval.py
"""
from __future__ import annotations

import json
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, ".")
from relay.config import app_config, retrieval_settings  # noqa: E402
from relay.retrieval import search as S  # noqa: E402
from relay.retrieval.context import fill  # noqa: E402
from relay.retrieval.index_build import load_model  # noqa: E402
from relay.retrieval.paper_mentions import named_papers  # noqa: E402
from relay.retrieval.rerank import load_reranker, rerank  # noqa: E402
from relay.retrieval.search import tokenize  # noqa: E402

CORPORA = {"c3": (Path("data/index"), Path("data/processed")),
           "c6": (Path("experiments/phase10/corpus/c6/index"), Path("experiments/phase10/corpus/c6/processed"))}
Q11 = next(json.loads(l) for l in open("eval/questions.jsonl", encoding="utf-8") if json.loads(l)["question_id"] == "q11")
GOLD = {"AIAYN:T3:r01": "base row (h = 8): BLEU 25.8, PPL 4.92", "AIAYN:T3:r02": "(A) rows incl. h = 1: BLEU 24.9, PPL 5.29"}
TABLE_REF = re.compile(r"\bTable\s+(\d+)\b", re.I)
CAP = app_config()["pressure"]["L0"]["context_cap"]


def table_ref_chunks(question: str, index: S.Index) -> list[str]:
    m = TABLE_REF.search(question)
    if not m:
        return []
    papers = named_papers(question)
    return sorted(cid for cid, c in index.chunks.items() if c.get("kind") == "table" and c.get("table_id") == f"T{m.group(1)}"
                  and (not papers or c["paper"] in papers))


def search_tableref(question: str, setting, index: S.Index, reranker) -> tuple[S.Evidence, dict]:
    """Same steps as relay.retrieval.search.search for a single-paper/global question, plus the table-ref pool."""
    added = table_ref_chunks(question, index) if setting.rerank else []
    if not added or len(named_papers(question)) >= 2:
        t0 = time.perf_counter()
        ev = S.search(question, setting, CAP, index, reranker if setting.rerank else None)
        return ev, {"added": [], "rerank_ms": None, "total_ms": round((time.perf_counter() - t0) * 1000, 1)}
    t0 = time.perf_counter()
    qvec = index.embed_query(question)
    cands = index.fused(question, qvec, setting.candidates)
    have = {c["chunk_id"] for c in cands}
    extra = [dict(index.chunks[cid], score_rrf=0.0, dense_cos=None) for cid in added if cid not in have]
    t1 = time.perf_counter()
    cands = rerank(question, cands + extra, reranker)
    rr = (time.perf_counter() - t1) * 1000
    passages, tokens = fill(cands[:setting.final_passages], min(CAP, setting.context_budget))
    ev = S.Evidence(passages=passages, top_dense_cos=max([c["dense_cos"] for c in cands if c["dense_cos"]] or [0.0]),
                    top_ce=cands[0]["score_ce"], tokens=tokens, candidates=cands)
    return ev, {"added": [c["chunk_id"] for c in extra], "rerank_ms": round(rr, 1),
                "total_ms": round((time.perf_counter() - t0) * 1000, 1)}


def stages(question: str, setting, index: S.Index, reranker) -> dict:
    """Where each gold chunk sits at every stage of the production search."""
    scores = index.bm25.get_scores(tokenize(question))
    bm25_order = [index.bm25_ids[i] for i in sorted(range(len(scores)), key=lambda i: -scores[i])]
    qvec = index.embed_query(question)
    dense = [cid for cid, _ in index.dense_ranking(qvec, len(index.chunks))]
    fused = [c["chunk_id"] for c in index.fused(question, qvec, setting.candidates)]
    t0 = time.perf_counter()
    ev = S.search(question, setting, CAP, index, reranker if setting.rerank else None)
    ms = round((time.perf_counter() - t0) * 1000, 1)
    ranked = [c["chunk_id"] for c in ev.candidates]
    final = [p["chunk_id"] for p in ev.passages]
    out = {"final": final, "context_tokens": ev.tokens, "search_ms": ms,
           "dropped_by_budget": [c for c in ranked[:setting.final_passages] if c not in final], "gold": {}}
    for g in GOLD:
        out["gold"][g] = {"bm25_rank": bm25_order.index(g) + 1, "dense_rank": dense.index(g) + 1,
                          "in_fused_top": fused.index(g) + 1 if g in fused else None,
                          "after_rerank": ranked.index(g) + 1 if g in ranked else None, "in_context": g in final}
    out["table3_in_context"] = [c for c in final if c.startswith("AIAYN:T3:")]
    return out


def rows_in_context(ev: S.Evidence) -> list[str]:
    return [f"{p['chunk_id']}: " + "; ".join(r["configuration"] + " " + ", ".join(
        f"{v['metric']}={v['value']}" for v in r["values"] if v["metric"] in ("h", "BLEU (dev)", "PPL (dev)"))
        for r in p.get("rows") or []) for p in ev.passages if p.get("table_id") == "T3" and p["paper"] == "AIAYN"]


def main() -> None:
    settings = retrieval_settings()
    model, reranker = load_model(), load_reranker()
    questions = [json.loads(l) for l in open("eval/questions.jsonl", encoding="utf-8")]
    tq = json.load(open("eval/phase7/table_question_set.json", encoding="utf-8"))
    questions += [{"question_id": q.get("id") or q.get("question_id"), "question": q["question"]} for q in (tq["questions"] if isinstance(tq, dict) else tq)]
    report = {"question": Q11["question"], "expected": Q11["expected_answer"], "gold": GOLD, "corpora": {}}
    for corpus, (idx, proc) in CORPORA.items():
        r = report["corpora"][corpus] = {}
        for size in sorted({s.piece_tokens for s in settings.values()}):
            index = S.Index(size, model, idx, proc)
            for name, st in settings.items():
                if st.piece_tokens != size:
                    continue
                base = stages(Q11["question"], st, index, reranker)
                ev, meta = search_tableref(Q11["question"], st, index, reranker)
                final = [p["chunk_id"] for p in ev.passages]
                r[name] = {"baseline": base, "tableref": {
                    "final": final, "added_candidates": meta["added"], "context_tokens": ev.tokens,
                    "rerank_ms": meta["rerank_ms"], "total_ms": meta["total_ms"],
                    "gold_in_context": {g: g in final for g in GOLD},
                    "gold_rank_after_rerank": {g: next((i + 1 for i, c in enumerate(ev.candidates) if c["chunk_id"] == g), None)
                                               for g in GOLD},
                    "table3_rows_in_context": rows_in_context(ev),
                    "other_tables_in_context": sorted({p["chunk_id"].rsplit(":", 1)[0] for p in ev.passages
                                                       if p.get("kind") == "table" and not p["chunk_id"].startswith("AIAYN:T3:")})}}
                print(corpus, name, "| baseline gold in context:", {g[-3:]: base["gold"][g]["in_context"] for g in GOLD},
                      "| tableref:", {g[-3:]: g in final for g in GOLD}, "| added", len(meta["added"]),
                      "| rerank", meta["rerank_ms"], "ms")
                if corpus == "c6":  # every question that names a table: what changes
                    affected = [q for q in questions if TABLE_REF.search(q["question"])]
                    r[name]["other_questions"] = []
                    for q in affected:
                        before = [p["chunk_id"] for p in S.search(q["question"], st, CAP, index, reranker if st.rerank else None).passages]
                        after = [p["chunk_id"] for p in search_tableref(q["question"], st, index, reranker)[0].passages]
                        r[name]["other_questions"].append({"id": q["question_id"], "question": q["question"],
                                                           "before": before, "after": after, "changed": before != after})
                    unaffected = [q for q in questions if not TABLE_REF.search(q["question"])][:5]
                    r[name]["unaffected_identical"] = all(
                        [p["chunk_id"] for p in S.search(q["question"], st, CAP, index, reranker if st.rerank else None).passages] ==
                        [p["chunk_id"] for p in search_tableref(q["question"], st, index, reranker)[0].passages] for q in unaffected)
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ")
    out = Path(f"experiments/phase10/out/q11_retrieval_{stamp}.json")
    with out.open("x", encoding="utf-8") as f:
        json.dump(report, f, indent=1, ensure_ascii=False)
    print(out.as_posix())


if __name__ == "__main__":
    main()
