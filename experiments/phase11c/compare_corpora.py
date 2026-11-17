"""Phase 11C (REAL-PAPER OFFLINE): Configuration G on c3 vs c6. Same production search(expand=True,
share_captions=True), setting B (20 candidates, 8 passages, 1,600 tokens), reranker, glossary, question sets
(30 original; 32 table [PROVISIONAL, assistant-labelled]; 12 new [PROVISIONAL, assistant-written]) and the Phase 11A
metrics. Gold quotes of eval/questions.jsonl are located per corpus (q09's quote was written against c3's unit-less
text; in c6 it is located with e4's unit insertions undone; the label is unchanged).
Also: q11 stage ranks on both corpora, token usage, and latency interleaved per question (c3, c6, c3, c6, ...).
Writes experiments/phase11c/out/compare_corpora_{UTC}.json."""
from __future__ import annotations

import json
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, ".")
sys.path.insert(0, "experiments/phase9")
sys.path.insert(0, "experiments/phase11")
sys.path.insert(0, "experiments/phase11b")
import retrieval_lab as L  # noqa: E402
from compare import ProdLab  # noqa: E402
from relay.retrieval import glossary  # noqa: E402
from relay.retrieval import search as S  # noqa: E402

CORPORA = {"c3": (Path("data/index"), Path("data/processed")), "c6": L.C6}
OUT = Path("experiments/phase11c/out")


def q11_stages(lab: ProdLab) -> dict:
    q = next(x for x in L.ORIG if x["question_id"] == "q11")["question"]
    st, index = L.SETTINGS["B"], lab.index[256]
    exp = glossary.expand(q)["expanded"]
    scores = index.bm25.get_scores(S.tokenize(exp))
    bm25 = [index.bm25_ids[i] for i in sorted(range(len(scores)), key=lambda i: -scores[i])]
    dense = [cid for cid, _ in index.dense_ranking(index.embed_query(exp), len(index.chunks))]
    fused = [c["chunk_id"] for c in index.fused(exp, index.embed_query(exp), st.candidates)]
    ev = S.search(q, st, L.CAP, index, lab.reranker, expand=True, share_captions=True)
    ranked = [c["chunk_id"] for c in ev.candidates]
    final = [p["chunk_id"] for p in ev.passages]
    rows = {"AIAYN:T3:r01": "base (h = 8): BLEU 25.8", "AIAYN:T3:r02": "(A) h = 1: BLEU 24.9"}
    out = {"expanded_query": exp, "final": final, "tokens": ev.tokens}
    for cid, what in rows.items():
        c = index.chunks[cid]
        out[cid] = {"holds": what, "row_in_chunk": any(what.split(":")[0].split(" ")[0] == r["configuration"] for r in c["rows"]),
                    "bm25_rank": bm25.index(cid) + 1, "dense_rank": dense.index(cid) + 1,
                    "fused_rank": fused.index(cid) + 1 if cid in fused else None,
                    "rerank_rank": ranked.index(cid) + 1 if cid in ranked else None,
                    "score_ce": next((round(x["score_ce"], 3) for x in ev.candidates if x["chunk_id"] == cid), None),
                    "in_final_context": cid in final}
    return out


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    labs, shared, report = {}, {}, {"rows": [], "summary": {}, "q11": {}}
    for corpus, dirs in CORPORA.items():
        labs[corpus] = ProdLab(dirs, shared.get("model"), shared.get("reranker"))
        shared |= {"model": labs[corpus].model, "reranker": labs[corpus].reranker}
        rows = L.evaluate(labs[corpus], ["PROD-G"])
        for r in rows:
            r["corpus"] = corpus
        report["rows"] += rows
        report["summary"][corpus] = L.summarise(rows)
        report["q11"][corpus] = q11_stages(labs[corpus])
    # per-question changes c3 -> c6
    by = {(r["corpus"], r["dataset"], r["question_id"]): r for r in report["rows"]}
    gains, losses = [], []
    for (c, ds, q), r in by.items():
        if c != "c6" or "units" not in r:
            continue
        b = by[("c3", ds, q)]
        tag = f"{ds}:{q} cov {b['coverage']:.2f}->{r['coverage']:.2f} mrr {b['mrr']:.2f}->{r['mrr']:.2f} all {b['all_gold']}->{r['all_gold']}"
        better = r["coverage"] > b["coverage"] or (r["coverage"] == b["coverage"] and (r["mrr"] - b["mrr"] > 0.05 or r["all_gold"] > b["all_gold"]))
        worse = r["coverage"] < b["coverage"] or r["mrr"] - b["mrr"] < -0.05 or r["all_gold"] < b["all_gold"]
        (gains if better and not worse else losses if worse else []).append(tag)
    report["changes_c3_to_c6"] = {"gains": gains, "losses": losses,
                                  "unanswerable_table_in_context_changed": [
                                      (q, by[("c3", ds, q)]["table_in_context"], r["table_in_context"])
                                      for (c, ds, q), r in by.items() if c == "c6" and "table_in_context" in r
                                      and r["table_in_context"] != by[("c3", ds, q)]["table_in_context"]]}
    # tokens and interleaved latency
    qs = [q["question"] for q in L.ORIG + L.TABLE_QS + L.NEW_QS]
    lat = {c: [] for c in CORPORA}
    for q in qs:
        for c in CORPORA:
            t0 = time.perf_counter()
            S.search(q, L.SETTINGS["B"], L.CAP, labs[c].index[256], labs[c].reranker, expand=True, share_captions=True)
            lat[c].append((time.perf_counter() - t0) * 1000)
    report["latency_ms"] = {c: {"median": round(statistics.median(v), 1), "p95": round(sorted(v)[int(0.95 * (len(v) - 1))], 1)}
                            for c, v in lat.items()}
    report["tokens"] = {c: {"mean": round(statistics.mean(r["tokens"] for r in report["rows"] if r["corpus"] == c), 1),
                            "max": max(r["tokens"] for r in report["rows"] if r["corpus"] == c)} for c in CORPORA}
    out = OUT / f"compare_corpora_{L.stamp()}.json"
    out.write_text(json.dumps(report, indent=1, ensure_ascii=False), encoding="utf-8")
    for c in CORPORA:
        for k, v in report["summary"][c].items():
            print(c, f"{k:28}", {x: v[x] for x in ("recall@5", "recall@8", "mrr", "coverage", "all_gold_questions",
                                                   "multi_unit_all_gold", "budget_retention")})
    print("c3->c6 gains:", gains)
    print("c3->c6 losses:", losses)
    print("unanswerable changes:", report["changes_c3_to_c6"]["unanswerable_table_in_context_changed"])
    print("q11:", json.dumps(report["q11"], ensure_ascii=False))
    print("latency:", report["latency_ms"], "tokens:", report["tokens"])
    print(out.as_posix())


if __name__ == "__main__":
    main()
