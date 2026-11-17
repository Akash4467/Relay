"""Phase 11F (REAL-PAPER OFFLINE; no API): q11, t10 and t29 on c3 and c6. For each: the original query, the chunks of
that corpus that hold each gold row (and whether a chunk holds it COMPLETE, i.e. with the cells that answer it), the
production retrieval (Configuration G, setting B) with BM25 / dense / fused / reranked ranks of those chunks, whether
all required evidence is in the final context, and the real verifier v4.2 (real NLI, 0.9) on a draft restating the
gold answer that cites the gold chunks (scripted text, not a model answer). Writes q11_t10_t29_audit.json."""
import json
import sys
from pathlib import Path

sys.path.insert(0, ".")
sys.path.insert(0, "experiments/phase11f")
from reconcile import holds, units_for  # noqa: E402  (the phase 11F gold matcher)
from relay import corpus  # noqa: E402
from relay.config import app_config, piece_sizes, retrieval_settings  # noqa: E402
from relay.eval.retrieval_recall import load_questions  # noqa: E402
from relay.retrieval import glossary  # noqa: E402
from relay.retrieval.index_build import load_model  # noqa: E402
from relay.retrieval.rerank import load_reranker  # noqa: E402
from relay.retrieval.search import policy_flags, search, tokenize  # noqa: E402
from relay.verify import nli  # noqa: E402
from relay.verify.verifier import verify  # noqa: E402

CFG = app_config()
B = retrieval_settings()["B"]
CAP = CFG["pressure"]["L0"]["context_cap"]
Q11 = next(q for q in load_questions(None) if q["question_id"] == "q11")
TQ = {q["question_id"]: q for q in json.loads(Path("eval/phase7/table_question_set.json").read_text(encoding="utf-8"))["questions"]}
# gold for q11 as rows of AIAYN Table 3 (its quotes locate the same rows)
Q11_ROWS = {"question_id": "q11", "paper": "AIAYN", "table_id": "T3", "answerable": True, "question": Q11["question"],
            "gold": {"rows": [{"id": ["Configuration base:"], "cells": ["BLEU / (dev) = 25.8"]},
                              {"id": ["(A): h = 1;"], "cells": ["BLEU / (dev) = 24.9"]}]}}
DRAFTS = {
    "q11": "With a single attention head the development BLEU is 24.9 [{c1}]. The base configuration reaches a "
           "development BLEU of 25.8 [{c0}].",
    "t10": "In the audio quality evaluation table, the causal column is checked for Mimi, adv. loss only, for Same, "
           "downsampled at 16kHz, and for Mimi, non adv. only [{cs}].",
    "t29": "In the extended Mimi codec ablation, the configuration with an ABX error of 31.3% has a VisQOL of 2.37 [{c0}].",
}


def main():
    model, rr = load_model(), load_reranker()
    v = CFG["verification"]
    nli_model = nli.load_nli(v["nli_model"])
    out = {}
    for name in ("c3", "c6"):
        c = corpus.selected(CFG, name)
        corpus.validate(c, piece_sizes())
        index = corpus.load_indexes(c, [256], model)[256]
        for qid, q in (("q11", Q11_ROWS), ("t10", TQ["t10"]), ("t29", TQ["t29"])):
            units = units_for(q, "table", index.chunks)
            holders = [sorted(cid for cid in index.chunks if holds(u, cid, index.chunks)) for u in units]
            complete = [sorted(cid for cid in index.chunks if holds(u, cid, index.chunks, True)) for u in units]
            ev = search(q["question"], B, CAP, index, rr, **policy_flags(CFG))
            exp = ev.expansion["expanded"] if ev.expansion else q["question"]
            sc = index.bm25.get_scores(tokenize(exp))
            bm25 = [index.bm25_ids[i] for i in sorted(range(len(sc)), key=lambda i: -sc[i])]
            dense = [x for x, _ in index.dense_ranking(index.embed_query(exp), len(index.chunks))]
            fused = [x["chunk_id"] for x in index.fused(exp, index.embed_query(exp), B.candidates)]
            ranked = [x["chunk_id"] for x in ev.candidates]
            final = [p["chunk_id"] for p in ev.passages]
            gold_ids = sorted({cid for h in holders for cid in h})
            ranks = {cid: {"bm25": bm25.index(cid) + 1, "dense": dense.index(cid) + 1,
                           "fused": fused.index(cid) + 1 if cid in fused else None,
                           "reranked": ranked.index(cid) + 1 if cid in ranked else None, "in_final_context": cid in final}
                     for cid in gold_ids}
            all_in = all(any(cid in final for cid in h) for h in holders)
            all_complete_in = all(any(cid in final for cid in h) for h in complete)
            cite = [h[0] if h else "NONE" for h in holders]
            draft = DRAFTS[qid].format(c0=cite[0], c1=cite[1] if len(cite) > 1 else cite[0], cs="; ".join(sorted(set(cite))))
            rep = verify(draft, ev.passages, set(index.chunks), nli_model, v["entail_min"], v["entail_min_flagged"])
            out[f"{name}/{qid}"] = {
                "query": q["question"], "expanded_retrieval_query": exp if ev.expansion and ev.expansion["terms"] else None,
                "gold_rows": [u["row"]["id"] + u["row"]["cells"] for u in units],
                "chunks_holding_each_gold_row": holders, "chunks_holding_it_complete": complete,
                "gold_chunk_ranks": ranks, "final_context": final,
                "all_gold_rows_in_final_context": all_in, "all_gold_rows_complete_in_final_context": all_complete_in,
                "verifier_on_gold_restating_draft": {"draft": draft, "accepted": rep.ok,
                                                     "failures": [(s[:70], w[:160]) for s, w in rep.failures()]}}
            print(name, qid, "| all gold in context:", all_in, "| complete:", all_complete_in, "| ranks:",
                  {k[-6:]: (r["fused"], r["reranked"], r["in_final_context"]) for k, r in ranks.items()},
                  "| verifier:", rep.ok, [w[:70] for _, w in rep.failures()][:2])
    Path("experiments/phase11f/q11_t10_t29_audit.json").write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
