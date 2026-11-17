"""Phase 8, investigation C (offline): MOSHI Table 3 vs Table 17 (two near-identical Mimi ablation tables).

1. Reproducibility: the 'base' re-chunking must equal the stored c4 chunks.
2. Stage trace (baseline, setting B): first rank of each table after BM25, dense, RRF fusion, re-rank, and in the context.
3. Variants, one change each (experiments/phase8/retrieval_common.py): base, full_caption, metadata, whole_table.
   For each, on 8 adversarial queries + the 8 table-set questions about T3/T17: is the correct table ranked ahead of
   the other, and is it (and the other) in the context? Then the full original + table sets for side effects.
Not tested: adding table identity to the query (no question names a table number, so it is never justified here).
Writes experiments/phase8/out/table_retrieval_{UTC}.json."""
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, ".")
sys.path.insert(0, "experiments/phase8")
from retrieval_common import CAP, SETTINGS, TABLE_QS, MemIndex, run, summarise, variant_chunks  # noqa: E402

from relay.retrieval.index_build import load_model
from relay.retrieval.rerank import load_reranker, rerank
from relay.retrieval.search import search

ADV = [  # (id, gold table, question) written to separate the two tables by their real differences
    ("a1", "T3", "In the Moshi Mimi codec ablation that includes human MUSHRA judgments, which configuration has the highest MUSHRA score?"),
    ("a2", "T3", "Which Mimi codec ablation in the Moshi paper reports MUSHRA alongside ABX, VisQOL and MOSNet?"),
    ("a3", "T3", "In the Mimi ablation with MUSHRA scores, what MUSHRA does the configuration without a Transformer in the decoder get?"),
    ("a4", "T3", "What MUSHRA score does the Mimi ablation report for the configuration with WavLM distillation but no split quantizer?"),
    ("a5", "T17", "In the additional appendix ablation on the Mimi codec, what ABX error does the configuration without any of the five components get?"),
    ("a6", "T17", "In the Moshi appendix's additional ablation on the Mimi codec, what VisQOL does the configuration with only a Transformer in the encoder get?"),
    ("a7", "T17", "In the additional ablation on the Mimi codec, which configuration with quantization rate only is listed, and what is its MOSNet?"),
    ("a8", "T17", "In the extended Mimi codec ablation without MUSHRA, what ABX does the configuration with Transformers in both encoder and decoder but no quantization rate get?"),
]
TQ = [(q["question_id"], q["table_id"], q["question"]) for q in TABLE_QS if q["table_id"] in ("T3", "T17")]
QUERIES = ADV + TQ
embedder, reranker = load_model(), load_reranker()


def first_rank(ids, tid):
    return next((i + 1 for i, c in enumerate(ids) if c.startswith(f"MOSHI:{tid}:")), None)


out = {"queries": [{"id": a, "gold": g, "question": q} for a, g, q in QUERIES], "variants": {}}
stored = {s: [json.loads(l) for l in open(f"data/corpus/c4/processed/chunks_{s}.jsonl", encoding="utf-8")] for s in (256, 512)}
for variant in ("base", "full_caption", "metadata", "whole_table"):
    chunks = {s: variant_chunks(variant, s) for s in (256, 512)}
    if variant == "base":
        same = all(json.dumps(chunks[s], sort_keys=True, ensure_ascii=False) == json.dumps(stored[s], sort_keys=True, ensure_ascii=False)
                   for s in (256, 512))
        out["base_reproduces_stored_c4"] = same
        if not same:
            raise SystemExit("base re-chunking differs from the stored c4 chunks: comparison not controlled")
    idx = {s: MemIndex(chunks[s], embedder) for s in (256, 512)}
    per_q = []
    for qid, gold, question in QUERIES:
        other = "T17" if gold == "T3" else "T3"
        rec = {"id": qid, "gold": gold}
        for st in "ABC":
            s = SETTINGS[st]
            ev = search(question, s, CAP, idx[s.piece_tokens], reranker if s.rerank else None)
            ids = [c["chunk_id"] for c in ev.candidates]
            ctx = [p["chunk_id"] for p in ev.passages]
            rg, ro = first_rank(ids, gold), first_rank(ids, other)
            rec[st] = {"gold_rank": rg, "other_rank": ro, "correct_ahead": rg is not None and (ro is None or rg < ro),
                       "gold_in_context": any(c.startswith(f"MOSHI:{gold}:") for c in ctx),
                       "other_in_context": any(c.startswith(f"MOSHI:{other}:") for c in ctx)}
            if variant == "base" and st == "B":  # stage trace
                index = idx[256]
                qvec = index.embed_query(question)
                bm = index.bm25_ranking(question, 20)
                de = [c for c, _ in index.dense_ranking(qvec, 20)]
                fu = [c["chunk_id"] for c in index.fused(question, qvec, 20)]
                rr = [c["chunk_id"] for c in rerank(question, index.fused(question, qvec, 20), reranker)]
                rec["trace_B"] = {stage: {"gold": first_rank(lst, gold), "other": first_rank(lst, other)}
                                  for stage, lst in (("bm25", bm), ("dense", de), ("rrf", fu), ("rerank", rr), ("context", ctx))}
        per_q.append(rec)
    side = summarise(run(idx, reranker))
    agg = {st: {"correct_ahead": f"{sum(r[st]['correct_ahead'] for r in per_q)}/{len(per_q)}",
                "gold_in_context": f"{sum(r[st]['gold_in_context'] for r in per_q)}/{len(per_q)}",
                "other_in_context": f"{sum(r[st]['other_in_context'] for r in per_q)}/{len(per_q)}"} for st in "ABC"}
    out["variants"][variant] = {"t3_t17": agg, "per_query": per_q, "side_effects": side,
                                "chunks": {s: len(chunks[s]) for s in (256, 512)}}
    print(f"\n### {variant}: chunks {out['variants'][variant]['chunks']}  T3/T17 {agg}")
    for k, v in side.items():
        print(f"   {k}: {v}")
    if variant == "base":
        for r in per_q:
            print(f"   trace {r['id']} ({r['gold']}): " + " | ".join(f"{k} {v['gold']}/{v['other']}" for k, v in r["trace_B"].items()))
p = Path(f"experiments/phase8/out/table_retrieval_{datetime.now(timezone.utc).strftime('%Y-%m-%dT%H-%M-%SZ')}.json")
with p.open("x", encoding="utf-8") as f:
    json.dump(out, f, indent=1, ensure_ascii=False)
print(p)
