"""q14 retrieval diagnosis (offline, no LLM/NLI): rank of AIAYN:5.1:c025 in the AIAYN slot (BM25, dense, fused,
reranked) and the final context of production search(), for the original question and candidate expansions.
usage: diagnose_q14.py c3|c6"""
import json
import sys

sys.path.insert(0, ".")
from relay import corpus  # noqa: E402
from relay.config import app_config, piece_sizes, retrieval_settings  # noqa: E402
from relay.retrieval.index_build import load_model  # noqa: E402
from relay.retrieval.rerank import load_reranker, rerank  # noqa: E402
from relay.retrieval.search import policy_flags, search  # noqa: E402

Q = "How is the input segmented in the original Transformer's translation experiments compared with H-Net?"
GOLD = "AIAYN:5.1:c025"
VARIANTS = {
    "original": Q,
    "+ byte-pair encoding, word-piece vocabulary": Q + " byte-pair encoding word-piece vocabulary",
    "+ byte-pair encoding, word-piece, tokens, vocabulary": Q + " byte-pair encoding word-piece tokens vocabulary",
    "+ encoded using byte-pair encoding": Q + " encoded using byte-pair encoding",
    "+ tokenization (generic word)": Q + " tokenization",
    "+ word-piece vocabulary, source-target vocabulary": Q + " word-piece vocabulary shared source-target vocabulary",
    "+ sentences were encoded, word-piece vocabulary": Q + " sentences were encoded word-piece vocabulary",
    "+ byte-pair, word-piece vocabulary, source-target": Q + " byte-pair word-piece vocabulary shared source-target vocabulary",
}
import sys as _s
if len(_s.argv) > 2:
    VARIANTS = {k: v for k, v in VARIANTS.items() if k in _s.argv[2:]}
name = sys.argv[1]
cfg = app_config()
c = corpus.selected(cfg, name)
corpus.validate(c, piece_sizes())
model, rr = load_model(), load_reranker()
idx = corpus.load_indexes(c, [256], model)[256]
B = retrieval_settings()["B"]
cap = cfg["pressure"]["L0"]["context_cap"]
out = {}
for label, q in VARIANTS.items():
    qv = idx.embed_query(q)
    bm = idx.bm25_ranking(q, 10_000, "AIAYN")
    de = [x for x, _ in idx.dense_ranking(qv, 10_000, "AIAYN")]
    fused = [x["chunk_id"] for x in idx.fused(q, qv, B.candidates // 2, "AIAYN")]
    rer = [x["chunk_id"] for x in rerank(q, idx.fused(q, qv, B.candidates // 2, "AIAYN"), rr)]
    # full production search with this text as the retrieval query (expansion off: we pass the expanded text)
    ev = search(q, B, cap, idx, rr, expand=False, share_captions=True)
    final = [p["chunk_id"] for p in ev.passages]
    r = {"bm25_rank_in_AIAYN": bm.index(GOLD) + 1 if GOLD in bm else None, "dense_rank_in_AIAYN": de.index(GOLD) + 1,
         "fused_rank_in_AIAYN_slot(10)": fused.index(GOLD) + 1 if GOLD in fused else None,
         "reranked_rank_in_AIAYN_slot": rer.index(GOLD) + 1 if GOLD in rer else None,
         "in_final_context": GOLD in final, "final_context": final,
         "hnet_in_context": sum(x.startswith("HNET") for x in final)}
    out[label] = r
    print(f"{label:55s} bm25 {r['bm25_rank_in_AIAYN']}  dense {r['dense_rank_in_AIAYN']}  fused {r['fused_rank_in_AIAYN_slot(10)']}  "
          f"reranked {r['reranked_rank_in_AIAYN_slot']}  IN CONTEXT {r['in_final_context']}  | HNET {r['hnet_in_context']} | {final}")
json.dump(out, open(f"assessment/e2e/retrieval_fix/diagnose_q14_{name}.json", "w"), indent=1)
