"""Phase 11A Stage 1 (REAL-PAPER OFFLINE): reproduce q11 exactly as in Phase 10C (production search, settings A/B/C,
corpora c3 and c6) and record every stage. Fails (exit 1) unless the Phase 10C failure is reproduced:
B: both gold chunks are candidates and neither reaches the context; c3/B ranks after rerank 10 (r01) and 9 (r02)."""
import json
import sys
from pathlib import Path

sys.path.insert(0, ".")
sys.path.insert(0, "experiments/phase10")
sys.path.insert(0, "experiments/phase11")
import q11_retrieval as p10  # noqa: E402
from retrieval_lab import OUT, stamp  # noqa: E402
from relay.config import retrieval_settings  # noqa: E402
from relay.retrieval import search as S  # noqa: E402
from relay.retrieval.index_build import load_model  # noqa: E402
from relay.retrieval.rerank import load_reranker  # noqa: E402


def main() -> int:
    model, reranker, st = load_model(), load_reranker(), retrieval_settings()
    q = p10.Q11["question"]
    report = {"question": q, "expected_answer": p10.Q11["expected_answer"], "gold": p10.GOLD, "runs": {}}
    for corpus, dirs in p10.CORPORA.items():
        for name, s in st.items():
            index = S.Index(s.piece_tokens, model, *dirs)
            stg = p10.stages(q, s, index, reranker if s.rerank else None)
            ev = S.search(q, s, p10.CAP, index, reranker if s.rerank else None)
            ce = {c["chunk_id"]: round(c["score_ce"], 3) for c in ev.candidates if c.get("score_ce") is not None}
            report["runs"][f"{corpus}/{name}"] = stg | {
                "ranked_after_rerank": [c["chunk_id"] for c in ev.candidates][:12],
                "ce_scores_top12": {c["chunk_id"]: ce.get(c["chunk_id"]) for c in ev.candidates[:12]} if ce else None,
                "gold_ce": {g: ce.get(g) for g in p10.GOLD} if ce else None,
                "final_evidence": [p["chunk_id"] for p in ev.passages]}
    b3 = report["runs"]["c3/B"]["gold"]
    ok = (b3["AIAYN:T3:r01"]["after_rerank"], b3["AIAYN:T3:r02"]["after_rerank"]) == (10, 9) and \
        not any(v["in_context"] for v in b3.values()) and all(v["in_fused_top"] for v in b3.values())
    report["matches_phase10c"] = ok
    OUT.mkdir(parents=True, exist_ok=True)
    out = OUT / f"stage1_q11_{stamp()}.json"
    out.write_text(json.dumps(report, indent=1, ensure_ascii=False), encoding="utf-8")
    for k, r in report["runs"].items():
        print(k, "| gold:", {g[-3:]: (v["bm25_rank"], v["dense_rank"], v["in_fused_top"], v["after_rerank"], v["in_context"])
                             for g, v in r["gold"].items()}, "| CE:", r["gold_ce"], "| dropped:", r["dropped_by_budget"])
    print("matches Phase 10C:", ok, "|", out.as_posix())
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
