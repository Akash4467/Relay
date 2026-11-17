"""Optional cross-encoder rerank (DESIGN §7 step 3). Kept only if Table 3 shows it earns its latency."""
from __future__ import annotations

RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


def load_reranker():
    from sentence_transformers import CrossEncoder  # heavy import, only when reranking is on
    return CrossEncoder(RERANK_MODEL, device="cpu")


def rerank(question: str, passages: list[dict], model) -> list[dict]:
    """Adds score_ce (raw logit, higher = more relevant) and sorts by it. Input order breaks ties."""
    if not passages:
        return []
    scores = model.predict([(question, p["text"]) for p in passages])
    for p, s in zip(passages, scores):
        p["score_ce"] = float(s)
    return sorted(passages, key=lambda p: -p["score_ce"])
