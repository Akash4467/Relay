"""Reciprocal Rank Fusion (DESIGN §7 step 2): rank-based, so BM25 and cosine scores never get compared."""
from __future__ import annotations

K = 60


def rrf(rankings: list[list[str]], k: int = K) -> list[tuple[str, float]]:
    """rankings: lists of ids, best first. Returns (id, score) sorted by fused score; ties keep first-seen order."""
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking, 1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda kv: -kv[1])
