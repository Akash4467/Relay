"""search(question, setting, context_cap) -> Evidence (DESIGN §7). The retrieval tool call of the pipeline.

BM25 top-k and dense top-k -> RRF -> [cross-encoder] -> final_passages -> context cap.
Policy "per_paper" (only when the question names >= 2 papers): candidates are retrieved inside each named
paper, reranked together, and the context reserves passages from every named paper before filling by rank.
CPU-heavy calls (embedding, cross-encoder) are plain sync calls here; the server runs search()
in its bounded thread pool so they never block the event loop.
"""
from __future__ import annotations

import json
import pickle
from dataclasses import dataclass, field
from pathlib import Path

import faiss
import numpy as np

from relay.config import RetrievalSetting, app_config
from relay.retrieval import glossary
from relay.retrieval.context import fill
from relay.retrieval.index_build import INDEX_DIR, QUERY_PREFIX, tokenize
from relay.retrieval.latex_parse import PROCESSED_DIR
from relay.retrieval.paper_mentions import named_papers
from relay.retrieval.rerank import rerank
from relay.retrieval.rrf import rrf


@dataclass
class Evidence:
    passages: list[dict]
    top_dense_cos: float
    top_ce: float | None
    tokens: int
    policy: str = "global"                       # "global" | "per_paper"
    named_papers: list[str] = field(default_factory=list)
    candidates: list[dict] = field(default_factory=list)  # fused (and reranked) list before the cap, for analysis
    expansion: dict | None = None                 # phase 11B: glossary.expand() result when expansion was on
    caption_sharing: bool = False                 # phase 11B: tokens counted on the caption-shared model view


class Index:
    """Loaded indexes for one piece size."""

    def __init__(self, size: int, embedder, index_dir: Path = INDEX_DIR, processed_dir: Path = PROCESSED_DIR):
        with (index_dir / f"bm25_{size}.pkl").open("rb") as f:
            saved = pickle.load(f)  # our own offline build output, never user-supplied
        self.bm25, self.bm25_ids = saved["bm25"], saved["ids"]
        self.faiss = faiss.read_index(str(index_dir / f"faiss_{size}.index"))
        self.faiss_ids = json.loads((index_dir / f"faiss_{size}.ids.json").read_text(encoding="utf-8"))
        chunks = [json.loads(l) for l in (processed_dir / f"chunks_{size}.jsonl").read_text(encoding="utf-8").splitlines()]
        self.chunks = {c["chunk_id"]: c for c in chunks}
        if set(self.bm25_ids) != set(self.chunks) or set(self.faiss_ids) != set(self.chunks):
            raise ValueError(f"index for size {size} is stale: rebuild with index_build")
        self.embedder = embedder

    def _in(self, cid: str, paper: str | None) -> bool:
        return paper is None or self.chunks[cid]["paper"] == paper

    def bm25_ranking(self, question: str, k: int, paper: str | None = None) -> list[str]:
        scores = self.bm25.get_scores(tokenize(question))
        order = np.argsort(-scores, kind="stable")
        return [self.bm25_ids[i] for i in order if scores[i] > 0 and self._in(self.bm25_ids[i], paper)][:k]

    def embed_query(self, question: str) -> np.ndarray:
        return np.asarray(self.embedder.encode([QUERY_PREFIX + question], normalize_embeddings=True), dtype="float32")

    def dense_ranking(self, qvec: np.ndarray, k: int, paper: str | None = None) -> list[tuple[str, float]]:
        # within one paper, rank the whole index and filter: the corpus is a few hundred vectors
        scores, idx = self.faiss.search(qvec, self.faiss.ntotal if paper else k)
        hits = [(self.faiss_ids[i], float(s)) for s, i in zip(scores[0], idx[0]) if i >= 0]
        return [(cid, s) for cid, s in hits if self._in(cid, paper)][:k]

    def fused(self, question: str, qvec: np.ndarray, k: int, paper: str | None = None) -> list[dict]:
        dense = self.dense_ranking(qvec, k, paper)
        cos = dict(dense)
        ranked = rrf([self.bm25_ranking(question, k, paper), [cid for cid, _ in dense]])[:k]
        return [dict(self.chunks[cid], score_rrf=s, dense_cos=cos.get(cid)) for cid, s in ranked]


def reserve(candidates: list[dict], papers: list[str], per_paper: int) -> list[dict]:
    """Ranked list with each named paper's best passages pulled to the front, round-robin (best of each paper
    first, then second best, ...), so even a tight token cap keeps at least one passage per paper."""
    picked: list[dict] = []
    for _ in range(per_paper):
        for paper in papers:
            nxt = next((c for c in candidates if c["paper"] == paper and c not in picked), None)
            if nxt is not None:
                picked.append(nxt)
    return picked + [c for c in candidates if c not in picked]


def policy_flags(cfg: dict) -> dict:
    """The phase 11B switches from config retrieval_policy, as search() keyword arguments (production callers)."""
    pol = cfg.get("retrieval_policy") or {}  # absent switches are off
    return {"expand": bool(pol.get("symbolic_query_expansion", False)),
            "share_captions": bool(pol.get("prompt_caption_sharing", False))}


def search(question: str, setting: RetrievalSetting, context_cap: int, index: Index, reranker=None,
           reserve_per_paper: int | None = None, expand: bool = False, share_captions: bool = False) -> Evidence:
    """context_cap is the pressure level's cap; the tighter of it and setting.context_budget applies.
    Phase 11B (Configuration G, off unless the caller passes the config flags): expand=True retrieves (BM25, dense,
    cross-encoder) with glossary.expand()'s query; the ORIGINAL question still decides the paper policy, and is what
    generation and verification see. share_captions=True counts the evidence budget on the caption-shared model view.
    Limits, scores and the returned chunks are otherwise unchanged."""
    if setting.rerank and reranker is None:
        raise ValueError("setting needs a reranker")
    expansion = glossary.expand(question) if expand else None
    query = expansion["expanded"] if expansion else question
    papers = named_papers(question)
    per_paper = len(papers) >= 2
    qvec = index.embed_query(query)
    if per_paper:
        k = max(1, setting.candidates // len(papers))
        candidates = sorted((c for p in papers for c in index.fused(query, qvec, k, p)),
                            key=lambda c: -c["score_rrf"])
    else:
        candidates = index.fused(query, qvec, setting.candidates)
    if setting.rerank:
        candidates = rerank(query, candidates, reranker)  # global rerank across papers
    ordered = candidates
    if per_paper:
        n = reserve_per_paper if reserve_per_paper is not None else app_config()["retrieval_policy"]["cross_paper_reserve_per_paper"]
        ordered = reserve(candidates, papers, n)
    passages, tokens = fill(ordered[:setting.final_passages], min(context_cap, setting.context_budget), share_captions)
    cos = [c["dense_cos"] for c in candidates if c["dense_cos"] is not None]
    return Evidence(
        passages=passages,
        top_dense_cos=max(cos, default=0.0),
        top_ce=candidates[0]["score_ce"] if setting.rerank and candidates else None,
        tokens=tokens,
        policy="per_paper" if per_paper else "global",
        named_papers=papers,
        candidates=candidates,
        expansion=expansion,
        caption_sharing=share_captions,
    )
