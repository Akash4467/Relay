"""Candidate retrieval R3 = R2 with the terminology expansion SCOPED to the slot of the paper it belongs to.
For evaluation only (production search unchanged).

Why (assessment/e2e/live/R2_p7_v47_c6, q14): R2 added AIAYN's phrases ("byte-pair encoding", "word-piece
vocabulary") to the one retrieval query used for EVERY paper's slot. That fixed the AIAYN slot (AIAYN:5.1:c025 first
in context), but it also reshuffled the H-Net slot: the H-Net abstract (learned dynamic chunking) dropped out and
SpaceByte (HNET:3:c041) became H-Net's protected top passage, so the answer's H-Net half described a baseline.

R3, only for questions that name >= 2 papers (production policy "per_paper"): each named paper's slot is retrieved
(BM25 + dense + fusion, k = candidates // papers as in production) and reranked with ITS OWN query: the question
(plus the symbolic glossary expansion when on, as in production) plus only that paper's R2 phrases. Candidates are then
ordered by cross-encoder score (fused score without a reranker); only a slot whose query was EXPANDED has its best
FUSED candidate placed first (R2 protected every slot, which lifted SpaceByte HNET:3:c041 in q14's un-expanded H-Net
slot; a cross-paper question without expansion, e.g. q15, keeps production's order), then production's named-paper
reserve, and production's fill() builds the context under the same
budget. Questions naming fewer than two papers go through R2 unchanged.
"""
from __future__ import annotations

from relay.candidate_r2 import search as r2
from relay.config import app_config
from relay.retrieval import glossary
from relay.retrieval import search as prod
from relay.retrieval.context import fill
from relay.retrieval.paper_mentions import named_papers
from relay.retrieval.rerank import rerank

VERSION = "r3"


def scoped_terms(question: str, paper: str) -> list[str]:
    out = []
    for t in r2.TERMS:
        if t["trigger"].search(question) and paper in t["phrases"]:
            out += [p for p in t["phrases"][paper][1] if p.lower() not in question.lower()]
    return out


def search(question: str, setting, context_cap: int, index, reranker=None, reserve_per_paper: int | None = None,
           expand: bool = False, share_captions: bool = False) -> prod.Evidence:
    papers = named_papers(question)
    if len(papers) < 2:
        return r2.search(question, setting, context_cap, index, reranker, reserve_per_paper, expand, share_captions)
    if setting.rerank and reranker is None:
        raise ValueError("setting needs a reranker")
    sym = glossary.expand(question) if expand else None
    base = sym["expanded"] if sym else question
    k = max(1, setting.candidates // len(papers))
    candidates, slot_terms, best_fused = [], {}, []
    for p in papers:
        terms = scoped_terms(question, p)
        slot_terms[p] = terms
        q = f"{base} {' '.join(terms)}" if terms else base
        fused = index.fused(q, index.embed_query(q), k, p)
        if fused and terms:  # protect only an EXPANDED slot's best fused candidate (the expansion's target)
            best_fused.append(max(fused, key=lambda c: c["score_rrf"]))
        candidates += rerank(q, fused, reranker) if setting.rerank else fused
    key = (lambda c: -c["score_ce"]) if setting.rerank else (lambda c: -c["score_rrf"])
    candidates = sorted(candidates, key=key)
    n = reserve_per_paper if reserve_per_paper is not None else app_config()["retrieval_policy"]["cross_paper_reserve_per_paper"]
    ordered = prod.reserve(candidates, papers, n)
    ordered = best_fused + [c for c in ordered if c not in best_fused]
    passages, tokens = fill(ordered[:setting.final_passages], min(context_cap, setting.context_budget), share_captions)
    cos = [c["dense_cos"] for c in candidates if c["dense_cos"] is not None]
    ev = prod.Evidence(passages=passages, top_dense_cos=max(cos, default=0.0),
                       top_ce=max((c["score_ce"] for c in candidates), default=None) if setting.rerank else None,
                       tokens=tokens, policy="per_paper", named_papers=papers, candidates=candidates, expansion=sym,
                       caption_sharing=share_captions)
    ev.term_expansion = [t for p in papers for t in slot_terms[p]]
    ev.term_expansion_by_paper = slot_terms
    return ev
