"""Pre-model sufficiency check (DESIGN §14.4): structural, never lexical (Akash's decision, §20.3 2026-11-12).

Refuse before calling the model only when there is nothing to answer from:
- no usable passage (non-empty text) reached the context after the token budget, or
- the question explicitly names several papers and one of them has no passage in the context.
Missing rare words are not evidence that the papers lack the answer; the model may still answer, and the
verifier decides whether its claims are supported.
"""
from __future__ import annotations

from dataclasses import dataclass

from relay.retrieval.paper_mentions import named_papers


@dataclass
class Sufficiency:
    ok: bool
    reason: str | None = None


def check(question: str, passages: list[dict]) -> Sufficiency:
    usable = [p for p in passages if p.get("text", "").strip()]
    if not usable:
        return Sufficiency(False, "no usable passage in the context")
    named = named_papers(question)
    if len(named) >= 2:
        missing = [paper for paper in named if not any(p["paper"] == paper for p in usable)]
        if missing:
            return Sufficiency(False, f"no passage from {', '.join(missing)}, which the question names")
    return Sufficiency(True)
