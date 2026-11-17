"""Candidate retrieval R2 = production search() + two narrow changes, for evaluation only (production unchanged).

Diagnosis (assessment/e2e/retrieval_fix/diagnose_q14_{c3,c6}.json): q14 asks how the input is "segmented" in the
original Transformer's "translation experiments"; the answering passage AIAYN:5.1:c025 says sentences were "encoded
using byte-pair encoding" and "split tokens into a 32000 word-piece vocabulary". No shared words: within AIAYN it
ranks 51st on BM25 and 36th on dense, outside the 10-candidate AIAYN slot. With its own terms added it ranks 1st on
BM25 and 1st after fusion, but the MiniLM cross-encoder then puts it 9th of 10 in the slot (AIAYN Table 3 rows, whose
repeated caption says "byte-pair encoding ... wordpiece", and AIAYN's conclusion outrank it). Neither change alone
puts it in the context; both together are needed.

1. Terminology expansion (TERMS): if the question contains a tokenization word and names AIAYN or no paper, the
   RETRIEVAL query gets AIAYN's own phrases, verbatim from AIAYN 5.1 ("byte-pair encoding", "word-piece vocabulary";
   checked by a test). H-Net and Moshi already use the question's words ("tokenization", "tokenizer"), so they get
   none: no H-Net passages are pulled in by the expansion. Generation and verification still see the original question.
2. Cross-paper slot protection: only when the question names >= 2 papers (search policy "per_paper"), each named
   paper's best FUSED candidate (BM25 + dense, before the cross-encoder) is placed first, then production's order
   (named-paper reserve, then reranked); the context is refilled with production's fill() under the same budget.
"""
from __future__ import annotations

import re

from relay.config import app_config
from relay.retrieval import search as prod
from relay.retrieval.context import fill
from relay.retrieval.paper_mentions import named_papers

VERSION = "r2"
TERMS = [{
    "concept": "input segmentation / tokenization",
    "trigger": re.compile(r"\b(segment(?:s|ed|ation|ing)?|tokeni[sz](?:e|ed|ation|er|ers|ing)|sub-?words?)\b", re.I),
    "phrases": {"AIAYN": ("AIAYN:5.1:c025", ["byte-pair encoding", "word-piece vocabulary"])},
}]


def term_expansion(question: str) -> list[str]:
    named = named_papers(question)
    out = []
    for t in TERMS:
        if t["trigger"].search(question):
            for paper, (_, phrases) in t["phrases"].items():
                if not named or paper in named:
                    out += [p for p in phrases if p.lower() not in question.lower()]
    return out


def search(question: str, setting, context_cap: int, index, reranker=None, reserve_per_paper: int | None = None,
           expand: bool = False, share_captions: bool = False) -> prod.Evidence:
    terms = term_expansion(question)
    query = f"{question} {' '.join(terms)}" if terms else question
    ev = prod.search(query, setting, context_cap, index, reranker, reserve_per_paper, expand, share_captions)
    if ev.policy == "per_paper":
        best = [max((c for c in ev.candidates if c["paper"] == p), key=lambda c: c["score_rrf"], default=None)
                for p in ev.named_papers]
        protect = [c for c in best if c is not None]
        n = reserve_per_paper if reserve_per_paper is not None else app_config()["retrieval_policy"]["cross_paper_reserve_per_paper"]
        ordered = prod.reserve(ev.candidates, ev.named_papers, n)
        ordered = protect + [c for c in ordered if c not in protect]
        ev.passages, ev.tokens = fill(ordered[:setting.final_passages], min(context_cap, setting.context_budget), share_captions)
    ev.term_expansion = terms  # recorded by the harness; production Evidence has no such field
    return ev
