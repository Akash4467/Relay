"""Verifier level 1, citation integrity (DESIGN §14.2). Deterministic.

Every sentence except the allowed "The papers do not state ..." sentence must cite at least one passage,
and every cited ID must exist in the corpus and have been retrieved for this request.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from relay.llm.prompts import NOT_STATED
from relay.verify.sentences import split_sentences

CHUNK_ID = re.compile(r"[A-Z]+:[0-9A-Za-z.]+:[cr]\d+")
_BRACKET = re.compile(r"\[([^\[\]]+)\]")


@dataclass
class SentenceCheck:
    sentence: str
    citations: list[str] = field(default_factory=list)
    reasons: list[str] = field(default_factory=list)  # failures: the sentence is rejected
    exempt: bool = False  # the "papers do not state" sentence
    flags: list[str] = field(default_factory=list)    # level 2 warnings: raise the support the sentence needs
    entailment: float | None = None                   # semantic support from the best cited premise

    @property
    def ok(self) -> bool:
        return not self.reasons


def citations_in(sentence: str) -> list[str]:
    """Chunk IDs inside [...] brackets; brackets that hold no chunk ID (e.g. "[0, 1]") are not citations."""
    ids: list[str] = []
    for bracket in _BRACKET.findall(sentence):
        parts = [p.strip() for p in re.split(r"[,;]", bracket)]
        if parts and all(CHUNK_ID.fullmatch(p) for p in parts):
            ids += [p for p in parts if p not in ids]
    return ids


def strip_citations(sentence: str) -> str:
    """The claim without its citations, punctuation re-attached: "Mimi runs at 12.5Hz [ID]." -> "... 12.5Hz."."""
    text = _BRACKET.sub(lambda m: "" if citations_in(m.group(0)) else m.group(0), sentence)
    return re.sub(r"\s+([.?!,;:])", r"\1", re.sub(r"\s{2,}", " ", text)).strip()


def check(answer: str, retrieved_ids: set[str], corpus_ids: set[str]) -> list[SentenceCheck]:
    out = []
    for sent in split_sentences(answer):
        c = SentenceCheck(sent, citations_in(sent))
        if strip_citations(sent).startswith(NOT_STATED):
            c.exempt = True
            if c.citations:
                c.reasons.append("the 'papers do not state' sentence must not cite a passage")
        elif not c.citations:
            c.reasons.append("no citation")
        for cid in c.citations:
            if cid not in corpus_ids:
                c.reasons.append(f"cites {cid}, which does not exist")
            elif cid not in retrieved_ids:
                c.reasons.append(f"cites {cid}, which was not retrieved for this question")
        out.append(c)
    return out
