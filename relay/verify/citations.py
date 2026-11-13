"""Verifier level 1, citation integrity (DESIGN §14.2). Deterministic.

Every factual sentence must cite at least one passage, and every cited ID must exist in the corpus and have been
retrieved for this request. The one sentence kind without a citation is an absence sentence, and it is checked too
(2026-11-13, after a false "The papers do not state ..." reached a draft in q11):
- it must be scoped to the evidence: "The retrieved passages do not state <missing part>." A claim that the
  *papers* do not state something is rejected: the model only saw the retrieved passages, so it cannot know;
- it must not cite a passage and must not contain a number (a value inside an absence claim is itself a claim).
Whether the retrieved passages really lack the missing part is not provable by entailment; that limit is recorded,
not hidden.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from relay.llm.prompts import NOT_STATED
from relay.verify.sentences import split_sentences

CHUNK_ID = re.compile(r"[A-Z]+:[0-9A-Za-z.]+:[cr]\d+")
_BRACKET = re.compile(r"\[([^\[\]]+)\]")
_UNSCOPED_ABSENCE = re.compile(
    r"^(?:the|these|both|all three)?\s*(?:three\s+)?(?:papers?|authors?|literature)\s+(?:do|does|did)\s+not\s+"
    r"(?:state|mention|say|report|specify|give|provide|discuss)", re.I)


@dataclass
class SentenceCheck:
    sentence: str
    citations: list[str] = field(default_factory=list)
    reasons: list[str] = field(default_factory=list)  # failures: the sentence is rejected
    absence: bool = False                             # "The retrieved passages do not state ..." (checked, not exempt)
    flags: list[str] = field(default_factory=list)    # level 2 warnings: raise the support the sentence needs
    entailment: float | None = None                   # semantic support from the best cited premise
    context: str | None = None                        # previous accepted sentence used to resolve "it"/"they"

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
        claim = strip_citations(sent)
        if claim.startswith(NOT_STATED):
            c.absence = True
            if c.citations:
                c.reasons.append("an absence sentence must not cite a passage")
            if re.search(r"\d", claim):
                c.reasons.append("an absence sentence must not contain a value: a value is a claim that needs a citation")
        elif _UNSCOPED_ABSENCE.match(claim):
            c.reasons.append("claims the papers do not state something; only the retrieved passages were seen, "
                             f"so write \"{NOT_STATED} ...\"")
        elif not c.citations:
            c.reasons.append("no citation")
        for cid in c.citations:
            if cid not in corpus_ids:
                c.reasons.append(f"cites {cid}, which does not exist")
            elif cid not in retrieved_ids:
                c.reasons.append(f"cites {cid}, which was not retrieved for this question")
        out.append(c)
    return out
