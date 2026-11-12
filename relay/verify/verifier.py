"""Whole-answer verification (DESIGN §14, revised 2026-11-12). Verification is an evidence consistency check,
not proof of factual correctness.

Level 1 (citations) rejects outright. Level 2 (numbers, terms, attribution) only flags. Semantic support
decides: each claim must be entailed by its cited passages with probability >= entail_min, or >= entail_min_flagged
when level 2 flagged it. A familiar wording never makes an unsupported claim acceptable, and an unfamiliar
wording never rejects a supported one.
"""
from __future__ import annotations

from dataclasses import dataclass

from relay.retrieval.index_build import index_text
from relay.verify import citations, evidence, nli
from relay.verify.citations import SentenceCheck, strip_citations


@dataclass
class Report:
    sentences: list[SentenceCheck]

    @property
    def ok(self) -> bool:
        return bool(self.sentences) and all(s.ok for s in self.sentences) and \
            not all(s.exempt for s in self.sentences)

    def failures(self) -> list[tuple[str, str]]:
        return [(s.sentence, "; ".join(s.reasons)) for s in self.sentences if not s.ok]

    def cited(self) -> list[str]:
        return list(dict.fromkeys(c for s in self.sentences for c in s.citations))


def premises(cited: list[dict]) -> list[str]:
    texts = [index_text(p) for p in cited]
    return texts + (["\n\n".join(texts)] if len(texts) > 1 else [])  # a claim may need two passages together


def verify(answer: str, passages: list[dict], corpus_ids: set[str], nli_model,
           entail_min: float, entail_min_flagged: float) -> Report:
    by_id = {p["chunk_id"]: p for p in passages}
    sentences = citations.check(answer, set(by_id), corpus_ids)
    evidence.check(sentences, by_id)
    todo = [s for s in sentences if s.ok and not s.exempt]
    pairs, owners = [], []
    for s in todo:
        claim = strip_citations(s.sentence)
        for premise in premises([by_id[c] for c in s.citations]):
            pairs.append((premise, claim))
            owners.append(s)
    for s, (ent, _contra) in zip(owners, nli.scores(nli_model, pairs)):
        s.entailment = max(s.entailment or 0.0, ent)
    for s in todo:
        need = entail_min_flagged if s.flags else entail_min
        if (s.entailment or 0.0) < need:
            why = f"not supported by the cited passage (entailment {s.entailment:.2f} < {need})"
            s.reasons.append(why + (f"; {'; '.join(s.flags)}" if s.flags else ""))
    return Report(sentences)
