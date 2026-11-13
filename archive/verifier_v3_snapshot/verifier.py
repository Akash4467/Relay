"""Whole-answer verification (DESIGN §14, revised 2026-11-13). Verification is an evidence consistency check,
not proof of factual correctness.

Level 1 (citations, absence wording) rejects outright. Level 2 (numbers, terms, attribution) only flags. Semantic
support decides: each claim must be entailed by its cited evidence with probability >= entail_min, or
>= entail_min_flagged when level 2 flagged it.

Evidence per claim (`premises`): each cited passage as indexed; for a table chunk, each row on its own as an explicit
record (paper, table, configuration, metric, split, value) plus all its rows as separate labelled lines, so a
sibling row is never read as a competing value for the same configuration and a comparison of two rows can still be
supported; and, when several passages are cited, all of them joined.

Context (`REFERENCE`): a claim that opens with a reference ("It", "They", "This", ...) is also scored with the
previous sentence placed before each premise, but only when that sentence was itself accepted. The claim is still
scored on its own: context only resolves what "it" means and cannot add support that the evidence lacks.
Policy v3 (2026-11-13): (1) claim and premise go through mathnorm.plain_math before NLI, so equivalent LaTeX
spellings meet as one text; (2) each clause of a sentence (claims.atomic_claims) must be entailed as well as the
whole sentence, so a supported clause cannot carry an unsupported one; a clause opening with a reference ("This
...") is also scored with the sentence's first clause before the premise; (3) table evidence is the cited row on
its own plus only the relevant rows (named or value-matched) across all cited tables, never a chunk's unrelated
sibling rows; tables.py rejects a value owned by an unnamed sibling row and an absence sentence that a retrieved row
contradicts. Nothing here accepts a claim without NLI entailment >= the threshold.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from relay.retrieval.index_build import PAPER_TITLES, index_text
from relay.verify import citations, evidence, nli, tables
from relay.verify.claims import atomic_claims
from relay.verify.mathnorm import plain_math
from relay.verify.citations import SentenceCheck, strip_citations

REFERENCE = re.compile(r"^(?:it|its|they|their|them|this|these|those|that|such|both|the latter|the former)\b", re.I)


@dataclass
class Report:
    sentences: list[SentenceCheck]

    @property
    def ok(self) -> bool:
        # an answer made only of absence sentences answers nothing: it is a refusal
        return bool(self.sentences) and all(s.ok for s in self.sentences) and \
            not all(s.absence for s in self.sentences)

    def failures(self) -> list[tuple[str, str]]:
        return [(s.sentence, "; ".join(s.reasons)) for s in self.sentences if not s.ok]

    def cited(self) -> list[str]:
        return list(dict.fromkeys(c for s in self.sentences for c in s.citations))


def _row_line(row: dict) -> str:
    # v3: one plain sentence per row (v2 "Configuration X: m is v; ..." scored lower on supported calibration rows)
    values = ", ".join(f"{v['metric']}{f' on the {v['split']} set' if v['split'] else ''} = {v['value']}"
                       for v in row["values"])
    return f"The configuration {row['configuration']} has {values}."


def _head(chunk: dict) -> str:
    return f"{PAPER_TITLES[chunk['paper']]}, {chunk['text'].split(chr(10), 1)[0]}"


def table_premises(chunk: dict) -> list[str]:
    """One premise per row: the table's caption and that row alone."""
    return [f"{_head(chunk)}\n{_row_line(r)}" for r in chunk["rows"]]


def premises(cited: list[dict], claim: str = "") -> list[str]:
    out = []
    for p in cited:
        out += table_premises(p) if p.get("kind") == "table" and p.get("rows") else [index_text(p)]
    # the rows the claim is about, across all cited tables, as labelled lines (comparisons, base vs variant)
    rows = tables.relevant_rows(claim, cited)
    heads = list(dict.fromkeys(_head(p) for p in cited if p.get("kind") == "table" and p.get("rows")))
    row_block = "\n".join(heads + [_row_line(r) for r in rows]) if rows else ""
    if len(rows) > 1:
        out.append(row_block)
    texts = [index_text(p) for p in cited if not (p.get("kind") == "table" and p.get("rows"))]
    if len(cited) > 1 and texts:  # a claim may need two passages together; tables add only their relevant rows
        out.append("\n\n".join(texts + ([row_block] if row_block else [])))
    return list(dict.fromkeys(out))


def _support(claim: str, prem: list[str], contexts: list[str], nli_model) -> float:
    pairs = [(plain_math(p), plain_math(claim)) for p in prem]
    pairs += [(plain_math(f"{c}\n{p}"), plain_math(claim)) for c in contexts for p in prem]
    return max(e for e, _ in nli.scores(nli_model, pairs))


def verify(answer: str, passages: list[dict], corpus_ids: set[str], nli_model,
           entail_min: float, entail_min_flagged: float) -> Report:
    by_id = {p["chunk_id"]: p for p in passages}
    sentences = citations.check(answer, set(by_id), corpus_ids)
    evidence.check(sentences, by_id)
    previous_accepted: str | None = None
    for s in sentences:
        if s.ok and s.absence:
            s.reasons += [f"absence contradicted: {r}" for r in tables.absence_contradicted(s.sentence, passages)]
        if not s.ok or s.absence:
            previous_accepted = None
            continue
        claim = strip_citations(s.sentence)
        cited = [by_id[c] for c in s.citations]
        s.reasons += tables.value_owner_mismatch(claim, cited)
        if not s.ok:
            previous_accepted = None
            continue
        prem = premises(cited, claim)
        parts = atomic_claims(claim)  # [claim] or [claim, clause 1, clause 2, ...]
        context = [previous_accepted] if previous_accepted and REFERENCE.match(claim) else []
        s.context = context[0] if context else None
        need = entail_min_flagged if s.flags else entail_min
        scores = []
        for i, part in enumerate(parts):
            # the whole sentence and its first clause may lean on the previous accepted sentence ("It ..."); a
            # later clause opening with a reference ("This worsens ...") leans on this sentence's first clause
            ctx = context if i <= 1 else ([parts[1]] if REFERENCE.match(part) else [])
            scores.append(_support(part, prem, ctx, nli_model))
        s.entailment = min(scores)
        if s.entailment < need:
            weakest = parts[scores.index(s.entailment)]
            where = "" if weakest == claim else f" (clause: \"{weakest}\")"
            why = f"not supported by the cited passage{where} (entailment {s.entailment:.2f} < {need})"
            s.reasons.append(why + (f"; {'; '.join(s.flags)}" if s.flags else ""))
        previous_accepted = claim if s.ok else None
    return Report(sentences)
