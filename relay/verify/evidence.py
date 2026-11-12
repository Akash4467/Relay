"""Verifier level 2, evidence signals (DESIGN §14.3, revised 2026-11-12). Deterministic.

Per claim, against the union of its cited passages (header + text, normalized), these raise warning
flags, never a rejection on their own:
- a number (with the unit right after it, e.g. "8 heads", "12.5 Hz") that is not in the cited passages;
- a name-like term absent from them: a digit, hyphen or underscore, a capital after the first letter
  (RQ-Transformer, H100), or a capitalized word that does not start a sentence (Mimi, Helium);
- a paper the claim names with no cited passage from that paper.
Whether a claim is accepted is decided by semantic support (verify/nli.py); a flag raises the support it
needs. Ordinary words are never flagged, however rare they are in the corpus: rarity flagged plain paraphrase
("lets", "parallelized") on the stress-test calibration pairs.
"""
from __future__ import annotations

import re

from relay.retrieval.index_build import index_text
from relay.retrieval.paper_mentions import named_papers, strip_mentions
from relay.verify.citations import SentenceCheck, strip_citations

WINDOW = 5
UNITS = {"ms", "s", "second", "m", "b", "k", "hz", "khz", "kbps", "bps", "token", "layer", "head", "day", "hour",
         "step", "gpu", "frame", "byte", "parameter", "param", "gb", "%", "dimension", "codebook", "quantizer"}
# "one" is left out: it is far more often a pronoun ("one for itself and one for the user") than a count
_NUM_WORDS = {w: str(i) for i, w in enumerate(
    "zero _ two three four five six seven eight nine ten eleven twelve".split()) if w != "_"}
_TOKEN = re.compile(r"[a-z][a-z0-9]*(?:[-_][a-z0-9]+)*|\d+(?:\.\d+)?|%")
_SURFACE = re.compile(r"[A-Za-z][A-Za-z0-9]*(?:[-_][A-Za-z0-9]+)*")


def normalize(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[‐-―−]", "-", text)           # unicode dashes and minus
    text = re.sub(r"(?<=\d),(?=\d{3}\b)", "", text)                # 4,500 -> 4500
    text = re.sub(r"(?<=\d)(?=[a-z%])", " ", text)                 # 12.5hz -> 12.5 hz, 160ms -> 160 ms
    return text


def tokens(text: str) -> list[str]:
    return [_NUM_WORDS.get(t, t) for t in _TOKEN.findall(normalize(text))]


def stem(t: str) -> str:
    for suffix in ("ing", "ed", "es", "s"):
        if len(t) > len(suffix) + 2 and t.endswith(suffix):
            return t[: -len(suffix)]
    return t


def is_number(t: str) -> bool:
    return bool(re.fullmatch(r"\d+(?:\.\d+)?", t))


def technical_terms(text: str) -> list[str]:
    """Normalized name-like words in order of appearance, paper references removed."""
    text = strip_mentions(text)
    out: list[str] = []
    for m in _SURFACE.finditer(text):
        w = m.group(0)
        starts_sentence = m.start() == 0 or text[:m.start()].rstrip().endswith((".", "?", "!", ":"))
        if re.search(r"\d|-|_", w) or re.search(r"[A-Z]", w[1:]) or (w[0].isupper() and not starts_sentence):
            t = normalize(w)
            if t not in out:
                out.append(t)
    return out


def _has_term(term: str, toks: list[str]) -> bool:
    s = stem(term)
    return any(t == term or stem(t) == s for t in toks)


def _number_ok(num: str, unit: str | None, toks: list[str]) -> bool:
    for i, t in enumerate(toks):
        if t == num and (unit is None or any(stem(u) == stem(unit) for u in toks[max(0, i - WINDOW): i + WINDOW + 1])):
            return True
    return False


def signals(claim: str, cited: list[dict]) -> list[str]:
    """Warning flags for one claim (citations already stripped) against its cited passages."""
    toks = [t for p in cited for t in tokens(index_text(p))]
    flags = []
    stoks = tokens(claim)
    for i, t in enumerate(stoks):
        if not is_number(t):
            continue
        nxt = stoks[i + 1] if i + 1 < len(stoks) else None
        unit = nxt if nxt and stem(nxt) in {stem(u) for u in UNITS} else None
        if not _number_ok(t, unit, toks):
            flags.append(f"number {t}{' ' + unit if unit else ''} not in the cited passage")
    for term in technical_terms(claim):
        if not _has_term(term, toks):
            flags.append(f"term '{term}' not in the cited passage")
    for paper in named_papers(claim):
        if not any(p["paper"] == paper for p in cited):
            flags.append(f"names {paper} but cites no {paper} passage")
    return flags


def check(sentences: list[SentenceCheck], passages: dict[str, dict]) -> list[SentenceCheck]:
    """Adds flags to sentences that passed level 1. passages: chunk_id -> retrieved chunk."""
    for c in sentences:
        if not c.exempt and c.ok:
            c.flags = signals(strip_citations(c.sentence), [passages[cid] for cid in c.citations])
    return sentences
