"""Sentence splitter shared by the chunker and the verifier (DESIGN §14.1)."""
from __future__ import annotations

import re

# words whose trailing period never ends a sentence
_ABBREV = {"e.g", "i.e", "al", "fig", "figs", "eq", "eqs", "sec", "secs", "vs", "cf", "resp", "approx",
           "no", "tab", "ref", "refs", "app", "etc"}
_BOUNDARY = re.compile(r"(?<=[.?!])\s+(?=[A-Z\[(\"“])|\n+")
_LEADING_CITATIONS = re.compile(r"^((?:\[[A-Z]+:[0-9A-Za-z.]+:[cr]\d+(?:[,;]\s*[A-Z]+:[0-9A-Za-z.]+:[cr]\d+)*\]\s*)+)")


def _protected(before: str) -> bool:
    word = re.search(r"(\S+)$", before)
    if not word:
        return False
    w = word.group(1).lower().rstrip(".?!")
    # "e.g." / "i.e." / "et al." / "Fig." / "§3." / single initials like "A."
    return w in _ABBREV or w.startswith("§") or (len(w) == 1 and w.isalpha())


def split_sentences(text: str) -> list[str]:
    out: list[str] = []
    start = 0
    for m in _BOUNDARY.finditer(text):
        if "\n" not in m.group(0) and _protected(text[start:m.start()]):
            continue
        piece = text[start:m.start()].strip()
        if piece:
            out.append(piece)
        start = m.end()
    tail = text[start:].strip()
    if tail:
        out.append(tail)
    # "...claim. [MOSHI:4.2:c017] Next" -> the citation belongs to the sentence before it
    merged: list[str] = []
    for s in out:
        lead = _LEADING_CITATIONS.match(s)
        if lead and merged:
            merged[-1] += " " + lead.group(1).strip()
            s = s[lead.end():].strip()
            if not s:
                continue
        merged.append(s)
    return merged
