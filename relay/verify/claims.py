"""Atomic claim decomposition before semantic verification (Akash, 2026-11-12): a sentence with several claims
must not pass because one part is supported.

Rule-based and conservative: split only at clause connectors (", and", ", but", ", while", ", so", ";",
", which", "because", "which is why", "so that"). ", and" closing a list ("A, B, and C") is not a connector, and
"since" is left out because it is often about time ("since the 2017 paper"). The result keeps the full sentence first, then each
clause; every one must be supported. The full sentence stays because "X because Y" can have X and Y both true
and still invent the causal link between them.
"""
from __future__ import annotations

import re

_CONNECTOR = re.compile(  # longer connectors first, so ", and this is why" is not split as ", and"
    r",?\s+(?:and this is why|which is why)\s+"
    r"|,\s+(?:and|but|while|whereas|so)\s+|;\s+|,\s+which\s+|\s+(?:because|so that)\s+",
    re.I)


def _clause(text: str, after_which: bool) -> str:
    text = text.strip().rstrip(".!?").strip()
    if after_which:  # "..., which worsens perplexity" -> "This worsens perplexity."
        text = "This " + text
    return text[:1].upper() + text[1:] + "." if text else ""


def atomic_claims(sentence: str) -> list[str]:
    """[sentence] for a single claim, else [sentence, clause 1, clause 2, ...]."""
    parts, start, after_which = [], 0, False
    for m in _CONNECTOR.finditer(sentence):
        if re.fullmatch(r",\s+and\s+", m.group(0), re.I) and "," in sentence[start:m.start()]:
            continue  # "A, B, and C" is one list, not two clauses
        parts.append(_clause(sentence[start:m.start()], after_which))
        after_which = m.group(0).strip(" ,").lower() == "which"
        start = m.end()
    parts.append(_clause(sentence[start:], after_which))
    parts = [p for p in parts if len(p.split()) >= 3]  # fragments too short to be a claim are not split off
    return [sentence] if len(parts) < 2 else [sentence] + parts
