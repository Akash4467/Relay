"""Which papers a question names explicitly. Drives the per-paper coverage policy (DESIGN §7, §20.3 2026-11-12)
and later the cross_paper tag in classify_query.

Only explicit references count. A bare "Transformer" does not name AIAYN: the other two papers use
"RQ-Transformer", "BPE Transformer" and "Temporal Transformer" for their own components and baselines.
"""
from __future__ import annotations

import re

_PATTERNS = {
    "AIAYN": re.compile(r"attention is all you need|vaswani|\boriginal transformer|\btransformer paper", re.I),
    "MOSHI": re.compile(r"\bmoshi\b", re.I),
    "HNET": re.compile(r"\bh-?net\b|dynamic chunking paper", re.I),
}


def named_papers(question: str) -> list[str]:
    """Paper codes named in the question, in library order."""
    return [code for code, pat in _PATTERNS.items() if pat.search(question)]


def strip_mentions(text: str) -> str:
    """Text with the paper references removed, so "Vaswani" or "H-Net" never count as content terms."""
    for pat in _PATTERNS.values():
        text = pat.sub(" ", text)
    return text
