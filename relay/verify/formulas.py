"""Formula consistency (verification policy v4, 2026-11-13). Deterministic; it only ever REJECTS.

The NLI model reads a formula as a bag of symbols: on the v3 calibration split it accepted a smoothing formula with
its weights swapped (0.974), and on the targeted set a softmax without its square root (0.901) and with Q and K
swapped (0.940). This verifier cannot check algebra, so the safe rule is narrow: every LaTeX formula in a claim
that states a relation (=, <, >, <=, >=, ≈) must appear in a cited passage, up to notation (mathnorm.plain_math with
spaces removed). Equivalent notation passes (\\frac vs \\tfrac, \\left( vs (, spacing commands); a rearranged but
equivalent formula (b·a for a·b) is rejected: conservative by design. Formulas written in plain text, without LaTeX
delimiters, are not checked here and are left to NLI alone.
"""
from __future__ import annotations

import re

from relay.retrieval.index_build import index_text
from relay.verify.mathnorm import plain_math

_MATH = re.compile(r"\\\((.+?)\\\)|\\\[(.+?)\\\]|\$\$(.+?)\$\$|\$(.+?)\$", re.S)
_RELATION = re.compile(r"=|<|>|≈")


def compact(text: str) -> str:
    return re.sub(r"\s+", "", plain_math(text.replace("&", " ")))


def formulas(sentence: str) -> list[str]:
    """LaTeX math segments of a sentence that state a relation."""
    segs = [next(g for g in m.groups() if g is not None) for m in _MATH.finditer(sentence)]
    return [s for s in segs if _RELATION.search(compact(s))]


def unsupported(sentence: str, cited: list[dict]) -> list[str]:
    evidence = [compact(index_text(p)) for p in cited]
    return [f"the formula {plain_math(f)!r} is not written this way in the cited passage (formulas must match the "
            f"passage up to notation; algebra is not checked)"
            for f in formulas(sentence) if not any(compact(f) in e for e in evidence)]
