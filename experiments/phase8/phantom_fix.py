"""Phase 8, investigation A: candidate extraction "e3" = e2 + invisible-spacing removal. Experimental only: it wraps
relay.retrieval.latex_parse without editing it; the e1/e2 modes and corpora c3/c4 are untouched.

Root cause: pylatexenc prints the argument of \\phantom ("$\\phantom{1}42.2$" -> "142.2", "\\phantom{00}1.0" ->
"001.0"), although LaTeX renders it as invisible space; \\hphantom and \\vphantom are already dropped. Fix: before
conversion, remove each \\phantom, \\hphantom and \\vphantom command together with its brace-matched argument (nested
phantoms vanish with their parent). No digits or other text are replaced."""
from __future__ import annotations

import re
import sys
from contextlib import contextmanager

sys.path.insert(0, ".")
from relay.retrieval import latex_parse as lp

_PHANTOM = re.compile(r"\\[hv]?phantom\s*(?=\{)")


def strip_phantoms(tex: str) -> str:
    """Remove invisible-spacing commands and their (brace-matched) arguments; everything else is kept as is."""
    out, pos = [], 0
    for m in _PHANTOM.finditer(tex):
        if m.start() < pos:  # inside an argument already removed (nested phantom)
            continue
        try:
            _, end = lp.group(tex, m.end())
        except ValueError:  # unbalanced braces: leave the text alone
            continue
        out.append(tex[pos:m.start()])
        pos = end
    out.append(tex[pos:])
    return "".join(out)


class _PhantomFreeL2T:
    def __init__(self, inner):
        self.inner = inner

    def latex_to_text(self, tex, *a, **k):
        return self.inner.latex_to_text(strip_phantoms(tex), *a, **k)


@contextmanager
def e3():
    """Within this block latex_parse converts text with phantoms removed (process-local, restored on exit)."""
    original = lp._L2T
    lp._L2T = _PhantomFreeL2T(original)
    try:
        yield
    finally:
        lp._L2T = original


def parse_paper_e3(code: str, raw_dir=lp.RAW_DIR) -> lp.Parsed:
    with e3():
        return lp.parse_paper(code, raw_dir, extraction="e2")
