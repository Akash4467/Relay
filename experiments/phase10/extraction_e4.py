"""Phase 10C, extraction e4 (experimental; production extraction in relay/retrieval/latex_parse.py is untouched).

e4 = e2 (✓/✗ marks, explicit blank marks, \\multirow carry-down) + e3 (\\phantom family removed, Phase 9A)
     + siunitx: \\SI{160}{\\ms} -> "160 ms" (pylatexenc does not know \\SI: it printed the number and dropped the unit
       macro \\ms, so MOSHI's "160 ms ... 230 ms" became "160 ... 230").

siunitx forms handled: \\SI[opt]{n}{unit}, \\si[opt]{unit}, \\num[opt]{n}, \\SIrange[opt]{a}{b}{unit},
\\qty[opt]{n}{unit} (siunitx v3 name of \\SI). A unit argument is built from known unit and prefix macros (\\milli
\\second -> ms, \\kilo\\hertz -> kHz, \\per -> /, \\percent -> %) and literal text (kbps). An unknown unit macro is
kept as its name without the backslash (never dropped silently) and recorded in UNKNOWN for the report. Nothing
outside these commands is rewritten. Brace matching reuses latex_parse.group; unbalanced input is left as is.
"""
from __future__ import annotations

import re
import sys
from contextlib import contextmanager
from pathlib import Path

sys.path.insert(0, ".")
sys.path.insert(0, "experiments/phase9")
from relay.retrieval import latex_parse as lp  # noqa: E402
import extraction_e3  # noqa: E402

VERSION = "e4"
PREFIX = {"yocto": "y", "zepto": "z", "atto": "a", "femto": "f", "pico": "p", "nano": "n", "micro": "µ",
          "milli": "m", "centi": "c", "deci": "d", "kilo": "k", "mega": "M", "giga": "G", "tera": "T", "peta": "P"}
UNIT = {"second": "s", "s": "s", "minute": "min", "hour": "h", "day": "d", "hertz": "Hz", "Hz": "Hz",
        "kHz": "kHz", "MHz": "MHz", "GHz": "GHz", "ms": "ms", "us": "µs", "ns": "ns", "metre": "m", "meter": "m",
        "gram": "g", "kg": "kg", "byte": "B", "bit": "bit", "percent": r"\%",  # escaped: the result is LaTeX again
 "decibel": "dB", "dB": "dB",
        "watt": "W", "joule": "J", "volt": "V", "ampere": "A", "kelvin": "K", "degree": "°", "celsius": "°C",
        "degreeCelsius": "°C"}
UNKNOWN: list[str] = []
_CMD = re.compile(r"\\(SIrange|SI|si|num|qty)(?![A-Za-z])")
_UNIT_TOKEN = re.compile(r"\\([A-Za-z]+)|([^\\]+)")


def _skip_ws(tex: str, i: int) -> int:
    while i < len(tex) and tex[i] in " \t\n":
        i += 1
    return i


def _optional(tex: str, i: int) -> int:
    i = _skip_ws(tex, i)
    if i < len(tex) and tex[i] == "[":
        end = tex.find("]", i)
        return i if end < 0 else end + 1
    return i


def _args(tex: str, i: int, n: int) -> tuple[list[str], int] | None:
    out = []
    for _ in range(n):
        i = _skip_ws(tex, i)
        if i >= len(tex) or tex[i] != "{":
            return None
        try:
            body, i = lp.group(tex, i)
        except ValueError:
            return None
        out.append(body)
    return out, i


def unit_text(unit: str) -> str:
    """\\milli\\second -> ms; \\kilo\\byte\\per\\second -> kB/s; literal text kept; unknown macros kept by name."""
    out, prefix = [], ""
    for macro, text in _UNIT_TOKEN.findall(unit):
        if text:
            out.append(prefix + text.strip())
            prefix = ""
        elif macro in PREFIX:
            prefix += PREFIX[macro]
        elif macro == "per":
            out.append("/")
        elif macro in ("square", "cubic"):
            continue  # shortcut: power prefixes are not in the papers; upgrade if a corpus uses them
        elif macro in UNIT:
            out.append(prefix + UNIT[macro])
            prefix = ""
        else:
            UNKNOWN.append(macro)
            out.append(prefix + macro)
            prefix = ""
    return "".join(out).replace(" / ", "/").strip()


def expand_siunitx(tex: str) -> str:
    out, pos = [], 0
    for m in _CMD.finditer(tex):
        if m.start() < pos:
            continue
        name = m.group(1)
        i = _optional(tex, m.end())
        got = _args(tex, i, {"SIrange": 3, "SI": 2, "qty": 2, "si": 1, "num": 1}[name])
        if got is None:  # malformed: leave the text alone
            continue
        a, end = got
        text = {"SI": lambda: f"{a[0].strip()} {unit_text(a[1])}", "qty": lambda: f"{a[0].strip()} {unit_text(a[1])}",
                "si": lambda: unit_text(a[0]), "num": lambda: a[0].strip(),
                "SIrange": lambda: f"{a[0].strip()}–{a[1].strip()} {unit_text(a[2])}"}[name]()
        out.append(tex[pos:m.start()] + text)
        pos = end
    out.append(tex[pos:])
    return "".join(out)


class _E4L2T:
    def __init__(self, inner):
        self.inner = inner

    def latex_to_text(self, tex, *a, **k):
        return self.inner.latex_to_text(expand_siunitx(extraction_e3.strip_phantoms(tex)), *a, **k)


@contextmanager
def e4():
    """Within this block latex_parse converts with e3's phantom removal and siunitx expansion (process-local)."""
    original = lp._L2T
    lp._L2T = _E4L2T(original)
    try:
        yield
    finally:
        lp._L2T = original


def parse_paper_e4(code: str, raw_dir: Path = lp.RAW_DIR) -> lp.Parsed:
    with e4():
        return lp.parse_paper(code, raw_dir, extraction="e2")
