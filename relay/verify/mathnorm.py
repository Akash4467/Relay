"""Plain-text form of LaTeX math, for the NLI input only (verification policy v3, 2026-11-13).

The same claim scored 0.935 or 0.631 depending on whether the model wrote `\\bigl(` or `\\!\\left(` (q05). Both the
claim and the premise are rewritten into one canonical plain form before scoring, so equivalent spellings meet as
the same text. Meaning is kept: numbers, operators, sub/superscripts and accents survive (`\\hat{x}` -> `x-hat`,
never plain `x`, since x-hat and x are different variables); only typesetting is dropped (sizing, spacing,
delimiters, font commands). Displayed answers and citations are never changed.
"""
from __future__ import annotations

import re
import unicodedata

_DROP = {"left", "right", "bigl", "bigr", "Bigl", "Bigr", "big", "Big", "biggl", "biggr", "displaystyle", "textstyle",
         "limits", "nolimits"}
_SPACE = {"quad", "qquad", "space", "enspace", "thinspace"}
_FONT = {"mathrm", "text", "textrm", "mathsf", "mathbf", "mathit", "mathcal", "mathbb", "mathds", "operatorname",
         "boldsymbol", "mathtt", "texttt", "textbf", "textit", "mathfrak", "bm"}
_ACCENT = {"hat": "hat", "widehat": "hat", "bar": "bar", "overline": "bar", "tilde": "tilde", "widetilde": "tilde",
           "vec": "vec", "dot": "dot"}
_FRAC = {"frac", "tfrac", "dfrac", "cfrac"}
_SYMBOL = {"geq": ">=", "ge": ">=", "leq": "<=", "le": "<=", "neq": "!=", "ne": "!=", "approx": "≈", "in": " in ",
           "cdot": "*", "times": "×", "top": "T", "intercal": "T", "Vert": "||", "lVert": "||", "rVert": "||",
           "|": "||", "mid": "|", "ldots": "...", "dots": "...", "cdots": "...", "to": "->", "rightarrow": "->",
           "leftarrow": "<-", "infty": "infinity", "sum": "sum", "prod": "prod", "forall": "for all", "partial": "d",
           "nabla": "grad", "sqrt": "sqrt", "log": "log", "exp": "exp", "sin": "sin", "cos": "cos", "max": "max",
           "min": "min", "arg": "arg", "lbrace": "{", "rbrace": "}", "{": "{", "}": "}", "%": "%", "_": "_"}
_COMBINING = {"̂": "hat", "̄": "bar", "̃": "tilde", "⃗": "vec", "̇": "dot"}
_UNICODE = {" ": " ", " ": " ", " ": " ", "‑": "-", "−": "-", "–": "-", "≥": ">=", "≤": "<=",
            "·": "*", "∈": " in ", "…": "..."}


def _group(s: str, i: int) -> tuple[str, int]:
    """The argument starting at s[i]: a {braced group} or a single character (TeX's \\tfrac12 shorthand)."""
    while i < len(s) and s[i] == " ":
        i += 1
    if i >= len(s):
        return "", i
    if s[i] != "{":
        if s[i] == "\\":
            m = re.match(r"\\[A-Za-z]+", s[i:])
            if m:
                return m.group(0), i + m.end()
        return s[i], i + 1
    depth, j = 0, i
    while j < len(s):
        depth += {"{": 1, "}": -1}.get(s[j], 0)
        if depth == 0:
            return s[i + 1:j], j + 1
        j += 1
    return s[i + 1:], len(s)


def _simple(x: str) -> bool:
    return bool(re.fullmatch(r"[\w.]+", x))


def _convert(s: str) -> str:
    out, i = [], 0
    while i < len(s):
        c = s[i]
        if c in "_^" and s[i + 1:i + 2] == "{":  # keep the group: k_{t-1} -> k_(t-1), never k_t - 1 (v4)
            arg, i = _group(s, i + 1)
            arg = _convert(arg).strip()
            out.append(f"{c}{arg if _simple(arg) else f'({arg})'}")
            continue
        if c != "\\":
            out.append(c)
            i += 1
            continue
        m = re.match(r"\\([A-Za-z]+)", s[i:])
        if not m:  # \, \! \; \: \| \{ \} \( \) \[ \]
            sym = s[i + 1:i + 2]
            out.append(_SYMBOL.get(sym, " " if sym in ",!;: " else ""))
            i += 2
            continue
        name, i = m.group(1), i + m.end()
        if name in _FRAC:
            num, i = _group(s, i)
            den, i = _group(s, i)
            num, den = _convert(num).strip(), _convert(den).strip()
            out.append(f"{num if _simple(num) else f'({num})'}/{den if _simple(den) else f'({den})'}")
        elif name in _ACCENT:
            arg, i = _group(s, i)
            out.append(f"{_convert(arg).strip()}-{_ACCENT[name]}")
        elif name in _FONT:
            arg, i = _group(s, i)
            out.append(_convert(arg))
        elif name in _DROP:
            pass
        elif name in _SPACE:
            out.append(" ")
        elif name in _SYMBOL:
            out.append(_SYMBOL[name])
        else:  # Greek letters and anything else: the name itself (\alpha -> alpha)
            out.append(name)
    return "".join(out)


def plain_math(text: str) -> str:
    """Canonical plain text for NLI scoring. Prose without LaTeX passes through with only whitespace tidied."""
    text = unicodedata.normalize("NFC", text)
    for k, v in _UNICODE.items():
        text = text.replace(k, v)
    text = unicodedata.normalize("NFD", text)  # x̂ -> x + combining hat
    text = re.sub(r"(\w)([̂̄̃⃗̇])", lambda m: f"{m.group(1)}-{_COMBINING[m.group(2)]}", text)
    text = unicodedata.normalize("NFC", text)
    text = text.replace("[EQ]", " ")
    text = re.sub(r"\\[()\[\]]|\$", " ", text)  # math delimiters
    text = _convert(text)
    text = re.sub(r"[{}]", "", text)  # remaining grouping braces: k_{t-1} -> k_t-1
    text = re.sub(r"\|+", lambda m: " " + " ".join(["||"] * max(1, len(m.group(0)) // 2)) + " ", text)
    text = re.sub(r"[ \t]*([=+<>/*]+)[ \t]*", r" \1 ", text)  # one space around operators
    text = re.sub(r"[ \t]*(?<![A-Za-z])-[ \t]*|[ \t]*-(?![A-Za-z])[ \t]*", " - ", text)  # minus, not a word hyphen
    text = re.sub(r"[ \t]*\^[ \t]*|[ \t]*_[ \t]*", lambda m: m.group(0).strip(), text)  # no space around ^ and _
    text = re.sub(r"[ \t]*\([ \t]*", " (", re.sub(r"[ \t]+\)", ")", text))
    text = re.sub(r"([_^]) \(", r"\1(", text)  # a sub/superscript group stays attached: k_(t - 1)
    text = re.sub(r"[ \t]+([.,;:])", r"\1", text)
    text = re.sub(r",(?=\S)(?!\d{3}\b)", ", ", text)  # [0,1] -> [0, 1]; 4,500 stays
    text = re.sub(r"[ \t\r\f\v]+", " ", text)  # line breaks stay: premises keep their labelled lines
    return re.sub(r" ?\n ?", "\n", text).strip()
