"""Fix 1 (candidate v4.8): the question's check-mark CONDITION must hold for the table row an answer sentence states.

Root cause (assessment/e2e/live/E25_R3b_p7_v47_c6, t29): asked for "the configuration with no marks in any of the five
component columns", the redraft answered "Configuration Row 5 reports ABX (↓) = 29.1% [MOSHI:T17:r03]". The sentence is
true of Row 5 (0.99), but Row 5 has a check under Quantization / Rate: a different row, with the condition dropped.

Rule (rejection only; reads only the retrieved row's column cells, never labels or prose):
  1. condition(question): parsed conservatively from row-selection phrasings:
       all checked   "check mark(s) in all (five) component columns", "checked in all ..."
       none checked  "no marks in any of the ... columns", "no marks at all", "with no marks"
       one column    "(a) check mark (only) under X", "check only under X", "with check marks under X and Y",
                     "no mark under X", "without a mark under X"   ("only" => every other mark column unchecked)
     The same column asked both checked and unchecked (a comparison, e.g. t15) => no condition (not enforced).
  2. For each accepted, non-absence sentence citing a table chunk: the rows it states are the cited chunk's rows
     whose non-mark cell values appear in the sentence. A cited table with no mark cells at all cannot show the
     condition (c3 rows) -> rejected when it states a value from such a table.
  3. If a stated row's mark cells do not satisfy the condition (and no stated row does) -> rejected.
A sentence stating no value of any cited row is not judged here (other checks still apply).
"""
from __future__ import annotations

import re

CHECK, BLANK = "✓", "(blank)"
_MARK_VALUES = {"✓", "✗", "✔", "✘", "(blank)", ""}
_NUM = re.compile(r"\d+(?:\.\d+)?")


def _norm(t: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", t.lower()).strip()


def _cols_in(phrase: str, columns: list[str]) -> list[str]:
    """Mark columns named by a question phrase ('Quantization Rate', 'both Transformer columns', 'RQ-Transformer')."""
    p = _norm(phrase)
    pw = set(p.split())
    hits = [c for c in columns if set(_norm(c).split()) <= pw or (_norm(c) and _norm(c) in p)]
    if not hits and "both" in pw:  # "both Transformer columns": every column whose name contains the word
        key = [w for w in pw if w not in {"both", "columns", "column", "and", "the"}]
        hits = [c for c in columns if key and all(k in _norm(c).split() for k in key)]
    return hits


_ALL = re.compile(r"check[\s-]?marks?\s+in\s+all|checked\s+in\s+all|marked\s+in\s+all", re.I)
_NONE = re.compile(r"no\s+marks?\s+(?:in\s+any|at\s+all)|with\s+no\s+marks?\b(?!\s+under)", re.I)
_ONLY = re.compile(r"(?:check[\s-]?marks?|check)\s+only\s+under\s+['\"‘’]?([^'\"‘’?,]+)", re.I)
_UNDER_YES = re.compile(r"(?:with\s+(?:a\s+)?check[\s-]?marks?|a\s+check[\s-]?mark(?:\s*\(✓\))?)\s+(?:under|in)\s+(?:the\s+)?"
                        r"['\"‘’]?(.+?)(?:,?\s+and\s+no\s+marks?\s+under|\s+report|\s+obtain|\s+reach|\?|$)", re.I)
_UNDER_NO = re.compile(r"(?:no\s+marks?|without\s+(?:a\s+)?marks?)\s+under\s+(?:the\s+)?['\"‘’]?(.+?)(?:\s+report|\s+obtain|\s+reach|\?|$)", re.I)


def condition(question: str) -> dict | None:
    """{'all': True} | {'all': False} | {'cols': [(phrase, checked)], 'only': bool} | None."""
    if len(re.findall(r"\brows?\s+with(?:out)?\b", question, re.I)) >= 2 or re.search(r"\bbetween\b|\bor\s+in\s+the\s+row\b", question, re.I):
        return None  # a comparison of two rows (t15, t23): each side has its own condition; not enforced
    if _ALL.search(question):
        return {"all": True}
    if _NONE.search(question):
        return {"all": False}
    m = _ONLY.search(question)
    if m:
        return {"cols": [(m.group(1).strip(), True)], "only": True}
    cols = [(m.group(1).strip(), True) for m in _UNDER_YES.finditer(question)]
    cols += [(m.group(1).strip(), False) for m in _UNDER_NO.finditer(question)]
    if not cols:
        return None
    if re.search(r"\bor\s+in\s+the\s+row\s+without\b|\bwithout\s+one\b", question, re.I):
        return None  # comparison of the checked and unchecked rows (t15): nothing to enforce
    return {"cols": cols, "only": False}


def _mark_cols(row: dict) -> list[str]:
    return [v.get("column") or v["metric"] for v in row["values"] if v["value"].strip() in _MARK_VALUES]


def _satisfies(row: dict, cond: dict) -> bool | None:
    marks = {v.get("column") or v["metric"]: v["value"].strip() == CHECK for v in row["values"]
             if v["value"].strip() in _MARK_VALUES}
    if not marks:
        return False
    if "all" in cond:
        return all(marks.values()) if cond["all"] else not any(marks.values())
    named = {}
    for phrase, want in cond["cols"]:
        parts = re.split(r",\s*|\s+and\s+", phrase) if not _cols_in(phrase, list(marks)) else [phrase]
        for part in parts:
            cs = _cols_in(part, list(marks))
            if not cs:
                return None  # a named column we cannot find: do not judge
            for c in cs:
                named[c] = want
    if any(marks[c] != w for c, w in named.items()):
        return False
    if cond.get("only") and any(marks[c] for c in marks if c not in named):
        return False
    return True


def _cell(t: str) -> str:
    return re.sub(r"[\s%]+", "", t.lower())


def stated_rows(sentence: str, chunk: dict) -> list[dict]:
    """Rows whose WHOLE cell value (e.g. '68.3±1.7') appears in the sentence; else the rows sharing the most numbers
    with it. A shared fragment such as '±1.7' alone never makes two rows equally 'stated'."""
    rows = chunk.get("rows") or []
    cells = lambda r: [v["value"] for v in r["values"] if v["value"].strip() not in _MARK_VALUES and _NUM.search(v["value"])]  # noqa: E731
    sent = _cell(sentence)
    full = [r for r in rows if any(_cell(v) in sent for v in cells(r))]
    if full:
        return full
    nums = set(_NUM.findall(sentence))
    score = {id(r): len(nums & {n for v in cells(r) for n in _NUM.findall(v)}) for r in rows}
    best = max(score.values(), default=0)
    return [r for r in rows if best and score[id(r)] == best]


def check(question: str, rep, passages: list[dict], strip) -> None:
    cond = condition(question)
    if cond is None:
        return
    by_id = {p["chunk_id"]: p for p in passages}
    for s in rep.sentences:
        if not s.ok or s.absence:
            continue
        claim = strip(s.sentence)
        for cid in s.citations:
            p = by_id.get(cid)
            if not p or p.get("kind") != "table" or not p.get("rows"):
                continue
            rows = stated_rows(claim, p)
            if not rows:
                continue
            if not any(_mark_cols(r) for r in p["rows"]):
                s.reasons.append(f"the question selects a row by its check marks, but the cited table rows of {cid} "
                                 "carry no marks: the evidence cannot show which row is meant")
                break
            verdicts = [_satisfies(r, cond) for r in rows]
            if None in verdicts:
                continue
            if not any(verdicts):
                labels = ", ".join(r["configuration"] for r in rows)
                s.reasons.append(f"the question selects the row by its check marks, but the cited row ({labels}) in "
                                 f"{cid} does not have those marks: it is a different row")
                break
