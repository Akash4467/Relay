"""Table-backed absence check (verification policy v4.2, phase 10C). Promoted VERBATIM from candidate v5's P6
(relay/candidate_v5/verifier.py, evaluated offline in Phase 8; that file is unchanged and still frozen). Rejects an
absence sentence only when a retrieved table shows the exact requested value is present; never accepts anything.
Rules (from the v5 docstring): 1. topic: every content word of the table caption is in the missing part; 2. row: it
names exactly one variant of a row; 3. value: the requested cells hold real values; 4. nothing unexplained: every
content word of the missing part is in the caption, a row label, a column, or is generic. Otherwise ambiguous: kept.
"""
from __future__ import annotations

import re

from relay.verify import absence, tables

GENERIC = {"accuracy", "accuracies", "score", "scores", "result", "results", "performance", "number", "numbers",
           "value", "values", "figure", "figures", "metric", "metrics", "benchmark", "benchmarks", "evaluation",
           "reported", "achieves", "achieved", "reach", "reaches", "obtained", "on", "for", "in", "its", "table",
           "no", "without"}

_NEG = re.compile(r"^\s*(?:no|without|w/o)\b\s*", re.I)

_MISSING = ("", "-", "–", "—")

def _words(text: str) -> set[str]:
    return {t for t in tables._toks(text) if len(t) >= 2 and t not in tables._STOP and t not in absence._STOP}

def _explained(word: str, vocab: set[str]) -> bool:
    return word in GENERIC or tables._present(word, vocab) or any(tables._present(v, {word}) for v in vocab)

def _split_label(label: str) -> tuple[str, str]:
    m = re.match(r"^(.*?)\s*\(([^()]*)\)\s*$", label)
    return (m.group(1), m.group(2)) if m else (label, "")

def _asks_negated(core: str, missing: str) -> bool:
    first = re.escape(core.split()[0]) if core.split() else "$^"
    return bool(re.search(rf"\b(?:no|without|w/o)\s+{first}", missing, re.I))

def _row_requested(label: str, labels: list[str], want: set[str], missing: str) -> bool:
    """Rule 2: does the absence sentence ask for exactly this row variant?"""
    base, qual = _split_label(label)
    if not _words(base) or not all(tables._present(t, want) for t in _words(base)):
        return False
    if qual:
        core = _NEG.sub("", qual)
        return bool(_words(core)) and all(tables._present(t, want) for t in _words(core)) and \
            bool(_NEG.match(qual)) == _asks_negated(core, missing)
    for other in labels:  # plain row: a mentioned sibling qualifier decides which variant is asked for
        b2, q2 = _split_label(other)
        core2 = _NEG.sub("", q2)
        if q2 and _words(b2) == _words(base) and _words(core2) and all(tables._present(t, want) for t in _words(core2)):
            if not _NEG.match(q2) or _asks_negated(core2, missing):
                return False  # the sentence asks for the qualified sibling, not the plain row
    return True

def table_absence_contradicted(sentence: str, passages: list[dict]) -> list[str]:
    """P6: the retrieved table rows that show the 'missing' information is present (empty when not shown)."""
    missing = absence.missing_part(sentence)
    want = _words(missing)
    if not want:
        return []
    by_table: dict[str, list[dict]] = {}
    for p in passages:
        if p.get("kind") == "table" and p.get("rows"):
            by_table.setdefault(f"{p['paper']}:{p.get('table_id')}", []).append(p)
    out = []
    for key, chunks in by_table.items():
        caption = re.sub(r"^Table\s+\w+:\s*", "", chunks[0]["text"].split("\n", 1)[0])
        topic = _words(caption) - GENERIC
        if len(topic) < 2 or not all(tables._present(t, want) for t in topic):
            continue  # 1. not this table's topic
        rows = tables.rows_of(chunks)
        columns = {v["column"] or v["metric"] for r in rows for v in r["values"]}
        named_cols = {c for c in columns if (m := _words(re.sub(r"\([^()]*\)", " ", c))) and
                      all(tables._present(t, want) for t in m)}
        vocab = topic | set().union(*(_words(r["configuration"]) for r in rows)) | \
            set().union(*(_words(c) for c in columns))
        if not all(_explained(w, vocab) for w in want):
            continue  # 4. something asked for is not about this table: ambiguous, keep the absence sentence
        labels = [r["configuration"] for r in rows]
        for r in rows:
            if not _row_requested(r["configuration"], labels, want, missing):
                continue  # 2. not the requested row variant
            values = {v["column"] or v["metric"]: v["value"] for v in r["values"]}
            needed = sorted(named_cols) if named_cols else sorted(columns)  # 3. the exact requested cells
            if all(str(values.get(c, "")).strip() not in _MISSING for c in needed):
                shown = ", ".join(f"{c} = {values[c]}" for c in needed)
                out.append(f"absence contradicted by retrieved table {key} row '{r['configuration']}': {shown}")
    return out
