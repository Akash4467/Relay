"""Candidate verifier v5 = frozen v4 + two optional changes, for offline comparison only.

P1 (citation brackets): gpt-oss often writes citations as 【ID】. A full-width bracket whose content is only chunk
IDs (separated by , or ;) is rewritten to [ID] before the unchanged v4 verify() runs, so every v4 check still
applies. Anything else in 【…】 (empty, malformed, gpt-oss's own "【source†L1】") is left as text: not a citation.

P6 (table-backed absence): reject an absence sentence ("The retrieved passages do not state X.") that v4 accepted,
only when a retrieved table shows the exact requested value is present. Rejection only, never an answer. All of
these must hold:
  1. topic: every content word of the table caption is in X ("Table 8: Evaluation of spoken question answering");
  2. row: X names exactly one variant of a row. "Base (Q)" needs every word of Base and Q, and a negated qualifier
     ("(no Inner Monologue)") only matches a negated request ("without/no Inner Monologue"). A plain "Base" row
     matches a request that mentions a sibling's qualifier only when that sibling is the negated variant (so
     "Moshi with Inner Monologue" is the plain "Moshi" row next to "Moshi (no Inner Monologue)");
  3. value: if X names metric columns, each of them has a real value (not "-") in that row; if X names none, the
     caption is the requested metric and every cell of the row has a real value. A caption and row label alone
     never suffice; one missing cell makes it ambiguous;
  4. nothing unexplained: every content word of X is in the caption, a row label or a column of that table, or is
     a generic word ("accuracy"). A leftover word ("Natural Questions") makes the evidence ambiguous: keep the
     sentence (safe refusal behaviour is unchanged).
"""
from __future__ import annotations

import re

from relay.verify import absence, tables
from relay.verify.citations import CHUNK_ID
from relay.verify.verifier import Report, verify as verify_v4

_FULLWIDTH = re.compile(r"【([^【】]*)】")
GENERIC = {"accuracy", "accuracies", "score", "scores", "result", "results", "performance", "number", "numbers",
           "value", "values", "figure", "figures", "metric", "metrics", "benchmark", "benchmarks", "evaluation",
           "reported", "achieves", "achieved", "reach", "reaches", "obtained", "on", "for", "in", "its", "table",
           "no", "without"}
_NEG = re.compile(r"^\s*(?:no|without|w/o)\b\s*", re.I)
_MISSING = ("", "-", "–", "—")


def normalize_citations(text: str) -> str:
    """P1: 【ID】 / 【ID; ID】 -> [ID] / [ID; ID]; other full-width brackets are left untouched."""
    def fix(m: re.Match) -> str:
        parts = [p.strip() for p in re.split(r"[,;]", m.group(1))]
        return f"[{m.group(1).strip()}]" if parts and all(CHUNK_ID.fullmatch(p) for p in parts) else m.group(0)
    return _FULLWIDTH.sub(fix, text)


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


def verify(answer: str, passages: list[dict], corpus_ids: set[str], nli_model, entail_min: float,
           entail_min_flagged: float, p1: bool = True, p6: bool = True) -> Report:
    """v4 verify() with P1 (before) and P6 (after) switched on or off; p1=p6=False is exactly v4."""
    report = verify_v4(normalize_citations(answer) if p1 else answer, passages, corpus_ids, nli_model, entail_min,
                       entail_min_flagged)
    if p6:
        for s in report.sentences:
            if s.ok and s.absence:
                s.reasons += table_absence_contradicted(s.sentence, passages)
    return report
