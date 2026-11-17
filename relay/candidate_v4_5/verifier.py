"""Candidate verifier v4.5c = frozen v4.2 + one REJECTION-ONLY check, for evaluation only.

Root cause (live B12 t05 and Phase 12 t05, a wrong answer shown to the user): c3 table rows carry no check-mark or
cross cells (0 chunks of c3 contain ✓ or ✗). Asked for "the configuration with a check mark in all five columns",
the model picks a row; the verifier accepts it because the cited row really holds the value it copied.

Check: a sentence that claims marks ("check mark", "checkmark", "tick", "cross", "marked", "no marks", ✓, ✗)
while citing table chunks and none of its cited table chunks contains a ✓ or ✗ is rejected: the evidence cannot
show which row has which marks. It can only add a rejection; every v4.2 decision is otherwise unchanged. On a corpus
whose rows carry their marks (c6), the check passes and v4.2 decides as before.
"""
from __future__ import annotations

import re

from relay.verify import verifier as v42

VERSION = "v4.5c"
MARK_CLAIM = re.compile(r"check[\s-]?marks?|checkmarks?|\bticks?\b|\bticked\b|\bcross(es)?\b|\bmarked\b|\bno marks?\b"
                        r"|\bmarks?\b|✓|✗|✔|✘", re.IGNORECASE)
MARKS = ("✓", "✗", "✔", "✘")


def verify(answer: str, passages: list[dict], corpus_ids: set[str], nli_model, entail_min: float,
           entail_min_flagged: float) -> v42.Report:
    rep = v42.verify(answer, passages, corpus_ids, nli_model, entail_min, entail_min_flagged)
    by_id = {p["chunk_id"]: p for p in passages}
    for s in rep.sentences:
        tables = [by_id[c] for c in s.citations if c in by_id and by_id[c].get("kind") == "table"]
        if s.ok and tables and MARK_CLAIM.search(v42.strip_citations(s.sentence)) \
                and not any(m in t["text"] for t in tables for m in MARKS):
            s.reasons.append("claims check marks or crosses, but the cited table rows carry no marks: the evidence "
                             "cannot show which row is marked")
    return rep
