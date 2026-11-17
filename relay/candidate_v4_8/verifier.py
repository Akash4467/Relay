"""Candidate verifier v4.8 = v4.7c + two isolated, rejection-only fixes (evaluation only; production unchanged).
  fix 1 (marks.py)           the question's check-mark condition must hold for the row a sentence states   (t29)
  fix 2 (absence_labels.py)  a "not stated" sentence is false when a retrieved row with that label reports it (q13)
verify_for(question, fix1, fix2) builds the per-question verifier (v4.7c's guard included). Modules fix1_only and
fix2_only expose each fix alone, so each can be measured in isolation; this module's verify_for has both (v4.8c).
"""
from __future__ import annotations

from relay.candidate_v4_7 import verifier as v47
from relay.candidate_v4_8 import absence_labels, marks
from relay.verify.citations import strip_citations

VERSION = "v4.8c"


def build(question: str, fix1: bool = True, fix2: bool = True, version: str = VERSION):
    base = v47.verify_for(question)

    def verify(answer, passages, corpus_ids, nli_model, entail_min, entail_min_flagged):
        rep = base(answer, passages, corpus_ids, nli_model, entail_min, entail_min_flagged)
        if fix1:
            marks.check(question, rep, passages, strip_citations)
        if fix2:
            absence_labels.check(rep, passages)
        return rep
    verify.VERSION = version
    return verify


def verify_for(question: str):
    return build(question, True, True, VERSION)
