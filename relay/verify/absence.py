"""Absence sentences contradicted by retrieved text (verification policy v4, 2026-11-13). Deterministic; REJECTS only.

"The retrieved passages do not state <missing part>." is a claim about the retrieved evidence, so the retrieved
evidence can refute it. v3 caught this only for table rows (tables.absence_contradicted). Here: the missing part's
content words (subject and measure; question and framing words such as "what", "which", "number", "value" dropped)
must not ALL occur in one sentence of one retrieved text passage. If they do, that sentence talks about exactly the
thing said to be missing and the absence sentence is rejected.

Limits, stated rather than hidden: it is lexical (the same words in one sentence), so a sentence that names the
subject and measure without giving a value also rejects (a false rejection the answer pays by refusing), and a
passage that reports the thing in other words is missed (a false acceptance). It never accepts anything and never
looks outside the retrieved passages, so it cannot infer a global absence from incomplete retrieval.
"""
from __future__ import annotations

import re

from relay.llm.prompts import NOT_STATED
from relay.verify.sentences import split_sentences

MIN_WORDS = 2  # one shared word is a mention, not a report
_STOP = set("""a an the of in on at to for by with from and or but as is are was were be been being this that these
those it its their there here what which who whom whose why how when where whether many much number numbers value
values amount exact exactly specific specifically any some each other about into than then do does did done state
stated states say says said report reported mention mentioned give given provide provided used during""".split())


def _variants(word: str) -> set[str]:
    out = {word}
    for suffix, repl in (("ies", "y"), ("es", ""), ("s", ""), ("ed", ""), ("d", ""), ("ing", "")):
        if word.endswith(suffix) and len(word) - len(suffix) >= 3:
            out.add(word[: -len(suffix)] + repl)
    return out


def _words(text: str) -> list[set[str]]:
    return [_variants(w) for w in re.findall(r"[a-z0-9]+", text.lower()) if w not in _STOP and len(w) >= 2]


def missing_part(sentence: str) -> str:
    return sentence[len(NOT_STATED):].strip().rstrip(".") if sentence.startswith(NOT_STATED) else ""


def contradicted(sentence: str, passages: list[dict]) -> list[str]:
    need = _words(missing_part(sentence))
    if len(need) < MIN_WORDS:
        return []
    for p in passages:
        if p.get("kind") == "table":
            continue  # table rows: tables.absence_contradicted
        for s in split_sentences(p["text"].replace("[EQ]", " ")):
            have = set().union(*_words(s)) if _words(s) else set()
            if all(v & have for v in need):
                return [f"{p['chunk_id']} says: \"{s[:120]}{'…' if len(s) > 120 else ''}\""]
    return []
