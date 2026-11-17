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


# ---- v4.2 (phase 10C): absence claims are recognised by meaning, not by one exact phrase
_VERB = (r"(?:state|provide|mention|report|specify|give|include|list|say|describe|indicate|show|present|discuss|"
         r"contain|define|disclose|document)")
_ABSENCE = re.compile(
    rf"\b(?:do|does|did)\s+not\s+{_VERB}\b|\b(?:don['’]t|doesn['’]t|didn['’]t)\s+{_VERB}\b|"
    rf"\b(?:is|are|was|were|has\s+been|have\s+been)\s+not\s+(?:{_VERB}[a-z]*|available|found)\b|"
    r"\bnot\s+(?:available|found|reported|stated|specified|mentioned|given|provided)\s+in\b|"
    r"\bno\s+(?:information|data|details?|values?|figures?|numbers?|mention|indication|statement)\b|"
    r"\bmakes?\s+no\s+mention\b|\bnowhere\b", re.I)
_SCOPE_RETRIEVED = re.compile(r"\b(?:retrieved|provided|given|supplied|above|these|available)\s+"
                              r"(?:passages?|excerpts?|context|evidence|sources?|text|chunks?)\b|\bthe\s+passages?\b",
                              re.I)
_SCOPE_PAPER = re.compile(r"\b(?:papers?|authors?|article|study|manuscript|literature)\b", re.I)


def classify(claim: str, cited: bool) -> str | None:
    """None: not an absence claim. Otherwise what it says the information is missing from: "retrieved" (the only
    scope the verifier can check), "paper" (whole-paper absence: never established from a limited retrieval) or
    "unscoped". A cited sentence without any scope word is an ordinary claim (e.g. "There is no recurrence [id].")."""
    if claim.startswith(NOT_STATED):
        return "retrieved"
    if not _ABSENCE.search(claim):
        return None
    if _SCOPE_RETRIEVED.search(claim):
        return "retrieved"
    if _SCOPE_PAPER.search(claim):
        return "paper"
    return None if cited else "unscoped"


def missing_part(sentence: str) -> str:
    """What the absence sentence says is missing: after the canonical phrase, else after the negated verb ("do not
    provide X"), else the subject of a passive ("X is not reported in the retrieved passages")."""
    if sentence.startswith(NOT_STATED):
        return sentence[len(NOT_STATED):].strip().rstrip(".")
    m = _ABSENCE.search(sentence)
    if not m:
        return ""
    clean = lambda t: _SCOPE_RETRIEVED.sub(" ", t).strip(" .,;:")  # noqa: E731
    after = re.sub(r"^\s*(?:about|on|regarding|for|of|in)\b", "", clean(sentence[m.end():]), flags=re.I).strip()
    before = re.sub(r"\b(?:in|from|by)\s*$", "", clean(sentence[:m.start()]), flags=re.I).strip()
    if len(_words(after)) >= MIN_WORDS or not before:
        return after
    return f"{before} {after}".strip() if _words(after) else before


def vague(sentence: str, passages: list[dict]) -> list[str]:
    """Fail closed (v4.2): an absence claim whose missing part has fewer than MIN_WORDS content words cannot be
    checked by the sentence rule. Accept it only if its one word appears in no retrieved passage at all."""
    need = _words(missing_part(sentence))
    if not need:
        return ["absence claim too vague to check: it does not say what is missing"]
    if len(need) >= MIN_WORDS:
        return []
    for p in passages:
        have = set().union(*_words(p["text"])) if _words(p["text"]) else set()
        if need[0] & have:
            return [f"absence claim names one term that {p['chunk_id']} mentions: cannot establish it is missing"]
    return []


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
