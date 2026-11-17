"""Structured table evidence (verification policy v3, 2026-11-13). Deterministic; it only ever REJECTS.

Table chunks carry structured rows (configuration; per cell metric, split, value). The NLI model reads a row as
text and ignores which configuration a value belongs to: on the v3 calibration split it accepted "GNMT + RL reaches
39.2" when 39.2 is the sibling Deep-Att row (0.949). Two structural checks close that gap without accepting
anything on their own:
- value ownership: a number in a table claim that appears only in rows of configurations the claim does not name,
  while the claim does name another configuration of the cited rows, is rejected;
- absence: "The retrieved passages do not state the <metric> of <configuration>" is rejected when a retrieved row of
  that configuration reports that metric.
Configurations with no name of two or more letters (e.g. "(A)") cannot be matched and are never used to reject.
"""
from __future__ import annotations

import re

from relay.verify.evidence import delatex, tokens

_STOP = {"model", "models", "the", "of", "to", "a", "an", "and", "with", "only", "acc", "per", "split", "dev",
         "test", "score"}
_REF_NUMBER = re.compile(r"\b(?:table|figure|fig|section|equation|eq|row|appendix)\.?\s*\(?(\d+)", re.I)
_NUMBER = re.compile(r"\d+(?:\.\d+)?")


def _toks(text: str) -> list[str]:
    return [t for t in tokens(delatex(text)) if not re.fullmatch(r"[\d.%]+", t)]


def _present(term: str, claim: set[str]) -> bool:
    return term in claim or (len(term) >= 3 and any(t.startswith(term) for t in claim))


def named(configuration: str, claim: set[str]) -> bool | None:
    """Does the claim name this configuration? None when the configuration has no usable name."""
    groups = re.findall(r"\(([^()]*)\)", configuration)
    outside = [t for t in _toks(re.sub(r"\([^()]*\)", " ", configuration)) if len(t) >= 2 and t not in _STOP]
    inside = [[t for t in _toks(g) if len(t) >= 2 and t not in _STOP] for g in groups]
    inside = [g for g in inside if g]
    if not outside and not inside:
        return None
    return all(_present(t, claim) for t in outside) and all(any(_present(t, claim) for t in g) for g in inside)


def numbers(text: str) -> set[str]:
    text = delatex(text)
    refs = {m.group(1) for m in _REF_NUMBER.finditer(text)}
    return {n for n in _NUMBER.findall(text) if n not in refs}


def _row_numbers(row: dict) -> set[str]:
    return {n for v in row["values"] for n in _NUMBER.findall(v["value"])}


def rows_of(chunks: list[dict]) -> list[dict]:
    return [r for c in chunks if c.get("kind") == "table" for r in c.get("rows") or []]


def relevant_rows(claim: str, chunks: list[dict]) -> list[dict]:
    """Rows the claim names, or whose values it quotes: the evidence a table claim is about."""
    words, nums = set(_toks(claim)), numbers(claim)
    return [r for r in rows_of(chunks) if named(r["configuration"], words) or (_row_numbers(r) & nums)]


def value_owner_mismatch(claim: str, chunks: list[dict]) -> list[str]:
    rows = rows_of(chunks)
    words = set(_toks(claim))
    mentioned = [r for r in rows if named(r["configuration"], words)]
    if not mentioned:
        return []  # the claim names no configuration of these rows: nothing to compare against
    reasons = []
    for n in sorted(numbers(claim)):
        owners = [r for r in rows if n in _row_numbers(r)]
        if owners and not any(r in mentioned for r in owners) and all(named(r["configuration"], words) is False
                                                                          for r in owners):
            reasons.append(f"the value {n} belongs to {', '.join(sorted({r['configuration'] for r in owners}))} "
                           f"in the cited table, not to {', '.join(sorted({r['configuration'] for r in mentioned}))}")
    return reasons


def absence_contradicted(sentence: str, chunks: list[dict]) -> list[str]:
    words = set(_toks(sentence))
    out = []
    for r in rows_of(chunks):
        if not named(r["configuration"], words):
            continue
        # the metric's main name, every word of it: "Training Cost (FLOPs) / EN-DE" -> training cost
        mains = {v["metric"]: [t for t in _toks(re.sub(r"\([^()]*\)", " ", v["metric"].split("/")[0]))
                               if len(t) >= 2 and t not in _STOP] for v in r["values"]}
        hit = sorted({" ".join(m) for m in mains.values() if m and all(_present(t, words) for t in m)})
        if hit:
            out.append(f"the retrieved row '{r['configuration']}' reports {', '.join(hit)}")
    return out
