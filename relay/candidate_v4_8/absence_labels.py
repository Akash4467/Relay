"""Fix 2 (candidate v4.8): a "not stated" sentence about a subject that a RETRIEVED table row reports is false.

Root cause (assessment/e2e/live/E25_R3b_p7_v47_c6, q13): "The retrieved passages do not state the average zero-shot
accuracy of the BPE Transformer." was accepted although HNET:T2:r01 ("Transformer (#FLOPs matched to GPT-3 Large)",
Average = 53.3) was in the context; the existing absence checks do not link "BPE Transformer" to the row label.

Rule (rejection only, conservative). An accepted absence sentence is rejected when one retrieved table row R has:
  1. label core: every content word of R's label outside parentheses appears as a whole word in the sentence, and
     the core is not generic ("Row 5", "Configuration", "ours", "baseline" only);
  2. no other row's distinguishing word: no word of the sentence occurs only in the labels of OTHER rows of the same
     table (e.g. "Inner Monologue" when R is "Moshi" and a sibling is "Moshi (no Inner Monologue)"); and if R's
     parenthetical is a negation ("no ...", "without ...", "w/o ..."), the sentence carries that negation;
  3. metric: a content word of the sentence names one of R's columns whose value is present (not "-", not blank).
Words in R's own parenthetical (e.g. "#FLOPs matched to GPT-3 Large") are neither required nor distinguishing.
"""
from __future__ import annotations

import re

_GENERIC = {"row", "rows", "configuration", "config", "model", "models", "ours", "baseline", "base model", "method",
            "the", "a", "an", "of", "with", "and", "to", "for", "in", "on", "matched"}
_GENERIC_METRIC = {"acc", "value", "values", "score", "scores", "rate", "of", "the"}
_NEG = re.compile(r"^\s*(no|without|w/o|non)\b", re.I)
_EMPTY = {"-", "", "(blank)", "—", "–"}


def _words(t: str) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9][a-z0-9\-+.]*[a-z0-9+]|[a-z0-9]", t.lower()) if w not in _GENERIC}


def _core_and_paren(label: str) -> tuple[str, list[str]]:
    parens = re.findall(r"\(([^()]*)\)", label)
    return re.sub(r"\([^()]*\)", " ", label).strip(), parens


def _table(cid: str) -> str:
    return cid.rsplit(":", 1)[0]


def contradicting_row(sentence: str, passages: list[dict]) -> tuple[str, dict] | None:
    sw = _words(sentence)
    tables: dict[str, list[tuple[str, dict]]] = {}
    for p in passages:
        if p.get("kind") == "table" and p.get("rows"):
            tables.setdefault(_table(p["chunk_id"]), []).extend((p["chunk_id"], r) for r in p["rows"])
    for rows in tables.values():
        labels = {r["configuration"]: _words(r["configuration"]) for _, r in rows}
        for cid, r in rows:
            core, parens = _core_and_paren(r["configuration"])
            cw = _words(core)
            if not cw or all(re.fullmatch(r"\d+|row", w) for w in cw) or not cw <= sw:
                continue
            own = labels[r["configuration"]]
            others = set().union(*(w for k, w in labels.items() if k != r["configuration"])) - own
            if sw & others:
                continue
            if any(_NEG.match(pz) for pz in parens) and not any(_NEG.match(pz) and _words(pz) <= sw for pz in parens) \
                    and not re.search(r"\b(no|without|w/o)\b", sentence, re.I):
                continue
            metric_hit = [v for v in r["values"] if v["value"].strip() not in _EMPTY
                          and (_words(v.get("column") or v["metric"]) - _GENERIC_METRIC) & sw]
            if metric_hit:
                return cid, {"row": r["configuration"], "column": metric_hit[0].get("column") or metric_hit[0]["metric"],
                             "value": metric_hit[0]["value"]}
    return None


def check(rep, passages: list[dict]) -> None:
    for s in rep.sentences:
        if not (s.ok and s.absence):
            continue
        hit = contradicting_row(s.sentence, passages)
        if hit:
            cid, h = hit
            s.reasons.append(f"absence contradicted: retrieved row '{h['row']}' ({cid}) reports {h['column']} = {h['value']}")
