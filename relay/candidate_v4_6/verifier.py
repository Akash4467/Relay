"""Candidate verifier v4.6c = v4.5c (frozen v4.2 + check-mark guard) + a deterministic EXACT-SUPPORT rule.
For evaluation only; production stays v4.2 until the owner approves a change.

Root cause (assessment/e2e/live/B12_p7_c6, 2026-11-14): the NLI model rejects sentences copied VERBATIM from the
cited passage ("This inherently sequential nature precludes parallelization within training examples, which becomes
critical at longer sequence lengths, as memory constraints limit batching across examples." = AIAYN:1:c002, 0.07;
"All models are trained on H100 GPUs, using FSDP and activation checkpointing." = MOSHI:4.4:c062, 0.41) and exact
copies of a cited table row (0.15-0.69). Sentence-level premises (v4.4c, iteration 003) fixed that but accepted
wrong facts, so this rule is stricter than NLI, not looser.

Rule (applies ONLY to a sentence whose every failure reason is NLI's 'not supported by the cited passage'; any
citation, absence, value-owner, formula or check-mark failure still rejects):
  quote  - the sentence, normalized (case, whitespace, typographic Unicode, trailing period; 'the authors'/'their'
           read as 'we'/'our'), is a contiguous substring of ONE cited text passage;
  row    - the sentence states >= 1 'column = value' pair, every pair equals a cell of ONE cited table row
           (normalized column and value; a claim value may add '%'), every content word of that row's label is in
           the sentence, and no content word that only a sibling row's label has is in the sentence.
The sentence is then supported by construction: it repeats what the passage says.
"""
from __future__ import annotations

import re

from relay.candidate_v4_5 import verifier as v45
from relay.verify import verifier as v42

VERSION = "v4.6c"
_TYPO = str.maketrans({"‑": "-", "‐": "-", "‒": "-", "–": "-", "−": "-", " ": " ",
                       " ": " ", " ": " ", "’": "'", "‘": "'", "“": '"', "”": '"'})
_PRON = [(r"\bthe authors'", "our"), (r"\bthe authors\b", "we"), (r"\btheir\b", "our"), (r"\bthem\b", "us")]
_NLI_ONLY = "not supported by the cited passage"
_PAIR = re.compile(r"([^=;,:]+?)\s*=\s*([-+]?[\d.]+(?:\s*±\s*[\d.]+)?\s*%?|[^\s;,]+)")
_STOP = {"the", "a", "an", "of", "in", "on", "with", "and", "has", "have", "table", "configuration", "row", "model",
         "models", "matched", "to", "for", "is", "are", "at", "by"}


def norm(t: str) -> str:
    t = t.translate(_TYPO).lower()
    t = re.sub(r"\s+", " ", t).strip().rstrip(".").strip()
    return t


def _quote_ok(claim: str, p: dict) -> bool:
    c = norm(claim)
    for pat, rep in _PRON:
        c = re.sub(pat, rep, c)
    return len(c.split()) >= 4 and c in norm(p["text"])


def _words(t: str) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9][a-z0-9.\-]*", norm(t)) if w not in _STOP}


def _val(v: str) -> str:
    return re.sub(r"\s+", "", norm(v))


def _row_ok(claim: str, p: dict, passages: list[dict] = ()) -> bool:
    pairs = [(norm(a).split(",")[-1].strip(), _val(b)) for a, b in _PAIR.findall(claim)]
    if not pairs or not p.get("rows"):
        return False
    table = p["chunk_id"].rsplit(":", 1)[0] + ":"  # sibling rows: every retrieved chunk of the same table
    rows_all = [r for q in [p, *passages] if q["chunk_id"].startswith(table) for r in q.get("rows") or []]
    labels = {r["configuration"]: _words(r["configuration"]) for r in rows_all}
    cw = _words(claim)
    parens = set().union(*(_words(x) for x in re.findall(r"\(([^()]*)\)", claim))) if "(" in claim else set()
    for r in p["rows"]:
        cells = {(norm(v.get("column") or v["metric"]), _val(v["value"])) for v in r["values"]}
        def match(col: str, val: str) -> bool:  # noqa: E306
            return any((col.endswith(cc) or cc.endswith(col)) and (val == cv or val == cv + "%") for cc, cv in cells)
        if not all(match(c, v) for c, v in pairs):
            continue
        own = labels[r["configuration"]]
        others = set().union(*(w for k, w in labels.items() if k != r["configuration"])) - own if len(labels) > 1 else set()
        colw = set().union(*(_words(v.get("column") or v["metric"]) for v in r["values"]))
        # a qualifier like "(no Inner Monologue)" must be the row's own (column names such as "BLEU / (dev)" are fine)
        if own <= cw and not (others & cw) and parens - colw <= own:
            return True
    return False


def verify(answer: str, passages: list[dict], corpus_ids: set[str], nli_model, entail_min: float,
           entail_min_flagged: float) -> v42.Report:
    rep = v45.verify(answer, passages, corpus_ids, nli_model, entail_min, entail_min_flagged)
    by_id = {p["chunk_id"]: p for p in passages}
    for s in rep.sentences:
        if s.ok or s.absence or not s.reasons or not all(r.startswith(_NLI_ONLY) for r in s.reasons):
            continue
        claim = v42.strip_citations(s.sentence)
        cited = [by_id[c] for c in s.citations if c in by_id]
        if any((_row_ok(claim, p, passages) if p.get("kind") == "table" and p.get("rows") else _quote_ok(claim, p)) for p in cited):
            s.reasons = []
            s.flags = s.flags + ["accepted by exact support (verbatim quote or exact table-row cells)"]
    return rep
