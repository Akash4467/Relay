"""Phase 5 experiment T: deterministic verification of single-row table claims (moved verbatim from
candidates.py so it can be tested on its own). Experimental only; not used by the pipeline."""
import re

from relay.candidate_v5.verifier import GENERIC, _row_requested, _words
from relay.verify import tables
from relay.verify.citations import strip_citations

BLOCK = re.compile(r"\b(than|better|worse|higher|lower|more|less|outperform\w*|improv\w*|exceed\w*|beat\w*|best|worst|"
                   r"not|no|never|nor|without|triple\w*|double\w*|twice|compared|versus|vs|only|all|every)\b", re.I)
VERBS = {"achieves", "achieve", "achieved", "reaches", "reach", "scores", "score", "gets", "obtains", "has", "have",
         "attains", "records", "yields", "accuracy", "accuracies", "respectively"}


def _compact(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", s.lower())


def table_supported(sentence: str, cited: list[dict], passages: list[dict]) -> str | None:
    if not cited or any(not (p.get("kind") == "table" and p.get("rows")) for p in cited):
        return None
    claim = strip_citations(sentence)
    # a qualifier that names a row variant ("(no Inner Monologue)") may appear as "with/without/no <qualifier>":
    # the row rule decides it, so it is not a negation here; any other negation or comparative falls back to NLI
    keys0 = {(p["paper"], p.get("table_id")) for p in cited}
    labels0 = [r["configuration"] for p in passages if p.get("kind") == "table" and (p["paper"], p.get("table_id")) in keys0
               for r in p.get("rows", [])]
    without_q = claim
    for q in {re.sub(r"^\s*(?:no|without|w/o)\s+", "", g, flags=re.I) for l in labels0 for g in re.findall(r"\(([^()]*)\)", l)}:
        if q.strip():
            without_q = re.sub(r"\b(?:with|without|no|w/o)\s+" + re.escape(q.strip()) + r"\b", " ", without_q, flags=re.I)
    if BLOCK.search(without_q):
        return None
    if not tables.numbers(claim):
        return None
    keys = {(p["paper"], p.get("table_id")) for p in cited}
    if len(keys) != 1:
        return None
    same = [p for p in passages if p.get("kind") == "table" and (p["paper"], p.get("table_id")) in keys]
    rows_all = tables.rows_of(same)
    labels = [r["configuration"] for r in rows_all]
    want = _words(claim)
    chosen = [r for r in tables.rows_of(cited) if _row_requested(r["configuration"], labels, want, claim)]
    if len(chosen) != 1:
        return None
    row = chosen[0]
    cells = {(v["column"] or v["metric"]): str(v["value"]).strip() for v in row["values"]}
    used = []
    for seg in re.split(r",|;|\band\b", claim):
        for n in re.findall(r"\d+(?:\.\d+)?", seg):
            if n in re.findall(r"\d+(?:\.\d+)?", row["configuration"]):
                continue  # part of the row label, e.g. "7B"
            hit = [c for c, v in cells.items() if v == n and _compact(re.sub(r"\([^()]*\)", "", c)) in _compact(seg)]
            if not hit:
                return None
            used += hit
    if not used:
        return None
    caption = re.sub(r"^Table\s+\w+:\s*", "", same[0]["text"].split("\n", 1)[0])
    vocab = _words(caption) | set().union(*(_words(l) for l in labels)) | set().union(*(_words(c) for c in cells))
    abbrevs = {t.rstrip(".").lower() for c in cells for t in c.split() if t.endswith(".") and len(t) <= 3}
    for w in want:
        if re.fullmatch(r"[\d.%]+", w) or w in GENERIC or w in VERBS:
            continue
        if tables._present(w, vocab) or any(tables._present(v, {w}) for v in vocab) or any(w.startswith(a) for a in abbrevs):
            continue
        return None
    return f"table-verified: row '{row['configuration']}': " + ", ".join(f"{c} = {cells[c]}" for c in dict.fromkeys(used))


