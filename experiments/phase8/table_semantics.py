"""Phase 8, investigation B (offline): how verifiers read ✓, ✗ and blank table cells.

Cases: experiments/phase8/table_semantics_cases.json (constructed claims about real c4 table rows; labels relative to
the cited table evidence). Each claim cites the c4 chunk holding its row.
Variants (the verifier code is never modified; only the evidence representation or the checker differs):
  A    frozen v4, original c4 evidence (✓, ✗, "(blank)"; empty LaTeX cells absent)
  B1   frozen v4, marks as tokens: [CHECKMARK], [CROSSMARK], [EMPTY_CELL] (empty cells restored from tables.jsonl)
  B2   frozen v4, marks as words: checked, crossed, empty (same restoration)
  C    deterministic mark checker alone: literal "row R is marked X under column K" claims only; abstains otherwise
       (abstention counts as a rejection); it never interprets what a mark means
  C+A  C, falling back to frozen v4 on the original evidence when C abstains (a combination, reported separately)
  D1/D2 candidate v5 (P1+P6) on the B1 / B2 evidence
Writes experiments/phase8/out/table_semantics_{UTC}.json."""
import copy
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, ".")
from relay.candidate_v5.verifier import verify as verify_v5
from relay.config import app_config
from relay.eval.verifier_v2_check import CachedNLI
from relay.retrieval import latex_parse as lp
from relay.retrieval.chunker import structured_row
from relay.verify import nli
from relay.verify.verifier import verify as verify_v4

cfg = app_config()
T = cfg["verification"]["entail_min"]
NLI = CachedNLI(nli.load_nli(cfg["verification"]["nli_model"]))
CASE_FILE = Path(sys.argv[sys.argv.index("--cases") + 1]) if "--cases" in sys.argv else \
    Path("experiments/phase8/table_semantics_cases.json")
CASES = json.loads(CASE_FILE.read_text(encoding="utf-8"))["cases"]
CHUNKS = [json.loads(l) for l in open("data/corpus/c4/processed/chunks_256.jsonl", encoding="utf-8")]
IDS = {c["chunk_id"] for c in CHUNKS}
TABLES = {(t["paper"], t["table_id"]): t for t in map(json.loads, open("data/corpus/c4/processed/tables.jsonl", encoding="utf-8"))}
REPR = {"B1": {"✓": "[CHECKMARK]", "✗": "[CROSSMARK]", "(blank)": "[EMPTY_CELL]", "": "[EMPTY_CELL]"},
        "B2": {"✓": "checked", "✗": "crossed", "(blank)": "empty", "": "empty"}}


def cited_chunk(case):
    paper, tid = case["table"].split(":")
    hits = [c for c in CHUNKS if c.get("rows") and c["paper"] == paper and c.get("table_id") == tid
            and any(all(s in l for s in case["locator"]) for l in c["text"].split("\n")[2:])]
    assert len(hits) == 1, (case["id"], len(hits))
    return hits[0]


def full_rows(chunk):
    """The chunk's rows with every column (empty cells included), taken from the table record."""
    table = TABLES[(chunk["paper"], chunk["table_id"])]
    out = []
    for r in chunk["rows"]:
        shown = {v["column"]: v["value"] for v in r["values"]}
        match = [t for t in table["rows"] if t["row_header"] == r["configuration"]
                 and all(t["cells"].get(k) == v for k, v in shown.items())]
        out.append(match[0] if len(match) == 1 else {"row_header": r["configuration"], "cells": shown})
    return out


def represent(chunk, mapping):
    """Same chunk, marks and empty cells written with `mapping`; rows and text lines rebuilt the way the chunker does."""
    c = copy.deepcopy(chunk)
    rows = [{"row_header": r["row_header"], "cells": {k: mapping.get(v, v) for k, v in r["cells"].items()}}
            for r in full_rows(chunk)]
    c["rows"] = [structured_row(r) for r in rows]
    head = c["text"].split("\n")[:2]
    c["text"] = "\n".join(head + [f"Configuration {lp.row_text(r)}" for r in rows])
    return c


# ---- C: deterministic literal mark checker
MARK = {"check": re.compile(r"\b(?:check(?:ed)?(?: mark)?|tick(?:ed)?)\b|✓", re.I),
        "cross": re.compile(r"\b(?:cross(?:ed)?(?: mark)?)\b|✗", re.I),
        "blank": re.compile(r"\b(?:blank|empty|no mark|unmarked|unspecified)\b", re.I)}
NEG = re.compile(r"\b(?:not|never|without|does|doesn't|isn't)\b", re.I)
FRAME = set("in the a an is are row with has have under cell table leaves marked mark of and for its".split())


def words(s):
    return {w for w in re.findall(r"[a-z0-9]+(?:[.-][a-z0-9]+)*", s.lower())}


def deterministic(case, chunk):
    claim = case["claim"]
    kinds = [k for k, rx in MARK.items() if rx.search(claim)]
    text_wo_mark = MARK["blank"].sub(" ", claim) if kinds == ["blank"] else claim
    if len(kinds) != 1 or NEG.search(text_wo_mark):
        return None, "abstain: not a single literal mark claim"
    rows = full_rows(chunk)
    table = TABLES[(chunk["paper"], chunk["table_id"])]
    cw = words(claim)
    cols = [h for h in table["headers"] if words(re.sub(r"\([^()]*\)", " ", h)) - {"k"} <= cw
            and words(re.sub(r"\([^()]*\)", " ", h))]
    cols = [h for h in cols if not any(h != o and words(h) < words(o) for o in cols)]  # most specific header
    if len(cols) != 1:
        return None, f"abstain: column not unique ({cols})"
    col = cols[0]

    def score(r):
        vals = [r["row_header"]] + [v for k, v in r["cells"].items() if k != col and v not in ("✓", "✗", "(blank)", "")]
        return sum(1 for v in vals if len(v) >= 3 and v.lower() in claim.lower())
    best = max(score(r) for r in rows)
    named = [r for r in rows if best and score(r) == best]
    if len(named) != 1:
        return None, f"abstain: row not unique ({len(named)})"
    row = named[0]
    vocab = words(table["caption"]) | words(" ".join(table["headers"])) | words(row["row_header"]) | \
        set().union(*(words(v) for v in row["cells"].values())) | FRAME | {"check", "checked", "cross", "crossed", "tick",
                                                                          "blank", "empty", "unspecified", "unmarked", "no"}
    left = {w for w in cw - vocab if not re.fullmatch(r"[\d.%]+", w)}
    if left:
        return None, f"abstain: unexplained words {sorted(left)}"
    value = row["cells"].get(col, "")
    actual = {"✓": "check", "✗": "cross", "(blank)": "blank", "": "blank"}.get(value)
    if actual is None:
        return None, f"abstain: cell is not a mark ({value!r})"
    return actual == kinds[0], f"cell {col} = {value!r}; claim says {kinds[0]}"


def _stem(w):
    return w[:-3] + "y" if w.endswith("ies") and len(w) > 4 else w[:-1] if w.endswith("s") and len(w) > 3 else w


def words2(s):
    return {_stem(w) for w in re.findall(r"[a-z0-9]+(?:\.[0-9]+)?", s.lower())}


def deterministic_v2(case, chunk):
    """C2 (revised after the first run, general rules only): the column is the phrase after the last 'under', or
    'the/a/blank <phrase> cell'; words are plural-stemmed and hyphen-split; rows are found by distinctive printed
    values. Abstains on negation, interpretation, unexplained words, or a non-mark cell."""
    claim = case["claim"]
    kinds = [k for k, rx in MARK.items() if rx.search(claim)]
    text_wo_mark = MARK["blank"].sub(" ", claim) if kinds == ["blank"] else claim
    if len(kinds) != 1 or NEG.search(text_wo_mark):
        return None, "abstain: not a single literal mark claim"
    m = re.search(r"\bunder\s+(.+?)\.?$", claim) or re.search(r"\b(?:the|a|blank)\s+(?!blank\b)([\w\s/().-]+?)\s+cell\b", claim)
    if not m:
        return None, "abstain: no column phrase"
    phrase = words2(m.group(1))
    table = TABLES[(chunk["paper"], chunk["table_id"])]
    cols = [h for h in table["headers"] if (hw := words2(re.sub(r"\([^()]*\)", " ", h))) and hw <= phrase]
    cols = [h for h in cols if not any(h != o and words2(h) < words2(o) for o in cols)]
    if len(cols) != 1:
        return None, f"abstain: column not unique ({cols})"
    col = cols[0]
    rows = full_rows(chunk)
    low = claim.lower()

    def score(r):
        vals = [r["row_header"]] + [v for k, v in r["cells"].items() if k != col and v not in ("✓", "✗", "(blank)", "")]
        return sum(1 for v in vals if (len(v) >= 3 or re.search(r"\d", v)) and v.lower() in low)
    best = max(score(r) for r in rows)
    named = [r for r in rows if best and score(r) == best]
    if len(named) != 1:
        return None, f"abstain: row not unique ({len(named)})"
    row = named[0]
    vocab = words2(table["caption"]) | words2(" ".join(table["headers"])) | words2(row["row_header"]) | \
        set().union(*(words2(v) for v in row["cells"].values())) | {_stem(w) for w in FRAME} | \
        {_stem(w) for w in ("check checked cross crossed tick blank empty unspecified unmarked no at for one".split())}
    left = {w for w in words2(claim) - vocab if not re.fullmatch(r"[\d.]+", w)}
    if left:
        return None, f"abstain: unexplained words {sorted(left)}"
    value = row["cells"].get(col, "")
    actual = {"✓": "check", "✗": "cross", "(blank)": "blank", "": "blank"}.get(value)
    if actual is None:
        return None, f"abstain: cell is not a mark ({value!r})"
    return actual == kinds[0], f"cell {col} = {value!r}; claim says {kinds[0]}"


def run_verifier(fn, case, passage):
    rep = fn(f"{case['claim']} [{passage['chunk_id']}]", [passage], IDS, NLI, T, T)
    s = rep.sentences[0]
    return {"accepted": rep.ok, "entailment": None if s.entailment is None else round(s.entailment, 3), "reasons": s.reasons}


results = []
for case in CASES:
    chunk = cited_chunk(case)
    b1, b2 = represent(chunk, REPR["B1"]), represent(chunk, REPR["B2"])
    det, why = deterministic(case, chunk)
    det2, why2 = deterministic_v2(case, chunk)
    r = {"id": case["id"], "label": case["label"], "category": case["category"], "claim": case["claim"], "chunk": chunk["chunk_id"],
         "A": run_verifier(verify_v4, case, chunk), "B1": run_verifier(verify_v4, case, b1), "B2": run_verifier(verify_v4, case, b2),
         "D1": run_verifier(verify_v5, case, b1), "D2": run_verifier(verify_v5, case, b2),
         "C": {"accepted": bool(det), "abstained": det is None, "why": why},
         "C2": {"accepted": bool(det2), "abstained": det2 is None, "why": why2}}
    r["C+A"] = {"accepted": r["A"]["accepted"] if det is None else det, "from": "v4" if det is None else "deterministic"}
    r["C2+A"] = {"accepted": r["A"]["accepted"] if det2 is None else det2, "from": "v4" if det2 is None else "deterministic"}
    # C2g (registered before the held-out run): a claim that asserts a mark but that C2 cannot resolve is rejected,
    # never handed to v4 (which misreads marks); only claims without a mark word fall back to v4
    mark_claim = any(rx.search(case["claim"]) for rx in MARK.values())
    r["C2g"] = {"accepted": det2 if det2 is not None else (False if mark_claim else r["A"]["accepted"]),
                "from": "deterministic" if det2 is not None else ("rejected: unresolved mark claim" if mark_claim else "v4")}
    results.append(r)


def metrics(v):
    s = [r for r in results if r["label"] == "supported"]
    u = [r for r in results if r["label"] == "unsupported"]
    sa, ua = sum(r[v]["accepted"] for r in s), sum(r[v]["accepted"] for r in u)
    return {"supported_accepted": sa, "supported_rejected": len(s) - sa, "unsupported_accepted": ua,
            "unsupported_rejected": len(u) - ua, "FAR": round(ua / len(u), 3), "FRR": round((len(s) - sa) / len(s), 3)}


VARIANTS = ("A", "B1", "B2", "C", "C+A", "C2", "C2+A", "C2g", "D1", "D2")
summary = {v: metrics(v) for v in VARIANTS}
summary["C"]["abstained"] = sum(r["C"]["abstained"] for r in results)
summary["C2"]["abstained"] = sum(r["C2"]["abstained"] for r in results)
out = Path(f"experiments/phase8/out/table_semantics_{datetime.now(timezone.utc).strftime('%Y-%m-%dT%H-%M-%SZ')}.json")
with out.open("x", encoding="utf-8") as f:
    json.dump({"case_file": CASE_FILE.as_posix(), "summary": summary, "results": results, "threshold": T}, f, indent=1,
              ensure_ascii=False)
print(f"{len(results)} cases: {sum(r['label'] == 'supported' for r in results)} supported, {sum(r['label'] == 'unsupported' for r in results)} unsupported")
for v, m in summary.items():
    print(f"{v:4} {m}")
print("\nper case (accept = 1):  id  label        category                        " + " ".join(VARIANTS) + " | A-entail")
for r in results:
    print(f"  {r['id']} {r['label'][:5]:5} {r['category'][:34]:34} " + " ".join(str(int(r[v]['accepted'])) for v in VARIANTS)
          + f" | {r['A']['entailment']}  C2: {r['C2']['why'][:60]}")
print(out)
