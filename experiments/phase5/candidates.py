"""Phase 5, investigation B/D (offline): two candidate verifier changes layered on v5, compared with frozen v4 and v5.
Experimental code only (never imported by the pipeline). No LLM or API calls; NLI is local.

T (table-aware, deterministic): a sentence that cites only table rows and v5 rejected *only* for low entailment is
  accepted when (a) it names exactly one row variant of the cited table (P6's sibling-qualifier rule), (b) every number
  in it equals a cell of that row in a column the same clause names, (c) it has no comparative or negation word, and
  (d) every content word is explained by the caption, row labels, column names (including abbreviations such as
  "Q." for "Questions") or generic words. Otherwise NLI decides, unchanged.
C (claim decomposition): a sentence v5 rejected *only* for low entailment is split into clauses (v4's atomic claims,
  plus ": " and ", confirming/validating/demonstrating/showing/indicating/suggesting that" tails) and accepted when
  every clause reaches the unchanged threshold on its own (the whole-sentence requirement is dropped).

Writes experiments/phase5/out/candidates_{UTC}.json and prints the comparison."""
import csv
import json
import re
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, ".")
from relay.candidate_v5.verifier import GENERIC, _row_requested, _words, verify as verify_v5
from relay.config import app_config
from relay.eval.verifier_v2_check import CachedNLI
from relay.verify import nli, tables
from relay.verify.citations import strip_citations
from relay.verify.claims import atomic_claims
from relay.verify.verifier import REFERENCE, _support, premises, verify as verify_v4
sys.path.insert(0, 'experiments/phase5')
from table_check import table_supported  # noqa: E402

cfg = app_config()
T = cfg["verification"]["entail_min"]
M = CachedNLI(nli.load_nli(cfg["verification"]["nli_model"]))
CHUNKS = {json.loads(l)["chunk_id"]: json.loads(l) for l in open("data/processed/chunks_256.jsonl", encoding="utf-8")}
IDS = set(CHUNKS)
NLI_ONLY = "not supported by the cited passage"
TAIL = re.compile(r",\s+(?:confirming|validating|demonstrating|showing|indicating|suggesting)\s+that\s+", re.I)


def decomposed_ok(s, cited: list[dict]) -> bool:
    claim = strip_citations(s.sentence)
    parts = atomic_claims(claim)[1:]
    if not parts:
        parts = [p.strip() for p in re.split(r":\s+|" + TAIL.pattern, claim) if p and p.strip()]
    if len(parts) < 2:
        return False
    prem = premises(cited, claim)
    ctx = [s.context] if s.context else []
    for j, part in enumerate(parts):
        c = ctx if j == 0 else ([parts[0]] if REFERENCE.match(part) else [])
        if _support(part, prem, c, M) < T:
            return False
    return True


def run_variant(name: str, answer: str, passages: list[dict]):
    if name == "v4":
        return verify_v4(answer, passages, IDS, M, T, T)
    rep = verify_v5(answer, passages, IDS, M, T, T)
    if name == "v5":
        return rep
    by = {p["chunk_id"]: p for p in passages}
    for s in rep.sentences:
        if s.ok or not s.reasons or not all(r.startswith(NLI_ONLY) for r in s.reasons):
            continue
        cited = [by[c] for c in s.citations if c in by]
        if "T" in name and table_supported(s.sentence, cited, passages):
            s.reasons = []
        elif "C" in name and decomposed_ok(s, cited):
            s.reasons = []
    return rep


VARIANTS = ["v4", "v5", "v5+T", "v5+C", "v5+T+C"]

# ---------- labelled verifier sets (judged sentence = the last one, as in the earlier evaluations)
SETS = [("v3_eval", "eval/verifier_v3_eval.jsonl", None), ("v3b_targeted", "eval/verifier_v3b_targeted.jsonl", None),
        ("v2_stress calibration", "eval/verifier_v2_stress.jsonl", "calibration"),
        ("held-out (evaluation only)", "eval/verifier_v2_stress.jsonl", "heldout")]
labelled = {}
for set_name, path, split in SETS:
    pairs = [json.loads(l) for l in open(path, encoding="utf-8")]
    pairs = [p for p in pairs if split is None or p.get("split") == split]
    labelled[set_name] = {v: {p["pair_id"]: [p["supported"], run_variant(v, p["answer"], [CHUNKS[c] for c in p["evidence"]]).sentences[-1].ok]
                              for p in pairs} for v in VARIANTS}

# ---------- real drafts with sentence grades
LIVE_LABELS = {  # assistant grades of the live v5 drafts (preliminary): (support, factual)
    ("q02", 1, 1): ("partly_supported", "true"), ("q02", 2, 1): ("partly_supported", "true"),
    ("q10", 1, 1): ("partly_supported", "true"), ("q10", 1, 2): ("partly_supported", "true"),
    ("q10", 1, 3): ("partly_supported", "true"), ("q10", 1, 4): ("supported", "true"),
    ("q10", 1, 5): ("unsupported", "true"), ("q10", 2, 1): ("partly_supported", "true"),
    ("q10", 2, 2): ("unsupported", "false"), ("q10", 2, 3): ("supported", "true"),
    ("q10", 2, 4): ("partly_supported", "true"), ("q12", 1, 1): ("supported", "true"),
    ("q12", 1, 2): ("absence_sentence", "false"), ("q12", 2, 1): ("supported", "true"),
    ("q12", 2, 2): ("partly_supported", "true")}
grades = {(g["question_id"], int(g["draft_n"]), int(g["sentence_n"])): (g["assistant_citation_support"], g["assistant_factual"])
          for g in csv.DictReader(open("eval/answer_quality_sentences.csv", encoding="utf-8")) if g["sentence_n"]}
sources = [("15-question run (v4 drafts)", "results/raw/answer_quality/real_2026-11-13T18-06-56Z.jsonl", grades, True),
           ("live v5 run", "results/live_validation/real_v5_2026-11-13T19-39-49Z.jsonl", LIVE_LABELS, False)]
sentence_rows = []
for src_name, path, labels, restore in sources:
    for r in map(json.loads, open(path, encoding="utf-8")):
        if r.get("kind") != "question":
            continue
        passages = [CHUNKS[p["chunk_id"]] for p in r["retrieval"]["passages"]] if restore else r["retrieval"]["passages"]
        for d in r["drafts"]:
            if d["verifier"] is None:
                continue
            reps = {v: run_variant(v, d["text"], passages) for v in VARIANTS}
            for i in range(len(reps["v4"].sentences)):
                sup, fact = labels.get((r["question_id"], d["n"], i + 1), ("", ""))
                sentence_rows.append({"source": src_name, "question_id": r["question_id"], "draft": d["n"], "sentence": i + 1,
                                      "text": reps["v4"].sentences[i].sentence, "support": sup, "factual": fact,
                                      **{v: reps[v].sentences[i].ok for v in VARIANTS},
                                      **{f"draft_ok_{v}": reps[v].ok for v in VARIANTS}})


def bucket(row):
    if row["support"] == "unsupported" or row["factual"] == "false":
        return "unsupported_or_false"
    if row["support"] == "partly_supported":
        return "partly_supported"
    if row["support"] in ("supported", "absence_sentence"):
        return "supported"
    return "ungraded"


summary = {}
for v in VARIANTS:
    lab = {s: Counter(("S" if sup else "U") + ("acc" if ok else "rej") for sup, ok in labelled[s][v].values()) for s in labelled}
    sent = Counter((bucket(r), r[v]) for r in sentence_rows)
    summary[v] = {"labelled": {s: dict(c) for s, c in lab.items()},
                  "sentences_accepted": {b: sent[(b, True)] for b in ("supported", "partly_supported", "unsupported_or_false")},
                  "sentences_total": {b: sent[(b, True)] + sent[(b, False)] for b in ("supported", "partly_supported", "unsupported_or_false")},
                  "drafts_accepted": len({(r["source"], r["question_id"], r["draft"]) for r in sentence_rows if r[f"draft_ok_{v}"]})}
changes = {v: [f"{r['source']} {r['question_id']} d{r['draft']} s{r['sentence']} ({bucket(r)}): {r['text'][:90]}"
               for r in sentence_rows if r[v] and not r["v5"]] for v in ("v5+T", "v5+C", "v5+T+C")}
lab_changes = {v: {s: [pid for pid in labelled[s][v] if labelled[s][v][pid][1] != labelled[s]["v5"][pid][1]] for s in labelled}
               for v in ("v5+T", "v5+C", "v5+T+C")}
out = Path(f"experiments/phase5/out/candidates_{datetime.now(timezone.utc).strftime('%Y-%m-%dT%H-%M-%SZ')}.json")
with out.open("x", encoding="utf-8") as f:
    json.dump({"summary": summary, "sentence_changes_vs_v5": changes, "labelled_changes_vs_v5": lab_changes,
               "labelled": labelled, "sentences": sentence_rows}, f, indent=1, ensure_ascii=False)
for v in VARIANTS:
    s = summary[v]
    print(f"{v:8} labelled S/U accepted: " + "; ".join(f"{k} {c.get('Sacc', 0)}/{c.get('Sacc', 0) + c.get('Srej', 0)} "
          f"| {c.get('Uacc', 0)}/{c.get('Uacc', 0) + c.get('Urej', 0)}" for k, c in s["labelled"].items()))
    print(f"         real sentences accepted (supported / partly / unsupported-or-false): "
          + " / ".join(f"{s['sentences_accepted'][b]} of {s['sentences_total'][b]}" for b in ("supported", "partly_supported", "unsupported_or_false"))
          + f" | drafts accepted {s['drafts_accepted']}")
for v, ch in changes.items():
    print(f"\n{v} newly accepted real sentences vs v5:", *ch, sep="\n   ")
    print(f"   labelled pairs changed vs v5: {lab_changes[v]}")
print("\n", out)
