"""Steps 5-6: replay (no API). For every recorded live draft (all runs, c3 and c6, each with its own question and
recorded evidence) and every labelled stress pair (no question), run v4.7c ONCE, then apply fix 1, fix 2 and both
on deep copies of that report. Reports every decision change. A change from reject to accept would mean a check was
weakened; the fixes are rejection-only, so the script asserts there is none. Writes replay.json."""
import copy
import glob
import json
import sys
from pathlib import Path

sys.path.insert(0, ".")
sys.path.insert(0, "assessment/improvement_loop/tools")
from relay.candidate_v4_7 import verifier as v47  # noqa: E402
from relay.candidate_v4_8 import absence_labels, marks  # noqa: E402
from relay.config import app_config  # noqa: E402
from relay.verify import nli  # noqa: E402
from relay.verify.citations import strip_citations  # noqa: E402
from verifier_compare_checkpointed import STRESS, answer_of  # noqa: E402

cfg = app_config()["verification"]
m = nli.load_nli(cfg["nli_model"])
CC = {c: {json.loads(l)["chunk_id"]: json.loads(l) for l in open(f"{d}/chunks_256.jsonl", encoding="utf-8")}
      for c, d in {"c3": "data/processed", "c6": "experiments/phase10/corpus/c6/processed"}.items()}
files = sorted(set(glob.glob("experiments/phase12/phase12b/c*/*.json") + glob.glob("assessment/improvement_loop/baseline/*/records/*.json")
                   + glob.glob("assessment/improvement_loop/iterations/*/live/records/*.json") + glob.glob("assessment/e2e/live/*/records/*.json")))
cases = []
for f in files:
    r = json.loads(Path(f).read_text(encoding="utf-8"))
    for i, d in enumerate(r.get("drafts") or []):
        if d.strip() != "NOT_IN_PAPERS":
            cases.append({"id": f"{f}#{i}", "kind": "draft", "question": r["question"], "corpus": r.get("corpus_version", "c3"),
                          "evidence": r["evidence_ids"], "text": d, "question_id": r["question_id"]})
for f in STRESS:
    for p in (json.loads(l) for l in open(f, encoding="utf-8") if l.strip()):
        cases.append({"id": f"stress:{Path(f).stem}:{p['pair_id']}", "kind": "stress", "question": "", "corpus": "c3",
                      "evidence": p["evidence"], "text": answer_of(p), "label": "supported" if p["supported"] else "unsupported"})
rows = []
for n, c in enumerate(cases, 1):
    C = CC[c["corpus"]]
    ev = [C[e] for e in c["evidence"]]
    base = v47.verify_for(c["question"])(c["text"], ev, set(C), m, cfg["entail_min"], cfg["entail_min_flagged"])
    judge = (lambda rep: rep.sentences[-1].ok) if c["kind"] == "stress" else (lambda rep: rep.ok)
    out = {"v4.7c": judge(base)}
    for name, f1, f2 in (("fix1", True, False), ("fix2", False, True), ("both", True, True)):
        rep = copy.deepcopy(base)
        if f1:
            marks.check(c["question"], rep, ev, strip_citations)
        if f2:
            absence_labels.check(rep, ev)
        out[name] = judge(rep)
        out[name + "_reasons"] = [x for s in rep.sentences for x in s.reasons if "check marks" in x or "retrieved row '" in x]
    rows.append({**{k: v for k, v in c.items() if k != "evidence"}, **out})
    if n % 50 == 0:
        print(n, "/", len(cases), flush=True)
weakened = [r for r in rows if any(r[k] and not r["v4.7c"] for k in ("fix1", "fix2", "both"))]
assert not weakened, weakened
changed = [r for r in rows if r["both"] != r["v4.7c"]]
summary = {"n_cases": len(rows), "n_drafts": sum(r["kind"] == "draft" for r in rows), "n_stress": sum(r["kind"] == "stress" for r in rows),
           "weakened (reject->accept)": 0,
           "changed_by": {k: sum(r[k] != r["v4.7c"] for r in rows) for k in ("fix1", "fix2", "both")},
           "stress_supported_newly_rejected": [r["id"] for r in changed if r.get("label") == "supported"],
           "stress_unsupported_newly_rejected": [r["id"] for r in changed if r.get("label") == "unsupported"],
           "changed": [{k: r[k] for k in ("id", "question_id", "v4.7c", "fix1", "fix2", "both", "fix1_reasons", "fix2_reasons", "text")
                        if k in r} for r in changed]}
Path("assessment/e2e/fix_v48/replay.json").write_text(json.dumps({"summary": summary, "rows": rows}, indent=1, ensure_ascii=False), encoding="utf-8")
print(json.dumps({k: v for k, v in summary.items() if k != "changed"}, indent=1))
for r in summary["changed"]:
    print("CHANGED", r["id"].replace("assessment/e2e/live/", "").replace("assessment/improvement_loop/", "")[:70], "| fix1", r["fix1"], "fix2", r["fix2"])
    print("      ", r["text"][:200].replace("\n", " "))
    print("      ", (r["fix1_reasons"] + r["fix2_reasons"])[:1])
