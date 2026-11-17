"""Replays every recorded live draft under v4.2 and v4.5c (each against its own corpus's chunks). v4.5c only adds a
rejection, so any change is v4.2-accept -> v4.5c-reject; each one is listed for review. Writes replay_v45.json."""
import glob
import json
import sys
from pathlib import Path

sys.path.insert(0, ".")
from relay.candidate_v4_5 import verifier as v45  # noqa: E402
from relay.config import app_config  # noqa: E402
from relay.verify import nli  # noqa: E402
from relay.verify import verifier as v42  # noqa: E402

cfg = app_config()
t = cfg["verification"]["entail_min"]
m = nli.load_nli(cfg["verification"]["nli_model"])
proc = {"c3": "data/processed", "c6": "experiments/phase10/corpus/c6/processed"}
C = {c: {json.loads(l)["chunk_id"]: json.loads(l) for l in open(f"{d}/chunks_256.jsonl", encoding="utf-8")} for c, d in proc.items()}
files = sorted(glob.glob("assessment/improvement_loop/baseline/*/records/*.json") + glob.glob("assessment/improvement_loop/iterations/iteration_002/live/records/*.json")
               + glob.glob("assessment/e2e/live/B12/records/*.json") + glob.glob("experiments/phase12/phase12b/c6/*.json"))
rows = []
for f in files:
    r = json.loads(Path(f).read_text(encoding="utf-8"))
    cv = r.get("corpus_version", "c3")
    for i, d in enumerate(r.get("drafts") or []):
        if d.strip() == "NOT_IN_PAPERS":
            continue
        ev = [C[cv][c] for c in r["evidence_ids"]]
        a = v42.verify(d, ev, set(C[cv]), m, t, t).ok
        b = v45.verify(d, ev, set(C[cv]), m, t, t)
        rows.append({"file": f, "corpus": cv, "question_id": r["question_id"], "draft": i, "v4.2": a, "v4.5c": b.ok,
                     "v4.5c_reasons": [x for s in b.sentences for x in s.reasons if "carry no marks" in x], "text": d})
changed = [x for x in rows if x["v4.2"] != x["v4.5c"]]
Path("assessment/e2e/fix/replay_v45.json").write_text(json.dumps({"n_drafts": len(rows), "changed": changed, "rows": rows}, indent=1, ensure_ascii=False), encoding="utf-8")
print("drafts", len(rows), "| v4.2 accepted", sum(x["v4.2"] for x in rows), "| v4.5c accepted", sum(x["v4.5c"] for x in rows))
for x in changed:
    print("  CHANGED", x["corpus"], x["question_id"], x["draft"], x["v4.2"], "->", x["v4.5c"], "|", x["text"][:110], "|", x["file"].split("/")[-3])
