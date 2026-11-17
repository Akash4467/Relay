"""Steps 1-3: reproduce the t29 and q13 failures with the baseline verifier v4.7c (recorded live drafts, recorded
c6 evidence), then show v4.8a (fix 1 only), v4.8b (fix 2 only) and v4.8c (both) on the same drafts."""
import json
import sys
from pathlib import Path

sys.path.insert(0, ".")
from relay.candidate_v4_7 import verifier as v47  # noqa: E402
from relay.candidate_v4_8 import fix1_only, fix2_only, verifier as v48  # noqa: E402
from relay.config import app_config  # noqa: E402
from relay.verify import nli  # noqa: E402

cfg = app_config()["verification"]
m = nli.load_nli(cfg["nli_model"])
C = {json.loads(l)["chunk_id"]: json.loads(l) for l in open("experiments/phase10/corpus/c6/processed/chunks_256.jsonl", encoding="utf-8")}
out = {}
for qid in ("t29", "q13"):
    r = json.loads(Path(f"assessment/e2e/live/E25_R3b_p7_v47_c6/records/{qid}.json").read_text(encoding="utf-8"))
    ev = [C[c] for c in r["evidence_ids"]]
    for i, d in enumerate(r["drafts"]):
        row = {}
        for name, build in (("v4.7c", v47.verify_for), ("v4.8a_fix1", fix1_only.verify_for),
                            ("v4.8b_fix2", fix2_only.verify_for), ("v4.8c_both", v48.verify_for)):
            rep = build(r["question"])(d, ev, set(C), m, cfg["entail_min"], cfg["entail_min_flagged"])
            row[name] = {"ok": rep.ok, "new_reasons": [x for s in rep.sentences for x in s.reasons
                                                       if "check marks" in x or "retrieved row" in x]}
        out[f"{qid}#{i}"] = {"draft": d, **row}
        print(f"{qid} D{i}: " + " | ".join(f"{k}={'ACCEPT' if v['ok'] else 'reject'}" for k, v in row.items() if k != "draft"))
        for k, v in row.items():
            if v["new_reasons"]:
                print(f"     {k}: {v['new_reasons'][0][:170]}")
Path("assessment/e2e/fix_v48/reproduce.json").write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
