"""Phase 10C, REAL-PAPER OFFLINE replay: every real gpt-oss draft on disk, re-verified with the current verifier (v4.2)
and the real NLI model (local; no API). Compared with the decision recorded when the draft was made.

Sources (read only):
  results/raw/answer_quality/real_2026-11-13T18-06-56Z.jsonl  (verifier v4; passages stored in each record)
  experiments/phase10/out/live/pipeline_*.json                 (v4 / v4.1; passages = c3 chunks by recorded id)
Writes experiments/phase10/out/replay_verifier_{UTC}.json.
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, ".")
from relay.config import app_config  # noqa: E402
from relay.eval.verifier_v2_check import CachedNLI  # noqa: E402
from relay.verify import nli  # noqa: E402
from relay.verify.verifier import verify  # noqa: E402

AQ = Path("results/raw/answer_quality/real_2026-11-13T18-06-56Z.jsonl")
LIVE = sorted(Path("experiments/phase10/out/live").glob("pipeline_*.json"))
C3 = {c["chunk_id"]: c for c in map(json.loads, open("data/processed/chunks_256.jsonl", encoding="utf-8"))}


def drafts() -> list[dict]:
    out = []
    for r in map(json.loads, AQ.open(encoding="utf-8")):
        if "drafts" not in r:
            continue
        for d in r["drafts"]:
            if d.get("kind") == "not_in_papers" or d["text"].strip() == "NOT_IN_PAPERS":
                continue
            out.append({"source": AQ.name, "question_id": r["question_id"], "n": d["n"], "text": d["text"],
                        "passages": [C3[p["chunk_id"]] for p in r["retrieval"]["passages"]], "recorded_ok": (d.get("verifier") or {}).get("accepted"),
                        "recorded_version": r["verification_policy_version"]})
    for f in LIVE:
        run = json.loads(f.read_text(encoding="utf-8"))
        version = (run.get("versions") or {}).get("verification_policy_version", "v4")
        for q in run["questions"]:
            for n, (text, v) in enumerate(zip(q.get("drafts") or [], q.get("verification") or []), 1):
                out.append({"source": f.name, "question_id": q["id"], "n": n, "text": text,
                            "passages": [C3[i] for i in q["evidence_ids"]], "recorded_ok": v["ok"],
                            "recorded_version": version})
    return out


def main() -> None:
    cfg = app_config()["verification"]
    model = CachedNLI(nli.load_nli(cfg["nli_model"]))
    corpus = set(C3)
    rows = []
    for d in drafts():
        rep = verify(d["text"], d["passages"], corpus, model, cfg["entail_min"], cfg["entail_min_flagged"])
        rows.append({k: d[k] for k in ("source", "question_id", "n", "recorded_version", "recorded_ok")} |
                    {"text": d["text"], "v42_ok": rep.ok, "v42_failures": rep.failures(),
                     "absence_sentences": [s.sentence for s in rep.sentences if s.absence]})
    changed = [r for r in rows if r["recorded_ok"] is not None and r["recorded_ok"] != r["v42_ok"]]
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ")
    out = Path(f"experiments/phase10/out/replay_verifier_{stamp}.json")
    with out.open("x", encoding="utf-8") as f:
        json.dump({"verifier": app_config()["verification_policy_version"], "drafts": len(rows),
                   "changed": changed, "rows": rows}, f, indent=1, ensure_ascii=False)
    print(f"{len(rows)} real drafts replayed; {len(changed)} decisions differ from the recorded ones")
    for r in changed:
        print(f"  {r['source'][:28]:28} {r['question_id']} draft {r['n']} ({r['recorded_version']}): "
              f"{'accept' if r['recorded_ok'] else 'reject'} -> {'accept' if r['v42_ok'] else 'reject'} | "
              f"{[w[:110] for _, w in r['v42_failures']][:2]}")
    print(out.as_posix())


if __name__ == "__main__":
    main()
