"""Offline verifier comparison (no API): production v4.2 vs a candidate, on
  (1) every labelled verifier stress file in eval/ (160 pairs, supported true/false, written in earlier phases), judged
      as relay.eval.verifier_v2_check.judge does (the last sentence; a 'claim' pair gets its evidence IDs as citation);
  (2) a replay of every live draft recorded in the given run directories (same draft text, same evidence chunks):
      accept/reject under each verifier, with the draft label from the run's draft_labels.json when one exists.
usage: verifier_offline.py <candidate module> <out.json> [live run dirs ...]
"""
from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

sys.path.insert(0, ".")
from relay.config import app_config  # noqa: E402
from relay.verify import nli  # noqa: E402
from relay.verify import verifier as v42  # noqa: E402

STRESS = ["eval/verifier_stress.jsonl", "eval/verifier_stress_v2.jsonl", "eval/verifier_v2_stress.jsonl",
          "eval/verifier_v3_eval.jsonl", "eval/verifier_v3b_targeted.jsonl"]


def chunks(path="data/processed/chunks_256.jsonl") -> dict:
    return {json.loads(l)["chunk_id"]: json.loads(l) for l in open(path, encoding="utf-8")}


def answer_of(p: dict) -> str:
    if "answer" in p:
        return p["answer"]
    return p["claim"].rstrip().rstrip(".") + " " + "".join(f"[{e}]" for e in p["evidence"]) + "."


def main(cand_mod: str, out: Path, runs: list[str]) -> None:
    cand = importlib.import_module(cand_mod)
    t = app_config()["verification"]["entail_min"]
    model = nli.load_nli(app_config()["verification"]["nli_model"])
    C = chunks()
    vs = {"v4.2": v42.verify, cand.VERSION: cand.verify}
    res = {"threshold": t, "candidate": cand_mod, "stress": {}, "live_replay": []}
    for f in STRESS:
        pairs = [json.loads(l) for l in open(f, encoding="utf-8") if l.strip()]
        rows = []
        for p in pairs:
            ev = [C[c] for c in p["evidence"]]
            d = {}
            for name, fn in vs.items():
                s = fn(answer_of(p), ev, set(C), model, t, t).sentences[-1]
                d[name] = {"ok": s.ok, "entailment": s.entailment}
            rows.append({"pair_id": p["pair_id"], "supported": p["supported"], "category": p.get("category"), **d})
        conf = {n: {"supported_accepted": sum(r["supported"] and r[n]["ok"] for r in rows),
                    "supported_rejected": sum(r["supported"] and not r[n]["ok"] for r in rows),
                    "unsupported_accepted": sum(not r["supported"] and r[n]["ok"] for r in rows),
                    "unsupported_rejected": sum(not r["supported"] and not r[n]["ok"] for r in rows)} for n in vs}
        changed = [r for r in rows if r["v4.2"]["ok"] != r[cand.VERSION]["ok"]]
        res["stress"][f] = {"n": len(rows), "confusion": conf, "changed": changed, "rows": rows}
        print(f, len(rows), {n: (c["supported_rejected"], c["unsupported_accepted"]) for n, c in conf.items()},
              "changed:", [(r["pair_id"], r["supported"], r[cand.VERSION]["ok"]) for r in changed])
    for run in runs:
        labels_path = Path(run) / "draft_labels.json"
        labels = json.loads(labels_path.read_text(encoding="utf-8"))["labels"] if labels_path.exists() else {}
        for f in sorted(Path(run).glob("records/*.json")) or sorted(Path(run).glob("*.json")):
            r = json.loads(f.read_text(encoding="utf-8"))
            if "drafts" not in r or not r.get("drafts"):
                continue
            ev = [C[c] for c in r["evidence_ids"]]
            for i, d in enumerate(r["drafts"]):
                if d.strip() == "NOT_IN_PAPERS" or i >= len(r.get("verification") or []):
                    continue
                dec = {n: fn(d, ev, set(C), model, t, t).ok for n, fn in vs.items()}
                key = f"{r['question_id']}#{i}"
                res["live_replay"].append({"run": run, "question_id": r["question_id"], "draft": i,
                                           "recorded_ok": r["verification"][i]["ok"], **dec,
                                           "label": labels.get(key), "text": d})
    lr = res["live_replay"]
    res["live_replay_summary"] = {
        "n_drafts": len(lr),
        "recorded_equals_v4.2_replay": sum(x["recorded_ok"] == x["v4.2"] for x in lr),
        **{n: {"accepted": sum(x[n] for x in lr),
               "accepted_labelled_correct": sum(x[n] and x["label"] == "correct" for x in lr),
               "false_acceptance (accepted, labelled mixed/wrong/unsupported)":
                   sum(x[n] and x["label"] in ("mixed", "wrong", "unsupported") for x in lr),
               "false_rejection (rejected, labelled correct)": sum(not x[n] and x["label"] == "correct" for x in lr),
               "unlabelled": sum(x["label"] is None for x in lr)} for n in vs}}
    print("live replay:", res["live_replay_summary"])
    for x in lr:
        if x["v4.2"] != x[cand.VERSION]:
            print("  flip", x["run"].split("/")[-1], x["question_id"], x["draft"], "->", x[cand.VERSION], x["label"],
                  "|", x["text"][:110])
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main(sys.argv[1], Path(sys.argv[2]), sys.argv[3:])
