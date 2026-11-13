"""Independent verifier evaluation (DESIGN §20.3, 2026-11-13). No LLM calls.

eval/verifier_v3_eval.jsonl: 42 answers written before any v3 change (sha256 recorded in the report), 7 categories
(direct, math, table, pronoun, compound, numeric, absence), 14 calibration / 28 evaluation. Each answer runs through
the production verifier at the configured threshold (no sweep); the LAST sentence is the judged claim. Separate from
the original 8 held-out pairs (eval/verifier_v2_stress.jsonl), which stay untouched.

    uv run python -m relay.eval.verifier_v3_eval --label v2_baseline   # before the v3 change
    uv run python -m relay.eval.verifier_v3_eval --label v3             # after it
Writes eval/reports/verifier_v3_eval_<label>.json (or <pairs stem>_<label>.json with --pairs), never overwrites. Prints calibration only unless --show-evaluation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import time
from pathlib import Path

from relay.config import app_config
from relay.eval.retrieval_recall import _git
from relay.eval.stress_verifier import metrics
from relay.eval.verifier_v2_check import judge
from relay.retrieval.latex_parse import PROCESSED_DIR
from relay.verify import nli

PAIRS = Path("eval/verifier_v3_eval.jsonl")
REPORTS = Path("eval/reports")


def main(label: str, pairs_path: Path = PAIRS) -> dict:
    stem = "verifier_v3_eval" if pairs_path == PAIRS else pairs_path.stem
    out = REPORTS / f"{stem}_{label}.json"
    if out.exists():
        raise FileExistsError(f"{out} exists: each label is evaluated once")
    cfg = app_config()
    t = cfg["verification"]["entail_min"]
    chunks = {json.loads(l)["chunk_id"]: json.loads(l)
              for l in (PROCESSED_DIR / "chunks_256.jsonl").read_text(encoding="utf-8").splitlines()}
    model = nli.load_nli(cfg["verification"]["nli_model"])
    nli.scores(model, [("warm up", "warm up")])
    pairs = [json.loads(l) for l in pairs_path.read_text(encoding="utf-8").splitlines()]
    splits = list(dict.fromkeys(p.get("split", "targeted") for p in pairs))
    decided, ms = {}, []
    for p in pairs:
        t0 = time.perf_counter_ns()
        decided[p["pair_id"]] = judge(p, chunks, model, t)
        ms.append((time.perf_counter_ns() - t0) / 1e6)
    result = {"meta": {"label": label, "threshold": t, "git_commit": _git("rev-parse", "HEAD"),
                       "git_dirty": bool(_git("status", "--porcelain", "--untracked-files=no")),
                       "pairs_file": str(pairs_path), "pairs_sha256": hashlib.sha256(pairs_path.read_bytes()).hexdigest(),
                       "nli_model": cfg["verification"]["nli_model"], "corpus_version": cfg["corpus_version"],
                       "verification_policy_version": cfg["verification_policy_version"]}}
    result["splits"] = splits
    for split in splits:
        sel = [p for p in pairs if p.get("split", "targeted") == split]
        result[split] = metrics(sel, [decided[p["pair_id"]][0] for p in sel])
        result[split]["confusion"] = {
            "supported_accepted": sum(p["supported"] and decided[p["pair_id"]][0] for p in sel),
            "supported_rejected": sum(p["supported"] and not decided[p["pair_id"]][0] for p in sel),
            "unsupported_accepted": sum(not p["supported"] and decided[p["pair_id"]][0] for p in sel),
            "unsupported_rejected": sum(not p["supported"] and not decided[p["pair_id"]][0] for p in sel)}
        result[split + "_detail"] = {p["pair_id"]: decided[p["pair_id"]][1] for p in sel}
    q = statistics.quantiles(ms, n=100)
    result["latency_ms_per_answer"] = {"n": len(ms), "mean": round(statistics.mean(ms), 1),
                                       "p50": round(q[49], 1), "p95": round(q[94], 1)}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    return result


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True)
    ap.add_argument("--show-evaluation", action="store_true")
    ap.add_argument("--pairs", type=Path, default=PAIRS)
    a = ap.parse_args()
    r = main(a.label, a.pairs)
    for split in [s for s in r["splits"] if a.show_evaluation or s != "evaluation"]:
        m = r[split]
        print(split, {k: m[k] for k in ("n", "false_acceptance_rate_pct", "false_rejection_rate_pct", "accuracy_pct",
                                         "confusion")})
        for k, v in r[split + "_detail"].items():
            print("  ", k, "ok" if v["ok"] else "REJ", v["entailment"], "; ".join(v["reasons"])[:110])
    print("latency:", r["latency_ms_per_answer"])
