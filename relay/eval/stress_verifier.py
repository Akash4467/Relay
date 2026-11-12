"""Verifier stress test (DESIGN §19.6): two NLI models x two acceptance policies on 40 claim/evidence pairs.

Every claim is scored against each of its cited passages and against all of them joined (only the passages
listed as its evidence, as indexed with paper/section header); support = best entailment, contradiction =
highest contradiction. Level 2 flags come from verify/evidence.py.

Policies:
  P1 entailment-gated:      accept iff entailment >= t
  P2 contradiction-or-flag: reject if contradiction >= tc; else a flagged claim needs entailment >= tf;
                            an unflagged claim is accepted

Thresholds are chosen on the calibration split only, minimizing 2 x false acceptances + false rejections
(an accepted invented finding costs twice a rejected paraphrase); ties go to the stricter setting. They are
then frozen and the held-out split is evaluated once. The report keeps every raw score and decision and is
never overwritten. No LLM calls; eval/questions.jsonl is not used.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import time
from pathlib import Path

from relay.eval.retrieval_recall import _git
from relay.retrieval.latex_parse import PROCESSED_DIR
from relay.retrieval.index_build import INDEX_DIR
from relay.verify import evidence, nli
from relay.verify.verifier import premises

PAIRS = Path("eval/verifier_stress.jsonl")
REPORT = Path("eval/reports/verifier_stress.json")
MODELS = ("cross-encoder/nli-deberta-v3-small", "cross-encoder/nli-deberta-v3-base")
GRID = [0.01, 0.02, 0.05] + [round(0.1 + 0.05 * i, 2) for i in range(17)] + [0.95, 0.98, 0.99]
FA_WEIGHT = 2
CLAIMS_PER_ANSWER = 4  # latency per answer is estimated as 4 claims (the prompt allows at most 6 sentences)


def load_pairs(path: Path = PAIRS) -> list[dict]:
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines()]


def score_pairs(pairs: list[dict], chunks: dict[str, dict], vocab: evidence.Vocabulary, model) -> list[dict]:
    rows = []
    for p in pairs:
        cited = [chunks[c] for c in p["evidence"]]
        t0 = time.perf_counter_ns()
        scored = nli.scores(model, [(prem, p["claim"]) for prem in premises(cited)])
        ms = (time.perf_counter_ns() - t0) / 1e6
        rows.append({"pair_id": p["pair_id"], "split": p["split"], "category": p["category"],
                     "supported": p["supported"], "flags": evidence.signals(p["claim"], cited, vocab),
                     "entail_by_premise": [round(e, 5) for e, _ in scored],
                     "contra_by_premise": [round(c, 5) for _, c in scored],
                     "entail": max(e for e, _ in scored), "contra": max(c for _, c in scored), "ms": round(ms, 2)})
    return rows


def p1(row: dict, t: float) -> bool:
    return row["entail"] >= t


def p2(row: dict, tc: float, tf: float) -> bool:
    if row["contra"] >= tc:
        return False
    return row["entail"] >= tf if row["flags"] else True


def metrics(rows: list[dict], accepted: list[bool]) -> dict:
    sup = [a for r, a in zip(rows, accepted) if r["supported"]]
    uns = [a for r, a in zip(rows, accepted) if not r["supported"]]
    fa, fr = sum(uns), sum(not a for a in sup)
    out = {"n": len(rows), "false_acceptances": fa, "false_rejections": fr,
           "false_acceptance_rate_pct": round(100 * fa / len(uns), 1) if uns else None,
           "false_rejection_rate_pct": round(100 * fr / len(sup), 1) if sup else None,
           "accuracy_pct": round(100 * (len(rows) - fa - fr) / len(rows), 1) if rows else None}
    cats: dict[str, dict] = {}
    for r, a in zip(rows, accepted):
        c = cats.setdefault(r["category"], {"supported": 0, "false_rejections": 0, "unsupported": 0, "false_acceptances": 0})
        if r["supported"]:
            c["supported"] += 1
            c["false_rejections"] += not a
        else:
            c["unsupported"] += 1
            c["false_acceptances"] += a
    out["by_category"] = cats
    return out


def _cost(m: dict) -> int:
    return FA_WEIGHT * m["false_acceptances"] + m["false_rejections"]


def calibrate_p1(rows: list[dict]) -> dict:
    # ties -> higher t (stricter)
    best = min(GRID, key=lambda t: (_cost(metrics(rows, [p1(r, t) for r in rows])), -t))
    return {"t": best}


def calibrate_p2(rows: list[dict]) -> dict:
    # ties -> lower tc and higher tf (both stricter)
    combos = [(tc, tf) for tc in GRID for tf in GRID]
    tc, tf = min(combos, key=lambda c: (_cost(metrics(rows, [p2(r, *c) for r in rows])), c[0], -c[1]))
    return {"tc": tc, "tf": tf}


def evaluate(rows: list[dict]) -> dict:
    """Calibrate on the calibration split, freeze, evaluate the held-out split once."""
    cal = [r for r in rows if r["split"] == "calibration"]
    held = [r for r in rows if r["split"] == "heldout"]
    out = {}
    for name, calibrate, decide in (("P1", calibrate_p1, p1), ("P2", calibrate_p2, p2)):
        params = calibrate(cal)
        out[name] = {
            "params": params,
            "calibration": metrics(cal, [decide(r, **params) for r in cal]),
            "heldout": metrics(held, [decide(r, **params) for r in held]),
            "heldout_decisions": {r["pair_id"]: decide(r, **params) for r in held},
            "calibration_decisions": {r["pair_id"]: decide(r, **params) for r in cal},
        }
    ms = [r["ms"] for r in rows]
    out["latency_ms_per_claim"] = {"mean": round(statistics.mean(ms), 1),
                                   "p50": round(statistics.quantiles(ms, n=100)[49], 1),
                                   "p95": round(statistics.quantiles(ms, n=100)[94], 1)}
    out["latency_ms_per_answer_est"] = {k: round(v * CLAIMS_PER_ANSWER, 1) for k, v in out["latency_ms_per_claim"].items()}
    return out


def main(models: tuple[str, ...] = MODELS, report: Path = REPORT) -> dict:
    if report.exists():
        raise FileExistsError(f"{report} exists: the held-out evaluation runs once and is never overwritten")
    pairs = load_pairs()
    chunks = {json.loads(l)["chunk_id"]: json.loads(l)
              for l in (PROCESSED_DIR / "chunks_256.jsonl").read_text(encoding="utf-8").splitlines()}
    vocab = evidence.Vocabulary(json.loads((INDEX_DIR / "idf_256.json").read_text(encoding="utf-8")))
    result = {"meta": {"git_commit": _git("rev-parse", "HEAD"),
                       "git_dirty": bool(_git("status", "--porcelain", "--untracked-files=no")),
                       "pairs_sha256": hashlib.sha256(PAIRS.read_bytes()).hexdigest(),
                       "harness_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                       "grid": GRID, "fa_weight": FA_WEIGHT, "claims_per_answer": CLAIMS_PER_ANSWER},
              "models": {}}
    for name in models:
        model = nli.load_nli(name)
        nli.scores(model, [("warm up", "warm up")])  # first call pays one-off setup cost
        rows = score_pairs(pairs, chunks, vocab, model)
        result["models"][name] = {"evaluation": evaluate(rows), "rows": rows}
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def summary(result: dict) -> str:
    lines = ["model | policy | params | held-out FAR % | FRR % | acc % | calibration FAR % | FRR % | ms/claim p50 / p95"]
    for name, m in result["models"].items():
        ev = m["evaluation"]
        lat = ev["latency_ms_per_claim"]
        for pol in ("P1", "P2"):
            h, c = ev[pol]["heldout"], ev[pol]["calibration"]
            lines.append(f"{name.split('/')[-1]} | {pol} | {ev[pol]['params']} | {h['false_acceptance_rate_pct']} | "
                         f"{h['false_rejection_rate_pct']} | {h['accuracy_pct']} | {c['false_acceptance_rate_pct']} | "
                         f"{c['false_rejection_rate_pct']} | {lat['p50']} / {lat['p95']}")
    return "\n".join(lines)


if __name__ == "__main__":
    argparse.ArgumentParser(description=__doc__).parse_args()
    print(summary(main()))
