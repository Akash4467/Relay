"""Verifier stress test (DESIGN §19.6): NLI models x acceptance policies x claim granularity.

Every claim is scored against each of its cited passages and against all of them joined (only the passages
listed as its evidence, as indexed with paper/section header); support = best entailment, contradiction =
highest contradiction. Level 2 flags come from verify/evidence.py.

Policies (per claim):
  P1 entailment-gated:      accept iff entailment >= t
  P2 contradiction-or-flag: reject if contradiction >= tc; else a flagged claim needs entailment >= tf;
                            an unflagged claim is accepted
Granularity:
  whole:  the sentence is one claim
  atomic: the sentence and each of its clauses (verify/claims.py); every one must be accepted

Thresholds are chosen per combination on the calibration split only, minimizing 2 x false acceptances +
false rejections (an accepted invented finding costs twice a rejected paraphrase); ties go to the stricter
setting. They are frozen, then the held-out split is evaluated once. The report keeps every raw score and
decision and is never overwritten. No LLM calls; eval/questions.jsonl is not used.

Round 1: eval/verifier_stress.jsonl -> eval/reports/verifier_stress.json (whole claims, two models).
Round 2: eval/verifier_stress_v2.jsonl -> eval/reports/verifier_stress_v2.json (fresh pairs).
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
from relay.verify import evidence, nli
from relay.verify.claims import atomic_claims
from relay.verify.verifier import premises

PAIRS = Path("eval/verifier_stress_v2.jsonl")
REPORT = Path("eval/reports/verifier_stress_v2.json")
MODELS = ("cross-encoder/nli-deberta-v3-base", "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli")
GRID = [0.01, 0.02, 0.05] + [round(0.1 + 0.05 * i, 2) for i in range(17)] + [0.95, 0.98, 0.99]
FA_WEIGHT = 2
CLAIMS_PER_ANSWER = 4  # latency per answer is estimated as 4 sentences (the prompt allows at most 6)


def load_pairs(path: Path = PAIRS) -> list[dict]:
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines()]


def score_claim(claim: str, cited: list[dict], model) -> dict:
    t0 = time.perf_counter_ns()
    scored = nli.scores(model, [(prem, claim) for prem in premises(cited)])
    return {"claim": claim, "flags": evidence.signals(claim, cited),
            "entail_by_premise": [round(e, 5) for e, _ in scored],
            "contra_by_premise": [round(c, 5) for _, c in scored],
            "entail": max(e for e, _ in scored), "contra": max(c for _, c in scored),
            "ms": round((time.perf_counter_ns() - t0) / 1e6, 2)}


def score_pairs(pairs: list[dict], chunks: dict[str, dict], model) -> list[dict]:
    rows = []
    for p in pairs:
        cited = [chunks[c] for c in p["evidence"]]
        claims = [score_claim(c, cited, model) for c in atomic_claims(p["claim"])]
        rows.append({"pair_id": p["pair_id"], "split": p["split"], "category": p["category"],
                     "supported": p["supported"], "claims": claims,
                     "ms_whole": claims[0]["ms"], "ms_atomic": round(sum(c["ms"] for c in claims), 2)})
    return rows


def p1(claim: dict, t: float) -> bool:
    return claim["entail"] >= t


def p2(claim: dict, tc: float, tf: float) -> bool:
    if claim["contra"] >= tc:
        return False
    return claim["entail"] >= tf if claim["flags"] else True


def decide(row: dict, policy, granularity: str, params: dict) -> bool:
    claims = row["claims"] if granularity == "atomic" else row["claims"][:1]
    return all(policy(c, **params) for c in claims)


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


def calibrate(rows: list[dict], policy, granularity: str) -> dict:
    """Grid search on calibration rows. Ties -> stricter: higher t / tf, lower tc."""
    def cost(params: dict) -> int:
        return _cost(metrics(rows, [decide(r, policy, granularity, params) for r in rows]))
    if policy is p1:
        return {"t": min(GRID, key=lambda t: (cost({"t": t}), -t))}
    tc, tf = min(((tc, tf) for tc in GRID for tf in GRID), key=lambda c: (cost({"tc": c[0], "tf": c[1]}), c[0], -c[1]))
    return {"tc": tc, "tf": tf}


def _latency(ms: list[float]) -> dict:
    q = statistics.quantiles(ms, n=100)
    return {"mean": round(statistics.mean(ms), 1), "p50": round(q[49], 1), "p95": round(q[94], 1)}


def evaluate(rows: list[dict]) -> dict:
    """Calibrate on the calibration split, freeze, evaluate the held-out split once."""
    cal = [r for r in rows if r["split"] == "calibration"]
    held = [r for r in rows if r["split"] == "heldout"]
    out = {}
    for gran in ("whole", "atomic"):
        for name, policy in (("P1", p1), ("P2", p2)):
            params = calibrate(cal, policy, gran)
            out[f"{name}-{gran}"] = {
                "params": params,
                "calibration": metrics(cal, [decide(r, policy, gran, params) for r in cal]),
                "heldout": metrics(held, [decide(r, policy, gran, params) for r in held]),
                "calibration_decisions": {r["pair_id"]: decide(r, policy, gran, params) for r in cal},
                "heldout_decisions": {r["pair_id"]: decide(r, policy, gran, params) for r in held},
            }
    for gran in ("whole", "atomic"):
        per_claim = _latency([r[f"ms_{gran}"] for r in rows])
        out[f"latency_ms_per_sentence_{gran}"] = per_claim
        out[f"latency_ms_per_answer_est_{gran}"] = {k: round(v * CLAIMS_PER_ANSWER, 1) for k, v in per_claim.items()}
    return out


def main(models: tuple[str, ...] = MODELS, pairs_path: Path = PAIRS, report: Path = REPORT) -> dict:
    if report.exists():
        raise FileExistsError(f"{report} exists: the held-out evaluation runs once and is never overwritten")
    pairs = load_pairs(pairs_path)
    chunks = {json.loads(l)["chunk_id"]: json.loads(l)
              for l in (PROCESSED_DIR / "chunks_256.jsonl").read_text(encoding="utf-8").splitlines()}
    result = {"meta": {"git_commit": _git("rev-parse", "HEAD"),
                       "git_dirty": bool(_git("status", "--porcelain", "--untracked-files=no")),
                       "pairs_file": str(pairs_path),
                       "pairs_sha256": hashlib.sha256(pairs_path.read_bytes()).hexdigest(),
                       "harness_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                       "grid": GRID, "fa_weight": FA_WEIGHT, "claims_per_answer": CLAIMS_PER_ANSWER},
              "models": {}}
    for name in models:
        model = nli.load_nli(name)
        nli.scores(model, [("warm up", "warm up")])  # first call pays one-off setup cost
        rows = score_pairs(pairs, chunks, model)
        result["models"][name] = {"evaluation": evaluate(rows), "rows": rows}
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def summary(result: dict) -> str:
    lines = ["model | combo | params | held-out FAR % | FRR % | acc % | calib FAR % | FRR % | ms/sentence p50 / p95"]
    for name, m in result["models"].items():
        ev = m["evaluation"]
        for combo in ("P1-whole", "P2-whole", "P1-atomic", "P2-atomic"):
            h, c = ev[combo]["heldout"], ev[combo]["calibration"]
            lat = ev[f"latency_ms_per_sentence_{combo.split('-')[1]}"]
            lines.append(f"{name.split('/')[-1]} | {combo} | {ev[combo]['params']} | {h['false_acceptance_rate_pct']} | "
                         f"{h['false_rejection_rate_pct']} | {h['accuracy_pct']} | {c['false_acceptance_rate_pct']} | "
                         f"{c['false_rejection_rate_pct']} | {lat['p50']} / {lat['p95']}")
    return "\n".join(lines)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pairs", default=str(PAIRS))
    ap.add_argument("--report", default=str(REPORT))
    args = ap.parse_args()
    print(summary(main(pairs_path=Path(args.pairs), report=Path(args.report))))
