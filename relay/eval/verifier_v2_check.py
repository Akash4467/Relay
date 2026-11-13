"""Local check of verifier v2 (DESIGN §20.3, 2026-11-13) before more Groq calls. No LLM calls.

1. Stress set eval/verifier_v2_stress.jsonl (20 fresh answers, 12 calibration / 8 held-out): each answer runs
   through the full production verifier (levels 1-2, context window, per-row table premises, absence checks);
   the LAST sentence is the judged claim. Held-out is evaluated once at the provisional threshold (config) and at
   the threshold the calibration split alone would choose (2 x false acceptances + false rejections, ties
   stricter). Latency is the wall time of verify() per answer, measured before any score caching.
2. Operational replays: the logged real drafts for q05, q11 and q16 rerun through verifier v2 / the pipeline.
   Observations, not evaluation data; nothing is tuned on them.
Writes eval/reports/verifier_v2_check.json and never overwrites it.
"""
from __future__ import annotations

import hashlib
import json
import statistics
import time
from pathlib import Path

from relay.config import app_config
from relay.eval.retrieval_recall import _git
from relay.eval.stress_verifier import FA_WEIGHT, GRID, metrics
from relay.llm import prompts
from relay.pipeline import Verifier, answer_from_evidence
from relay.retrieval.latex_parse import PROCESSED_DIR
from relay.retrieval.search import Evidence
from relay.verify import nli
from relay.verify.verifier import verify

PAIRS = Path("eval/verifier_v2_stress.jsonl")
REPORT = Path("eval/reports/verifier_v2_check.json")
REAL_CALLS = Path("results/raw/real/2026-11-12.jsonl")
REAL_RUNS = Path("results/raw/e2e/2026-11-12.jsonl")


class CachedNLI:
    """Same predictions, each (premise, hypothesis) pair computed once: threshold sweeps reuse the scores."""

    def __init__(self, model):
        self.model, self.config, self.cache = model, model.config, {}

    def predict(self, pairs):
        todo = [p for p in pairs if p not in self.cache]
        if todo:
            for p, logits in zip(todo, self.model.predict(todo)):
                self.cache[p] = logits
        return [self.cache[p] for p in pairs]


def judge(pair: dict, chunks: dict, model, t: float) -> tuple[bool, dict]:
    report = verify(pair["answer"], [chunks[c] for c in pair["evidence"]], set(chunks), model, t, t)
    s = report.sentences[-1]
    return s.ok, {"sentence": s.sentence, "ok": s.ok, "absence": s.absence, "entailment": s.entailment,
                  "context": s.context, "flags": s.flags, "reasons": s.reasons,
                  "all_sentences_ok": [x.ok for x in report.sentences]}


def evaluate(pairs: list[dict], chunks: dict, model, provisional: float) -> dict:
    timed = []
    for p in pairs:  # first pass: real compute, timed
        t0 = time.perf_counter_ns()
        judge(p, chunks, model, provisional)
        timed.append((time.perf_counter_ns() - t0) / 1e6)
    cached = CachedNLI(model)
    rows = {p["pair_id"]: p for p in pairs}

    def run(split: str, t: float) -> tuple[dict, dict]:
        sel = [p for p in pairs if p["split"] == split]
        decided = {p["pair_id"]: judge(p, chunks, cached, t) for p in sel}
        m = metrics([{"supported": p["supported"], "category": p["category"]} for p in sel],
                    [decided[p["pair_id"]][0] for p in sel])
        return m, {k: v[1] for k, v in decided.items()}

    def cost(t: float) -> int:
        m, _ = run("calibration", t)
        return FA_WEIGHT * m["false_acceptances"] + m["false_rejections"]

    calibrated = min(GRID, key=lambda t: (cost(t), -t))
    out = {"provisional_threshold": provisional, "calibration_choice": calibrated}
    for name, t in (("provisional", provisional), ("calibration_choice", calibrated)):
        cal, cal_detail = run("calibration", t)
        held, held_detail = run("heldout", t)
        out[name] = {"threshold": t, "calibration": cal, "heldout": held,
                     "calibration_detail": cal_detail, "heldout_detail": held_detail}
    q = statistics.quantiles(timed, n=100)
    out["latency_ms_per_answer"] = {"n": len(timed), "mean": round(statistics.mean(timed), 1),
                                    "p50": round(q[49], 1), "p95": round(q[94], 1)}
    out["pairs"] = {k: {"supported": v["supported"], "category": v["category"], "split": v["split"]}
                    for k, v in rows.items()}
    return out


def replays(chunks: dict, model, t: float) -> dict:
    """q05 and q11 drafts through verifier v2; q16's NOT_IN_PAPERS through the pipeline."""
    calls = [json.loads(l) for l in REAL_CALLS.read_text(encoding="utf-8").splitlines()]
    runs = {r["question_id"]: r for r in (json.loads(l) for l in REAL_RUNS.read_text(encoding="utf-8").splitlines())}
    drafts = {"q05": [calls[2]["text"], calls[3]["text"]], "q11": [calls[4]["text"], calls[5]["text"]],
              "q16": [calls[6]["text"]]}
    out = {}
    for qid in ("q05", "q11"):
        passages = [chunks[c] for c in runs[qid]["passages"]]
        out[qid] = []
        for d in drafts[qid]:
            rep = verify(d, passages, set(chunks), model, t, t)
            out[qid].append({"answer_ok": rep.ok, "sentences": [
                {"sentence": s.sentence, "ok": s.ok, "absence": s.absence, "entailment": s.entailment,
                 "context": s.context, "flags": s.flags, "reasons": s.reasons} for s in rep.sentences]})
    ev = Evidence([chunks[c] for c in runs["q16"]["passages"]], 0.0, None, 0)
    calls_made = []
    res = answer_from_evidence(runs["q16"]["question"], ev, lambda m: calls_made.append(m) or drafts["q16"][0],
                               Verifier(model, t, t), set(chunks))
    out["q16"] = {"status": res.status, "refusal_reason": res.refusal_reason, "text": res.text,
                  "llm_calls": len(calls_made), "not_in_papers_is_control_signal": res.text != prompts.NOT_IN_PAPERS}
    return out


def main(report: Path = REPORT) -> dict:
    if report.exists():
        raise FileExistsError(f"{report} exists: held-out pairs are evaluated once")
    cfg = app_config()
    chunks = {json.loads(l)["chunk_id"]: json.loads(l)
              for l in (PROCESSED_DIR / "chunks_256.jsonl").read_text(encoding="utf-8").splitlines()}
    model = nli.load_nli(cfg["verification"]["nli_model"])
    nli.scores(model, [("warm up", "warm up")])
    pairs = [json.loads(l) for l in PAIRS.read_text(encoding="utf-8").splitlines()]
    t = cfg["verification"]["entail_min"]
    result = {"meta": {"git_commit": _git("rev-parse", "HEAD"),
                       "git_dirty": bool(_git("status", "--porcelain", "--untracked-files=no")),
                       "pairs_sha256": hashlib.sha256(PAIRS.read_bytes()).hexdigest(),
                       "nli_model": cfg["verification"]["nli_model"], "corpus_version": cfg["corpus_version"],
                       "verification_policy_version": cfg["verification_policy_version"]},
              "stress": evaluate(pairs, chunks, model, t), "replays": replays(chunks, model, t)}
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    return result


if __name__ == "__main__":
    r = main()["stress"]
    for name in ("provisional", "calibration_choice"):
        x = r[name]
        print(f"{name} t={x['threshold']}: held-out FAR {x['heldout']['false_acceptance_rate_pct']}% "
              f"FRR {x['heldout']['false_rejection_rate_pct']}% acc {x['heldout']['accuracy_pct']}% | calibration FAR "
              f"{x['calibration']['false_acceptance_rate_pct']}% FRR {x['calibration']['false_rejection_rate_pct']}%")
    print("latency per answer:", r["latency_ms_per_answer"])
