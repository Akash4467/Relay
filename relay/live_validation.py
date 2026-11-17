"""Live validation of a verifier on a few frozen questions (design B: candidate v5 live, compared later with the
existing live v4 run). Not run yet: it needs the owner's explicit approval after the security issue is resolved.

Every output goes under results/live_validation/ (run files, and the client's provider-call log in
results/live_validation/provider_calls/), never under results/raw/. Hard caps: 10 provider calls and 25,000 tokens
**cumulative across every run**, counted from that provider log (every attempt: retries and failed calls
included) and checked before the models load; BudgetedClient then refuses any call that could cross a cap. No
override exists. --max-run-calls (1-10, default 10) can only lower the calls one run may make; on --resume the
calls already made since that run started count against it. The verifier must be named (v4 or v5); there is no
provider fallback (Groq or the stand-in only). Reuses the evaluation runner: governor, pacing, NLI self-check, stop
conditions, "started" lines and --resume.

    LLM_PROVIDER=groq ALLOW_REAL=1 uv run --env-file .env python -m relay.live_validation --verifier v5 \
        --ids q02,q10,q12 --max-run-calls 6
"""
from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path

from relay.candidate_v5.verifier import verify as verify_v5
from relay import corpus
from relay.config import app_config, retrieval_settings
from relay.eval.answer_quality import (SET, BudgetedClient, load_set, nli_selfcheck, run, sha256, started,
                                       used_since)
from relay.eval.retrieval_recall import _git
from relay.governor.governor import Governor
from relay.llm.client import from_env
from relay.llm.hosted import RetryBudget
from relay.pipeline import Verifier
from relay.verify import nli

OUT = Path("results/live_validation")
PROTECTED = Path("results/raw")
MAX_CALLS, MAX_TOKENS = 10, 25_000
DESIGN_B = ("q02", "q09", "q10", "q12", "q13")
VERIFIERS = {"v4": None, "v5": verify_v5}  # None = answer_from_evidence's frozen v4 default


def inside(path: Path, root: Path = OUT) -> Path:
    """Only paths under results/live_validation/, never under results/raw/."""
    p = path.resolve()
    if not p.is_relative_to(root.resolve()) or p.is_relative_to(PROTECTED.resolve()):
        raise SystemExit(f"refusing to write {path}: live-validation outputs go under {root} only")
    return path


def run_limit(n: int) -> int:
    """A per-run call limit may lower the hard cap, never raise it."""
    if not isinstance(n, int) or isinstance(n, bool) or not 1 <= n <= MAX_CALLS:
        raise SystemExit(f"--max-run-calls must be a whole number from 1 to {MAX_CALLS} (the hard cap)")
    return n


def budgeted_client(client, raw_dir: Path, overhead: int, max_run_calls: int = MAX_CALLS,
                    run_started: str | None = None) -> BudgetedClient:
    """The caps, already charged with every call this validation has made before (all runs, from its log; failed
    calls and retries included). The run may make at most max_run_calls; on --resume the calls made since that run
    started (run_started) count against it. Whichever is tighter wins: the run limit, 10 calls or 25,000 tokens."""
    run_limit(max_run_calls)
    b = BudgetedClient(client, MAX_CALLS, MAX_TOKENS, overhead)
    b.calls, b.tokens = used_since(raw_dir, client.source, "1970-01-01T00:00:00+00:00", overhead)
    if b.calls >= MAX_CALLS or b.tokens >= MAX_TOKENS:
        raise SystemExit(f"cap reached: {b.calls}/{MAX_CALLS} calls, {b.tokens}/{MAX_TOKENS} tokens already used")
    in_run = used_since(raw_dir, client.source, run_started, overhead)[0] if run_started else 0
    if in_run >= max_run_calls:
        raise SystemExit(f"run limit reached: {in_run}/{max_run_calls} calls already made in this run")
    b.max_calls = min(MAX_CALLS, b.calls + max_run_calls - in_run)
    return b


def main(ids: list[str], verifier: str, resume: Path | None = None, out: Path = OUT,
         max_run_calls: int = MAX_CALLS) -> dict:
    from relay.retrieval.index_build import load_model
    from relay.retrieval.rerank import load_reranker
    from relay.retrieval.search import Index

    run_limit(max_run_calls)
    if verifier not in VERIFIERS:
        raise SystemExit(f"--verifier must be one of {sorted(VERIFIERS)}")
    if not ids or len(set(ids)) != len(ids) or set(ids) - set(DESIGN_B):
        raise SystemExit(f"--ids must be distinct questions from {DESIGN_B}")
    spec, frozen = load_set()  # refuses if eval/questions.jsonl changed
    by_id = {q["question_id"]: q for q in frozen}
    questions = [by_id[i] for i in ids]
    cfg = app_config()
    if (cfg["verification_policy_version"], cfg["models"]["primary"]["id"]) != \
            (spec["verification_policy_version"], spec["model"]):
        raise SystemExit("config differs from the frozen set (verifier v4 config, model)")
    raw_dir = inside(out / "provider_calls", out)
    client = from_env(cfg["timeouts"], raw_dir)
    if client.source == "real" and os.environ.get("ALLOW_REAL") != "1":
        raise SystemExit("real calls need ALLOW_REAL=1 and the owner's explicit approval")
    overhead = cfg["reservation"]["request_overhead_tokens"]
    resumed, run_started = {}, None
    if resume:
        inside(resume, out)
        first, begun, cut_off = started(resume)
        if first.get("source") != client.source or first.get("verifier") != verifier:
            raise SystemExit(f"{resume} was made with another provider or verifier")
        if first.get("max_run_calls") != max_run_calls:
            raise SystemExit(f"{resume} was started with --max-run-calls {first.get('max_run_calls')}; resume with the same")
        questions = [q for q in questions if q["question_id"] not in begun]
        run_started = first["started_at"]
        resumed = {"resumes": run_started, "not_reasked_cut_off": sorted(cut_off)}
    budgeted = budgeted_client(client, raw_dir, overhead, max_run_calls, run_started)

    primary = cfg["models"]["primary"]
    setting = retrieval_settings()[spec["retrieval_setting"]]
    ver = cfg["verification"]
    selected = corpus.selected(cfg)  # phase 11D: the configured corpus, validated before any call
    corpus.validate(selected, [setting.piece_tokens])
    index = corpus.load_indexes(selected, [setting.piece_tokens], load_model())[setting.piece_tokens]
    ctx = {
        "cfg": cfg, "setting": setting, "setting_name": spec["retrieval_setting"], "index": index,
        "reranker": load_reranker() if setting.rerank else None, "client": budgeted, "governor": Governor.from_config(),
        "primary": primary, "corpus_ids": set(index.chunks), "verify_fn": VERIFIERS[verifier],
        "retry_budget": RetryBudget.from_config(cfg["retry"]),
        "reserve": int(cfg["reservation"]["output_p95_tokens"][primary["id"]] * (1 + cfg["reservation"]["safety_margin"])),
        "verifier": Verifier(nli.load_nli(ver["nli_model"]), ver["entail_min"], ver["entail_min_flagged"]),
        "versions": {"source": client.source, "model": primary["id"], "reasoning_effort": primary["reasoning_effort"],
                     "prompt_version": cfg["prompt_version"], "corpus_version": cfg["corpus_version"],
                     "verifier": verifier, "verification_policy_version_config": cfg["verification_policy_version"],
                     "nli_model": ver["nli_model"], "entail_min": ver["entail_min"]},
    }
    header = {"started_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "design": "B",
              "verifier": verifier, "set_sha256": sha256(SET), "questions_sha256": spec["questions_sha256"],
              "ids": [q["question_id"] for q in questions], "caps": {"max_calls": MAX_CALLS, "max_tokens": MAX_TOKENS},
              "max_run_calls": max_run_calls, "run_call_ceiling": budgeted.max_calls,
              "already_used": {"calls": budgeted.calls, "tokens": budgeted.tokens},
              "git_commit": _git("rev-parse", "HEAD"), "git_dirty": bool(_git("status", "--porcelain")),
              "nli_selfcheck": nli_selfcheck(ctx["verifier"]), **resumed, **ctx["versions"]}
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ")
    path = inside(resume or out / f"{client.source}_{verifier}_{stamp}.jsonl", out)
    return run(questions, ctx, path, header, resume=bool(resume))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--verifier", required=True, choices=sorted(VERIFIERS))
    ap.add_argument("--ids", required=True, help=f"comma-separated, from {','.join(DESIGN_B)}")
    ap.add_argument("--resume", type=Path)
    ap.add_argument("--max-run-calls", type=int, default=MAX_CALLS,
                    help=f"calls this run may make, 1 to {MAX_CALLS}; can only lower the hard cap")
    a = ap.parse_args()
    print(json.dumps(main(a.ids.split(","), a.verifier, a.resume, max_run_calls=a.max_run_calls), indent=1))
