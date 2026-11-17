"""Real-model answer-quality evaluation over the frozen set eval/answer_quality_set.json.

Each question goes through the same functions the server's /ask handler uses (relay/main.py work()):
search -> answer_from_evidence -> HostedLLM (governor reserve -> call -> settle, retry policy). Everything is
kept: every draft (rejected ones too), the verifier's per-sentence decisions, the retrieved passages with their
text, gold-passage presence, calls, tokens and timings. One JSON line per question, appended and flushed as soon as
the question finishes, to a new file results/raw/answer_quality/{source}_{UTC}.jsonl that is never overwritten.

Hard limits (frozen in the set file): a call is refused before it is sent if it could push the run past
max_calls or max_tokens (worst case: prompt + max_completion_tokens). The run stops on: budget, a per-day 429,
any 4xx other than 429, 2 provider errors in a row, or any other error (e.g. the NLI model failing mid-run).
Nothing is retried or rerun by hand.

Before the first provider call the NLI model must pass a self-check (one entailment, one contradiction pair).
A "started" line is written before each question. --resume FILE continues an interrupted run in the same file:
questions already started are never asked again (one cut off mid-way is reported, not re-asked), and the calls
and tokens already used are re-read from the provider log, which records every attempt.

    # real (only after approval): the full frozen set, caps from the set file
    LLM_PROVIDER=groq ALLOW_REAL=1 uv run --env-file .env python -m relay.eval.answer_quality
    # dry run against the stand-in (it only has fixtures for q05/q11/q16)
    LLM_PROVIDER=standin GROQ_API_KEY=dummy uv run python -m relay.eval.answer_quality --ids q05,q11,q16 \
        --out-dir <scratch> --raw-dir <scratch>
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

from relay import corpus
from relay.config import app_config, retrieval_settings
from relay.eval.retrieval_recall import _git, gold_chunks
from relay.governor.governor import Governor
from relay.llm import prompts
from relay.llm.client import RAW_DIR, LLMError, from_env
from relay.llm.hosted import DailyExhausted, GovernorReject, HostedLLM, MESSAGE_OVERHEAD_TOKENS, RetryBudget
from relay.llm.tokenizer import count
from relay.pipeline import Verifier, answer_from_evidence
from relay.retrieval.search import policy_flags, search
from relay.verify import nli
from relay.verify.verifier import verify

SET = Path("eval/answer_quality_set.json")
OUT_DIR = Path("results/raw/answer_quality")


class BudgetExceeded(Exception):
    """The next call could take the run past its call or token cap."""


class BudgetedClient:
    """ChatClient wrapper that counts every attempt (retries included) and refuses one that could break the caps.
    Usage is the provider's total_tokens; a failed call counts its worst case, since its usage is unknown."""

    def __init__(self, client, max_calls: int, max_tokens: int, request_overhead: int):
        self.client, self.max_calls, self.max_tokens, self.overhead = client, max_calls, max_tokens, request_overhead
        self.calls = self.tokens = 0

    def __getattr__(self, name):
        return getattr(self.client, name)

    def complete(self, model, messages, max_tokens, *args, **kwargs):
        worst = self.overhead + sum(count(m["content"]) + MESSAGE_OVERHEAD_TOKENS for m in messages) + max_tokens
        if self.calls + 1 > self.max_calls or self.tokens + worst > self.max_tokens:
            raise BudgetExceeded(f"next call needs up to {worst} tokens; used {self.calls}/{self.max_calls} calls, "
                                 f"{self.tokens}/{self.max_tokens} tokens")
        self.calls += 1
        try:
            call = self.client.complete(model, messages, max_tokens, *args, **kwargs)
        except LLMError:
            self.tokens += worst
            raise
        self.tokens += (call.usage or {}).get("total_tokens") or worst
        return call


SELFCHECK = ("We trained our models on one machine with 8 NVIDIA P100 GPUs.",
             "The models were trained on 8 P100 GPUs.", "The models were trained on 16 TPUs.")


def nli_selfcheck(verifier: Verifier) -> dict:
    """The loaded NLI model must entail and contradict two fixed pairs before any provider call is made."""
    premise, entailed, contradicted = SELFCHECK
    (e1, c1), (e2, c2) = nli.scores(verifier.nli_model, [(premise, entailed), (premise, contradicted)])
    if not (e1 >= verifier.entail_min and e2 < 0.5 and c2 >= 0.5):
        raise SystemExit(f"NLI self-check failed (entailed {e1:.3f}, contradicted entail {e2:.3f} / contra {c2:.3f})")
    return {"entailed": round(e1, 3), "contradicted_entail": round(e2, 3), "contradicted_contra": round(c2, 3)}


def used_since(raw_dir: Path, source: str, since: str, overhead: int) -> tuple[int, int]:
    """Calls and tokens in the provider log since `since` (retries and failed attempts included)."""
    start, calls, tokens = datetime.fromisoformat(since), 0, 0
    for f in sorted((raw_dir / source).glob("*.jsonl")):
        for c in map(json.loads, f.read_text(encoding="utf-8").splitlines()):
            if datetime.fromisoformat(c["started_at"]) < start:
                continue
            req = c["request"]
            worst = overhead + sum(count(m["content"]) + MESSAGE_OVERHEAD_TOKENS for m in req["messages"]) + \
                req["max_completion_tokens"]
            calls += 1
            tokens += (c.get("usage") or {}).get("total_tokens") or worst
    return calls, tokens


def started(out: Path) -> tuple[dict, set[str], set[str]]:
    """An earlier run file: its header, the questions it started, and those started but never recorded."""
    recs = [json.loads(l) for l in out.read_text(encoding="utf-8").splitlines() if l]
    begun = {r["question_id"] for r in recs if r["kind"] == "started"}
    recorded = {r["question_id"] for r in recs if r["kind"] == "question"}
    return recs[0], begun, begun - recorded


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_set(path: Path = SET) -> tuple[dict, list[dict]]:
    """The frozen set and its questions, in set order; refuses if questions.jsonl changed since freezing."""
    spec = json.loads(path.read_text(encoding="utf-8"))
    qfile = Path(spec["questions_file"])
    if sha256(qfile) != spec["questions_sha256"]:
        raise SystemExit(f"{qfile} changed since the set was frozen: refusing to run")
    qs = {q["question_id"]: q for q in map(json.loads, qfile.read_text(encoding="utf-8").splitlines())}
    return spec, [qs[i] for i in spec["ids"]]


def sentences(report) -> list[dict]:
    return [{"sentence": s.sentence, "citations": s.citations, "accepted": s.ok, "absence": s.absence,
             "entailment": s.entailment, "flags": s.flags, "reasons": s.reasons, "context": s.context}
            for s in report.sentences]


def draft_records(drafts: list[str], reports: list, ev, corpus_ids: set[str], verifier: Verifier,
                  verify_fn=None) -> list[dict]:
    """One record per generated draft with the verifier's decision. Reports missing because the request stopped
    mid-way are recomputed locally (deterministic, no LLM) and marked so."""
    out, reports = [], list(reports)
    for i, text in enumerate(drafts):
        rec = {"n": i + 1, "text": text, "kind": "redraft" if i else "draft"}
        if text == prompts.NOT_IN_PAPERS:
            rec["verifier"] = None  # the model declined; nothing to verify
        else:
            recomputed = not reports
            try:
                report = reports.pop(0) if reports else (verify_fn or verify)(
                    text, ev.passages, corpus_ids, verifier.nli_model, verifier.entail_min, verifier.entail_min_flagged)
            except Exception as e:  # keep the draft even if the verifier cannot run now
                rec["verifier"] = {"accepted": None, "error": f"{type(e).__name__}: {e}"}
            else:
                rec["verifier"] = {"accepted": report.ok, "recomputed_after_stop": recomputed,
                                   "sentences": sentences(report)}
        out.append(rec)
    return out


def gold(q: dict, ev, chunks: dict) -> dict:
    if not q["answerable"]:
        return {"quotes": [], "any_in_context": None, "all_in_context": None}
    got = {p["chunk_id"] for p in ev.passages}
    try:
        sets = gold_chunks(q, chunks)
    except ValueError as e:
        return {"quotes": [], "error": str(e), "any_in_context": None, "all_in_context": None}
    quotes = [{"section": e["section"], "quote": e["quote"], "chunk_ids": sorted(ids), "in_context": bool(ids & got)}
              for e, ids in zip(q["evidence"], sets)]
    return {"quotes": quotes, "any_in_context": any(x["in_context"] for x in quotes),
            "all_in_context": all(x["in_context"] for x in quotes)}


def ask_one(q: dict, ctx: dict, answer=answer_from_evidence) -> tuple[dict, Exception | None]:
    """One question end to end. Returns the record and the exception that stopped it (if any)."""
    cfg, setting, index = ctx["cfg"], ctx["setting"], ctx["index"]
    started = datetime.now(timezone.utc).isoformat(timespec="milliseconds")
    t0 = time.perf_counter_ns()
    ev = search(q["question"], setting, cfg["pressure"]["L0"]["context_cap"], index, ctx["reranker"], **policy_flags(cfg))
    retrieval_ms = round((time.perf_counter_ns() - t0) / 1e6, 1)
    retry = cfg["retry"]
    llm = HostedLLM(ctx["client"], ctx["governor"], ctx["primary"], ctx["reserve"],
                    request_overhead=cfg["reservation"]["request_overhead_tokens"], question_id=q["question_id"],
                    deadline=None, max_wait_s=300.0,  # an evaluation waits for the governor instead of a user deadline
                    max_retries=retry["max_transport_retries_per_call"], backoff_base_s=retry["backoff_base_ms"] / 1000,
                    backoff_cap_s=retry["backoff_cap_ms"] / 1000, budget=ctx["retry_budget"])
    drafts: list[str] = []

    def recording(messages):
        text = llm(messages)
        drafts.append(text.strip())
        return text

    result, stop = None, None
    try:
        chosen = {"verify_fn": ctx["verify_fn"]} if ctx.get("verify_fn") else {}  # absent: the frozen v4 default
        result = answer(q["question"], ev, recording, ctx["verifier"], ctx["corpus_ids"], **chosen)
    except Exception as e:  # recorded, never swallowed: must_stop() ends the run on anything but a provider error
        stop = e
    total_ms = round((time.perf_counter_ns() - t0) / 1e6, 1)
    ok = [c for c in llm.calls if c["status"] == "ok"]
    rec = {
        "kind": "question", "started_at": started, "question_id": q["question_id"], "question": q["question"],
        "category": q["category"], "answerable": q["answerable"], "split": q["split"], "paper_id": q["paper_id"],
        "expected_answer": q["expected_answer"], "evidence": q["evidence"], **ctx["versions"],
        "retrieval": {"setting": ctx["setting_name"], "policy": ev.policy, "context_tokens": ev.tokens,
                      "ms": retrieval_ms, "gold": gold(q, ev, index.chunks),
                      # kind/table_id/rows: without them a replay cannot rebuild v4's table premises
                      "passages": [{k: p.get(k) for k in ("chunk_id", "paper", "section_id", "section_title", "page",
                                                          "n_tokens", "score_rrf", "score_ce", "dense_cos", "kind",
                                                          "table_id", "rows", "text")}
                                   for p in ev.passages]},
        "drafts": draft_records(drafts, result.reports if result else [], ev, ctx["corpus_ids"], ctx["verifier"],
                                ctx.get("verify_fn")),
        "final": None if result is None else {"status": result.status, "text": result.text,
                                              "citations": result.citations, "refusal_reason": result.refusal_reason},
        "stopped_by": None if stop is None else {"type": type(stop).__name__, "message": str(stop)},
        "calls": llm.calls, "retries": llm.retries,
        "tokens": {"in": sum(c["tokens_in"] for c in ok), "out": sum(c["tokens_out"] for c in ok),
                   "reasoning": sum(c.get("reasoning_tokens") or 0 for c in ok),
                   "total": sum(c["tokens_in"] + c["tokens_out"] for c in ok)},
        "timing_ms": {"retrieval": retrieval_ms, "llm": round(sum(c.get("total_ms") or 0 for c in llm.calls), 1),
                      "governor_wait": round(sum(c["governor_wait_ms"] for c in llm.calls), 1),
                      "verification": result.verification_ms if result else None, "total": total_ms},
    }
    return rec, stop


def must_stop(e: Exception | None, consecutive_errors: int) -> str | None:
    if e is None:
        return None
    if isinstance(e, (BudgetExceeded, DailyExhausted)):
        return type(e).__name__
    if isinstance(e, LLMError) and e.call.status and 400 <= e.call.status < 500 and e.call.status != 429:
        return f"HTTP {e.call.status}"
    if not isinstance(e, (LLMError, GovernorReject)):
        return f"{type(e).__name__}: {e}"
    if consecutive_errors >= 2:
        return "2 provider errors in a row"
    return None


def pace(ctx: dict, sleep=time.sleep) -> float:
    """Wait until the governor could cover a whole question (draft + redraft), so per-minute pacing never costs a
    question its reply. A very long wait (daily budget gone) is not waited out: the next call stops the run."""
    if ctx["governor"] is None:
        return 0.0
    # shortcut: assumes prompts of ~2,000 tokens (measured 1,564-1,812 + redraft growth); raise it if rejects recur
    need = 2 * (2000 + ctx["reserve"])
    waited = 0.0
    for _ in range(3):  # the estimate can be optimistic after provider headers resync the buckets
        wait = ctx["governor"].wait_estimate(ctx["primary"]["id"], need)
        if not 0 < wait <= 600:
            break
        sleep(wait + 1)
        waited += wait + 1
    return waited


def run(questions: list[dict], ctx: dict, out: Path, header: dict, answer=answer_from_evidence,
        resume: bool = False) -> dict:
    """Ask each question in order, appending one line per question; stop on the first stop condition."""
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("a" if resume else "x", encoding="utf-8") as f:  # "x": never overwrite an earlier run
        f.write(json.dumps({"kind": "resume" if resume else "run"} | header, ensure_ascii=False) + "\n")
    consecutive, stopped, done = 0, None, 0
    for q in questions:
        with out.open("a", encoding="utf-8") as f:  # before any call: a resumed run never asks it again
            f.write(json.dumps({"kind": "started", "question_id": q["question_id"]}) + "\n")
        paced_s = pace(ctx)
        rec, err = ask_one(q, ctx, answer)
        rec["timing_ms"]["paced_before"] = round(paced_s * 1000, 1)
        consecutive = consecutive + 1 if err is not None else 0
        with out.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        done += 1
        stopped = must_stop(err, consecutive)
        if stopped:
            break
    client = ctx["client"]
    end = {"kind": "end", "finished_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
           "questions_attempted": done, "questions_planned": len(questions), "stopped_by": stopped,
           "calls": client.calls, "tokens": client.tokens}
    with out.open("a", encoding="utf-8") as f:
        f.write(json.dumps(end, ensure_ascii=False) + "\n")
    return end


def main(ids: list[str] | None = None, out_dir: Path = OUT_DIR, raw_dir: Path = RAW_DIR,
         max_calls: int | None = None, max_tokens: int | None = None, allow_rerun: bool = False,
         resume: Path | None = None) -> dict:
    from relay.retrieval.index_build import load_model
    from relay.retrieval.rerank import load_reranker
    from relay.retrieval.search import Index

    spec, questions = load_set()
    cfg = app_config()
    if cfg["verification_policy_version"] != spec["verification_policy_version"]:
        raise SystemExit(f"verifier is {cfg['verification_policy_version']}, the set is frozen for "
                         f"{spec['verification_policy_version']}")
    primary = cfg["models"]["primary"]
    if primary["id"] != spec["model"]:
        raise SystemExit(f"primary model is {primary['id']}, the set is frozen for {spec['model']}")
    client = from_env(cfg["timeouts"], raw_dir)
    real = client.source == "real"
    if real:
        if os.environ.get("ALLOW_REAL") != "1":
            raise SystemExit("real run needs ALLOW_REAL=1 (and the owner's approval of the budget)")
        if ids or max_calls or max_tokens:
            raise SystemExit("a real run uses the whole frozen set and its caps; subsets and overrides are stand-in only")
        if list(out_dir.glob("real_*.jsonl")) and not allow_rerun and not resume:
            raise SystemExit(f"a real run already exists in {out_dir}; a rerun needs explicit approval (--allow-rerun)")
    if ids:
        known = {q["question_id"]: q for q in map(json.loads, Path(spec["questions_file"]).read_text(
            encoding="utf-8").splitlines())}
        questions = [known[i] for i in ids]
    caps = {"max_calls": max_calls or spec["budget"]["max_calls"], "max_tokens": max_tokens or spec["budget"]["max_tokens"]}
    budgeted = BudgetedClient(client, caps["max_calls"], caps["max_tokens"], cfg["reservation"]["request_overhead_tokens"])
    resumed = {}
    if resume:
        first, begun, cut_off = started(resume)
        if first.get("source") != client.source or first.get("set_sha256") != sha256(SET):
            raise SystemExit(f"{resume} was not made by this set and provider")
        questions = [q for q in questions if q["question_id"] not in begun]
        budgeted.calls, budgeted.tokens = used_since(raw_dir, client.source, first["started_at"],
                                                     cfg["reservation"]["request_overhead_tokens"])
        resumed = {"resumes": first["started_at"], "not_reasked_cut_off": sorted(cut_off),
                   "already_used": {"calls": budgeted.calls, "tokens": budgeted.tokens}}

    setting = retrieval_settings()[spec["retrieval_setting"]]
    ver = cfg["verification"]
    selected = corpus.selected(cfg)  # phase 11D: the configured corpus, validated before any call
    corpus.validate(selected, [setting.piece_tokens])
    index = corpus.load_indexes(selected, [setting.piece_tokens], load_model())[setting.piece_tokens]
    ctx = {
        "cfg": cfg, "setting": setting, "setting_name": spec["retrieval_setting"], "index": index,
        "reranker": load_reranker() if setting.rerank else None, "client": budgeted, "governor": Governor.from_config(),
        "primary": primary, "corpus_ids": set(index.chunks), "retry_budget": RetryBudget.from_config(cfg["retry"]),
        "reserve": int(cfg["reservation"]["output_p95_tokens"][primary["id"]] * (1 + cfg["reservation"]["safety_margin"])),
        "verifier": Verifier(nli.load_nli(ver["nli_model"]), ver["entail_min"], ver["entail_min_flagged"]),
        "versions": {"source": client.source, "model": primary["id"], "reasoning_effort": primary["reasoning_effort"],
                     "prompt_version": cfg["prompt_version"], "corpus_version": cfg["corpus_version"],
                     "verification_policy_version": cfg["verification_policy_version"],
                     "generation_config_version": cfg["generation_config_version"], "nli_model": ver["nli_model"],
                     "entail_min": ver["entail_min"]},
    }
    header = {"started_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "source": client.source,
              "set_file": str(SET), "set_sha256": sha256(SET), "questions_sha256": spec["questions_sha256"],
              "ids": [q["question_id"] for q in questions], "caps": caps, "git_commit": _git("rev-parse", "HEAD"),
              "git_dirty": bool(_git("status", "--porcelain")), "nli_selfcheck": nli_selfcheck(ctx["verifier"]),
              **resumed, **ctx["versions"]}
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ")
    return run(questions, ctx, resume or out_dir / f"{client.source}_{stamp}.jsonl", header, resume=bool(resume))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ids", help="comma-separated question IDs (stand-in dry runs only)")
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    ap.add_argument("--raw-dir", type=Path, default=RAW_DIR, help="where the client logs provider calls")
    ap.add_argument("--max-calls", type=int)
    ap.add_argument("--max-tokens", type=int)
    ap.add_argument("--allow-rerun", action="store_true")
    ap.add_argument("--resume", type=Path, help="continue an interrupted run file; started questions are not re-asked")
    a = ap.parse_args()
    print(json.dumps(main(a.ids.split(",") if a.ids else None, a.out_dir, a.raw_dir, a.max_calls, a.max_tokens,
                          a.allow_rerun, a.resume), indent=1))
