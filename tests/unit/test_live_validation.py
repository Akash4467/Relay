"""Verifier injection and the live-validation runner: no network, no models, no provider calls."""
import json

import pytest

import relay.pipeline as pipeline
from relay import live_validation as lv
from relay.candidate_v5.verifier import verify as verify_v5
from relay.eval import answer_quality as aq
from relay.pipeline import Verifier, answer_from_evidence
from relay.retrieval.search import Evidence
from relay.verify.verifier import verify as verify_v4
from tests.unit.test_answer_quality import FakeCall, FakeClient, ScriptedLLM, ctx, http_error, lines, question
from tests.unit.test_pipeline import CORPUS, FakeLLM, P
from tests.unit.test_verifier import FakeNLI

FULLWIDTH = "Mimi uses 8 quantizers 【MOSHI:3.3.1:c030】."  # read by v5 (P1) and, since phase 10 L1, by v4.1
NOCITE = "Mimi uses 8 quantizers."  # rejected by every verifier ("no citation")
VERIFIER = Verifier(FakeNLI({"Mimi uses 8 quantizers.": 0.95}), 0.5, 0.8)
EV = Evidence(passages=P, top_dense_cos=0.8, top_ce=None, tokens=100)


# ---------------------------------------------------------------- A. verifier injection
def test_default_is_the_frozen_v4_verify(monkeypatch):
    assert pipeline.verify is verify_v4
    seen = []
    monkeypatch.setattr(pipeline, "verify", lambda *a, **k: seen.append("v4") or verify_v4(*a, **k))
    r = answer_from_evidence("How many quantizers does Mimi use?", EV, FakeLLM(FULLWIDTH, FULLWIDTH), VERIFIER, CORPUS)
    assert seen == ["v4"] and r.status == "answered"  # v4.1 (phase 10, L1) reads 【…】: accepted at once
    r = answer_from_evidence("How many quantizers does Mimi use?", EV, FakeLLM(NOCITE, NOCITE), VERIFIER, CORPUS)
    assert seen == ["v4", "v4", "v4"] and r.status == "refused"
    assert all("no citation" in s.reasons for rep in r.reports for s in rep.sentences)


def test_candidate_v5_only_when_passed_explicitly():
    r = answer_from_evidence("How many quantizers does Mimi use?", EV, FakeLLM(FULLWIDTH), VERIFIER, CORPUS,
                             verify_fn=verify_v5)
    assert (r.status, r.citations, r.llm_calls) == ("answered", ["MOSHI:3.3.1:c030"], 1)
    assert lv.VERIFIERS == {"v4": None, "v5": verify_v5}  # "v4" maps to the default, v5 must be named


def test_runner_passes_the_chosen_verifier_through(monkeypatch, tmp_path):
    monkeypatch.setattr(aq, "HostedLLM", ScriptedLLM)
    monkeypatch.setattr(aq, "search", lambda *a, **k: EV)
    # v4.1 and v5 now agree on 【…】 (phase 10, L1), so a verifier that rejects everything shows the pass-through
    reject_all = lambda draft, *a, **k: verify_v4(NOCITE, *a, **k)  # noqa: E731
    for chosen, status in ((None, "answered"), (reject_all, "refused")):
        ScriptedLLM.script = [FULLWIDTH, FULLWIDTH]
        c = ctx() | ({"verify_fn": chosen} if chosen else {})
        out = tmp_path / f"{status}.jsonl"
        aq.run([question("q02")], c, out, {})
        assert lines(out)[1]["final"]["status"] == status


# ---------------------------------------------------------------- B. protected paths
@pytest.mark.parametrize("bad", ["results/raw/real/x.jsonl", "results/raw/answer_quality/x.jsonl",
                                 "results/results.csv", "eval/answer_quality_results.csv",
                                 "results/live_validation/../raw/x.jsonl", "/tmp/x.jsonl"])
def test_outputs_cannot_leave_live_validation(bad):
    with pytest.raises(SystemExit, match="live-validation outputs go under"):
        lv.inside(lv.Path(bad))


def test_live_validation_paths_are_allowed():
    assert lv.inside(lv.Path("results/live_validation/provider_calls")).as_posix() == "results/live_validation/provider_calls"


def test_live_validation_root_is_not_inside_results_raw():
    assert not lv.OUT.resolve().is_relative_to(lv.PROTECTED.resolve())


# ---------------------------------------------------------------- B. caps
def provider_log(tmp_path, entries):
    log = tmp_path / "real" / "2026-11-14.jsonl"
    log.parent.mkdir(parents=True)
    req = {"messages": [{"role": "user", "content": "hi"}], "max_completion_tokens": 900}
    log.write_text("\n".join(json.dumps({"started_at": f"2026-11-14T10:00:{i:02d}.000+00:00", "request": req,
                                         "usage": u}) for i, u in enumerate(entries)) + "\n", encoding="utf-8")


def test_caps_are_fixed_at_10_calls_and_25000_tokens():
    assert (lv.MAX_CALLS, lv.MAX_TOKENS) == (10, 25_000)


def test_caps_count_every_earlier_call_including_failed_ones(tmp_path):
    client = type("C", (), {"source": "real"})()
    provider_log(tmp_path, [{"total_tokens": 2000}] * 8 + [None])  # 8 ok calls + 1 failed (worst case counted)
    b = lv.budgeted_client(client, tmp_path, 67)
    assert b.calls == 9 and b.tokens == 8 * 2000 + 67 + 1 + 4 + 900
    b.client = FakeClient(FakeCall(500))
    b.complete("m", [{"role": "user", "content": "hi"}], 900)  # call 10 of 10
    with pytest.raises(aq.BudgetExceeded):
        b.complete("m", [{"role": "user", "content": "hi"}], 900)  # an 11th call is refused before sending


def test_runner_refuses_to_start_once_a_cap_is_reached(tmp_path):
    client = type("C", (), {"source": "real"})()
    provider_log(tmp_path, [{"total_tokens": 100}] * 10)
    with pytest.raises(SystemExit, match="cap reached: 10/10 calls"):
        lv.budgeted_client(client, tmp_path, 67)
    provider_log(tmp_path / "t", [{"total_tokens": 25_000}])
    with pytest.raises(SystemExit, match="25000/25000 tokens"):
        lv.budgeted_client(client, tmp_path / "t", 67)


def test_token_cap_refuses_a_call_that_could_cross_it():
    b = lv.BudgetedClient(FakeClient(FakeCall(1)), lv.MAX_CALLS, lv.MAX_TOKENS, 67)
    b.tokens = lv.MAX_TOKENS - 500  # the next call's worst case (~970 tokens) would cross 25,000
    with pytest.raises(aq.BudgetExceeded):
        b.complete("m", [{"role": "user", "content": "hi"}], 900)
    assert b.calls == 0


def test_runner_stops_safely_when_the_budget_runs_out_mid_run(monkeypatch, tmp_path):
    monkeypatch.setattr(aq, "HostedLLM", ScriptedLLM)
    monkeypatch.setattr(aq, "search", lambda *a, **k: EV)
    ScriptedLLM.script = [NOCITE, aq.BudgetExceeded("cap")]  # rejected draft, then the redraft hits the cap
    out = tmp_path / "r.jsonl"
    end = aq.run([question("q02"), question("q10"), question("q12")], ctx(), out, {})
    assert (end["stopped_by"], end["questions_attempted"]) == ("BudgetExceeded", 1)
    assert lines(out)[1]["drafts"][0]["text"] == NOCITE  # the paid draft is kept


# ---------------------------------------------------------------- B. guards before anything is loaded
def test_runner_guards(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "groq")
    monkeypatch.setenv("GROQ_API_KEY", "dummy")
    monkeypatch.delenv("ALLOW_REAL", raising=False)
    with pytest.raises(SystemExit, match="--verifier"):
        lv.main(["q02"], "v6")
    with pytest.raises(SystemExit, match="--ids"):
        lv.main(["q04"], "v5")  # not a design-B question
    with pytest.raises(SystemExit, match="--ids"):
        lv.main(["q02", "q02"], "v5")
    with pytest.raises(SystemExit, match="config differs from the frozen set"):  # config is v4.1, the set is v4
        lv.main(["q02", "q10", "q12"], "v5")
    real = lv.app_config()
    monkeypatch.setattr(lv, "app_config", lambda: real | {"verification_policy_version": "v4"})
    with pytest.raises(SystemExit, match="ALLOW_REAL"):
        lv.main(["q02", "q10", "q12"], "v5")


# ---------------------------------------------------------------- per-run call limit
REAL = type("C", (), {"source": "real"})()
MSG = [{"role": "user", "content": "hi"}]


def log_at(dir_, entries):
    """Provider log with (started_at, usage) entries; usage None = a failed call."""
    f = dir_ / "real" / "2026-11-14.jsonl"
    f.parent.mkdir(parents=True, exist_ok=True)
    req = {"messages": MSG, "max_completion_tokens": 900}
    with f.open("a", encoding="utf-8") as out:
        for at, usage in entries:
            out.write(json.dumps({"started_at": at, "request": req, "usage": usage}) + "\n")


def test_six_calls_allowed_and_the_seventh_rejected_before_sending(tmp_path):
    b = lv.budgeted_client(REAL, tmp_path, 67, max_run_calls=6)
    b.client = FakeClient(*[FakeCall(1000)] * 7)
    for _ in range(6):
        b.complete("m", MSG, 900)
    with pytest.raises(aq.BudgetExceeded):
        b.complete("m", MSG, 900)
    assert b.calls == 6 and len(b.client.results) == 1  # the 7th was never sent


def test_failed_calls_count_toward_the_run_limit(tmp_path):
    b = lv.budgeted_client(REAL, tmp_path, 67, max_run_calls=6)
    b.client = FakeClient(*[http_error(503, "upstream_error")] * 3 + [FakeCall(1000)] * 4)
    for _ in range(3):
        with pytest.raises(aq.LLMError):
            b.complete("m", MSG, 900)
    for _ in range(3):
        b.complete("m", MSG, 900)
    with pytest.raises(aq.BudgetExceeded):
        b.complete("m", MSG, 900)
    assert b.calls == 6


def test_resume_keeps_the_runs_calls_and_the_cumulative_total(tmp_path):
    log_at(tmp_path, [("2026-11-14T09:00:00.000+00:00", {"total_tokens": 1000})] * 2)    # an earlier run
    log_at(tmp_path, [("2026-11-14T10:00:05.000+00:00", {"total_tokens": 1000})] * 3 +    # this run, before
           [("2026-11-14T10:00:09.000+00:00", None)])                                     # the cut (1 failed)
    b = lv.budgeted_client(REAL, tmp_path, 67, max_run_calls=6, run_started="2026-11-14T10:00:00+00:00")
    assert b.calls == 6 and b.max_calls == 8  # 6 made overall, 4 of them in this run: 2 more allowed
    b.client = FakeClient(*[FakeCall(1000)] * 3)
    b.complete("m", MSG, 900)
    b.complete("m", MSG, 900)
    with pytest.raises(aq.BudgetExceeded):
        b.complete("m", MSG, 900)  # the run's 7th call, counting the calls before the resume
    log_at(tmp_path / "full", [("2026-11-14T10:00:05.000+00:00", {"total_tokens": 1})] * 6)
    with pytest.raises(SystemExit, match="run limit reached: 6/6"):
        lv.budgeted_client(REAL, tmp_path / "full", 67, max_run_calls=6, run_started="2026-11-14T10:00:00+00:00")


def test_the_global_cap_still_wins_over_a_run_limit(tmp_path):
    log_at(tmp_path, [("2026-11-14T09:00:00.000+00:00", {"total_tokens": 100})] * 8)  # 8 of 10 used before
    b = lv.budgeted_client(REAL, tmp_path, 67, max_run_calls=6)
    assert b.max_calls == 10  # only 2 more, not 6


def test_token_cap_still_enforced_with_a_run_limit(tmp_path):
    log_at(tmp_path, [("2026-11-14T09:00:00.000+00:00", {"total_tokens": 24_500})])
    b = lv.budgeted_client(REAL, tmp_path, 67, max_run_calls=6)
    b.client = FakeClient(FakeCall(1))
    with pytest.raises(aq.BudgetExceeded):
        b.complete("m", MSG, 900)  # calls remain, but the worst case would cross 25,000 tokens
    assert b.calls == 1


@pytest.mark.parametrize("bad", [11, 100, 0, -1, 6.5, True, "6"])
def test_a_run_limit_cannot_raise_the_hard_cap_or_be_invalid(tmp_path, bad):
    with pytest.raises(SystemExit, match="max-run-calls must be a whole number from 1 to 10"):
        lv.budgeted_client(REAL, tmp_path, 67, max_run_calls=bad)
    with pytest.raises(SystemExit, match="max-run-calls"):
        lv.main(["q02"], "v5", max_run_calls=bad)  # refused before anything is loaded


def test_default_run_limit_is_the_hard_cap(tmp_path):
    assert lv.budgeted_client(REAL, tmp_path, 67).max_calls == lv.MAX_CALLS == 10
