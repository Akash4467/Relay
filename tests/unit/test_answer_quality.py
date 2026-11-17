"""The answer-quality runner: frozen set, hard budget, stop conditions, every draft kept. No network, no models."""
import json
from pathlib import Path

import pytest

from relay.eval import answer_quality as aq
from relay.llm.client import Call, LLMError
from relay.pipeline import Verifier
from relay.retrieval.search import Evidence
from tests.unit.test_verifier import CORPUS, P, FakeNLI

GOOD = "Mimi uses 8 quantizers [MOSHI:3.3.1:c030]."
BAD = "Mimi uses 9 quantizers [MOSHI:3.3.1:c030]."


def http_error(status: int, kind: str) -> LLMError:
    return LLMError(kind, Call(model="m", started_at="t", t_send_ns=0, status=status, error="x"))


class FakeCall:
    def __init__(self, total):
        self.usage = {"total_tokens": total}


class FakeClient:
    def __init__(self, *results):
        self.results = list(results)

    def complete(self, model, messages, max_tokens, *a, **kw):
        r = self.results.pop(0)
        if isinstance(r, Exception):
            raise r
        return r


class ScriptedLLM:
    """Stands in for HostedLLM: replies or raises from a shared script, records ok calls like HostedLLM."""
    script: list = []

    def __init__(self, *a, **kw):
        self.calls, self.retries = [], 0

    def __call__(self, messages):
        item = ScriptedLLM.script.pop(0)
        if isinstance(item, Exception):
            raise item
        self.calls.append({"status": "ok", "tokens_in": 100, "tokens_out": 10, "governor_wait_ms": 0.0, "total_ms": 1.0})
        return item


def ctx(client=None):
    cfg = {"pressure": {"L0": {"context_cap": 1600}}, "reservation": {"request_overhead_tokens": 67},
           "retry": {"max_transport_retries_per_call": 2, "backoff_base_ms": 500, "backoff_cap_ms": 8000}}
    index = type("Idx", (), {"chunks": {p["chunk_id"]: p for p in P}})()
    return {"cfg": cfg, "setting": None, "setting_name": "B", "index": index, "reranker": None, "governor": None,
            "client": client or aq.BudgetedClient(FakeClient(), 33, 105000, 67), "primary": {}, "reserve": 0,
            "retry_budget": None, "corpus_ids": CORPUS, "versions": {"verification_policy_version": "v4"},
            "verifier": Verifier(FakeNLI({"Mimi uses 8 quantizers.": 0.95}), 0.5, 0.8)}


def question(qid, answerable=True):
    return {"question_id": qid, "question": "How many quantizers does Mimi use?", "category": "direct",
            "answerable": answerable, "split": "test", "paper_id": "MOSHI", "expected_answer": "8", "evidence": []}


@pytest.fixture
def scripted(monkeypatch):
    monkeypatch.setattr(aq, "HostedLLM", ScriptedLLM)
    monkeypatch.setattr(aq, "search", lambda *a, **k: Evidence(passages=P, top_dense_cos=0.8, top_ce=None, tokens=50))
    return ScriptedLLM


def lines(path: Path, skip_started: bool = True) -> list[dict]:
    recs = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines()]
    return [r for r in recs if not (skip_started and r["kind"] == "started")]


def test_budget_refuses_a_call_that_could_break_a_cap_and_counts_usage():
    msgs = [{"role": "user", "content": "hi"}]
    b = aq.BudgetedClient(FakeClient(FakeCall(500), http_error(503, "upstream_error")), 3, 2000, 67)
    b.complete("m", msgs, 900)
    assert (b.calls, b.tokens) == (1, 500)  # the provider's usage, not the worst case
    with pytest.raises(LLMError):
        b.complete("m", msgs, 900)
    assert b.calls == 2 and b.tokens > 1400  # a failed call counts its worst case
    with pytest.raises(aq.BudgetExceeded):
        b.complete("m", msgs, 900)  # 1,4xx used + up to ~970 more > 2,000
    assert b.calls == 2  # refused before sending
    with pytest.raises(aq.BudgetExceeded):
        aq.BudgetedClient(FakeClient(), 0, 10**6, 67).complete("m", msgs, 900)  # call cap


def test_answered_refused_and_rejected_drafts_are_all_kept(scripted, tmp_path):
    scripted.script = [BAD, GOOD, BAD, BAD]
    out = tmp_path / "run.jsonl"
    end = aq.run([question("q04"), question("q17", answerable=False)], ctx(), out, {"source": "standin"})
    run, first, second, last = lines(out)
    assert (run["kind"], last["kind"], end["stopped_by"], end["questions_attempted"]) == ("run", "end", None, 2)
    assert first["final"]["status"] == "answered" and [d["verifier"]["accepted"] for d in first["drafts"]] == [False, True]
    assert first["drafts"][0]["text"] == BAD and not first["drafts"][0]["verifier"]["recomputed_after_stop"]
    assert second["final"]["status"] == "refused" and len(second["drafts"]) == 2  # both rejected drafts kept
    s = first["drafts"][0]["verifier"]["sentences"][0]
    assert s["citations"] == ["MOSHI:3.3.1:c030"] and s["accepted"] is False and s["reasons"]
    assert first["retrieval"]["passages"][0]["text"] == P[0]["text"]  # cited text kept for grading
    assert {"kind", "table_id", "rows"} <= set(first["retrieval"]["passages"][0])  # replays rebuild table premises
    assert first["tokens"]["total"] == 220 and first["verification_policy_version"] == "v4"


def test_budget_stop_mid_question_keeps_the_draft_and_stops_the_run(scripted, tmp_path):
    scripted.script = [BAD, aq.BudgetExceeded("cap")]
    out = tmp_path / "run.jsonl"
    end = aq.run([question("q04"), question("q06")], ctx(), out, {})
    rec = lines(out)[1]
    assert end["stopped_by"] == "BudgetExceeded" and end["questions_attempted"] == 1
    assert rec["final"] is None and rec["stopped_by"]["type"] == "BudgetExceeded"
    assert rec["drafts"][0]["text"] == BAD and rec["drafts"][0]["verifier"]["recomputed_after_stop"]
    with pytest.raises(FileExistsError):
        aq.run([question("q04")], ctx(), out, {})  # a run file is never overwritten


@pytest.mark.parametrize("errors, attempted, stopped", [
    ([http_error(401, "bad_request")], 1, "HTTP 401"),
    ([http_error(503, "upstream_error"), GOOD, http_error(503, "upstream_error"), GOOD], 3, None),
    ([http_error(503, "upstream_error"), http_error(500, "upstream_error")], 2, "2 provider errors in a row"),
])
def test_stop_conditions(scripted, tmp_path, errors, attempted, stopped):
    scripted.script = list(errors)
    end = aq.run([question("q04"), question("q06"), question("q07")], ctx(), tmp_path / "r.jsonl", {})
    assert (end["questions_attempted"], end["stopped_by"]) == (attempted, stopped)


def test_the_frozen_set_matches_the_question_file():
    spec, qs = aq.load_set()
    assert len(qs) == 15 and [q["question_id"] for q in qs] == spec["ids"]
    assert [q["question_id"] for q in qs if q["answerable"]] == spec["answerable_ids"] and len(spec["answerable_ids"]) == 10
    assert [q["question_id"] for q in qs if not q["answerable"]] == spec["unanswerable_ids"]
    assert set(spec["required_elements"]) == set(spec["answerable_ids"])
    assert not set(spec["authorship"]["unconfirmed_and_excluded"]) & set(spec["ids"])
    assert (spec["retrieval_setting"], spec["verification_policy_version"], spec["budget"]) == \
        ("B", "v4", {"max_calls": 33, "max_tokens": 105000})


def test_a_changed_question_file_is_refused(tmp_path):
    q = tmp_path / "q.jsonl"
    q.write_text(Path("eval/questions.jsonl").read_text(encoding="utf-8") + "\n", encoding="utf-8")
    spec = json.loads(aq.SET.read_text(encoding="utf-8")) | {"questions_file": str(q)}
    (tmp_path / "set.json").write_text(json.dumps(spec), encoding="utf-8")
    with pytest.raises(SystemExit, match="changed since the set was frozen"):
        aq.load_set(tmp_path / "set.json")


def test_real_runs_need_allow_real_the_full_set_and_no_earlier_run(monkeypatch, tmp_path):
    monkeypatch.setenv("LLM_PROVIDER", "groq")
    monkeypatch.setenv("GROQ_API_KEY", "dummy")
    monkeypatch.delenv("ALLOW_REAL", raising=False)
    with pytest.raises(SystemExit, match="the set is frozen for v4"):  # config is v4.1 since phase 10 (L1)
        aq.main(out_dir=tmp_path, raw_dir=None)
    real = aq.app_config()
    monkeypatch.setattr(aq, "app_config", lambda: real | {"verification_policy_version": "v4"})
    with pytest.raises(SystemExit, match="ALLOW_REAL"):
        aq.main(out_dir=tmp_path, raw_dir=None)
    monkeypatch.setenv("ALLOW_REAL", "1")
    with pytest.raises(SystemExit, match="stand-in only"):
        aq.main(ids=["q04"], out_dir=tmp_path, raw_dir=None)
    with pytest.raises(SystemExit, match="stand-in only"):
        aq.main(max_tokens=10**6, out_dir=tmp_path, raw_dir=None)
    (tmp_path / "real_earlier.jsonl").write_text("{}\n", encoding="utf-8")
    with pytest.raises(SystemExit, match="already exists"):
        aq.main(out_dir=tmp_path, raw_dir=None)


def test_pace_waits_until_a_whole_question_fits():
    class Gov:
        def __init__(self, waits):
            self.waits = list(waits)

        def wait_estimate(self, model, tokens):
            assert tokens == 2 * (2000 + 770)
            return self.waits.pop(0)

    slept = []
    c = ctx() | {"primary": {"id": "m"}, "reserve": 770}
    assert aq.pace(c | {"governor": Gov([12.0, 0.0])}, slept.append) == 13.0 and slept == [13.0]
    assert aq.pace(c | {"governor": Gov([5000.0])}, slept.append) == 0.0  # daily budget gone: don't wait it out


def test_an_unexpected_error_mid_question_is_recorded_and_stops_the_run(scripted, tmp_path, monkeypatch):
    def broken_nli(*a, **k):  # e.g. the NLI model running out of memory after the model already replied
        raise MemoryError("not enough memory")
    monkeypatch.setattr("relay.verify.verifier.nli.scores", broken_nli)
    scripted.script = [GOOD, GOOD]
    out = tmp_path / "run.jsonl"
    end = aq.run([question("q04"), question("q06")], ctx(), out, {})
    rec = lines(out)[1]
    assert end["stopped_by"].startswith("MemoryError") and end["questions_attempted"] == 1
    assert rec["final"] is None  # never turned into an answer
    assert rec["drafts"][0]["text"] == GOOD and rec["drafts"][0]["verifier"]["accepted"] is None  # draft kept


def test_resume_never_reasks_a_started_question_and_rereads_the_budget(scripted, tmp_path):
    out = tmp_path / "run.jsonl"
    scripted.script = [GOOD]
    aq.run([question("q04")], ctx(), out, {"started_at": "2026-11-13T10:00:00+00:00"})
    with out.open("a", encoding="utf-8") as f:  # interrupted after q06 started: no record for it
        f.write(json.dumps({"kind": "started", "question_id": "q06"}) + "\n")
    first, begun, cut_off = aq.started(out)
    assert first["kind"] == "run" and begun == {"q04", "q06"} and cut_off == {"q06"}
    scripted.script = [GOOD]
    aq.run([question("q07")], ctx(), out, {"x": 1}, resume=True)
    asked = [r["question_id"] for r in lines(out) if r["kind"] == "question"]
    assert asked == ["q04", "q07"] and [r["kind"] for r in lines(out)].count("resume") == 1

    log = tmp_path / "raw" / "real" / "2026-11-13.jsonl"
    log.parent.mkdir(parents=True)
    req = {"messages": [{"role": "user", "content": "hi"}], "max_completion_tokens": 900}
    log.write_text("\n".join(json.dumps(c) for c in [
        {"started_at": "2026-11-13T09:59:59.000+00:00", "request": req, "usage": {"total_tokens": 1}},  # before the run
        {"started_at": "2026-11-13T10:00:05.000+00:00", "request": req, "usage": {"total_tokens": 2000}},
        {"started_at": "2026-11-13T10:01:00.000+00:00", "request": req, "usage": None}]) + "\n")  # failed: worst case
    assert aq.used_since(tmp_path / "raw", "real", "2026-11-13T10:00:00+00:00", 67) == (2, 2000 + 67 + 1 + 4 + 900)


def test_nli_selfcheck_blocks_a_broken_model():
    import math

    class LabelledNLI(FakeNLI):  # logits in FakeNLI's label order: contradiction, entailment, neutral
        def predict(self, pairs):
            return [[math.log(0.005), math.log(0.99), math.log(0.005)] if "8 P100" in h else
                    [math.log(0.98), math.log(0.01), math.log(0.01)] for _, h in pairs]

    assert aq.nli_selfcheck(Verifier(LabelledNLI({}), 0.9, 0.9))["entailed"] == 0.99
    with pytest.raises(SystemExit, match="self-check failed"):
        aq.nli_selfcheck(Verifier(FakeNLI({}), 0.9, 0.9))  # entails nothing


def test_report_interval_and_owner_grade_override():
    from scripts import make_answer_quality as maq
    assert maq.wilson(0, 15) == "0.0% (0-20%)" and maq.wilson(5, 5) == "100.0% (57-100%)"
    g = {"assistant_label": "partially_correct", "confirmed_label": ""}
    assert maq.label(g) == "partially_correct" and maq.label(g | {"confirmed_label": "factually_wrong"}) == "factually_wrong"
