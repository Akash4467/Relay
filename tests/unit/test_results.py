"""results.csv and the tables, built from tiny synthetic raw files (never the real results)."""
import csv
import json

import pandas as pd
import pytest

from scripts import make_results_csv as mrc
from scripts import make_tables as mt


@pytest.mark.parametrize("text,seconds", [("1m26.4s", 86.4), ("705ms", 0.705), ("17.48s", 17.48), ("", None), ("x", None)])
def test_groq_durations(text, seconds):
    assert mrc.seconds(text) == (pytest.approx(seconds) if seconds is not None else None)


def write(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")


def test_results_csv_joins_client_and_server_views(tmp_path, monkeypatch):
    raw = tmp_path / "raw"
    monkeypatch.setattr(mrc, "RAW", raw)
    monkeypatch.setattr(mrc, "OUT", tmp_path / "results.csv")
    monkeypatch.setattr(mrc, "HISTORICAL_V2_SHA256", mrc.v2_digest([]))  # no v2 rows in this fake raw folder
    write(raw / "loadgen" / "steady_x.jsonl", [
        {"request_id": "steady-0000", "scenario": "steady", "source": "stand-in", "question_id": "q16",
         "start_time": "2026-11-13T01:00:00Z", "patience_ms": 15000, "admitted": True, "first_content_ms": 1800.0,
         "total_ms": 1801.0, "gave_up": False, "done_status": "ok", "done_reason": "model: not in papers",
         "done_tier": "refused"},
        {"request_id": "steady-0001", "scenario": "steady", "source": "stand-in", "question_id": "q05",
         "start_time": "2026-11-13T01:00:30Z", "patience_ms": 4000, "admitted": True, "first_content_ms": None,
         "total_ms": 4000.0, "gave_up": True, "done_status": "", "done_tier": ""},
        {"request_id": "smoke-0000", "scenario": "smoke", "source": "stand-in", "question_id": "q05",
         "start_time": "2026-11-13T00:00:00Z", "patience_ms": 1, "total_ms": 1, "gave_up": False, "done_status": "ok"}])
    (raw / "loadgen" / "steady_x.summary.json").write_text(json.dumps({"scenario": "steady", "duration_s": 600}))
    write(raw / "traces" / "2026-11-13.jsonl", [
        {"request_id": "steady-0000", "retrieval_setting": "B", "retries": 1, "tokens_in": 1487, "tokens_out": 37,
         "model": "openai/gpt-oss-120b", "retrieval_ms": 1000.0, "verification_ms": 0.0, "model_ttft_ms": 800.0,
         "queue_length_at_arrival": 2, "verification_policy_version": "v4",
         "calls": [{"rate_headers": {"x-ratelimit-limit-tokens": "8000", "x-ratelimit-reset-requests": "1m26.4s"}}]}])
    rows = mrc.main()
    assert [r["request_id"] for r in rows] == ["steady-0000", "steady-0001"]  # smoke runs are not a load shape
    with (tmp_path / "results.csv").open(encoding="utf-8") as f:
        header = next(csv.reader(f))
    assert header[:12] == mrc.REQUIRED  # the brief's columns, exact names, this order
    assert header[-1] == "verification_policy_version" and {r["verification_policy_version"] for r in rows} == {"v4"}
    # steady-0001 has no trace (gave up early): it takes its run's version; v4 stand-in rows also get their own file
    assert len(list(csv.DictReader((tmp_path / "v4_load_test_results.csv").open(encoding="utf-8")))) == 2
    first, second = rows
    assert (first["status"], first["retries"], first["tokens_in"], first["scenario_duration_s"]) == ("ok", 1, 1487, 600)
    assert first["ratelimit_reset_requests_s"] == pytest.approx(86.4) and first["queue_length_at_arrival"] == 2
    assert first["answer_grade"] == ""  # stand-in replies are never graded
    assert (second["status"], second["failure_reason"], second["tokens_in"]) == ("timeout", "gave_up", 0)
    assert second["answer_tier"] is None  # gave up: received nothing, whatever the server did later


def test_real_refusals_are_graded_by_rule():
    q = {"q16": {"answerable": False}, "q05": {"answerable": True}}
    row = {"request_id": "r", "source": "real", "status": "ok", "answer_tier": "refused"}
    assert mrc.grade(row | {"question_id": "q16"}, q, {}) == "refused_correctly"
    assert mrc.grade(row | {"question_id": "q05"}, q, {}) == "refused_wrongly"
    assert mrc.grade(row | {"question_id": "q05", "answer_tier": "normal"}, q, {}) == ""  # needs a manual grade
    assert mrc.grade(row | {"question_id": "q05"}, q, {"r": "correct"}) == "correct"


def test_tables_are_computed_from_the_csv_only(tmp_path, monkeypatch):
    rows = []
    for i in range(10):
        rows.append({"request_id": f"s-{i}", "scenario": "steady", "source": "stand-in", "start_time": "2026-11-13T01:00:00Z",
                     "first_token_ms": 1000 + 100 * i, "total_ms": 1000 + 100 * i,
                     "status": ["ok", "ok", "ok", "ok", "ok", "ok", "rate_limited", "rate_limited", "timeout", "error"][i],
                     "retries": 1 if i == 0 else 0, "tokens_in": 1000 if i < 6 else 0, "tokens_out": 100 if i < 6 else 0,
                     "model": "openai/gpt-oss-120b", "scenario_duration_s": 300, "queue_length_at_arrival": i % 4,
                     "answer_grade": "", "retrieval_setting": "B"})
    for r in rows:
        r["verification_policy_version"] = "v2"
    rows += [{"request_id": f"v4-{i}", "scenario": "steady", "source": "stand-in", "status": "error", "total_ms": 1,
              "verification_policy_version": "v4"} for i in range(3)]  # later runs never change the v2 tables
    pd.DataFrame(rows).reindex(columns=mrc.REQUIRED + mrc.EXTRA + ["verification_policy_version"]).to_csv(
        tmp_path / "r.csv", index=False)
    monkeypatch.setattr(mt, "CSV", tmp_path / "r.csv")
    monkeypatch.setattr(mt, "OUT", tmp_path / "t.md")
    md = mt.main()
    steady_a = next(l for l in md.splitlines() if l.startswith("| steady |") and "stand-in" in l)
    assert "| 10 | 300 | 2.00 |" in steady_a and "1450 / 1855 / 1891* (N=10)" in steady_a
    steady_b = [l for l in md.splitlines() if l.startswith("| steady |")][1]
    # 6 ok, 2 rate-limited, 1 timeout; 0.1 retries per request; peak queue 3; 1,100 tokens per answer;
    # cost: 6 x (1000 x 0.15 + 100 x 0.60) / 1e6 = $0.00126 -> per 1,000 succeeded answers: $0.21
    assert steady_b == "| steady | 60.0 | 20.0 | 10.0 | 0.10 | 3 | 1100 (N=6) | $0.21 |"
    assert "| burst | not measured | 0 |" in md and "| not measured* |" in md and "stand-in replies are fixtures" in md


def test_retrieval_csv_and_table_3b_keep_the_frozen_and_diagnostic_runs_apart(tmp_path, monkeypatch):
    from scripts import make_retrieval_csv as mr

    def report(split, corpus, pct):
        return {"meta": {"split": split, "corpus_version": corpus, "run_at": "t", "git_commit": "x", "git_dirty": False,
                         "settings": {"A": {"rerank": False}}},
                "results": [{"setting": "A", "piece_tokens": 256, "final_passages": 8, "answerable": 7,
                             "any_gold_retrieved_pct": pct, "all_gold_retrieved_pct": 50.0, "quote_recall_pct": 60.0,
                             "mean_context_tokens": 1468.9, "max_context_tokens": 1599}]}
    paths = {}
    for name, (split, corpus, pct) in {"frozen": ("test", "c1", 100.0), "diag": ("test", "c3", 85.7),
                                       "dev": ("dev", "c3", 75.0)}.items():
        paths[name] = tmp_path / f"{name}.json"
        paths[name].write_text(json.dumps(report(split, corpus, pct)), encoding="utf-8")
    monkeypatch.setattr(mr, "REPORTS", {str(paths["frozen"]): "frozen test result", str(paths["diag"]): "post-rebuild "
                                        "diagnostic", str(paths["dev"]): "dev"})
    rows = mr.main(tmp_path / "retrieval_eval.csv")
    assert [r["measurement"] for r in rows] == ["frozen test result", "post-rebuild diagnostic", "dev"]
    md = mt.coverage(tmp_path / "retrieval_eval.csv")
    # dev c3 | test c3 (diagnostic) | test c1 (frozen) | context tokens; a missing setting is never filled in
    assert "| A | 75.0 (N=7) | 85.7 (N=7) | 100.0 (N=7) |" in md  # dev c3 | test c3 diagnostic | test c1 frozen
    assert "| B | not measured | not measured | not measured |" in md
    assert mt.coverage(tmp_path / "missing.csv") == "not measured"


def test_rebuild_from_raw_reproduces_the_frozen_v2_rows():
    assert mrc.v2_digest(mrc.load_rows()) == mrc.HISTORICAL_V2_SHA256


def test_a_changed_historical_row_stops_the_build(tmp_path, monkeypatch):
    raw = tmp_path / "raw"
    monkeypatch.setattr(mrc, "RAW", raw)
    monkeypatch.setattr(mrc, "OUT", tmp_path / "results.csv")
    write(raw / "loadgen" / "steady_2026-11-12T19-50-05.jsonl", [  # a historical file name, different content
        {"request_id": "steady-0000", "scenario": "steady", "source": "stand-in", "question_id": "q16",
         "start_time": "t", "patience_ms": 1, "total_ms": 1, "gave_up": False, "done_status": "ok"}])
    with pytest.raises(SystemExit, match="historical v2 rows changed"):
        mrc.main()
    assert not (tmp_path / "results.csv").exists()


def test_a_row_without_a_version_outside_the_history_stops_the_build(tmp_path, monkeypatch):
    raw = tmp_path / "raw"
    monkeypatch.setattr(mrc, "RAW", raw)
    write(raw / "loadgen" / "steady_new.jsonl", [
        {"request_id": "steady-0000", "scenario": "steady", "source": "stand-in", "question_id": "q16",
         "start_time": "t", "patience_ms": 1, "total_ms": 1, "gave_up": False, "done_status": "ok"}])
    with pytest.raises(SystemExit, match="no verification_policy_version"):
        mrc.load_rows()
