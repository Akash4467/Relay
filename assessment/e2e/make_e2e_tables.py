"""Writes assessment/e2e/e2e_tables.md: the latest real answer-quality measurement for the write-up, GENERATED from the
evaluation's own files (CLAUDE.md rule 4: numbers in the write-up are never typed by hand). scripts/build_writeup.py
inserts it at the {{E2E}} marker. Reads only:
  assessment/improvement_loop/baseline/baseline_metrics.json   production c3 + p4 + v4.2 (25 questions, live)
  assessment/e2e/live/E25_R3b_p7_v47_c6_metrics.json + ledger_e25_r3b.json    previous candidate (one run)
  assessment/e2e/live/E25_V48_metrics.json + ledger_e25_v48.json               latest candidate (one run)
  assessment/e2e/fix_v48/replay.json                                         offline replay of the v4.8c safeguards
    uv run python assessment/e2e/make_e2e_tables.py
"""
from __future__ import annotations

import json
import re
from pathlib import Path

E2E = Path("assessment/e2e")
OUT = E2E / "e2e_tables.md"
ROWS = [
    ("Production (c3, p4, v4.2)", "assessment/improvement_loop/baseline/baseline_metrics.json", None),
    ("Candidate v4.7c (c6, R3b, p7)", "assessment/e2e/live/E25_R3b_p7_v47_c6_metrics.json", "assessment/e2e/live/ledger_e25_r3b.json"),
    ("Candidate v4.8c (c6, R3b, p7)", "assessment/e2e/live/E25_V48_metrics.json", "assessment/e2e/live/ledger_e25_v48.json"),
]


def load(p: str) -> dict:
    return json.loads(Path(p).read_text(encoding="utf-8"))


def frac(m: dict, k: str) -> str:
    return f"{m[k]['n']}/{m[k]['d']}"


def main() -> str:
    lines = ["## Latest measurement (same 25 questions: 20 answerable, 5 not; real Groq, setting B)", "",
             "| Configuration | Correct | Partly | Wrong | Refused | Unanswerable refused | Cites retrieved | p50 s | Requests / tokens |",
             "|---|---|---|---|---|---|---|---|---|"]
    for name, metrics, ledger in ROWS:
        m = load(metrics)["all"]
        if ledger:
            runs = load(ledger)["runs"]
            assert len(runs) == 1, ledger
            r = next(iter(runs.values()))
            spend = f"{r['requests']} / {r['tokens']:,}"
        else:
            spend = "several runs"
        lines.append(f"| {name} | {frac(m, 'correct_answer')} | {frac(m, 'partially_correct')} | {frac(m, 'wrong_answer')} | "
                     f"{frac(m, 'false_refusal')} | {frac(m, 'correct_refusal')} | {frac(m, 'citation_validity_shown_answers')} | "
                     f"{m['total_ms']['p50'] / 1000:.1f} | {spend} |")
    rep = load("assessment/e2e/fix_v48/replay.json")["summary"]
    tests = re.search(r"(\d+) passed", Path("assessment/e2e/fix_v48/test_run.txt").read_text(encoding="utf-8")).group(1)
    small = load("assessment/e2e/live/V48_R3b_p7_c6_grades.json")["grades"]
    small_led = next(iter(load("assessment/e2e/live/ledger_v48.json")["runs"].values()))
    n_ok = sum(v["grade"] == "correct_and_fully_supported" for v in small.values())
    lines += ["", f"Live rows: one run per question, AI-graded; no wrong answer in one run is not a zero error rate. A small "
                  f"live check of v4.8c came first ({len(small)} questions: {n_ok} correct, {len(small) - n_ok} refused). Offline, "
                  f"its two safeguards changed only the {rep['changed_by']['both']} target drafts of {rep['n_cases']} replayed "
                  f"cases and did not have to act live; {tests} tests pass. Source: assessment/e2e/."]
    text = "\n".join(lines) + "\n"
    OUT.write_text(text, encoding="utf-8")
    return text


if __name__ == "__main__":
    print(main())
