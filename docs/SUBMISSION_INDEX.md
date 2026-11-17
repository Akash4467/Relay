# The Relay: final submission index

Cited question answering over three arXiv papers, built to stay predictable under bursts and rate limits.
Index of the final submission files. On 2026-11-14 they were moved from `submission/final/` into the repository's root folders; the paths below are the new locations.

## Contents

| Path | What it is |
|---|---|
| `writeup/relay_writeup.pdf` | **The two-page write-up** (2 A4 pages, rendered by scripts/render_writeup.py) |
| `writeup/relay_writeup.md`, `writeup/relay_writeup.template.md` | Its built Markdown and the template (tables inserted from results/tables.md by scripts/build_writeup.py) |
| `results/results.csv` | The brief's raw file: one row per load-test or smoke request (173 rows), the brief's 12 columns first, plus `verification_policy_version` (all rows v2) |
| `results/tables.md`, `results/retrieval_eval.csv` | Tables 1, 2a, 2b, 3 and 3b, generated from results.csv and the retrieval reports |
| `eval/answer_quality_*.csv`, `answer_quality_report.md` | The 15-question real-Groq evaluation (verifier v4, setting B): question-level results, final grades, and per-sentence verifier decisions and citation support. **Grades are preliminary: assistant-graded, not yet confirmed by the owner** |
| `eval/answer_quality_set.json` | The frozen question set (IDs, rubric, budget; pinned to the hash of eval/questions.jsonl) |
| `results/raw/answer_quality/real_2026-11-13T18-06-56Z.jsonl` | Raw record of that run: every draft, verifier decision, retrieved passage and call (no request headers or keys) |
| `assessment/e2e/live/E25_V48_requests.csv` | **Latest request-level evidence:** the candidate (c6 + R3b + p7 + v4.8c) 25-question real-Groq run, one row per request (41: 40 generation calls plus the model-availability check). Built only from the run's raw records by `assessment/e2e/make_requests_csv.py`; fields the records do not hold are left blank |
| `assessment/e2e/live/ERRATA.md` | That run's metadata labels its verifier "v4.2". The verifier actually used is the recorded `verifier_module` (relay.candidate_v4_8.verifier). Neither v4.8c-only safeguard fired, so the run does not separate v4.8c from v4.7c |
| `eval/verifier_v4_report_corrected.md` | Phase 4 investigation of verifier v4 (counts corrected to 19 verified drafts, 5 model refusals) |
| `eval/verifier_v5_offline_comparison_corrected.md`, `results/candidate_v5/` | **Offline candidate only**: v5 (P1, P6) replayed on saved drafts; never deployed or run live; 5 final answers unknown |
| `docs/relay_architecture.pdf` (+ `docs/architecture.md`) | How the pipeline works, start to end (status section updated to the real evaluation) |
| `docs/conversation_history/claude_code_conversation_export_2026-11-13.txt` | Unedited Claude Code export. **Partial:** it ends on 2026-11-13 around 12:45 and must be re-exported (see docs/COMPLETENESS_AUDIT.md) |
| `docs/COMPLETENESS_AUDIT.md` | Each brief requirement: completed, partial, missing or not verifiable, with evidence |

## Which results are which

- **Load tests** (Tables 2a, 2b, 3; results.csv): Go stand-in except `real_smoke`; verifier **v2**. The stand-in replays
  recorded answers, so these measure operations (admission, limits, retries, latency), never answer quality.
- **Answer quality, earlier**: 15 questions, real Groq, verifier **v4** (production), 24 calls. Not in results.csv.
- **Answer quality, latest**: the candidate c6 + R3b + p7 + v4.8c on the 25 frozen questions, real Groq, one run per question,
  assistant-graded (write-up §3; requests in `assessment/e2e/live/E25_V48_requests.csv`). **Not deployed**: production stays
  c3 + p4 + v4.2.
- **Verifier v5**: offline candidate replay only. Not deployed, not live-tested, not part of the evaluated system.

## Reproduce (from the repository root; no API calls)

```
uv run python scripts/make_results_csv.py && uv run python scripts/make_tables.py   # results.csv, tables.md
uv run python scripts/make_answer_quality.py                                        # answer-quality CSV and report
uv run python -m relay.candidate_v5.evaluate                                        # v5 offline comparison (new file)
uv run python scripts/build_writeup.py writeup/relay_writeup.template.md writeup/relay_writeup.md
uv run python scripts/render_writeup.py --src writeup/relay_writeup.md --out writeup/relay_writeup.pdf
```

Code, the full raw results (`results/raw/`) and the design log (`docs/DESIGN.md` §20.3) are in the repository.
No API key or credential is included in these files.
