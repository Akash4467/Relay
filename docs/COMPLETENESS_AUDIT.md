# Completeness audit against the assignment brief

Status values: **Completed** (done and evidenced in the repository), **Partial**, **Missing**, **Not verifiable**
(cannot be confirmed from the repository). Answer-quality grades are preliminary (assistant-graded, unconfirmed).

| # | Requirement (brief) | Status | Evidence / gap |
|---|---|---|---|
| 1 | At least four pipeline steps | Completed | relay/pipeline.py: retrieve → structural check → draft → verify → redraft → answer/refuse |
| 2 | At least one tool call | Completed (interpretation) | Retrieval (relay/retrieval/search.py) is the tool step the pipeline calls each request; there is no model-initiated function calling |
| 3 | At least one branch or loop | Completed | Structural-check branch (refuse without a model call); one-redraft loop |
| 4 | One step retrieves passages from the library | Completed | BM25 + dense + RRF + cross-encoder; exact arXiv versions (data/raw/*/manifest.json) |
| 5 | Every answer cites; say so when the papers do not | Completed | Citation-required prompt; absence sentences; refusal text |
| 6 | Citation verification before sending | Completed, weak in practice | relay/verify (v4): it ignored 【ID】 citations, 21 of 46 sentences in the real run |
| 7 | Refusal behaviour | Completed | 5/5 unanswerable refused; but 9/10 answerable also refused (preliminary grades) |
| 8 | Real Groq free tier, own key, real measurements | Completed | 31 real calls, 58,617 tokens, no 429 (results/raw/real/); Table 1 |
| 9 | Stand-in: slow, fail, limit-refuse, slower under load, streaming; calibrated; differences stated | Completed | standin/ (Go); calibrated from 5 real answer calls; faults and slowdown are guesses (write-up §2) |
| 10 | At least three load shapes; raw results for both | Completed | steady, burst, overload, rpm_halved, daily_exhausted, retrieval A/B/C; results/raw/ |
| 11 | Table 1, 2a, 2b, 3 filled; "not measured" where not measured | Completed | Generated from results.csv (results/tables.md); copied into the write-up |
| 12 | Answer quality on at least 15 questions you wrote yourself, and tokens per answer | Partial | Run done: 15 questions, 3,231 tokens per question. Grades not yet confirmed by the owner; authorship of the 15 questions is recorded as "User idea N" notes (several reworded by the assistant) and needs the owner's confirmation |
| 13 | Retrieval: at least three settings with answer quality, tokens per answer, time to first token | Partial | A/B/C tokens and latency measured (stand-in); retrieval-only coverage measured; live answer quality measured for B only; Table 3 quality columns say "not measured" |
| 14 | results.csv, one row per request, brief's columns; every table figure computable from it | Partial | Exists with all 12 columns (173 rows, v2). The 15-question run's rows and grades are in eval/answer_quality_results.csv, not in results.csv (results.csv was not to be changed) |
| 15 | Two pages, headings in the brief's order | Completed | writeup/relay_writeup.pdf: 2 pages; all 12 required headings present in the brief's order (checked programmatically) |
| 16 | Where the time goes; limits; retries; what the person sees; unsure/left out/learned | Completed | Write-up §4–§8 (from traces and results.csv) |
| 17 | How the library misleads a naive retriever (three ways) | Completed | Write-up section, three example questions |
| 18 | Feedback on Vaani | Completed (owner-written) | Owner's own text, kept verbatim. It does not name something confusing or one change; it says why |
| 19 | How I worked with the assistant (three prompts, one error caught, one overrule, what is yours) | Partial | Owner's own text, kept verbatim. It does not quote three specific prompts or name the export file |
| 20 | Real, uncleaned conversation history | Partial | docs/conversation_history/: unedited export, no credentials found, but it ends on 2026-11-13 around 12:45 and misses the later work (real evaluation, Phase 4, v5, this submission) |
| 21 | Clear distinction between user and AI contributions | Partial / not verifiable | The owner's section states it; the repository cannot show who decided what beyond the conversation history |
| 22 | No key in the submission; within provider terms | Completed | Scans: no key, bearer token or credential patterns; raw logs store no headers; one account |
| 23 | Submit within 4 days | Not verifiable | Depends on the submission date |

**Missing:** none outright. **Partial:** 12, 13, 14, 19, 20, 21 (and 6, 7 work as designed but perform poorly).

## Owner actions before submitting
1. Confirm or change the grades (eval/answer_quality_grades.csv, `confirmed_label`) and the authorship of the
   15 questions; rerun `uv run python scripts/make_answer_quality.py` if any grade changes, and check the report text.
2. Re-export the full conversation with `/export` at the very end, check it for secrets, and replace the partial copy.
3. Optionally add one line to "How I worked with the assistant" naming the export file (the PDF must stay at 2 pages).
4. Rotate the Groq API key after the machine is cleaned (an unidentified process was found running during the work).
