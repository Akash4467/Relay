# Phase 12: controlled live Groq evaluation (c3 vs c6)

Experiment ID `phase12-live-c3-c6-2026-11-14`. Every number below comes from the files in `experiments/phase12/`.

- Runner: `run_phase12.py`.
- Analysis: `analyse_phase12.py`.
- Budget: `ledger.json`.

**Approvals.** The owner gave both approvals on 2026-11-14. Both are recorded in `phase12b/experiment_manifest.json` and `ledger.json`.
- "Accept risk again": use the current key. The earlier machine-security finding is still unresolved: no full scan has been run and the key has not been rotated.
- "Raise cap to 100,000": the token cap went from 30,000 to 100,000.

**What did not change.**
- c6 was not activated.
- `config/app.yaml` was not changed (`corpus.active: c3`, `corpus_version: c3`).
- No production code, corpus file, prompt, verifier, threshold, Phase 9B data or Phase 10/11 artifact was modified.
- Nothing was committed, pushed or deployed.

## 1. Starting state

- HEAD: `4688baf`, on branch `phase-1-retrieval` with uncommitted changes from Phases 10–11.
- The last change to any `relay/` source file was at 06:46 UTC. The first Phase 12 call was at 07:55 UTC.
- Configuration:
  - prompt p4;
  - verifier v4.2 (NLI `MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli`, threshold 0.9);
  - Configuration G (symbolic query expansion plus prompt-only caption sharing);
  - retrieval setting B (256-token pieces, cross-encoder reranker, context cap 2500);
  - model `openai/gpt-oss-120b`, `reasoning_effort` low, temperature 0.2, `max_tokens` 900.
- Before the live run, the runner was dry-run offline against a mock provider, writing to the scratchpad only. That dry run found two bugs in the runner itself, not in production code:
  - a redraft was counted as an initial call;
  - the prompt-identity check missed the question inside redraft requests.
  
  Both were fixed before any real call.

## 2. Phase 12A: connectivity (5 requests, 1,137 tokens)

| Check | Result |
|---|---|
| Auth and model list | HTTP 200. 11 models listed, including `openai/gpt-oss-120b` |
| Streaming completion | HTTP 200. Content streamed, `finish_reason` stop, usage in the stream; first content at 459 ms |
| Token accounting (real p4 prompt over 3 c3 passages) | Local estimate 979 vs provider 978 prompt tokens (−0.1%) |
| Rate-limit headers | Present: 8,000 TPM and 1,000 RPD limits |
| Invalid key | HTTP 401, handled as `bad_request`; the key appears nowhere in the artifacts |
| 1 ms first-byte timeout | Raised as `timeout` |

The gate (`phase12a/connectivity_results.json`) passed 7 of 7 checks.

## 3. Model and provider diagnostics (`phase12a/provider_diagnostics.json`)

**12B provider calls (38 in total):**
- All 38 returned HTTP 200 with `finish_reason` stop.
- There were no 429s, no 5xx errors, no timeouts and no transport retries.
- Usage was always reported in the OpenAI `usage` field.

**Tokens and latency per call:**
- Tokens: median 1,997, maximum 2,501.
- The local prompt-token estimate was off by ±1 token on every call.
- The local reservation was never exceeded (overrun 0 on every call).
- Model time to first content: median 722 ms. Whole call: median 829 ms.

**Rate-limit headroom:**
- The lowest remaining-TPM header seen was 2,830.
- The lowest remaining-RPD header seen was 960.

## 4. Question set and label provenance (`phase12b/questions.json`)

The question set was fixed before the run, by rules written into the file.

| Group | Questions | Rule | Labels |
|---|---|---|---|
| Ordinary answerable | q01, q02, q07, q15 | Original set; non-table categories; every gold unit retrieved on both corpora in Phase 11F | `eval/questions.jsonl`: project labels, not independently reviewed |
| Table-heavy, c6 more complete | t01, t05, t13 | Phase 11F `complete_evidence == gained`; the first question of each of the first three tables | `eval/phase7`: provisional, assistant-authored |
| Targeted | q11, t10, t29 | Required by the brief | q11 project labels; t10/t29 provisional |
| Unanswerable | q16, q27 | Original refusal questions on two different papers | Project labels |

**The table-heavy group was chosen because c6 holds more complete evidence for it.** Its results confirm that offline finding. They are not an unbiased estimate of c6's effect on table questions.

None of the seven unknown-provenance artifacts or three unknown-provenance stand-in fixtures was read.

## 5. Retrieval results (per corpus, from the live runs)

| | c3 | c6 |
|---|---|---|
| Answerable questions with every gold unit in the final context | 9/10 | 7/10 |
| Answerable questions with every gold unit **complete** in the final context (the answering cells present) | 4/10 | 7/10 |

**Different evidence on the two corpora** (5 of 12 questions):
- c3 has the rows but without the check marks: t01, t05, t13, t10, t29.
- c6 has complete rows for t01, t05 and t13.
- On c6, the gold rows for t10 and t29 are not retrieved.

**Identical first prompts on both corpora** (7 of 12 questions: q01, q02, q07, q15, q11, q16, q27): the evidence and the first prompt were byte-for-byte the same. On these questions the corpus cannot cause any outcome difference. Any difference comes from model sampling.

**Identity checks:** all passed on all 24 runs.
- Every evidence chunk the verifier received equals the selected corpus's original chunk.
- Every prompt contains the original question.
- No prompt contains the expanded retrieval query.

## 6. Real-model answer results

Outcomes for answered, answerable replies are **assistant judgments** (`phase12c/answer_judgments.json`), not an independent review. Every other outcome is derived mechanically.

| Outcome | c3 | c6 |
|---|---|---|
| Correct answer | 1 | 3 |
| Partially correct | 0 | 1 |
| Incorrect | **1** | 0 |
| Unsupported (answered an unanswerable question) | 0 | 0 |
| Correct abstention | 2/2 | 2/2 |
| False refusal | 8 | 6 |
| Invalid citation (in a shown reply) | 0 | 0 |
| Provider failure | 0 | 0 |
| Incomplete (budget or timeout) | 0 | 0 |
| **Answerable correct or partial** | **1/10** | **4/10** |

Rejected drafts: 8 runs on c3 and 6 on c6 needed a redraft. One c6 draft (q07) had uncited sentences, and the redraft fixed them.

**One wrong answer was shown to the user (c3/t05).** It reported a MUSHRA score of 65.9 ± 1.7; the expected answer is 64.0 ± 1.7.
- The c3 rows of MOSHI Table 3 carry no check marks, so the model took the first row.
- The verifier accepted the answer, because the cited row really does contain 65.9.
- This is a wrong answer from the current production configuration. It is documented here. No production code was changed (per the brief).

## 7. Paired comparison (12 pairs; raw counts)

Outcomes are ranked: correct answer = correct abstention, above partially correct, above everything else.

- **c6 better: 4**
  - t01, t05, t13: different evidence, so this is a corpus effect;
  - q11: identical prompt, so this is sampling.
- **c3 better: 1**
  - q01: identical prompt, so this is sampling. The c6 redraft scored 0.89 against the 0.9 threshold.
- **Same: 7**

**Attributable to the corpus: c6 better on 3 of the 5 questions where the evidence differed. c3 better on 0.**
- Those 3 are exactly the questions chosen because c6 has complete rows.
- On the other two (t10, t29), both corpora failed.

No statistical significance is claimed: 12 pairs, one sample each, temperature 0.2.

## 8. q11, t10, t29 (`phase12c/q11_t10_t29_analysis.json`)

**q11** (identical evidence on both corpora):
- `AIAYN:T3:r01`, the base row, is not in context on either corpus.
- c6: the model answered 24.9 for one head and said the passages do not give the base value. The verifier accepted it. Partially correct.
- c3: the model gave both 24.9 and 25.8, citing r02 for both. 25.8 is not in r02, so it was rightly rejected. The 24.9 sentence scored only 0.07, a false rejection. The redraft declined.
- The difference between the corpora is sampling. The retrieval gap from Phase 11F is confirmed live.

**t10** (both corpora declined with NOT_IN_PAPERS; no wrong answer was shown):
- c3: T4 rows r03/r04 are in context, but without the causal marks.
- c6: no T4 row is retrieved.

**t29:**
- c3: the model answered 31.3%, the expected value, but it cannot be derived from c3's unmarked rows (the same first-row guess was wrong for t05). The verifier rejected it at 0.76, partly because of "number 5" from the question's "five component columns".
- c6: the gold row `MOSHI:T17:r01` is not in context. The model answered 29.1% from r03, which is wrong and was rightly rejected.
- Both corpora refused, so no wrong answer was shown.

## 9. Failure breakdown (`phase12c/failure_analysis.json`)

| Layer | c3 | c6 |
|---|---|---|
| Verification: every draft rejected although complete evidence was in context | q02, q07, q15 | q01, q02, q07, q15 |
| Retrieval or extraction: rows in context but the answering cells missing | t01, t05 (wrong answer accepted), t13, t10, t29 | none |
| Retrieval: a gold unit not in context | q11 | q11, t10, t29 |
| Generation | 0 as primary layer | 0 as primary layer |
| Infrastructure | 0 | 0 |

**Verifier false rejections dominate the ordinary questions:**
- Ordinary answerable outcome: only 1 of 8 runs answered (4 questions × 2 corpora).
- The drafts were correct, by my judgment. The verifier rejected:
  - "eight … P100 GPUs" against "8 NVIDIA P100 GPUs" (q15, entailment 0.01);
  - "h=8 … heads" against "h=8 parallel attention layers, or heads" (q02, 0.01).
- For q07, the c3 drafts included one inaccurate sentence, which was correctly rejected.

## 10. Latency and tokens

**Wall time:** 8.5 minutes for 24 runs, sequential, from 07:56 to 08:04 UTC plus the last request.

| | c3 | c6 |
|---|---|---|
| Total request time, median (p90) | 28.1 s (32.6 s) | 15.2 s (31.7 s) |
| Governor wait, median | 22.3 s | 12.6 s |
| Generation, median | 1.7 s | 1.2 s |
| Verification, median (max) | 0.49 s (9.6 s) | 0.96 s (9.0 s) |
| Retrieval, median | 0.83 s | 0.82 s |
| First byte, median | 445 ms | 421 ms |
| LLM calls | 20 | 18 |
| Tokens in / out / of which reasoning | 37,560 / 2,588 / 1,432 | 34,202 / 2,314 / 1,204 |

**Wall time is dominated by the local rate governor, not by Groq.**
- Requests ran back to back. Each one reserved about 2,500 prompt tokens plus 770 output tokens against the 8,000 TPM bucket.
- Groq never came close to its own limit (lowest remaining TPM: 2,830).

The median gap between corpora reflects how many runs on each corpus needed a redraft and waited for capacity. It is not a corpus property.

## 11. Budget use (`ledger.json`)

| | Used | Cap |
|---|---|---|
| 12A connectivity requests | 5 | 5 |
| 12B initial generation calls | 24 | 24 |
| 12B provider requests (redrafts included) | 38 | 48 |
| Tokens (12A + 12B) | 1,137 + 76,664 = **77,801** | 100,000 |
| Calls refused by the cap | 0 | — |

Phase 10 used 29,826 tokens earlier the same day, so the day's total is about 107.6K of Groq's roughly 200K tokens per day.

## 12. Limitations

- **Sample size:** 12 questions, one live sample per question and corpus. Sampling alone flipped two pairs (q01, q11).
- **Label provenance:** question labels are project or provisional labels, and the correctness grades are mine. No independent reviewer checked either.
- **Selection bias:** the table-heavy questions were selected because c6 holds more complete evidence for them.
- **Missing coverage:**
  - no stress or concurrency test;
  - no 429 or daily-quota path exercised live;
  - only setting B tested.
- **Run order:** for each question the two corpora ran back to back. Order alternated between questions, but time-of-day effects are not controlled.

## 13. Did c6 show a meaningful practical advantage?

**Yes, on table questions whose answer depends on check marks, and only there.**
- On the 3 questions chosen for that property: c6 answered all 3 correctly. c3 answered none, and showed one **wrong** answer (t05).
- c6 showed no regression where the evidence was identical.
- c6 did not fix t10 or t29. It loses those gold rows in retrieval; c3 has them, but without marks. Both corpora refused, so neither was harmful.

**The biggest observed problem is not the corpus.** It is verifier v4.2 rejecting correct answers to ordinary questions, which neither corpus affects.

## 14. Recommendation

**Prepare a c6 activation proposal. Do not activate it from this experiment.**
- The proposal should cite this live evidence: 3 of 3 vs 0 of 3 on mark-dependent table questions, and c3's accepted wrong answer on t05.
- It should state the known c6 retrieval losses (t10, t29) and the selection bias.
- Activation still needs the owner's separate decision.

**Separately (out of scope here; no code changed): investigate verifier false rejections on ordinary questions.** On the 4 ordinary questions, 7 of 8 runs refused although complete evidence was in context. Examples: number words against digits, and "layers, or heads". This is a higher-impact fix than either corpus choice. It needs its own phase, and the verifier is frozen (v4.2), so any change needs owner approval.
