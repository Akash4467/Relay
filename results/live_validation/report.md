# Live validation: candidate verifier v5 on q02, q10, q12 (real Groq)

Three questions, so no statistical claim is possible. Grades are assistant-graded and preliminary.

## 1. Command executed

```
LLM_PROVIDER=groq ALLOW_REAL=1 uv run --env-file .env python -m relay.live_validation --verifier v5 --ids q02,q10,q12 --max-run-calls 6
```

## 2. Security preflight status

**Not passed; the owner chose to proceed.** At the final preflight the unidentified process `svcruntime9227_c10b.exe` (PID 29340) was still running and Defender showed no completed full or offline scan. The owner stated that the old Groq key was revoked and replaced; `.env` was modified at 2026-11-14 01:08 local time (consistent with a new key; its content was not opened). The owner then explicitly instructed to proceed despite the security issue. No credential was logged: the provider log stores request bodies and responses, never headers.

## 3. Model and verifier versions

- Model `openai/gpt-oss-120b` (reasoning_effort low, Groq), prompt p3, corpus c3, retrieval setting B.
- Verifier: candidate **v5** (P1 full-width citations + P6 table-backed absence) on top of frozen v4 (NLI MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli, threshold 0.9); v4 itself unchanged. NLI self-check passed (0.991 / 0.999).
- Limits: per run 6 calls (ceiling 6), cumulative 10 calls / 25,000 tokens.

## 4. Per-question results

| Question | First draft | Redraft | Final response | Grade (draft → redraft → final) | Calls | Tokens | Total ms |
|---|---|---|---|---|---|---|---|
| q02 | rejected | rejected | refused | correct_full_cited → correct_full_cited → **false_refusal** | 2 | 4,062 | 5,731 |
| q10 | rejected | rejected | refused | partially_correct → partially_correct → **false_refusal** | 2 | 4,738 | 10,112 |
| q12 | rejected | rejected | refused | factually_wrong → correct_partially_cited → **false_refusal** | 2 | 3,931 | 4,056 |

- **q02:** Both drafts correct (8 heads, d_k = d_v = 64) and cite the gold chunk in 【…】, which P1 now parses. Rejected by NLI on the clause "The base Transformer uses eight parallel attention heads" (0.012): the passage says "we employ h=8" without "base" (the named-subject gap, P4).
- **q10:** Gold passages not retrieved. Drafts true but miss the stage structure. The redraft fixed two of four sentences (0.954, 0.958) after v5's feedback; two near-verbatim sentences stayed rejected (0.054, 0.021). One accepted sentence quotes "a Multi-stage / Hierarchy value of 51": a corpus parsing artifact (the paper's \ding{51} tick mark stored as 51).
- **q12:** P6 rejected the first draft's false absence claim, showing the exact retrieved row (Moshi: 26.6 / 62.3 / 22.8). The redraft then gave all six required numbers, correctly, citing MOSHI:T8:r01 for the with-Inner-Monologue row; NLI rejected it (0.062) because that row is labelled "Moshi", not "with Inner Monologue". No wrong answer was delivered.

## 5. API usage

- Calls: **6** of 6 allowed (all HTTP [200]; retries 0; no 429).
- Tokens: input 11,516, output 1,215 (of which reasoning 512), total **12,731** (provider log: 12,731; budget used 12,731 of 25,000).
- Time per question: q02 5.7 s, q10 10.1 s, q12 4.1 s; the runner also waited 0.0 s, 15.1 s, 25.4 s before each question for the per-minute token limit.

## 6. v4 versus v5 (different runs: drafts differ)

| Question | v4 run (2026-11-13): first / redraft / final | v5 live: first / redraft / final |
|---|---|---|
| q02 | False / False / refused → false_refusal | False / False / refused → false_refusal |
| q10 | False / False / refused → false_refusal | False / False / refused → false_refusal |
| q12 | True /  / answered → factually_wrong | False / False / refused → false_refusal |

- Full-width citation parsing failures: v4 run 8 sentences written in 【…】 (unparsed by v4); v5 live 0.
- Wrong answers delivered: v4 1 (q12); v5 0. Correct answers delivered: 0 in both. False refusals: v4 2, v5 3.

## 7. Incomplete questions

None: all three questions ran to a final response within the limits; the run was not stopped.

## 8. Did P1 and P6 behave as expected?

- **P1: yes, mechanically.** q02's 【AIAYN:3.2.2:c014】 was read as a citation (no "no citation" rejection), so the rejection now reflects content. It did not change the outcome: the NLI model rejects "The base Transformer uses eight heads" because the passage says "we employ h=8".
- **P6: yes.** It blocked q12's false absence claim with the exact row evidence, and its feedback produced a correct, complete redraft. That redraft was then falsely rejected by NLI (the row label is "Moshi", the claim says "with Inner Monologue"), so the outcome moved from a wrong answer (v4) to a false refusal (v5).
- **Feedback quality:** better redrafts in 2 of 3 (q12 fully correct; q10 two more sentences accepted), none delivered.
- **Unsupported claims accepted:** none delivered. One accepted q10 sentence quotes a parsing artifact ("value of 51", the paper's tick mark); it was not delivered because other sentences failed.

## 9. Is a larger 15-question evaluation justified?

**Not yet.** v5 did what it was designed to do and removed the wrong answer, but all three answerable questions were still refused because of the NLI named-subject/qualifier gap (Phase 4 cause P4), which v5 does not touch. A 15-question run of v5 would mostly re-measure that gap. Fix or redesign the entailment step first (and the Table 6 tick-mark parsing), evaluate offline, then decide on a larger live run.

Files: `real_v5_2026-11-13T19-39-49Z.jsonl` (full run record: passages, drafts, verifier decisions and scores, feedback via the redraft drafts, final responses, calls), `provider_calls/real/2026-11-13.jsonl` (6 provider calls, no headers), `per_question.csv`, `v4_vs_v5.csv`, this report.
