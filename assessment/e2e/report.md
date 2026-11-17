# The Relay: end-to-end assessment (in progress)

Status (2026-11-14): sections 1–3 are measured offline. Section 4 has the first live batch (setting B, 12 questions, 40,750 tokens). The latest full live measurement is section 15: c6 + R3b + p7 + v4.8c on the 25 frozen questions. Settings A and C have not been measured live.

Nothing in production has changed:
- `config/app.yaml`: corpus c3, prompt p4, verifier v4.2, setting B.
- No commits.

Every number below comes from the files named next to it.

## 1. What is measured, and what is not

### Write-up Table 3 has no answer-quality figures for A, B or C

- Every load-test run used the stand-in, which replays fixed replies. A replayed reply says nothing about answer quality.
- Only 3 real questions were run for the write-up, and all 3 were refused.
- `writeup/relay_writeup.md` lines 50 and 63–71 say this themselves.

### The historical results do not describe the current production system

| Area | Historical result | Current production |
|---|---|---|
| Retrieval | Write-up Table 3b: N = 7–8 answerable questions, corpora c1/c3, before Configuration G (Phase 11B) | Configuration G on c3 |
| Load tests | Verifier v2 (CLAUDE.md "Current phase") | Verifier v4.2 |
| Prompt | p2/p3 era | p4 |

### Real answer quality measured so far

- Phase 12 and the improvement-loop baseline: 25 questions on c3, setting B only.
  - 2 of 20 answerable questions answered correctly;
  - 16 of 20 falsely refused;
  - 5 of 5 unanswerable questions correctly refused.
  - Source: `assessment/improvement_loop/baseline/baseline_metrics.json`.
- The same 25 questions on the candidate stacks, setting B (one run each; section 15 has the full table):
  - c6 + R3b + p7 + v4.7c: 8 of 20 correct, 4 partially correct, 2 wrong answers shown (`live/E25_R3b_p7_v47_c6_metrics.json`);
  - c6 + R3b + p7 + v4.8c: 9 of 20 correct, 5 partially correct, 0 wrong answers shown, 5 of 5 unanswerable correctly refused (`live/E25_V48_metrics.json`).
- No real measurement exists for settings A or C.

### Gold evidence annotations

| Question set | Questions | Label status |
|---|---|---|
| Original | 30 | Project labels; gold quotes located by the assistant |
| Phase 7 table set | 32 | Provisional, written by the assistant |
| Phase 11A new cases | 12 | Provisional |

- No label was edited for this assessment.
- **Every** gold unit (77 of 77) matches at least one chunk, on both c3 and c6, so no annotation points at missing text.
- **The gold sets are stricter than the questions need.** Several questions are counted as retrieval failures although equivalent evidence was in context:
  - q03: the live answer was correct without the second gold chunk (MOSHI:7:c125);
  - n10: AIAYN:T3:r01 in context also states h = 8;
  - q10: HNET:3.1:c049 in context states that 2-stage beats 1-stage.
- Retrieval coverage therefore **under-counts**. Real answer grading (section 4) has to be the deciding measure.

### Does the verifier reject correct answers?

**Yes.** In the loop baseline, 9 recorded drafts that I labelled correct were rejected, including sentences copied exactly from the cited chunk (iteration 002). Iteration 003, which targets this, was later completed and **rejected**: it raised false acceptances (section 7). The exact-support rule in v4.6c and later reduced these rejections without new false acceptances (section 7); some correct answers are still refused (section 15).

## 2. Retrieval A, B and C on c3, the production corpus (`retrieval/baseline_c3_production/`)

**Question set:** 74 questions (49 answerable, 77 gold units, 25 unanswerable), run through production `search()` with the Configuration G flags.

**Facets** were assigned by rule from each question's existing category (see `retrieval_eval.py`):

| Facet | Answerable questions |
|---|---|
| Table-heavy | 36 |
| Numerical | 22 |
| Cross-paper | 2 |
| Terminology mismatch | 8 |

Per-question results are in `per_question_refined.jsonl`.

| | A (256, 8, no rerank) | B (256, 8, rerank) | C (512, 4, rerank) |
|---|---|---|---|
| Recall@k (k = final passages), mean | 0.903 | 0.893 | 0.891 |
| MRR | 0.670 | 0.673 | 0.755 |
| Every gold unit in context (questions) | 40/49 | 41/49 | 42/49 |
| **Complete evidence in context (questions)** | **20/49** | **20/49** | **21/49** |
| Gold units complete in context | 39/77 | 38/77 | 38/77 |
| Unanswerable table questions: target table in context | 10/10 | 10/10 | 9/10 |
| Retrieval latency p50 / p95 (ms) | 46 / 63 | 1,126 / 1,240 | 1,950 / 2,045 |
| Context tokens, mean | 1,500 | 1,489 | 1,316 |

### Failure causes (`failure_analysis.json`)

| Cause | A | B | C | Questions (B) |
|---|---|---|---|---|
| Extraction: no c3 chunk holds the row's answering cells (check marks) | 25 units | 26 | 26 | 21 table questions (t01–t32) |
| Terminology mismatch: neither BM25 nor dense puts it in the 20-candidate pool | 6 | 6 | 3 | q03, q05, q10, q14 |
| Reranking: in the pool, cut below the final 8 | — | 6 | 7 | q06, q09, q10, q11, n10 |
| Fusion ranking (A only) | 5 | — | — | n10, q03, q09, q10, q13 |
| Embedding or BM25 retrieval: one ranker found it, fusion cut it | 1 | 1 | 3 | q06 (B) |
| Context budget dropped it | 1 | 0 | 0 | t28 (A) |

### The write-up's three named failure patterns, now measured

| Pattern | Question | Measurement |
|---|---|---|
| Terminology | q14 | Gold AIAYN:5.1:c025 ("byte-pair … word-piece") ranks 345th in BM25 and 238th in dense for "segmented". |
| Related work outranks results | q13 | Fusion ranking cuts gold in A and C. B places it. |
| Rows without the base row | q11 | The base row AIAYN:T3:r01 is dense rank 6, but the reranker pushes it to 9th. |
| Rows without the base row | n10 | The h=8 sentence goes from dense rank 13 to reranked 13th. |

### Choosing between A, B and C on retrieval alone

There is no clear winner:
- C has the best MRR and complete coverage by one question, and costs 0.8 s more per request than B.
- A is 25× faster than B and only one complete-coverage question behind.

## 3. Retrieval candidates (each compared against the same 74 questions)

### R1: the c6 corpus, extraction e4 (`retrieval/candidate_R1_c6/`, `retrieval/compare_baseline_vs_R1_c6.json`)

Run offline with c6 selected in memory only; c6 is not activated.

| | A | B | C |
|---|---|---|---|
| Complete evidence, questions | 20 → **38**/49 | 20 → **39**/49 | 21 → **40**/49 |
| Complete-evidence questions lost | 0 | 0 | 0 |
| Gold rows newly out of context | t06, t07, t10 | t10, t29 | t29, t30 |
| MRR | 0.670 → 0.603 | 0.673 → 0.624 | 0.755 → 0.769 |
| Retrieval p50 (ms) | 26 | 733 | 1,294 |

- All the gains are table questions whose marks or cells c3 never extracted. The rows lost were incomplete on c3 anyway.
- MRR falls in A and B because c6's complete rows compete with each other.
- Remaining failures on c6 with setting B (10): q03, q05, q06, q09, q10, q11, q14, n10, t10, t29.

### Not yet tested

- **Reranker demotion** (q06, q09, q10, q11, n10, t29): each needs a diagnosis of why MiniLM demotes it.
- **Terminology** (q14, q05, q03): the glossary covers symbols, not vocabulary such as "segmented" → "byte-pair".

I won't add context or tokens to rescue these until a cause is shown and the cost and benefit are measured.

## 4. Real answer quality, setting B (production): first live batch, 2026-11-14

**Owner approval:** setting B only, 12 questions, hard cap of 50,000 tokens.

**Run:** production configuration (c3, prompt p4, verifier v4.2, `openai/gpt-oss-120b`), real Groq.
- 21 requests (including 1 model check) and **40,750 tokens**.
- No 429, no provider error, no timeout. The Groq daily limit was not hit.

**Files:**
- Questions: `live/questions_B12.json`, fixed before the run.
- Raw requests and responses, retrieval, drafts, verifier decisions and traces: `live/B12/`.
- Grades: `live/B12_grades.json`. These are the assistant's grades against the papers, not an independent review.
- Metrics: `live/B12_metrics.json`.

### Answer grades

| Grade | Count | Questions |
|---|---|---|
| Correct and fully supported | 1/10 | q01 |
| Partially correct | 0/10 | — |
| Incorrect or unsupported (**shown to the user**) | **1/10** | **t05**: says 65.9 ± 1.7, the paper's answer is 64.0 ± 1.7 |
| Incorrectly refused | 8/10 | q04, q07, q12, q13, q15, q14, t13, n10 |
| Correctly refused | 2/2 | q16, q20 |

### Rates

| Measure | Result |
|---|---|
| Unsupported or wrong answers among the replies shown | 1 of 2 (t05) |
| Unanswerable questions answered | 0/2 |
| Citations naming a retrieved chunk, in shown answers | 2/2 |
| Citations that actually support the shown answer | 1/2 (t05 cites the wrong row) |
| Complete gold evidence in context | 6/10 |
| Draft claim sentences the verifier accepted | 12/29 |
| Redraft success | 1/8 |

### Tokens and latency

- **Tokens per successful answer:** 4,340 (q01, 2 calls).
- **Tokens per question:** p50 3,688 in and 280 out.
- **Time to first token:** p50 754 ms, p95 1,871 ms.
- **Total time:** p50 27.4 s, p95 32.5 s, of which a median 22.5 s is the local governor waiting on the per-minute token bucket.
- **The one correct answer took 10.9 s.**

### Where each failure starts (trace through the pipeline)

**Verification rejected a correct draft: 5 questions.**
- q04: the passage says "We use Q = 8 …" while the draft says "Mimi uses …" (0.40).
- q07: correct description of the decoder sub-layers (0.46 and 0.74).
- q13: the comparison sentence cites only one of the two rows it compares.
- q15: "The original Transformer …" (0.01).
- n10: "8 heads [AIAYN:T3:r01]", where the cited base row does hold h = 8 (0.04).

**Generation ignored evidence in context: q12.** The row with Inner Monologue (MOSHI:T8:r01) was in context. The model claimed those accuracies were "not stated", and the verifier correctly rejected that.

**Extraction:**
- **t05** is a **critical correctness failure**. The c3 rows carry no check marks, so the model guessed a row, and the verifier accepted it because the cited row really holds 65.9. This is the same failure Phase 12 recorded.
- t13: the guessed row was wrong, and the verifier rejected it.

**Retrieval (terminology): q14.** The byte-pair / word-piece passage was not retrieved, and the model correctly declined on the evidence it had.

### What this changes

On production setting B, the binding constraint is **verification false rejection** (5 of 10). After that come extraction on c3 (2 of 10, one of them a wrong answer shown) and generation and retrieval (1 each).

Two candidates already address parts of this:
- **c6** removes the t05/t13 extraction failure in retrieval: complete evidence in B rises from 20 to 39 of 49 questions.
- **Redraft prompt p5r** (improvement-loop iteration 002) rescued q04 and q13 in an earlier live run.

Neither is activated.

### Not measured

- Live answer quality for settings A and C.
- Live answer quality on c6 beyond Phase 12's 12 questions.
- Write-up Table 3 therefore still cannot be filled for A and C.

## Files

| Path | Contents |
|---|---|
| `assessment/e2e/retrieval_eval.py` | The A/B/C retrieval evaluation: production `search()`, corpus selected in memory, refuses to overwrite a run, no NLI model |
| `assessment/e2e/retrieval_analysis.py` | Splits the "missing cells" cause into extraction versus row context |
| `assessment/e2e/retrieval_compare.py` | Per-question gains and losses between two runs |
| `assessment/e2e/retrieval/baseline_c3_production/` | `per_question.jsonl`, `per_question_refined.jsonl`, `summary.json`, `failure_analysis.json` |
| `assessment/e2e/retrieval/candidate_R1_c6/` | The same files for c6 |

**Reproduce** (offline, about 5 minutes per run, under 2 GB of memory):
```
uv run python assessment/e2e/retrieval_eval.py <new label> [--corpus c6]
uv run python assessment/e2e/retrieval_analysis.py assessment/e2e/retrieval/<label> <data/processed | experiments/phase10/corpus/c6/processed>
uv run python assessment/e2e/retrieval_compare.py <baseline dir> <candidate dir> <out.json>
```

## 5. Fix attempt 1 (2026-11-14): prompt p6, check-mark guard v4.5c, and c6

Owner-approved cap: 90,000 tokens. Used: **74,027 tokens in 35 requests**. Grades are in `live/B12_fix_grades.json`; they are the assistant's grades, not an independent review.

**Groq's daily limit (429, organization tokens-per-day)** stopped arm 2 at q14. The runner stops on it and never retries.

### Root cause and probes

Correct drafts were worded in ways the NLI verifier cannot match. When the same facts are written in the passage's own terms, they are accepted (`fix/probe_phrasing.txt`):

| Draft wording | Score |
|---|---|
| "The original Transformer …" | 0.01 |
| "The authors trained their Transformer models on one machine with 8 NVIDIA P100 GPUs" | 0.96 |
| "The base Transformer uses 8 attention heads [T3:r01]" | 0.04 |
| "In Table 3, the base configuration has h = 8" | 0.99 |
| The same phrasings with wrong values (5 probes) | all 0.00 |

### Candidates

| Candidate | Files | Change |
|---|---|---|
| p6 | `relay/candidate_p6/prompts.py` | Rules 8–10: keep the passage's subject; write table values as "In Table N, the <row> configuration has <col> = <v>"; a comparison cites every row it uses; plus the p5r redraft instruction |
| v4.5c | `relay/candidate_v4_5/verifier.py` | v4.2 plus a rejection-only rule: a check-mark claim citing table rows that carry no marks is rejected |

**Offline checks:**
- Replaying all 77 recorded live drafts (`fix/replay_v45.json`), v4.5c changes exactly 2 decisions. Both are the wrong t05 answers that reached users on c3 (Phase 12 and B12). Every other decision is unchanged, including every correct c6 answer that relies on marks.
- Full offline suite: all exit 0, 510 Python tests (`fix/offline_tests.json`).

### Live results (setting B)

| Question | B12 (c3, p4, v4.2) | Arm 2 (c3, p6, v4.5c) | Arm 1 (c6, p6, v4.5c) |
|---|---|---|---|
| q01 | correct | refused (verification) | refused (verification) |
| q04 | refused | **correct** | **correct** |
| q07 | refused | **correct** | refused (verification) |
| q12 | refused | refused (table-row wording) | refused (table-row wording) |
| q13 | refused | **correct** | partially correct |
| q15 | refused | refused (wrote "original") | refused (wrote "original") |
| q14 | refused | not run (429) | **WRONG: answered a different question** |
| t05 | **WRONG (65.9)** | not run (429) | **correct (64.0)** |
| t13 | refused | not run (429) | **correct** |
| n10 | refused | not run (429) | refused (table-row wording) |
| q16 | correct refusal | correct refusal | not run (cap) |
| q20 | correct refusal | correct refusal | not run (cap) |

**On the 6 answerable questions that ran in all three configurations:**

| Configuration | Correct | Partially correct | Wrong |
|---|---|---|---|
| B12 | 1 | 0 | 0 |
| Arm 2 (c3, p6) | 3 | 0 | 0 |
| Arm 1 (c6, p6) | 1 | 1 | 0 |

The evidence on these 6 is the same on both corpora, so the gap between the arms is sampling noise (q01 and q07 flip between runs). **This is directional only, with 6 to 10 questions per arm.**

**On all of arm 1 (c6 + p6 + v4.5c):**
- Answerable: 3 correct and 1 partially correct out of 10, 1 wrong, 5 refused.
- Unanswerable: not run.

### Remaining causes, in order of impact

1. **Verifier rejects close paraphrases of table rows** (q12, n10): "Web Questions" against the row's "Web Q.", "BLEU / (dev)". It also rejects near-verbatim prose (q01, q07 at 0.18–0.85).
2. **The model does not always follow the phrasing rules:** q15 still writes "original".
3. **New critical failure, q14 on c6:** the gold passage is not retrieved, so the model answers about the H-Net paper's baselines. Each sentence is true of its cited passage, but the reply does not answer the question. The verifier checks support, not relevance.

### Next candidates

All are testable **offline by replaying these recorded drafts**, with no API calls:
- the v4.4c sentence-level premises (iteration 003);
- column-name aliases taken from the table's own header, for table claims;
- a question-relevance check.

Nothing has been activated in production.

## 6. Fix attempt 2, in progress (2026-11-14, after the owner replaced the Groq key)

**Owner decision, recorded:** the API key was replaced after Groq's daily 429, and the owner asked to continue testing. This overrides CLAUDE.md hard rule 3 (no extra keys or accounts to get around limits). I flagged that once to the owner.

### Rejected offline, before any live spend: focused table-cell premises

The idea: give the NLI model the caption plus only the cells whose values the claim states (scratch probe p6.py).

- It rescues none of the true table claims (q12 stays at 0.01; n10 goes from 0.59 to 0.72, still below 0.9).
- It **creates a false acceptance**: "the base configuration has a development **perplexity** of 25.8" rises from 0.65 to 0.93, but 25.8 is BLEU.

So it was not built.

### Built: prompt p7 (`relay/candidate_p7/prompts.py`; tests: `tests/unit/test_candidate_p7.py`, 10/10 with p6/v4.5c)

p7 is p6 plus redraft feedback built from the evidence. For each rejected sentence, the redraft message shows:
- the exact cited row ("Moshi: Web Q. = 26.6; LlaMA Q. = 62.3; …"), or the closest sentence of the cited passage;
- the words in the sentence that the passage lacks, for example "original". The vocabulary that rule 8 asks for ("authors", "paper", …) is never listed.

The verifier is unchanged.

### Waiting on memory

| Item | Status |
|---|---|
| Verifier v4.4c comparison (iteration 003, checkpointed) | 58 cases done |
| Memory guard (1.0 GB floor, not lowered) | Stopped the run at 0.84 GB free |
| Free memory now | About 1.5 GB; the largest user is VALORANT |
| Live p7 run | Also loads the NLI model, so it waits for memory too |

## 7. Fix attempt 3 (2026-11-14): exact-support verifier v4.6c. Best configuration so far

### Why p7 alone did not help (`live/B12_p7_c6/`, 44,417 tokens)

With p7 the model **did** copy the hinted evidence, and the NLI verifier still rejected verbatim text:
- the AIAYN:1:c002 sentence, copied exactly, scored 0.07;
- "All models are trained on H100 GPUs, using FSDP and activation checkpointing", verbatim, scored 0.41;
- exact copies of table rows scored 0.15–0.69.

### Iteration 003, verifier v4.4c (sentence-level premises): rejected offline

The completed comparison is in `assessment/improvement_loop/iterations/iteration_003/`. False acceptances rose from 6 to 9 on the stress set (new ones: base step time 1.0 s, "twice as wide", the formula without √) and from 2 to 3 on live drafts.

### v4.6c (`relay/candidate_v4_6/verifier.py`; tests: `tests/unit/test_candidate_v4_6.py`)

v4.6c is v4.5c (v4.2 plus the check-mark guard) plus a **deterministic exact-support rule**. It applies **only** to a sentence whose sole failure is a low NLI score. Such a sentence is accepted if:
- **Quote:** after normalisation it is a contiguous substring of one cited passage, of at least 4 words; or
- **Table row:** every "column = value" pair equals a cell of one cited row, the row's label words are present, no sibling row's distinguishing words or parenthetical qualifiers appear, and a claimed value may add only "%".

Citation, absence, value-owner, formula and check-mark checks are never overridden.

### Offline gate (`fix/compare_v46/`: 268 cases, 160 stress plus 108 live drafts)

| | v4.2 | v4.6c |
|---|---|---|
| False acceptances, stress | 6 | **6 (no new)** |
| False acceptances, live drafts | 2 | **0** (the two wrong t05 answers are now rejected) |
| False rejections, stress | 22 | 20 |
| False rejections, live drafts | 33 | 30 |
| Exact copies still rejected | 7 of 15 | **0 of 15** |

- The 4 flips on unlabelled p7 drafts were checked by hand; all are correct answers.
- The 14 hand-written false claims in the unit tests are all rejected: wrong value, wrong row, wrong column (PPL for BLEU), sibling label, added qualifier.
- Full Python suite: 518 passed.

### Live (`live/B12_p7_v46_c6/`): c6, p7, v4.6c, setting B, the same 12 questions; 20 requests, 43,794 tokens

| Question | B12 (production: c3, p4, v4.2) | **c6, p7, v4.6c** |
|---|---|---|
| q01 | correct | refused |
| q04 | refused | **correct** |
| q07 | refused | **correct** |
| q12 | refused | refused |
| q13 | refused | **correct** |
| q15 | refused | refused |
| q14 | refused | refused (safe) |
| t05 | **WRONG (65.9)** | **correct (64.0)** |
| t13 | refused | **correct** |
| n10 | refused | **correct** |
| q16, q20 | correct refusal | correct refusal |

**Totals (assistant grades, `live/B12_p7_v46_c6_grades.json`; metrics in `live/B12_p7_v46_c6_metrics.json`):**

| | Production (B12) | c6, p7, v4.6c |
|---|---|---|
| Correct | 1/10 | **6/10** |
| Wrong | 1/10 | **0/10** |
| Incorrectly refused | 8/10 | 4/10 |
| Correctly refused | 2/2 | 2/2 |
| Unsupported answers | 0/2 | 0/2 |
| Valid citations in shown answers | — | 9/9 |
| Redraft success | 1/8 | 4/7 |

**Cost and latency per correct answer:**
- 2,072–5,230 tokens.
- 16–39 s total, mostly the local governor waiting on the per-minute token bucket.
- Time to first token: p50 833 ms.

### Still refused, all safely

| Question | Cause |
|---|---|
| q15 | The model keeps writing "original Transformer" despite the hint |
| q12 | The model adds "(with Inner Monologue)" to a row labelled "Moshi" |
| q01 | A paraphrase mixing two sentences |
| q14 | Retrieval: the byte-pair / word-piece passage is not found |

### Limits

- 12 questions, one sample each, assistant grades. The result is directional, not significant.
- Readability: answers often read as copied rows or sentences ("In Table 2, the … has Average / acc ↑ = 55.5").
- Nothing is activated. Production remains c3, p4, v4.2.

## 8. Larger live evaluation: 25 frozen questions (2026-11-14)

**Run:** c6 + p7 + v4.6c, setting B, real Groq.
- Questions: the improvement loop's frozen dataset (`assessment/improvement_loop/datasets/datasets.json`, sha256 verified before the run). It has the 15 original answerable questions, the table failures t01, t05, t10, t13 and t29, and 5 unanswerable questions.
- **39 requests, 88,630 tokens** against an owner-approved cap of 95,000.
- No provider errors, no 429.
- **Production default not switched (owner's instruction).**

**Files:**
- Records: `live/E25_p7_v46_c6/`
- Grades: `live/E25_p7_v46_c6_grades.json` (the assistant's grades, not independent)
- Metrics: `live/E25_p7_v46_c6_metrics.json`
- The baseline is the production live run on the same 25 questions: `assessment/improvement_loop/baseline/baseline_metrics.json`.

| Same 25 questions | Production (c3, p4, v4.2) | Candidate (c6, p7, v4.6c) |
|---|---|---|
| Correct and fully supported | 2/20 | **10/20** |
| Partially correct | 1/20 | 2/20 |
| Wrong or unsupported answer shown | 1/20 (t05) | 1/20 (**q14**) |
| Incorrectly refused | 16/20 | 7/20 |
| Correctly refused | 5/5 | 5/5 |
| Unanswerable questions answered | 0/5 | 0/5 |
| Valid citations in shown answers | 5/5 | 21/21 |
| Complete gold evidence in context | 8/20 | 11/20 |
| Redraft success | 2/17 | 8/13 |
| Time to first token, p50 / p95 | 799 / 1,371 ms | 744 / 1,277 ms |
| Total time, p50 / p95 | 28.6 / 34.2 s | 32.3 / 41.4 s |
| Input tokens per question, p50 | 3,639 | 4,004 |

Per correct answer, the median is 4,743 tokens and 32.4 s.

### The one wrong answer, q14 (critical)

- Retrieval does not find AIAYN:5.1:c025 (byte-pair ~37,000 / word-piece 32,000).
- The model therefore answers with the H-Net paper's Transformer baseline: "the Transformer uses the GPT-2 tokenizer", which is HNET:3.1:c053.
- Each sentence is supported by its citation, but the reply does not answer the question.
- Production refused q14 instead, so this is a new wrong answer, while the old one (t05) is gone.

### Remaining refusals (7)

| Cause | Questions |
|---|---|
| Retrieval: gold not in context | q10, t10, t29, q11 (base row), q05 (one of two units) |
| Wording | q08 (close paraphrase, not an exact quote); q12 ("Web Questions" for "Web Q.", plus "(with Inner Monologue)") |

### Before any production switch

1. **Fix or guard q14.**
   - Retrieval: the terminology gap between "segmented" and "byte-pair / word-piece".
   - Or a guard against answering about a different paper than the one the question names. The question says "original Transformer's translation experiments", while the answer cites only HNET.
2. **Readability.** Answers repeat the papers' voice ("we employ", "The authors trained their models") and raw row formats.
3. **Sample size.** 20 answerable questions, one sample each, assistant grades. The result is directional.

## 9. The q14 guard: verifier v4.7c (2026-11-14)

**Code:** `relay/candidate_v4_7/verifier.py`; tests in `tests/unit/test_candidate_v4_7.py`. Full suite: 522 passed.

**Rule.** v4.7c is v4.6c plus a rejection-only, question-aware guard. Every paper the question names (via `named_papers`) must be covered by the answer, in one of two ways:
- an accepted sentence citing that paper; or
- an accepted "The retrieved passages do not state …" sentence that names it.

Otherwise the first sentence is rejected, with a reason the redraft can act on. Questions that name no paper are unaffected.

**How it is wired.** The production verifier is given no question, so the candidate is built per question with `verify_for(question)`. The experiment harness binds it; production is unchanged.

**Offline check** (`fix/replay_v47_guard.json`). Over all 45 answers shown in every recorded live run (production and candidates, c3 and c6), the guard blocks **exactly 2**: the two misleading q14 answers (Llama-3 tokenizer, GPT-2 tokenizer). Nothing else changes, including the correct cross-paper q15 answers.

**Live check** (`live/G4_p7_v47_c6/`): c6 + p7 + v4.7c on 4 questions; 8 requests, 16,846 tokens, cap 20,000. Grades: `live/G4_p7_v47_c6_grades.json`.

| Question | Result |
|---|---|
| q14 | The redraft repeated the misleading answer, and the **guard blocked it**: a safe refusal |
| q15 | Correct; the guard passes because both papers are cited |
| q20 | Correctly refused |
| q01 | Refused by NLI on a paraphrase. It was answered correctly in E25 with the same setup (sampling variance); the guard is not involved |

### Estimate for the full candidate stack (c6 + p7 + v4.7c) on the 25 questions

This is **inferred, not a single live run.** It combines E25 (v4.6c) with the guard, because the replay shows the guard changes only q14 among the shown answers. The q14 effect itself is confirmed live.

| | Count |
|---|---|
| Correct | 10/20 |
| Partially correct | 2/20 |
| **Wrong** | **0/20** |
| Refused | 8/20 |
| Correctly refused | 5/5 |

Run-to-run variance is about ±1–2 answers per run (q01 and q07 have each flipped between runs).

### Still open

- **Retrieval:** q14 (the byte-pair / word-piece passage), q10, q11 (base row), t10, t29, q05.
- **Wording:** q08, q12; q01 is borderline.
- **Readability:** answers echo the papers' voice and raw table rows.

Production is still c3 + p4 + v4.2. Nothing has been activated.

## 10. q14 retrieval fix: candidate R2 (2026-11-14)

### Diagnosis (`retrieval_fix/diagnose_q14_{c3,c6}.json`; scratch variants in `retrieval_fix/diagnose_q14.py`)

The question says "segmented" and "translation experiments". The passage AIAYN:5.1:c025 says "encoded using byte-pair encoding" and "split tokens into a 32000 word-piece vocabulary" on "WMT 2014". The two share no words.

| Query | BM25 rank (AIAYN) | Dense rank (AIAYN) | Fused rank (AIAYN slot of 10) | Reranked (slot) | In context? |
|---|---|---|---|---|---|
| Original question | 51 | 36 | not in slot | — | no |
| Plus its own terms | 1 | 5–21 | 1–4 | **5–9**, demoted by the cross-encoder | no |

A generic word ("tokenization") does not help. The terms that win BM25 also match the Table 3 caption, which every T3 row repeats, and those rows plus AIAYN's conclusion outrank the passage at the reranker. **Neither change alone works.**

### R2 (`relay/candidate_r2/search.py`; tests: `tests/unit/test_candidate_r2.py`; full suite 526 passed)

1. **Terminology expansion.** It is triggered by tokenization wording (segment…, tokeniz…, subword) when the question names AIAYN or no paper. It adds AIAYN's verbatim phrases "byte-pair encoding" and "word-piece vocabulary"; a test checks they are verbatim in c3 and c6. It affects the retrieval query only.
2. **Cross-paper slot protection.** For questions naming 2 or more papers, each named paper's best fused candidate is placed first, and the context is refilled under the same budget.

### Retrieval regression (74 questions × A/B/C, against the c6 baseline; `retrieval/compare_R1_c6_vs_R2_c6.json`, `retrieval_fix/isolation_check.txt`)

- **No question lost anything** in any setting (complete evidence, any gold, units in context, MRR).
- The expansion fired **only on q14**.
- The final context changed for **q14 only** (all settings) and for q15 in B, where only the order changed and both gold passages stayed.
- For q14, AIAYN:5.1:c025 is now **first in the context** (it was reranked 16th; the protection lifted it). Gold units in context rose from 60 to 62.
- q14's third gold unit (HNET:3:c044, the GPT-2 baseline) is still not retrieved.

### Offline verification (`retrieval_fix/verify_q14_scripted.txt`; hand-written drafts, v4.7c with the q14 question)

| Draft | Result |
|---|---|
| Complete correct answer (AIAYN 5.1 plus H-Net "byte-level" from HNET:3.2:c061) | **Accepted** (0.97, 0.99, and exact support) |
| AIAYN-only answer | Rejected by the guard (no H-Net part) |
| The misleading GPT-2 answer | Rejected |

### Live (`live/R2_p7_v47_c6/`; grades in `live/R2_p7_v47_c6_grades.json`): 12 requests, 23,538 tokens, cap 30,000

| Question | Result |
|---|---|
| q14 | **Partially correct.** The AIAYN half is correct from the right passage. The H-Net half describes SpaceByte, a baseline, instead of H-Net |
| q15 | Correct |
| q07 | Correct |
| t05 | Correct |
| q16 | Correctly refused |
| q04 | Refused. The context is identical to E25 (where it was correct); the redraft added a false absence sentence (sampling) |

### Open

- The expanded query is also used for the **H-Net slot**. That drops the H-Net abstract (learned dynamic chunking) and protects SpaceByte (HNET:3:c041).
- Next refinement: **apply the expansion only to the slot of the paper whose phrases it adds.** This needs per-paper queries inside search, and is not built yet.

## 11. Retrieval R3b: the expansion scoped to its paper's slot (2026-11-14)

**Code:** `relay/candidate_r3/search.py`; tests in `tests/unit/test_candidate_r3.py`. Full suite: 530 passed.

**Change.** For questions naming 2 or more papers, each paper's slot is retrieved and reranked with its own query. Only AIAYN's slot gets AIAYN's phrases, and only an expanded slot has its best fused candidate protected. Questions naming fewer papers go through R2 unchanged.

**First attempt (R3, every slot protected), offline.** The H-Net slot still led with SpaceByte (HNET:3:c041). That passage is the H-Net slot's best fused candidate even for the original question, so it was the protection, not the expansion, that lifted it. Fix: protect expanded slots only (R3b).

### Retrieval regression for R3b (74 × A/B/C against the c6 baseline; `retrieval/compare_R1_c6_vs_R3b_c6.json`, `retrieval_fix/isolation_check_r3b.txt`)

- **No question loses anything.**
- The context changes **only for q14**. q15 is identical to production again.
- q14's setting-B context, in order:
  1. AIAYN:5.1:c025 (byte-pair / word-piece);
  2. Table 3 rows;
  3. HNET:3.2:c061 ("H-Net (space) and H-Net (2-stage) are byte-level");
  4. HNET:3.1:c053 ("All models but Transformer take raw byte inputs");
  5. HNET:2.1.2:c019.
  
  There is no SpaceByte.
- Setting C still misses AIAYN 5.1 for q14, because 4 passages of 512 tokens leave too little room.

**Offline verification** (`retrieval_fix/verify_q14_scripted_r3b.txt`): the complete correct answer is accepted; the AIAYN-only and misleading answers are rejected.

### Live (`live/R3b_p7_v47_c6/`; grades in `live/R3b_p7_v47_c6_grades.json`)

**Groq's daily limit (429)** stopped the run at q15 after 6,779 tokens. The 429 names a different organization from the earlier limit, so the replacement key belongs to a separate account; that is the owner's decision, recorded in section 6. q15, q16, q20, q04 and q13 were not run.

**q14 is partially correct, with both halves on target:**

> "Sentences were encoded using byte-pair encoding, which has a shared source-target vocabulary of about 37000 tokens [AIAYN:5.1:c025]. H-Net (space) and H-Net (2-stage) are byte-level, while the Transformers use the Llama-3 tokenizer [HNET:3.2:c061]."

- Missing: the 32,000 word-piece vocabulary and H-Net's learned chunking.
- "the Transformers" in the second sentence are H-Net's own baselines, which could mislead.

**History of q14:**

| Configuration | q14 result |
|---|---|
| Production | refused |
| c6 + p7 + v4.6c | misleading |
| v4.7c (with the guard) | safely refused |
| R2 | H-Net half off-target |
| R3b | both halves correct, but incomplete |

### Regression status

- Offline, the final context of **every other question is identical** to the candidate stack evaluated in E25, so their answers are unaffected apart from sampling.
- The live regression batch is **pending** Groq's daily reset: q15, q16, q20, q04, q13.

### Current best stack (not activated)

**c6 + R3b retrieval + p7 prompt + v4.7c verifier.**

## 12. R3b regression, resumed (2026-11-14, after the owner installed another Groq key; owner's decision, as in section 6)

**Run:** `live/R3b_p7_v47_c6_resume/`; 9 requests, **18,566 tokens**, cap 30,000. Grades: `live/R3b_p7_v47_c6_resume_grades.json`. Production is unchanged (verified before the run: c3, p4, v4.2).

| Question | E25 (c6 + p7 + v4.6c) | R3b stack (c6 + R3b + p7 + v4.7c) | Same context as E25? |
|---|---|---|---|
| q15 | correct | **correct** | yes |
| q16 | correctly refused | correctly refused | yes |
| q20 | correctly refused | correctly refused | yes |
| q13 | correct | partially correct (omits the XL match; confusing absence sentence) | yes |
| q04 | correct | refused (a false "frame rate not stated", repeated in the redraft) | yes |
| q14 | wrong | **partially correct** (section 11) | changed (the fix) |

The context check is in `live/R3b_resume_context_check.txt`.

### Reading the results

- The contexts of the regression questions are identical to E25, so **R3b did not cause** the q13 and q04 changes. They come from generation variance.
- **q04's false absence sentence is recurring:** 2 of the last 2 runs.
  - The p7 redraft hints cover cited sentences only. An absence sentence has no citation, so it gets no hint.
  - The model ignores "never repeat a listed sentence unchanged".

### Candidate next fix (not built)

When the **only** rejected sentences of a final draft are absence sentences that a retrieved passage contradicts, drop them and keep the verified rest. This is a pipeline change:
- Every remaining sentence is still verified.
- What is removed is a false claim that something is missing.
- It needs its own offline gate and live check.

## 13. Clean 25-question measurement of the R3b stack (2026-11-14)

**Run:** c6 + R3b + p7 + v4.7c, setting B, the 25 frozen questions (sha256 verified). 41 requests, **93,002 tokens**, cap 95,000; no provider errors.
- Records: `live/E25_R3b_p7_v47_c6/`
- Grades: `live/E25_R3b_p7_v47_c6_grades.json` (assistant grades, not independent)
- Metrics: `live/E25_R3b_p7_v47_c6_metrics.json`

Production is unchanged.

| Same 25 questions | Production (c3, p4, v4.2) | E25 (c6, p7, v4.6c) | **E25 (c6, R3b, p7, v4.7c)** |
|---|---|---|---|
| Correct and fully supported | 2/20 | 10/20 | **8/20** |
| Partially correct | 1/20 | 2/20 | 4/20 |
| **Wrong or unsupported shown** | 1/20 (t05) | 1/20 (q14) | **2/20 (t29, q13)** |
| Incorrectly refused | 16/20 | 7/20 | 6/20 |
| Correctly refused | 5/5 | 5/5 | 5/5 |
| Citations naming retrieved chunks | 5/5 | 21/21 | 22/22 |
| Redraft success | 2/17 | 8/13 | 10/15 |
| Total time p50 / p95 (s) | 28.6 / 34.2 | 32.3 / 41.4 | 34.5 / 40.8 |

### What changed between the two candidate runs

- **q14:** wrong → **partially correct.** The R3b fix works.
- **q10:** refused → partially correct.
- **q02:** refused → correct.
- **t29:** refused → **WRONG**.
- **q13:** correct → **WRONG**.
- q01: correct → refused.
- q04 and q09 have the same grade in both runs.

All of these except q14 have identical retrieval contexts in the two runs, so they reflect generation variance plus two new failure modes, not R3b.

### Two new failure modes (critical): true sentences that are not the answer

1. **t29: the redraft drops the question's condition.**
   - The first draft said "the configuration with no marks … 29.1%". That was rejected, and it is false: the cited row (Row 5) has a ✓ under Quantization.
   - The p7 redraft restated only the row: "Configuration Row 5 reports ABX (↓) = 29.1%", a true sentence (0.99).
   - The check-mark guard does not fire, because the sentence no longer mentions marks.
   - The gold row MOSHI:T17:r01 is not retrieved on c6 (known).
2. **q13: a false absence statement is accepted.**
   - The redraft dropped the correct 53.3 sentence and wrote "The retrieved passages do not state … the BPE Transformer", although that row (HNET:T2:r01, labelled "Transformer") was in context.
   - The absence checks do not link "BPE Transformer" to the row label "Transformer".

### Conclusion

- The candidate stack answers **4–5× more questions correctly** than production (8–10 against 2 out of 20).
- It still shows **1–2 wrong or unsupported answers per 20**; production showed 1.
- The wrong answers are now true-but-irrelevant sentences and false absence claims.
- **Not ready for production.**

### Next fixes, each needing its own offline gate and live check

1. **Question-aware mark condition.** When the question selects a row by marks ("no marks in any column", "check mark in all five", "check mark under X"), the cited row's mark cells must match that condition. This would reject t29's Row 5.
2. **Absence claims about a configuration that is in context.** Reject "not stated" when a retrieved row label is contained in the claimed subject (for example "Transformer" within "BPE Transformer") and that row holds the named metric. This is q13.
3. **q04 (section 12):** drop contradicted absence sentences. **On hold at the owner's request.**

## 14. Verifier fixes for t29 and q13: candidate v4.8 (2026-11-14)

Production is unchanged. No production file has been modified since the earlier phases (checked by modification time).

### Code

- `relay/candidate_v4_8/marks.py`: fix 1.
- `relay/candidate_v4_8/absence_labels.py`: fix 2.
- `relay/candidate_v4_8/verifier.py`: v4.8c, both fixes.
- `fix1_only.py` (v4.8a) and `fix2_only.py` (v4.8b) isolate each fix.
- Tests: `tests/unit/test_candidate_v4_8.py`, 11 tests. Full suite: **541 passed**.

Both fixes are **rejection-only** and are built on v4.7c.

**Fix 1: check-mark condition.**
- It parses the question's row-selection condition: all checked; no marks; checked or unchecked under a named column; "only under X"; and combinations.
- Comparisons (t15, t23) are not enforced.
- It identifies the row a sentence states from its **whole cell values**. A shared fragment such as "±1.7" is not enough; that was a bug found and fixed during testing.
- It rejects the sentence if that row's **mark cells** do not satisfy the condition, or if the cited table has no mark cells at all.
- It never infers marks from labels or prose.

**Fix 2: table-label-aware absence.** A "not stated" sentence is rejected when a retrieved row:
- has its label core (outside the parentheses) present as whole words in the sentence, ignoring generic labels such as "Row 5";
- is not contradicted by another row's distinguishing words or a negation qualifier in its own label;
- reports a metric that the sentence names, with a non-empty value.

### Offline validation (`fix_v48/`)

**Reproduction** (`reproduce.txt`), on the recorded live drafts with their recorded c6 evidence:

| Draft | Baseline v4.7c | Fix 1 only | Fix 2 only | Both |
|---|---|---|---|---|
| t29 D1 "Configuration Row 5 reports ABX = 29.1%" | **accepted** | **rejected** | accepted | rejected |
| q13 D1 "… do not state … the BPE Transformer" | **accepted** | accepted | **rejected** | rejected |

**Unit cases on real c6 rows:**
- Every condition type is tested with the right row and a wrong row.
- A c3 row without marks is rejected.
- Comparisons are not enforced, and a sentence stating no row value is not judged.
- Legitimate absence claims pass:
  - "frame rate";
  - the base BLEU when the base row is not retrieved;
  - "with Inner Monologue" when only the "no Inner Monologue" row is retrieved;
  - "Row 3";
  - a subject with no metric named.
- When the base row *is* retrieved, the base-BLEU absence claim is rejected.

**Replay** (`replay.json`): 401 cases (241 recorded live drafts from every run, c3 and c6, each with its own question; plus 160 stress cases).

| | Count |
|---|---|
| Decisions changed | **exactly 2**: the t29 and q13 target drafts |
| Rejected → accepted (a weakened check) | **0** |
| Stress cases changed | 0 |
| Correct check-mark answers still accepted (t05 64.0, t13 36.8, t01 ✗) | all |

### False positives, false negatives and ambiguity

- **False positives** (a correct answer rejected): none in the replay or the unit cases.
- **Known false negative (fix 2):** if both "Moshi" and "Moshi (no Inner Monologue)" are retrieved, a false "not stated … Moshi with Inner Monologue" is **not** caught, because the conservative sibling rule skips it. v4.2's own absence checks caught that case in q12.
- **Not judged (fix 1):** a condition naming a column that cannot be matched to the table, and sentences that state no value of the cited rows.
- **Ambiguous evidence:**
  - comparison questions;
  - "which rows have a check mark" (t10), which is enforced row by row;
  - questions that ask *about* a mark (t01, t19, t28) rather than select a row by it, which are not parsed.

### Live check (`live/V48_R3b_p7_c6/`; grades in `live/V48_R3b_p7_c6_grades.json`)

c6 + R3b + p7 + v4.8c; 11 requests, **22,906 tokens**, cap 35,000.

| Question | Result |
|---|---|
| q13 | **Correct** (all three values) |
| q04 | **Correct** |
| t05 | Correct |
| t13 | Correct |
| t01 | Correct |
| q16 | Correctly refused |
| t29 | Refused safely (the model declined; the gold row is still not retrieved on c6) |

- **Neither fix had to reject anything live.** The model did not repeat either failure in this sample.
- The live run shows that the fixes **do not block correct answers**, including fix 1 passing the correct rows in t05 and t13.
- That they **block the failures** is shown offline only.

## 15. Full 25-question measurement of the v4.8c candidate (2026-11-14)

**Run:** c6 + R3b + p7 + v4.8c, setting B, the 25 frozen questions (sha256 verified). It completed with no 429 and no provider errors: 41 requests, **93,658 tokens**, cap 95,000.
- Records: `live/E25_V48/`
- Grades: `live/E25_V48_grades.json` (assistant grades, not independent)
- Metrics: `live/E25_V48_metrics.json`

Production is unchanged (c3, p4, v4.2).

| Same 25 questions | Production | c6 + p7 + v4.6c | c6 + R3b + p7 + v4.7c | **c6 + R3b + p7 + v4.8c** |
|---|---|---|---|---|
| Correct and fully supported | 2/20 | 10/20 | 8/20 | **9/20** |
| Partially correct | 1/20 | 2/20 | 4/20 | 5/20 |
| **Wrong or unsupported shown** | 1/20 | 1/20 | 2/20 | **0/20** |
| Incorrectly refused | 16/20 | 7/20 | 6/20 | 6/20 |
| Correctly refused | 5/5 | 5/5 | 5/5 | 5/5 |
| Unanswerable questions answered | 0/5 | 0/5 | 0/5 | 0/5 |
| Citations naming retrieved chunks | 5/5 | 21/21 | 22/22 | 25/25 |
| Redraft success | 2/17 | 8/13 | 10/15 | 11/15 |
| Total time, p50 (s) | 28.6 | 32.3 | 34.5 | 33.4 |
| Time to first token, p50 (ms) | 799 | 744 | 670 | 781 |
| Input tokens per question, p50 | 3,639 | 4,004 | 4,313 | 4,216 |

### Highlights

- **The first run with no wrong or unsupported answer shown:** 14 answers shown, every one correct or partially correct.
- **q05 (the routing-module formula) was answered correctly for the first time.**
- **q11 was answered for the first time.** It is partially correct: 24.9 for one head, plus an honest "base BLEU not stated" (the base row is not retrieved, and fix 2 correctly lets a legitimate absence stand).
- The fix-1 and fix-2 rejection paths did not fire in this run. The model did not repeat the t29 or q13 failures (t29 declined, and q13 was correct with all three values).

### Remaining

| Kind | Questions |
|---|---|
| Refusals | q01, q07, q08, q12 (wording / verifier); t10, t29 (retrieval on c6, safe declines) |
| Partial answers | q06, q09, q10, q11, q14 (missing evidence or omitted detail) |
| Readability | Copied rows and the papers' own voice |

- **q14 is still partially correct.** It gives the byte-pair encoding with the ~37000-token shared vocabulary (AIAYN:5.1:c025) and H-Net's byte-level input, but omits the 32000 word-piece vocabulary. It also lists the tokenizers of H-Net's Transformer baselines (GPT-2, Llama-3), which is true but may confuse.
- **The v4.8c safeguards** passed offline (section 14), but neither had to intervene in the small live test (section 14) or in this run.
- **Known strict-matching limitation (fix 2):** when both "Moshi" and "Moshi (no Inner Monologue)" rows are retrieved, a false "not stated … Moshi with Inner Monologue" claim is not caught (section 14).

### Caution

- One sample per question, assistant grades, 20 answerable questions. Run-to-run variation is about ±1–2 answers.
- 0 wrong answers in one run does not prove the rate is 0.
