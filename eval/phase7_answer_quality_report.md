# Phase 7: answer quality with corpus c3 vs c4, and symbol interpretation

**No new model output exists in this phase** (no API calls, by instruction; the earlier security concern is treated
as unresolved). Everything below uses saved drafts, constructed verifier probes, or is marked **pending**. Nothing was
generated or substituted. Frozen verifier v4 (threshold 0.9) is used throughout; candidate v5 is not used.

Artifacts: `eval/phase7/answer_quality_comparison.csv`, `eval/phase7/table_question_grades.csv`,
`eval/phase7/raw/answer_replay.jsonl`, `eval/phase7/raw/symbol_trace.json`. Code: `experiments/phase7/answer_replay.py`,
`experiments/phase7/symbol_trace.py`.

## 1. Original 15 questions: historical c3 run vs counterfactual c4 replay

- **c3 (measured live, 2026-11-13; setting B; v4):** 0 correct answers, 1 incorrect answer delivered (q12), 5 correct
  refusals, 9 false refusals. Grades are **assistant-graded and not confirmed by a human**. 24 calls, 48,470 tokens.
- **c4 (counterfactual replay, not a measurement of answer quality):** the same 19 saved drafts verified by frozen v4
  against each question's c4 setting-B context. **0 of 19 verdicts change; all 15 outcomes are identical.** The
  setting-B context differs only for q04 and q10.
- **The real c4 answer quality is pending.** The drafts were written from c3 contexts, so the replay only shows that
  c4 does not change the verdict on the same drafts. Whether the model answers better with c4's evidence needs new
  generation (two questions' contexts changed under B; under C, q10 gained gold evidence).

| Outcome class (original 15) | c3 live | c4 replay | c4 with new generation |
|---|---|---|---|
| 1. Correct answer with sufficient evidence | 0 | 0 | pending |
| 2. Correct refusal | 5 | 5 | pending |
| 3. False refusal | 9 | 9 | pending |
| 4. Unsupported answer | 0 | 0 | pending |
| 5. Incorrect answer delivered | 1 (q12) | 1 (q12) | pending |

Failure attribution for c3 (a question can have several; from the assistant grades):
- 6, retrieval failure: q06, q10, q14.
- 7, verification failure: q02, q04, q07, q08, q09, q10, q13 (citation brackets), and q12 (a false acceptance).
- 8, generation failure: q13 (a misread table row), q14 (the model declined).

## 2. Repaired-table set (32 questions): pending

No model answers exist for these questions, so `table_question_grades.csv` lists every answer and grade as
**pending**. What is known is retrieval completeness under B: complete evidence reaches the context for 3/24
answerable questions with c3 and 19/24 with c4 (see the retrieval report).

## 3. Symbol interpretation: traced end to end

Trace: extracted table → corpus → chunk → retrieved context (setting B, using the question that targets the cell) →
model context → generated answer → verifier. The verifier step uses **constructed probe claims** (test inputs written
for this check, not model output), each cited to the chunk holding the row.

| Symbol | Cell in c3 / c4 | Reaches the model context | Generated answer | Verifier v4 on c4 (probe: entailment, decision) |
|---|---|---|---|---|
| ✓ (H-Net, Multi-stage / Hierarchy) | "51" / "✓" | yes / yes | pending | correct "marked with a check": 0.989 **accept**; wrong "cross": 0.831 reject |
| ✗ (MANTa, Multi-stage / Hierarchy) | "55" / "✗" | yes / yes | pending | correct "cross": 0.787 **reject**; wrong "check": **0.988 accept** ✘ |
| (blank) from empty `\xmark` (RVQGAN, causal) | column absent / "(blank)" | yes / yes | pending | correct "cell is blank": 0.997 accept; "not causal" (a blank read as no): 0.133 reject; wrong "RVQGAN is causal": **0.958 accept** ✘ |
| Empty LaTeX cell (MOSHI T6, Depthwise, length 538) | absent / "" (not written in chunk text) | the row yes, the cell no | pending | correct "no mark": 0.45 reject; reading the blank as "no": 0.544 reject |
| `[symbol]` | 0 / 0 occurrences | - | - | not testable on the corpus (unit-tested only) |

**Findings:**
- **Retrieval and context are correct in c4:** ✓, ✗ and (blank) arrive in the model's context exactly as printed
  (c3 delivered "51", "55" or nothing).
- **The verifier misreads two of the three marks.** On c4, frozen v4 accepts a claim that *contradicts* a ✗ cell and a
  claim that turns a blank into "causal", while rejecting the correct reading of ✗. The symbols are not unknown tokens
  to the NLI model (✓ → token 52211, ✗ → tokens 507 and 125373), so the cause is the model's reading, not the
  tokenizer: it appears to treat "the row has a value in this column" as support for any mark. This is a **new
  false-acceptance risk that appears only with c4**: in c3 those cells were missing or numeric.
- **Interpretations the table does not define:** "MANTa does not support multi-stage hierarchies" (reading ✗) was
  rejected; "H-Net supports multi-stage hierarchies" was accepted on both corpora because the caption states it.
- **An empty LaTeX cell is not written into chunk text,** so no claim about it can be verified; questions about such
  cells (t18) can only be answered from the other values in the row.
- **How the model interprets the marks is pending:** it needs generation.

## 4. Tokens
c3 live: 48,470 tokens for the original 15 questions (3,231 per question). c4: none (no generation).

## 5. Answers to the decision questions

1. **Does c4 improve retrieval on the original set?** Partly. Setting C improves (any gold 86.7% → 93.3%, MRR 0.583 →
   0.644, via q10), B is unchanged, and A loses one quote (q11, indirect). All corrupted evidence leaves the contexts.
2. **Does c4 retrieve the eight repaired tables correctly?** The evidence is much more complete (questions complete
   under B: 3 → 19 of 24), but table and row retrieval got slightly worse: t10 lost its table under A and B, MOSHI T3
   and T17 are confused, and tables split into more chunks crowd the context. MOSHI Table 5 values remain wrong in
   both corpora (`\phantom`).
3. **Does c4 improve answer correctness?** Not established. Saved drafts give identical verdicts; new generation is
   pending.
4. **Does it reduce false refusals?** Not established (same reason).
5. **Does it introduce new unsupported answers?** None observed, because no new answers exist. But the verifier probes
   show a concrete new risk: v4 accepts wrong readings of ✗ and blank cells that only c4 contains.
6. **Does symbol interpretation work?** In retrieval and context, yes. In the verifier, no: ✗ and (blank) are misread.
   In the model: pending.
7. **Best-supported retrieval setting:** C (best on both datasets for complete evidence and on the original set).
   Small samples; not significant.
8. **Is c4 ready for production evaluation?** **No, it needs more work first:**
   - make the verifier read marks (for example, render ✓/✗/blank as words in table premises), which needs approval
     since it changes the verifier;
   - fix the `\phantom` extraction defect (MOSHI Table 5);
   - reduce table crowding and the T3/T17 confusion;
   - then run a generation-based c3-vs-c4 evaluation on both datasets, after security clearance and API approval.

## Human-review status
- Original 15 grades: assistant-graded, unconfirmed.
- Table question set: assistant-authored, review pending.
- Table answers and grades: none exist (pending).
- No grade in this report is human-confirmed.
