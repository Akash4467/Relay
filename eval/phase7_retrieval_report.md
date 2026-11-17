# Phase 7: retrieval with corpus c3 vs c4

Offline; no generation and no API calls. Conditions frozen in `eval/phase7/conditions.json` (git state, c3 file
hashes, c4 manifest and file hashes, model, embedding, settings, verifier v4 at 0.9, dataset hashes). Query-level
data: `eval/phase7/retrieval_comparison.csv`; raw rows with full contexts: `eval/phase7/raw/retrieval_rows.jsonl`;
aggregates: `eval/phase7/raw/retrieval_summary.json`. Code: `experiments/phase7/retrieval_eval.py`.

## Datasets
1. **Original:** the 30 questions of `eval/questions.jsonl` (15 answerable, 15 unanswerable); gold = evidence quotes.
2. **Repaired-table set:** `eval/phase7/table_question_set.json`, 32 questions on the eight repaired tables (24
   answerable, 8 unanswerable). Written by the assistant from the LaTeX source, validated mechanically
   (`experiments/phase7/validate_question_set.py`: unique gold rows, satisfiable evidence, every expected number
   in the LaTeX source); **independent human review pending**. See `eval/phase7/annotation_guide.md`.
The two datasets are reported separately and never aggregated together.

## Settings and metrics
Settings A (256-token pieces, 8 passages, no re-rank), B (256, 8, re-rank; production default), C (512, 4, re-rank);
same embedder (BAAI/bge-small-en-v1.5), cross-encoder, RRF and context cap for both corpora.
- Original: any gold / quote recall in the final context; MRR and Recall@5 over the 20 ranked candidates.
- Table set: *table in context* (a chunk of the gold table), table MRR / Recall@5 (first candidate from the table);
  *row in context* (a row line holding all of the gold row's identifying strings); *row complete* (that line also holds
  the needed marks and values); *question complete* (all gold rows complete); row MRR.
- Corrupted evidence: context lines with a 51/55 mark artifact (HNET/MOSHI Table 6) or a `\phantom`-inflated value
  (MOSHI Table 5).
- Not available: precision@k and nDCG with graded relevance (no graded labels exist).

## Original dataset (15 answerable questions)

| Setting | Corpus | Any gold | Quote recall | MRR | R@5 | Corrupted lines in context (all 30 q.) | Missing evidence |
|---|---|---|---|---|---|---|---|
| A | c3 | 86.7% | 67.6% | 0.569 | 0.578 | 11 | q03, q14 |
| A | c4 | 86.7% | 64.7% | 0.602 | 0.578 | 0 | q03, q14 |
| B | c3 | 73.3% | 58.8% | 0.582 | 0.572 | 11 | q06, q10, q11, q14 |
| B | c4 | 73.3% | 58.8% | 0.582 | 0.572 | 0 | q06, q10, q11, q14 |
| C | c3 | 86.7% | 64.7% | 0.583 | 0.756 | 11 | q10, q14 |
| C | c4 | **93.3%** | 67.6% | **0.644** | 0.778 | 0 | q14 |

The c3 numbers reproduce the Phase 6 and earlier reports.

**Investigated questions:**
- **q10:** improved under C (gold rank 12 → 1; 0 → 1 of 3 quotes) and A (rank 2 → 1). Under B it is still missing
  (rank 9 → 10), and c4's context holds **five HNET Table 6 chunks out of six**. Table 6 now spans five chunks instead of
  three (longer row lines), and they crowd out the prose evidence. In c3 the same context held corrupted Table 6 rows
  (51/55).
- **q11:** regressed under A (2 → 1 of 2 quotes; AIAYN:T3:r01 replaced by AIAYN:3.2.2:c014). All AIAYN chunks are
  textually identical in c3 and c4, so this is an indirect effect of corpus-wide BM25 statistics (455 → 465 chunks);
  setting A has no re-ranker. B and C unchanged.
- **q06:** unchanged in all settings (B: gold at rank 11, beyond the 8 passages; A and C retrieve it).
- **q14:** unchanged: no gold among the candidates in any setting or corpus (the vocabulary mismatch noted before).

## Repaired-table dataset (24 answerable questions, 31 gold rows)

| Setting | Corpus | Table in context | Table MRR | Rows in context | Rows complete | Questions complete | Row MRR | Corrupted lines | Unanswerable: table in context |
|---|---|---|---|---|---|---|---|---|---|
| A | c3 | 24/24 | 0.843 | 30/31 | 5/31 | 3/24 | 0.732 | 73 | 8/8 |
| A | c4 | 21/24 | 0.697 | 26/31 | 22/31 | 18/24 | 0.564 | 12 | 6/8 |
| B | c3 | 24/24 | 0.979 | 30/31 | 5/31 | 3/24 | 0.743 | 76 | 8/8 |
| B | c4 | 23/24 | 0.938 | 27/31 | 23/31 | **19/24** | 0.651 | 12 | 8/8 |
| C | c3 | 24/24 | 0.938 | 31/31 | 5/31 | 3/24 | 0.854 | 76 | 7/8 |
| C | c4 | 24/24 | 0.931 | 29/31 | 25/31 | **19/24** | 0.835 | 12 | 6/8 |

- **Improvement:** with c3 the right table and row are almost always retrieved, but the evidence is incomplete: marks
  are missing or corrupted. Only the 3 questions that need no mark (t09, t18, t26) are complete. With c4, 18-19 of 24
  are complete.
- **Regressions in c4** (where row or table retrieval dropped):
  - **t10 (MOSHI T4, "which rows have ✓ under causal"):** under A and B the table drops out of the context (fine under C).
  - **MOSHI T3 vs T17:** two near-duplicate Mimi ablation tables, which in c4 carry the same mark columns, are
    confused. t06 and t07 under A got T17 instead of T3; t29 (B, C) and t30 (C) got T3 rows instead of the T17 row.
  - **Crowding:** tables with longer rows split into more chunks (HNET T6 3 → 5, MOSHI T17 up to 7) and fill the
    context.
- **Still incomplete in c4 because of corpus defects:** t13-t15. MOSHI Table 5's `$\phantom{1}42.2$` is extracted as
  142.2 in both corpora (also 140.3 and 136.8): 12 corrupted lines remain in c4 contexts, all from this table.
- **Mark-only rows:** t22 (MOSHI T13: four multi-stream rows identical except for their marks) cannot be identified in
  c3 at all; in c4 it is complete in every setting.

## Limitations
- Small sets: 15 and 24 answerable questions; one question moves an aggregate by 4-7 points. No statistical claim.
- The table set is assistant-authored and not yet independently reviewed.
- Retrieval completeness says the evidence reached the model, not that the model or the verifier reads it
  correctly (see the symbol trace in the answer-quality report).
- `[symbol]` occurs nowhere in either corpus (0 occurrences), so it is covered only by unit tests.
