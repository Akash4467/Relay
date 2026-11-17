# Phase 8 — Investigation C: MOSHI Table 3 vs Table 17

Offline, no generation, no change to `relay/` or to any corpus.

- **Code:** `experiments/phase8/table_retrieval.py`, `experiments/phase8/variant_diff.py`, `experiments/phase8/retrieval_common.py`.
- **Outputs:** `experiments/phase8/out/table_retrieval_2026-11-13T21-28-12Z.json` and `variant_diff_2026-11-13T21-57-51Z.json`.

## Why the two tables are confused

| | MOSHI:T3 | MOSHI:T17 |
|---|---|---|
| LaTeX label | `tab:mimi_ablations` | `tab:mimi_ablations_appendix` |
| Section in the corpus | 5.1 Text Language Modeling (**misassigned**; the table floats there) | A Additional ablation on Mimi codec |
| Columns | the same 5 component columns + ABX, VisQOL, MOSNet, **MUSHRA** | the same 5 + ABX, VisQOL, MOSNet |
| Caption | the same first sentence; T3 adds "and collect human judgments with a MUSHRA protocol" | the same first sentence |
| c4 chunks (256) | 3 chunks of 2 rows | 7 chunks |

`short_caption` keeps only the first sentence, so the chunk text of the two tables differs only in rows and the MUSHRA
column. No chunk mixes rows of the two tables.

## Set-up

- **Queries:** 16 in total.
  - 8 adversarial queries, written for this test (a1–a4 → T3, a5–a8 → T17). Each separates the two tables by a real
    difference, such as "with MUSHRA", "appendix" or "additional ablation".
  - The 8 table-set questions about these tables: t05–t08 → T3, t29–t32 → T17. t08 and t31 are unanswerable but still
    target one table.
- **Measured:**
  - whether the first chunk of the correct table ranks ahead of the first chunk of the other table among the candidates;
  - whether each table reaches the context.
- **Variants:** one change each, built with the existing chunker in-process.

| Variant | Change |
|---|---|
| base | none. **Re-chunking equals the stored c4 chunks exactly** (checked; the run stops otherwise), and its metrics reproduce Phase 7 c4 |
| full_caption | every table chunk repeats the whole caption |
| metadata | each table chunk gets a line `Source: section <id> <title>; label <latex label>` |
| whole_table | a table is split only beyond 2 × piece size |

**Not tested: adding the table's identity to the query.** No question names a table number, so it is never justified.

## Per-stage trace (base, setting B): first rank of gold / other table

| Query | BM25 | Dense | RRF | Re-rank | Context |
|---|---|---|---|---|---|
| a1 T3 | 2/9 | 17/4 | 5/6 | **1**/8 | 1/– |
| a2 T3 | 1/4 | 10/1 | 7/**1** | **1**/5 | 1/5 |
| a3 T3 | 1/8 | 20/5 | 4/2 | **1**/4 | 1/4 |
| a4 T3 | 1/4 | 14/2 | 8/**1** | **1**/4 | 1/4 |
| a5 T17 | 2/12 | 3/– | 2/16 | **1**/3 | 1/3 |
| a6 T17 | 1/9 | 1/9 | 1/9 | **1**/6 | 1/6 |
| a7 T17 | 2/10 | 2/10 | 2/10 | **1**/8 | 1/– |
| a8 T17 | 4/1 | 2/15 | 1/7 | 4/**1** | 4/1 |
| t05 T3 | 1/5 | –/– | 3/10 | **1**/4 | 1/4 |
| t06 T3 | 1/2 | –/4 | 9/**1** | **1**/4 | 1/4 |
| t07 T3 | 1/5 | 17/5 | 8/4 | **1**/5 | 1/5 |
| t08 T3 (unans.) | 12/1 | 10/1 | 9/2 | 3/4 | 3/4 |
| t29 T17 | 3/10 | 2/11 | 2/8 | **1**/4 | 1/4 |
| t30 T17 | 1/8 | 2/13 | 1/9 | **1**/7 | 1/– |
| t31 T17 (unans.) | 9/1 | –/– | 16/2 | 5/**1** | 5/1 |
| t32 T17 | 1/8 | 1/8 | 1/8 | **1**/4 | 1/4 |

### What the trace shows

- **Dense ranks T17 ahead of T3 for every T3 query where it finds either table** (all except t05, where it finds neither).
  After RRF, T17 is still ahead for a2, a3, a4, t06, t07 and t08.
- **The cross-encoder is what separates the tables.** After re-ranking, the gold table is first in 14 of 16 queries.
- The two failures share one pattern: the query mentions **MUSHRA in order to exclude it**.
  - a8: "the extended ablation **without MUSHRA**".
  - t31: "the table **without a MUSHRA column**".

  The word pulls in T3. Neither BM25 nor the cross-encoder models the negation.
- **Both tables usually reach the context together.** The other table is in the context in 13 of 16 queries under B. The
  model must then pick the right one from the rows; the right table is present in 16 of 16.

## Variants on the 16 queries (correct table ahead / gold in context / other in context)

| Variant | A | B | C |
|---|---|---|---|
| base | 9 · 10 · 9 | **14 · 16** · 13 | 9 · 13 · 11 |
| full_caption | 10 · 10 · 5 | 12 · 13 · 5 | 9 · 13 · 11 |
| metadata | 8 · 11 · 8 | 13 · 15 · 9 | **13** · 14 · 7 |
| whole_table | 9 · 7 · 4 | 9 · 13 · 11 | 7 · 13 · 11 |

- **full_caption:**
  - The other table drops out of the context (B: 13 → 5).
  - But B loses a5 and t08 on ranking. Gold leaves the context for a8, t08 and t31, because whole-caption chunks of T3 now
    match "MUSHRA" queries even more strongly.
- **metadata:**
  - C improves from 9 to 13: a5, a6, t29 and t30 are now correct. The appendix section title "Additional ablation on Mimi
    codec" matches the T17 queries.
  - B loses t08 (unanswerable), and A loses a1.
  - It cannot fix the misassigned T3 section, because the metadata line repeats the wrong section, 5.1.
- **whole_table:**
  - Worse in B (14 → 9) and C (9 → 7). Fewer, larger chunks give the reranker less to separate the tables with.

## Side effects on the original 15 and the table set (`variant_diff`, per question against base)

| Variant | Original set | Table set (complete, 24 answerable) |
|---|---|---|
| full_caption | q11 A +1 quote; q13 A, B −1 quote | A 18 → 17 (+t06, −t29, −t32); B 19 → 17 (+t10, −t11, −t28, −t30); C 19 |
| metadata | **q11 B 0 → 1 quote** (any-gold 11 → 12/15); q13 A −1 quote; MRR B .582 → .552 | A 18; **B 19 → 20** (+t10, +t29, −t30); C 19 (+t30, −t25); rows in context B 27 → 30 |
| whole_table | q11 A +1, q11 B 0 → 2, **q11 C 2 → 0**, q13 A/B −1, q13 C +1 | A 18 → 17; B 19; C 19 → 18 |

## Answers

- **Can retrieval distinguish MOSHI Tables 3 and 17?**
  - **Partly, and only with the re-ranker.** Under B the correct table is ranked first in 14 of 16 queries and is always in
    the context. But the wrong table is also in the context in 13 of 16.
  - Without the re-ranker (A), or with 512-token pieces and 4 passages (C), it is 9 of 16.
  - Queries that name MUSHRA in order to exclude it fail in every setting and variant.
- **Does table-aware chunking improve retrieval without harming unrelated questions?**
  - **No.** whole_table cuts duplicate table chunks and fixes q11 B and t10.
  - But it regresses q11 C, q13 A/B, t05 A, t30 B and t32 C, and it makes T3/T17 discrimination worse.
- **Of the four variants, metadata** (section and label line on each table chunk) **is the only one with a net gain on both
  sets.** It is small and mixed:
  - C +4 on T3/T17;
  - table B +1 complete;
  - original B +1 any-gold;
  - but q13 A −1 quote, two T3/T17 losses, and a lower B MRR.

  It needs the misassigned T3 section fixed first: that is a parser defect, recorded but not fixed here. It also needs a
  held-out query set before any claim.

The adversarial queries were written by the assistant after inspecting the tables. They are not an independent test set.
