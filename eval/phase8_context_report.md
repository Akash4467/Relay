# Phase 8 — Investigation D: context crowding

Offline, no generation. Code: `experiments/phase8/context_crowding.py` with the shared `experiments/phase8/retrieval_common.py`.
Output: `experiments/phase8/out/context_crowding_2026-11-13T21-43-19Z.json`.

## Set-up

- **Index:** the unchanged c4 index, built in memory from the c4 processed files.
  - Its re-chunking was checked to equal the stored c4 chunks byte for byte (`base_reproduces_stored_c4: true` in
    `table_retrieval_2026-11-13T21-28-12Z.json`).
  - Its baseline reproduces every Phase 7 c4 number.
- **Policies:** each re-picks passages from the **same ranked candidates** that production `search()` returns, then calls the
  production `fill()`. Settings A/B/C, cap 2500, budget 1600, are unchanged.

| Policy | Rule |
|---|---|
| baseline | production path |
| identity | the policy hook with no change. **Equals baseline on all 3 × 47 questions** (checked; the run stops otherwise) |
| max2_per_table | at most 2 chunks of one table; later candidates move up |
| max1_per_table | at most 1 chunk of one table |
| table_half_budget | table chunks may use at most 50% of the context budget |
| whole_table_index | table-aware chunking (a table is split only beyond 2 × piece size) with baseline assembly |

Metrics are as in Phase 7. *Duplicate table chunk rate* is the number of extra chunks of an already-present table, divided by
the number of passages.

## Results

### Original 15 questions: any gold / quote recall (34 quotes) / duplicate rate / mean context tokens

| Policy | A | B | C |
|---|---|---|---|
| baseline | 13/15 · 22 · .099 · 1474 | 11/15 · 20 · .126 · 1461 | 14/15 · 23 · .062 · 1271 |
| max2_per_table | 13/15 · 21 · .047 · 1476 | **12/15** · 20 · .041 · 1473 | 14/15 · 23 · .045 · 1259 |
| max1_per_table | 13/15 · 20 · 0 · 1477 | **12/15** · 19 · 0 · 1485 | 14/15 · 23 · 0 · 1247 |
| table_half_budget | 13/15 · **23** · .075 · 1482 | 11/15 · 19 · .068 · 1478 | 14/15 · 23 · .009 · 1251 |
| whole_table_index | 13/15 · 22 · .036 · 1474 | **12/15** · 21 · .034 · 1496 | 13/15 · 22 · 0 · 1230 |

### Repaired-table set, 24 answerable: questions complete / rows in context (31) / duplicate rate

| Policy | A | B | C |
|---|---|---|---|
| baseline | 18 · 26 · .357 | 19 · 27 · .453 | 19 · 29 · .279 |
| max2_per_table | 15 · 23 · .143 | **13** · 21 · .165 | 18 · 28 · .231 |
| max1_per_table | **11** · 20 · 0 | **8** · 16 · 0 | 15 · 25 · 0 |
| table_half_budget | 15 · 23 · .209 | 15 · 23 · .210 | 15 · 25 · .057 |
| whole_table_index | 17 · 25 · .170 | 19 · 29 · .203 | 18 · 28 · .064 |

Mean context tokens barely move (1230–1515 in every cell): `fill()` always fills the budget. Crowding changes **what** is in
the context, not how much.

## The investigated questions

| Case | Baseline context | What changes |
|---|---|---|
| q10, B | 5 of 6 passages are HNET Table 6 chunks (r05, r02, r01, r04, r03); 0/3 quotes | max2 / max1: 1/3 quotes (`HNET:4:c080` gets in). half-budget: still 0/3. whole_table: 0/3 (Table 6 is now 2 chunks, but other prose outranks the gold) |
| q10, A | 1 prose + 5 Table 6 chunks; 1/3 quotes | no policy adds a quote (1/3 everywhere) |
| q11, A | 3 AIAYN Table 3 chunks; 1/2 quotes (r01 missing) | half-budget and whole_table: **2/2**. max2 / max1: 1/2 |

The q11 regression from Phase 7 was a matter of which Table 3 piece reached the context: r01 holds the gold quote. Capping or
merging Table 3 changes that. It is not a prose-vs-table crowding case.

## Side effects (per question, against baseline)

- **max2_per_table:**
  - Gains: q10 B, plus t06/t07 under A. In both t06 and t07 the right MOSHI T3 chunk replaces T17 chunks.
  - Losses: 12 question-setting cases become incomplete (5 under A, 6 under B, 1 under C; 9 distinct questions: t01, t02,
    t03, t17, t25, t28, t29, t30, t32), and q13 drops a quote under A and B.
  - Cause: a question about row k of a multi-chunk table needs chunk k, which is often the 3rd or 4th-ranked chunk of that
    table.
- **max1_per_table:** the same pattern, worse. 23 question-setting cases lose completeness (8 under A, 11 under B, 4 under
  C); q12 and q13 drop a quote under A and B.
- **table_half_budget:**
  - Gain: q11 A.
  - Losses: 11 question-setting cases on the table set (3 under A, 4 under B, 4 under C; t01, t09, t25, t28, t29, t30, t32),
    and q13 B drops a quote.
- **whole_table_index:**
  - Gains: q11 A/B, t10 A/B (Phase 7's t10 regression: the table is back in the context, all 3 rows, complete under B),
    and q13 C.
  - Losses: q11 C (2 → 0 quotes), q13 A/B (2 → 1), t05 A, t30 B, t32 C.
  - It is also worse on T3 vs T17 (see the table-retrieval report).

## Answer

Crowding is real: one table can take 5 of 6 passages (q10 B, HNET Table 6). But **every rule that limits table chunks also
removes the specific row chunk that table questions need**. They lose 1–11 complete table questions per setting (B: 4, 6
and 11). In exchange they rescue one original question: q10 B for the per-table caps, q11 A for the half-budget rule.

**No assembly policy should proceed.** Table-aware chunking (whole_table) is the most balanced: duplicates fall from
.126/.453 to .034/.203 under B, and it fixes q11 B and t10. But it trades one or two specific questions in each setting and
regresses T3/T17 discrimination, so it is not a clear win either.

A fix that limits a table by *rows the question does not need* would require knowing which row the question needs. That is
a different design: row-aware ranking inside a table. It is not tested here.

**Not tested:** generation from these contexts (no live calls), and settings other than A/B/C at cap 2500.
