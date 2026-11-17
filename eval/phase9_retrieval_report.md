# Phase 9A — Retrieval: c4 vs c5

Offline, no generation.

- **Code:** `experiments/phase9/evaluate_retrieval.py`.
- **Output:** `experiments/phase9/out/retrieval_eval_2026-11-13T22-17-05Z.json`, with per-question rows for both corpora
  and `changes_c4_to_c5`.
- **Command:** `uv run python experiments/phase9/evaluate_retrieval.py`. Exit code 0.

## Set-up

- **Corpora:** both loaded from disk: c4 from `data/corpus/c4`, c5 from `experiments/phase9/corpus/c5`.
- **Configuration:** identical for both, the production `search()` with settings A/B/C from
  `config/retrieval_settings.yaml`, pressure L0 cap 2500 and budget 1600.
- **Datasets:** always reported separately.
  - The original 15-question set.
  - The 32-question repaired-table set (24 answerable). **Its labels are provisional:** they are assistant-made and
    independent human review is pending.
  - 16 MOSHI T3/T17 probes: a1–a8 from Phase 8, plus the 8 table-set questions about these tables.
- **Baseline check:** the c4 rows reproduce the Phase 7/8 c4 numbers exactly: any gold 13/11/14 of 15, quote recall
  22/20/23 of 34, MRR .602/.582/.644, table complete 18/19/19.
- **Row locator:** t13–t15 have expected rows written against c4's inflated values, with the printed values in
  `paper_cells`. The evaluator also accepts the row locator with `paper_cells` substituted, and applies this to c4 and c5
  alike. The dataset file is unchanged. In c4 the paper's value is absent, so those rows stay incomplete there, exactly
  as in Phase 7.

## Original 15 questions

| | A c4 | A c5 | B c4 | B c5 | C c4 | C c5 |
|---|---|---|---|---|---|---|
| any gold in context | 13/15 | 13/15 | 11/15 | 11/15 | 14/15 | 14/15 |
| complete gold | 7/15 | 7/15 | 7/15 | 7/15 | 9/15 | 9/15 |
| quote recall (34) | 22 | 22 | 20 | **19** | 23 | 23 |
| MRR | .602 | **.569** | .582 | **.550** | .644 | .644 |
| quotes lost to the budget | 1 | 1 | 2 | 3 | 0 | 0 |
| duplicate table chunk rate | .099 | .078 | .126 | .147 | .062 | .045 |
| mean context tokens | 1474 | 1477 | 1461 | 1458 | 1271 | 1285 |
| missing | q03 q14 | same | q06 q10 q11 q14 | same | q14 | same |

## Repaired-table set (provisional labels): 24 answerable, 31 gold rows

| | A c4 | A c5 | B c4 | B c5 | C c4 | C c5 |
|---|---|---|---|---|---|---|
| correct table in context | 21 | **24** | 23 | **24** | 24 | 24 |
| table MRR | .697 | .865 | .938 | .979 | .931 | **1.000** |
| rows in context | 26 | 27 | 27 | 29 | 29 | **31** |
| **questions complete** | 18 | **21** | 19 | **22** | 19 | **24** |
| complete if nothing were lost to the budget | 18 | 23 | 19 | 24 | 20 | 24 |
| competing table (same paper) in context | 3 | 5 | 7 | 10 | 8 | 5 |
| duplicate table chunk rate | .357 | .411 | .453 | .497 | .279 | .313 |
| passages dropped by the budget | 32 | 64 | 42 | 77 | 24 | 29 |
| unanswerable: table in context (8) | 6 | 7 | 8 | 7 | 6 | 7 |

## MOSHI Table 3 vs Table 17 (16 probes)

| | A c4 | A c5 | B c4 | B c5 | C c4 | C c5 |
|---|---|---|---|---|---|---|
| correct table ranked first | 9 | **12** | 14 | 13 | 9 | **13** |
| correct table in context | 10 | **15** | 16 | 15 | 13 | 14 |
| wrong table in context | 9 | 8 | 13 | 11 | 11 | **4** |
| exclusion queries correct (a8, t29, t31) | 2/3 | 2/3 | 1/3 | 1/3 | 0/3 | 1/3 |
| wrong first | a2 a3 a4 t06 t07 t08 t31 | a2 t06 t08 t31 | a8 t31 | a8 **t08** t31 | a5 a6 a8 t08 t29 t30 t31 | a8 t08 t31 |

**Negation and exclusion are unsolved.**

- a8 ("…ablation **without MUSHRA**") and t31 ("…table **without a MUSHRA column**") rank Table 3 first in every
  setting, in both corpora.
- t29 ("the table reporting **only** ABX, VisQOL and MOSNet") is fixed in C only.

## Per-question changes c4 → c5

Complete list in `changes_c4_to_c5`. Small MRR shifts are listed only for the original set.

### Gains

| Question | Setting | Change | Cause (ablation) |
|---|---|---|---|
| t13, t14, t15 | A, B, C | incomplete → **complete** | e3: the printed perplexities (36.8, 42.2, 40.3) are now in the row |
| t07 | A | table now in context, row found, complete | metadata |
| t06 | A | table now in context (not complete in c5) | metadata; the section fix then removes its row again |
| t10 | A | table in context (1 row) | metadata |
| t10 | B | incomplete → **complete** (3 rows) | metadata |
| t29, t30 | C | incomplete → **complete** | metadata |
| t08, t31 (unanswerable) | C / A | target table now in context | metadata / section |
| probes a3, a4, t07 (A); a5, a6, t29, t30 (C) | | correct table now first | metadata |
| competing table removed | A t07 t12 t20; B t12 t29; C t10 t29 t30 t32 | | metadata / section |

### Regressions

| Question | Setting | Change | Cause |
|---|---|---|---|
| **t32** | A | complete → incomplete | the gold row chunk is in the top 8 but **dropped by the budget**; the context holds five T17 chunks (r06, r02, r07, r01, r03) |
| **t30** | B | complete → incomplete | the same: the needed T17 chunk is dropped by the budget |
| **q13** | B | 2 → 1 of 3 quotes | HNET Table 2 chunks (r10, r09, r07, r15, r14) fill the context; prose `HNET:3.1:c051` is dropped by the budget |
| q10 | A | MRR 1.0 → 0.5 | an HNET Table 6 chunk now outranks the prose. Any-gold is unchanged (1 of 3) |
| q12 | B | MRR 1.0 → 0.5 | its gold is MOSHI Table 8 (r01, r02); prose `MOSHI:5.5:c090` now ranks first. Both gold chunks are still in the context (metadata) |
| t08 (unanswerable) | B | T3 leaves the context; probe t08 now ranks T17 first | metadata |
| t04 | A | table MRR 1.0 → 0.5 | metadata |
| competing table added | A t13 t14 t15 t16; B t01 t13 t14 t23; C t11 | | metadata |

### Focus questions

- **q10:**
  - A: still 1 of 3 quotes; MRR 1.0 → 0.5.
  - B: still 0 of 3. The c5 context is five HNET Table 6 chunks, and the prose candidates (1:c012, 3.1:c049, 3:c043)
    are dropped by the budget.
  - Crowding is unchanged or slightly worse.
- **q11:** unchanged in quotes. A has 1 of 2; B has 0 of 2 (MRR .125 → .143); C has 2 of 2.
- **q13:** A unchanged (2 of 3); B regresses (above); C unchanged (1 of 3).
- **t05:** complete in every setting in both corpora; table MRR under A .333 → 1.0.
- **t30:** B regresses (budget); C gains.
- **t32:** A regresses (budget); B and C unchanged (complete).

## Answer

c5 retrieves more complete table evidence:

- table questions complete rise by 3, 3 and 5 under A, B and C;
- the correct table is in the context for 24/24 answerable questions in every setting.

On the original set, c5 gives no gain and small losses (MRR in A/B, one quote in B). The gains are on the provisional
table set; two of its questions regress through the token budget. These are retrieval measurements only. **No claim about
answer quality follows from them.**
