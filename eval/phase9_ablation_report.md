# Phase 9A — Ablation: which change does what

Offline.

- **Code:** `experiments/phase9/ablation.py`.
- **Output:** `experiments/phase9/out/ablation_2026-11-13T22-39-25Z.json`, with per-question rows for each variant,
  `changes_vs_c4` and `changes_c5_vs_section_meta`.
- **Command:** `uv run python experiments/phase9/ablation.py`. Exit code 0.

## Design: one factor at a time

All variants are built in memory by `build_c5.chunk_corpus`, the same function that built c5. They are evaluated with
the same code as `evaluate_retrieval.py`.

| Variant | Extraction | T3 section | Metadata line |
|---|---|---|---|
| c4 | e2 | 5.1 | no |
| e3_only | **e3** | 5.1 | no |
| section_only | e2 | **5.2** | no |
| metadata_only | e2 | 5.1 | **yes** (it names 5.1) |
| section_meta | e2 | 5.2 | yes |
| c5 | e3 | 5.2 | yes |

**Controls, all passed:**

- the `c4` variant equals the stored c4 chunks;
- the `c5` variant equals the stored c5 chunks;
- the in-memory c4 and c5 indexes give the same context as the disk indexes for every question and setting.

c5 − section_meta isolates e3 in the presence of the other two changes. section_meta − metadata_only isolates the section
fix in the presence of the metadata line.

**What could not be separated cleanly:** the metadata line contains the section title. So "metadata with the corrected
section" is only available together with the section fix (section_meta). Its section effect is reported as
section_meta vs metadata_only.

## Results: questions complete (table set, provisional) and T3/T17 correct-first

| Variant | Table complete A / B / C | T3/T17 correct first A / B / C | Original quote recall A / B / C | Original MRR A / B / C |
|---|---|---|---|---|
| c4 | 18 / 19 / 19 | 9 / 14 / 9 | 22 / 20 / 23 | .602 / .582 / .644 |
| e3_only | **21 / 22 / 22** | 9 / 14 / 9 | 22 / 20 / 23 | .602 / .582 / .644 |
| section_only | 18 / 19 / 19 | **8** / 14 / 9 | 22 / 20 / 23 | .602 / .582 / .644 |
| metadata_only | 19 / 19 / 21 | **14** / 13 / **13** | 22 / **19** / 23 | **.569 / .550** / .644 |
| section_meta | 18 / 19 / 21 | 12 / 13 / 13 | 22 / 19 / 23 | .569 / .550 / .644 |
| c5 | **21 / 22 / 24** | 12 / 13 / 13 | 22 / 19 / 23 | .569 / .550 / .644 |

Correct table in context (table set, 24): c4 21/23/24; metadata_only, section_meta and c5 all 24/24/24; e3_only and
section_only unchanged from c4.

## Attribution

### e3: clean and additive

- **Alone vs c4:** exactly t13, t14 and t15 become complete in A, B and C (9 question-setting cases). The only other
  change is t06 A table MRR .111 → .125.
- **On top of section + metadata (c5 vs section_meta):** the same 9 cases and nothing else.
- No change on the original set.
- These gains come from the extraction fix, not from metadata.

The gain is measured with the `paper_cells` row locator; c4 cannot pass because the printed value is absent.

### Section fix: neutral to slightly harmful

- **Alone vs c4:** no change in B or C. In A:
  - a1 is no longer correct-first and leaves the context;
  - a3 leaves the context;
  - t07 and t08 table MRR fall;
  - the wrong table leaves a8's context (the only gain).
- **Given metadata (section_meta vs metadata_only):** A loses t06 (complete → incomplete) and the correct-first ranking
  of a2 and t06. It gains t31's table in the context and removes T17 as a competitor for t29, t30 and t32. B: no change.
  C: one competitor removed (t12).

### Metadata line: the driver of table identification, and of every cost

- **Gains:**
  - correct table in context 24/24 in every setting;
  - table MRR up everywhere;
  - T3/T17 correct-first A 9 → 14 and C 9 → 13;
  - wrong table in context C 11 → 4;
  - newly complete: t06, t07 (A), t10 (B), t29, t30 (C).
- **Costs** (about 46 extra tokens per table chunk, so more evidence is dropped by the fixed budget):
  - t32 A and t30 B lose their gold row to the budget;
  - q13 B loses a quote;
  - t08 B (unanswerable) loses its table;
  - T3/T17 B 14 → 13;
  - original MRR A .602 → .569 and B .582 → .550 (q10 A, q12 B);
  - competing tables in context A 3 → 8 and B 7 → 10.
- All three original-set regressions in c5 (q10 A MRR, q12 B MRR, q13 B quote) appear in `metadata_only` and not in
  `e3_only` or `section_only`.

## Interactions

- **e3 is independent of the other two:** its effect is identical alone and in combination.
- **The section fix interacts with metadata only under A,** and the net effect is negative. Under A, the correct-first
  count falls 14 → 12 and table completeness 19 → 18.
