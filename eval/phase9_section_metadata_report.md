# Phase 9A — MOSHI Table 3 section correction (section metadata s1)

Offline.

- **Code:** `experiments/phase9/section_correction.py`.
- **Outputs:** `experiments/phase9/out/section_correction.json` and `experiments/phase9/corpus/c5/processed/section_corrections.json`.
- **Command:** `uv run python experiments/phase9/section_correction.py`. Exit code 0, `ALL CHECKS PASSED`.

## Where Table 3 is, and why the parser says 5.1

The parser assigns a table to the section whose source text contains its float.

| Evidence | Finding |
|---|---|
| LaTeX source | `\label{tab:mimi_ablations}` is at `sections/evaluation.tex` line 47. That is after `\subsection{Text Language Modeling}` (line 26) and before `\subsection{Audio Tokenization}` (line 90). |
| Rendered PDF | "Table 3:" is on **page 22**. The 5.1 heading is on page 21 and the 5.2 heading on page 23, so the table is printed among 5.1's pages. |
| Prose citing it | 3.3.1 (2 paragraphs), 3.3.2 (1), **5.2 (3)**: the 5.2 Metrics paragraph, "Results - Semantic tokens" and "Results - Acoustic tokens". **5.1 never cites it.** |
| Content | Mimi codec ablation (ABX, VisQOL, MOSNet, MUSHRA). 5.1 is about the text model Helium. |

So the misassignment is **semantic, not positional**. In both the source and the PDF the table sits inside 5.1, but it
belongs to 5.2, the evaluation subsection that cites and discusses it.

**s1 rule:** the table is assigned to the evaluation subsection whose prose cites it. Only Table 3 is changed, as the
phase scope requires.

## The correction

| | Original (c4) | Corrected (s1) |
|---|---|---|
| table record `section_id` | 5.1 | 5.2 (`section_id_parsed: 5.1` kept) |
| table chunks (256: r01–r03; 512: r01–r02) | 5.1 Text Language Modeling | 5.2 Audio Tokenization |
| full-caption chunk `MOSHI:5.1:c070` (256) | 5.1 | 5.2 (the ID keeps "5.1" so citations stay stable) |

The correction is applied **after chunking**. Chunk IDs, order, boundaries and text are unchanged; only the section
fields change.

What this changes in retrieval is the index header that BM25 and dense see: "paper title | section_id section_title".
Once metadata k1 is applied, it also changes the section named in the metadata line.

All five checks are true:

- only Table 3 changed;
- Table 3's ID, label, caption, headers and rows are unchanged;
- Table 17 is unchanged;
- 5.2 cites Table 3;
- 5.1 does not.

## Finding: the same placement affects seven more MOSHI tables (not corrected)

MOSHI puts each evaluation table's float just **before** the `\subsection` heading of the section that discusses it.

| Table | Assigned | Cited in (same chapter) |
|---|---|---|
| T2 `tab:llm_eval` | 5 | 5.1 |
| T4 `tab:mimi_baselines` | 5.1 | 5.2 |
| T5 `tab:ablation-rq` | 5.2 | 5.3 |
| T6 `tab:ablation-delays` | 5.2 | 5.3 |
| T7 `tab:textless` | 5.3 | 5.4, 5.5 |
| T8 `tab:qa` | 5.4 | 5.5 |
| T9 `tab:multistream` | 5.5 | 5.6 |
| T1 `tab:hparams` | 3.2.1 | 3.4.1 (and 4.4, 5.3); a hyperparameter table cited from several chapters, so a different case |

Table 17 (the appendix ablation) is cited by no paragraph at all.

## Effect on retrieval (from the ablation; details in `phase9_ablation_report.md`)

- **Section correction alone:**
  - No change in settings B and C.
  - In A (no re-ranker) it is slightly worse. Probe a1 loses the correct-table-first ranking and drops out of the
    context; a3 drops out of the context; t07/t08 table MRR falls.
  - Original set: no change.
- **Given the metadata line:**
  - Under A, T3/T17 correct-first falls from 14 to 12 (a2, t06) and table-set completeness from 19 to 18 (t06).
  - In exchange, T17 no longer competes in the context for t29, t30 and t32, and t31's table enters the context.
  - B: no change. C: one competing table removed (t12).

**Changing the section label does not resolve the Table 3 vs Table 17 confusion.** "5.2 Audio Tokenization" does not
separate the two tables better than "5.1 Text Language Modeling". The tables are separated by their rows, by MUSHRA and
by the appendix title, and the re-ranker carries most of that.
