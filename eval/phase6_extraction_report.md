# Phase 6: extraction repair (e2) and corpus c4

Offline only. No model, verifier, prompt, threshold or pipeline change. The default extraction (e1), the default
corpus (c3: `data/processed`, `data/index`) and `config/app.yaml` (`corpus_version: c3`) are unchanged. Evidence:
`experiments/phase6/out/extraction_validation_2026-11-13T20-18-12Z.json`.

## Root cause (from Phase 5, confirmed in the source)

- HNET and MOSHI define `\newcommand{\cmark}{\ding{51}}`; HNET also `\newcommand{\xmark}{\ding{55}}`; MOSHI defines
  `\newcommand{\xmark}{}` (blank in its PDF) and also uses `\checkmark`. AIAYN uses none of them.
- `latex_parse.to_text` hands the expanded LaTeX to pylatexenc, which renders the unknown `\ding{51}` as its
  argument "51" (`\ding{55}` → "55") and `\checkmark` as an empty string.
- Effects in c3: "51"/"55" as cell values (HNET Table 6, MOSHI Table 6); ticks lost entirely in MOSHI tables. Because
  `parse_tabular` (e1) drops empty cells and uses column 0 as the row label, MOSHI Tables 3 and 17 (Mimi ablations)
  lost all five configuration columns and their rows had an empty label.

## Extraction changes (opt-in extraction version "e2"; e1 stays the default)

All in `relay/retrieval/latex_parse.py`, selected with `parse_paper(code, raw_dir, extraction="e2")` /
`main(..., extraction="e2")` through a context variable. The e1 code path is untouched.

1. `map_symbols` (before pylatexenc, in `to_text`): `\ding{51}`, `\ding{52}`, `\checkmark` → ✓; `\ding{55}`,
   `\ding{56}` → ✗; any other `\ding{N}` → `[symbol]` (never the bare number). Nothing else is replaced.
2. Paper macros with no arguments and an empty body (MOSHI `\xmark`) expand to an internal blank marker: a table cell
   made only of it becomes `(blank)` (the PDF shows a blank there); in prose and captions it is removed, as before.
3. `parse_tabular_e2`: cells are read with their `\multirow` span, then a grid pass fills each span: `\multirow{n}`
   downward, `\multirow{-n}` (HNET Table 1) for this row and the n-1 rows above, as LaTeX draws it. Every column keeps
   its position in each row's `cells` (`""` when empty). A first column consisting only of ✓/✗/blank stays a column;
   the row label becomes "Row k". Header/data split, header naming, group labels and the e1 row-label fill-down are
   unchanged.
4. `row_text` (latex_parse) and `structured_row` (chunker) skip `""` cells, so empty positions are kept in the table
   record but not spelled out in chunk text. e1 data never holds `""`, so both are no-ops for c1-c3 (verified below).

## Before and after (real tables)

| Table | e1 (c3) | e2 (c4) |
|---|---|---|
| HNET T6, "Isotropic" row | `Autoregressive: 55` | `Autoregressive: ✗` |
| HNET T6, H-Net row | `Autoregressive = 51; … Multi-stage / Hierarchy = 51` | `Autoregressive = ✓; … Multi-stage / Hierarchy = ✓` |
| MOSHI T3 (Mimi ablation), row 1; source `\checkmark & \checkmark & \checkmark & \xmark & \xmark & 23.3\%` | label `""`; cells: ABX 23.3%, VisQOL 2.91, MOSNet 2.89, MUSHRA 65.9±1.7 | label `Row 1`; Quantization Rate ✓, Transformer in encoder ✓, in decoder ✓, WavLM distillation (blank), Split quantizer (blank), then the same metrics |
| MOSHI T6, row `[0, 0, …]` | `Depthwise / Parametrization: 51`; Inner Monologue missing | `✓`; Inner Monologue `(blank)` |
| HNET T1, SpaceByte++ row; source `\multirow{-2}{*}{Byte}` one row below | Input missing | `Input: Byte`, `L^0: 8192` |

## Unit tests (`tests/unit/test_latex_parse_e2.py`): 16 passed

Every supported macro (6 forms); unknown `\ding{99}` → `[symbol]`; e1 still renders `\ding{51}` as "51"; ordinary
numbers and prose without symbols identical in e1 and e2; mixed text and symbols; downward `\multirow` (exact span, no
over-carry) and empty cells keeping their column; first-column marks with a blank mark not filled down; a plain table
identical in e1 and e2; a full fake paper defining `\cmark` via `\ding` and `\xmark` as empty (tables and prose);
upward `\multirow{-2}` with no "-2" artifact; unknown extraction version refused.

## Validation against the three papers (in memory, before any corpus was built)

All checks passed on the final run (an earlier run failed and was fixed, see below):

1. The 53 rows changed by the Phase 5 symbol mapping were re-identified (53) and now hold marks; no 51/55 survives in
   the 8 affected tables (HNET T6; MOSHI T3, T4, T5, T6, T13, T16, T17).
2. Prose: 0 paragraphs changed; paragraph counts equal.
3. Every row keeps every column in header order (MOSHI T10/T11: see limitations; checked against e1 instead).
4. Same tables, same headers, same row counts in all 31 tables.
5. No 51/55 value from a mark macro anywhere.
6. Every numeric value of e1 is still present in e2, table by table (HNET T1: 22 → 24, the upward fills).

16 of 31 tables differ from e1. In the 7 without marks or spans (AIAYN T2, T3; MOSHI T1, T12; HNET T4; and most rows of
HNET T2, T3) the only difference is the kept empty positions. Every value or label change is a mark (51/55 → ✓/✗; lost
ticks → ✓ or (blank)) or an upward-multirow fill (HNET T1-T3: "Byte"/"8192").

The first validation run failed and stopped the build: (a) e2 turned `\multirow{-2}{*}{Byte}` into "-2*Byte" because
its span pattern accepted only positive numbers (fixed: negative spans fill upward; test added); (b) MOSHI T10 rows
failed the column check because their header list repeats "54.3", a pre-existing e1 defect (below).

## Corpus c4 (`data/corpus/c4/`, manifest `data/corpus/c4/manifest.json`)

Built with `scripts/build_corpus.py --version c4 --extraction e2 --out data/corpus/c4` (refuses an existing directory
or the default ones). Same stages and configuration as c3: piece sizes 256/512, BAAI/bge-small-en-v1.5, BM25 + FAISS.
The manifest records source tarball/PDF/source-tree hashes, code hashes, chunking and embedding configuration, build
time, git commit and every output hash. Reproducibility control: the same script with e1 rebuilt c3 into a scratch
directory **byte-identical** to the stored c3 (all processed files, chunks, FAISS, BM25 and IDF files).

c4: 465 chunks at 256 tokens (c3: 455), 260 at 512 (255). AIAYN chunks identical; MOSHI 213/238 and HNET 164/173
identical (256). In the 8 tables every chunk keeps the caption and the "Columns:" line, every row line is labelled,
structured rows match row lines, 175 marks appear in chunk text, and no mark artifact remains ("= 51" only as the
start of the genuine value 519).

## Remaining limitations

- MOSHI T10/T11 (pre-existing, also in e1): a data row is read as the header, so the header list repeats "54.3"/"49.8"
  and the cells dict collapses those columns. Not changed here.
- HNET T1 headers are partly wrong in e1 and e2 (`\multirowcell` header, "col6"-"col8"). Not changed here.
- MOSHI's blank `\xmark` is shown as "(blank)", faithful to the PDF; that it means "not used" is left to the reader.
- Table premises in the verifier (`_row_line`) would print ✓/✗; how NLI scores them is untested (not in scope).
- c4 is not used by the pipeline. Switching would need approval, a `corpus_version` bump and a fresh retrieval and
  answer evaluation.
