# Phase 9A — Extraction e3 (`\phantom` removal)

Offline. Code: `experiments/phase9/extraction_e3.py`. Output: `experiments/phase9/out/e3_changes.json`.
Command: `uv run python experiments/phase9/extraction_e3.py`. Exit code 0, `ALL CHECKS PASSED`.

## What e3 is

e3 is e2 plus one change. Before conversion, every `\phantom`, `\hphantom` and `\vphantom` is removed together with its
brace-matched argument:

- a nested phantom goes with its parent;
- unbalanced braces leave the text unchanged;
- no characters are replaced.

It wraps `relay.retrieval.latex_parse` in-process and restores the converter on exit, also on error. The production
parser, e1, e2, c3 and c4 are untouched. The control build in `build_c5.py` re-ran e2 and reproduced every c4 processed
file byte for byte.

## Validation: e2 vs e3 on all three papers

Every section, paragraph, figure caption, table caption, header, row header and cell was compared.

A changed cell is verified on its **PDF page**. The cells of its row, in order and with blanks omitted as the PDF does,
must appear as **consecutive lines** in that table's region of the page (from "Table N:" to the next caption). The row
must match with the e3 value and must not match with the e2 value.

| Paper/table | Row | Column | e2 (c4) | e3 | PDF page | Row matches PDF with e3 / e2 |
|---|---|---|---:|---:|---|---|
| MOSHI T5 | [0, 1, 2, 3, 4, 5, 6, 7] | Perplexity | 142.2 | 42.2 | 25 | yes / no |
| MOSHI T5 | [0, 1, 2, 3, 4, 5, 6, 7] (✓) | Perplexity | 140.3 | 40.3 | 25 | yes / no |
| MOSHI T5 | [0, 2, 2, 2, 2, 2, 2, 2] (✓) | Perplexity | 136.8 | 36.8 | 25 | yes / no |
| MOSHI T6 | [0, 0, 0, 0, 0, 0, 0, 0] | Semantic Token / Weight | 001.0 | 1.0 | 25 | yes / no |
| MOSHI T6 | [0, 1, 1, 1, 1, 1, 1, 1] | Semantic Token / Weight | 001.0 | 1.0 | 25 | yes / no |
| MOSHI T6 | [0, 2, 2, 2, 2, 2, 2, 2] | Semantic Token / Weight | 001.0 | 1.0 | 25 | yes / no |

The checks in `e3_changes.json` are all true:

- exactly these six cell changes;
- every changed row matches the PDF with e3, and none with e2;
- each new value equals the old value with only leading characters removed, so no visible digit disappears;
- the caption change below is the expected one and matches the PDF;
- no other difference in any section, paragraph, figure, table structure, header or row header.

## New finding: a seventh change, missed in Phase 8

**MOSHI Table 18's caption** (`tab:results_toxicity`, PDF page 61) contains three colour swatches written as
`\colorbox{…}{\phantom{x}}`.

- e2 prints them as "x": "…≥ 99 are gray x, unsafe … are Orange x, and highly unsafe … are Red x."
- The PDF prints a coloured box with no text: "gray\n, …".
- e3 removes the "x". The validator checks every removed word against the PDF caption by tokens: "gray x" is absent and
  "gray ," is present, and likewise for Orange and Red.

This text **is** in c4: in one caption chunk at 256 tokens and in 11 table chunks at 512 tokens.

**The Phase 8 extraction report was wrong on two points.**

- It said the swatches' text "is not in the corpus".
- It said there were "no row header, paragraph or caption changes".

Phase 8's validation compared figure captions but not table captions. The Phase 8 report is left unchanged as a
historical record, and this report supersedes it on this point.

## Defect inventory after e3

| Defect | Status in e3 |
|---|---|
| `\phantom` digits in table cells (MOSHI T5 ×3, T6 ×3) | fixed, PDF-verified |
| `\phantom{x}` swatch text in MOSHI T18 caption | fixed, PDF-verified |
| `\hphantom` (16 occurrences) | already dropped by pylatexenc; unchanged |
| table section assignment (`phase9_section_metadata_report.md`) | not an extraction defect; not addressed by e3 |

## Tests

All tests are listed in `phase9_decision_report.md`.

- `experiments/phase9/test_phase9.py` covers e3 with 10 parametrised `strip_phantoms` cases (including nested,
  unbalanced, `\phantomx`, the swatch, and visible digits on both sides).
- It also covers the converter being restored after use and after an error, the pinned e2 defect, PDF row matching,
  table-region bounds and the caption word-diff.
- The existing extraction tests (`tests/unit/test_latex_parse*.py`) pass unchanged.
