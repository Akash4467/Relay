# Annotation guide: repaired-table question set (`table_question_set.json`)

**Status:** written by the assistant (Claude) and validated mechanically
(`experiments/phase7/validate_question_set.py`). **Independent human review is pending.** The set is separate from
the original 15-question evaluation and is never merged with it.

## How questions were written
- Four questions per repaired table (HNET T6; MOSHI T3, T4, T5, T6, T13, T16, T17): 24 answerable, 8 unanswerable.
- Types: symbol_cell, row_column_symbol, row_column_value, symbol_column, symbol_configuration_value,
  distinguish_configurations, blank_cell, supported_comparison, unsupported_comparison, unanswerable.
- Each question names the table by its content ("the Mimi codec ablation table that reports MUSHRA"), never by a
  number a reader could not see in the question's context.
- Expected answers come from the paper's LaTeX source. The validator checks that every number in an expected answer
  occurs in that table's LaTeX float.

## Symbols and blanks
- Marks are referred to as printed: ✓ (check), ✗ (cross), blank. No question asks what a mark means beyond the
  column it is in, because no caption or text in these papers defines the symbols.
- A blank cell is never treated as "no", false or zero. Expected answers say "blank" or "no mark".
- In c4, a blank produced by MOSHI's empty `\xmark` macro is written `(blank)`; a cell left empty in the LaTeX is an
  empty string (both are blank in the PDF). Gold strings use whichever form the corpus actually contains.
- `[symbol]` (unknown `\ding`) does not occur in any of these tables, so it has no question; it is covered by unit tests.

## Gold evidence
- `gold.rows[]`: one entry per table row the answer needs. `id`: strings that locate the row's line in both corpora
  (row label and printed values). `cells`: strings that must also be on that line for the evidence to be complete
  (marks, values). `paper_cells`: the value as printed in the paper where the corpus text is known to be wrong.
- Matching is on chunk row lines ("Configuration ...") of chunks belonging to the named table, so chunk IDs do not
  matter across corpora.
- Validation: each gold row's id matches exactly one c4 row line (id plus marks where rows differ only by marks); id
  plus cells are on that line in c4; the id also locates a row line in c3.
- Unanswerable questions have no gold rows; the correct outcome is a refusal or a caveat.

## Known corpus defects recorded in the set
- t13-t15 (MOSHI T5): the paper writes `$\phantom{1}42.2$`; pylatexenc renders the invisible digit, so c3 and c4 both
  show 142.2 / 140.3 / 136.8 instead of 42.2 / 40.3 / 36.8. The evidence for these questions is wrong in both
  corpora (found while writing this set; not fixed in c4).
- t22 (MOSHI T13): the four multi-stream rows at temperature 0.8 differ only in their marks, so in c3 (marks lost)
  the row cannot be identified at all.

## Grading rules (for when model answers exist)
- Correct: every element of the expected answer is present and correct, with a citation that contains it.
- An answer that turns a blank into "no"/"not used"/"false" is graded wrong, unless the paper states that meaning.
- For t13-t15, an answer quoting the corpus value (e.g. 136.8) is factually wrong, even though it is "supported" by
  the retrieved text.
- Grades are labelled `assistant_graded` until a human confirms them.
