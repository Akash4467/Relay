# Phase 9B — Review package for the table-mark verifier benchmark

**Outcome A: independent review not yet available.** The package is ready and the evaluation code is validated. No
reviewed-label result exists, and no performance claim is made in this phase.

Offline only:

- no Groq or API calls, and no generation;
- no change to `relay/`, `config/`, `results/`, c3, c4, c5, v4 or v5;
- no commits.

## What was read first

- The Phase 8 reports: verifier semantics and decision.
- The Phase 9 decision and ablation reports.
- `experiments/phase8/table_semantics.py`, its two case files and all five Phase 8 table-semantics outputs.

C2g, as Phase 8 built it, has three parts:

- C2, a literal mark checker. It resolves claims of the form "row R is marked X under column K"; otherwise it abstains.
- A rule that rejects an unresolved claim containing a mark word.
- v4 for claims with no mark word.

Phase 8 reported 0/18 + 0/8 false acceptances and FRR 30.8% on dev (tuned) and 37.5% on held-out. Its labels were all
assigned by the assistant.

## The package: `eval/phase9b_human_review/`

| File | Content |
|---|---|
| `review_instructions.md` | Who may review, the label scope, the allowed labels and evidence bases, how to record disagreements, and what not to do |
| `development_review.csv` | 31 cases |
| `heldout_review.csv` | 16 cases |
| `table_evidence.md` | Per table: the full LaTeX environment, the c4 extraction and the PDF page |
| `pdf_pages/` | 6 rendered pages: HNET p21; MOSHI p22, p23, p25, p37, p40 |
| `disagreement_log.csv` | Header only |
| `review_status.md` | Awaiting review; preparer flags; one open decision |

Each case row has 26 fields from the package and 7 empty reviewer fields.

**Package fields:**

- claim and category;
- paper, table, LaTeX label and caption;
- cited c4 chunk;
- row number, row header and the full extracted row;
- column;
- the cell as extracted;
- the original LaTeX cell, with file, line and `\multirow` origin;
- the paper's mark macros and a note on the table;
- PDF page and image;
- current label, rationale and author;
- preparer flags.

The builder (`experiments/phase9b/build_review_package.py`) locates every case mechanically:

- **Row:** the chunk row that holds the Phase 8 locator strings, matched to exactly one table row.
- **Table source:** the LaTeX `table` environment holding the label. Its data rows must equal the extracted row count, and
  they do for all six tables.
- **Cell:** the column's cell in that row. If the cell is empty, the builder follows a `\multirow` from the rows above.
- **PDF page:** the page holding the caption "Table N:".

The columns were assigned by the assistant in `experiments/phase9b/review_targets.json`. Labels and claims are copied
unchanged, and a test checks this.

Two builder bugs were found while checking the output and fixed before the package was frozen in its current form:

- **Wrong HNET page:** the page search matched the text reference "Table 6." on page 1. It now matches only the
  caption "Table 6:", which is on page 21.
- **Wrong MOSHI T3 cell:** the first-column cell was read as the row header, so s24 showed `(empty)`. It now shows
  `(blank)`, and its LaTeX is `\xmark`.

## Finding: MOSHI's cross mark prints nothing

MOSHI's `main.tex` line 44 is `\newcommand{\xmark}{}`. HNET defines `\xmark` as `\ding{55}`, a visible ✗.

So in MOSHI T3, T4 and T6, every cell the authors wrote as a cross is **blank on the page**. The PDF page image confirms
this for T3. c4 keeps the source distinction:

- `\xmark` becomes `(blank)`;
- a truly empty LaTeX cell becomes `''`.

On the page, the two are indistinguishable.

This matters for the labels:

- Fourteen of the 47 cases are about a MOSHI `\xmark` cell: s13–s17, s19, s21–s24, h06, h08, h09 and h11.
- In T3, an ablation table, the authors' notation is `\checkmark` for a component used and `\xmark` for one left out.
  The 5.2 prose reads the columns that way: "improvement in MUSHRA when adding a Transformer in the decoder".
- In T4, prose states "RVQGAN and SemantiCodec are non-causal, while Mimi is fully causal".

Phase 8's rules forbid two readings:

- a blank read as "no";
- a check read as a property.

Under those rules, five claims are labelled unsupported that the paper's prose or notation arguably supports: s10, s14,
s23, s29 and h15. They are flagged for the reviewer; their labels are unchanged.

Phase 8's verifier notes also said the T4 blank "is never explained". That is true of SpeechTokenizer only; the prose
does explain RVQGAN and SemantiCodec.

## Distribution of the cases

The Phase 9B brief asks for categories with no cases in the current set:

- **Wrong table.** No claim cites one table while describing another.
- **Ambiguous labels.** No case is labelled `ambiguous`.
- **Multi-row matches as a category.** Rows that match several candidates (s26–s28, h01, h12) are only labelled
  supported or unsupported.
- **Multi-section evidence.**

The review instructions add an `ambiguous` label. These categories belong in the adversarial set (§9 of the brief),
which the brief schedules for after the freeze.

| Category | Dev | Held-out |
|---|---|---|
| Correct check or cross | 8 | 6 |
| Wrong mark | 2 | 3 |
| Wrong row, column or configuration | 8 | 3 |
| Blank described as blank, empty or unspecified | 4 | 2 |
| Blank interpreted as false, zero or true | 5 | 0 |
| Mark interpreted as a property | 1 | 1 |
| Unsupported comparison | 2 | 0 |
| Caption-supported interpretation | 1 | 0 |
| Mark claimed on a text cell | 0 | 1 |

## The evaluation code is validated (provisional labels, development set only)

`experiments/phase9b/evaluate_verifiers.py` runs the variants A, B1, B2, C and D.

- C is C2 alone, and its abstentions are scored separately.
- D is C2g.
- It loads the Phase 8 checker functions unchanged from `table_semantics.py`, through the syntax tree, so its top-level
  run is not executed.

The infrastructure check ran on the **development** set with the unreviewed Phase 8 labels:

```
uv run python experiments/phase9b/evaluate_verifiers.py --provisional --split development \
  --check-phase8 experiments/phase8/out/table_semantics_2026-11-13T21-09-22Z.json
```

- **Exit code:** 0.
- **Output:** `experiments/phase9b/out/verifiers_provisional_development_2026-11-14T00-04-55Z.json`.
- **Result:** for all 31 cases and every variant, the accept/reject decision, and for C the abstain decision, is
  **identical** to the stored Phase 8 run.

These are Phase 8's own numbers, reproduced. They are not a new or reviewed result.

The held-out set was **not** run. The code refuses it in provisional mode, and in frozen mode it refuses while the
manifest is not frozen. That was tested and also confirmed by running it: "benchmark not frozen (status:
awaiting_independent_review)".

## The freeze procedure

`evaluate_verifiers.py freeze` writes `experiments/phase9b/benchmark/{development,heldout}_v1.json`, records their
SHA-256 hashes in the manifest, and marks it frozen. It refuses when any of these holds:

- a case lacks a reviewer label, ID or date;
- the reviewer is the assistant;
- a disagreement has no adjudicated label;
- the version is already frozen;
- a benchmark file already exists.

It validates both splits before writing anything; a test caught an earlier version that left a partial file. Every case
keeps its Phase 8 label as `label_phase8`, along with its reviewer and date. A frozen run checks the hash of each
benchmark file before loading it.

## Tests

`uv run pytest -q experiments/phase9b`: 15 passed. They cover:

- scoring, with abstention kept separate, plus the ambiguous-case counts;
- the C2g outcome lists;
- the provisional held-out refusal;
- the freeze gate: incomplete review, self-review, unadjudicated disagreement, refreeze, tampering;
- the Phase 8 loader;
- `\multirow` and empty-cell resolution;
- labels and claims unchanged, and reviewer columns empty;
- the macro definitions;
- no overwrite of filled-in review files.
