# Instructions for reviewers: table-mark claims (Phase 9B)

You will check 47 short claims about cells in six tables from two arXiv papers:

- **HNET:** arXiv 2507.07955v2, `data/raw/HNET/paper.pdf`.
- **MOSHI:** arXiv 2410.00037v1, `data/raw/MOSHI/paper.pdf`.

For each claim, record two independent judgements: what the **table alone** shows, and what the **whole paper** shows.

**Who may review:** anyone who did not write or label these cases. The case author cannot review them.

**Work only from this folder** (`reviewer/`) and the two papers. Do not open the other files in
`eval/phase9b_human_review/`, or anything under `experiments/`, until you have finished. They contain earlier labels and
verifier results, which would bias your judgement.

## What is in this folder

| File | Use |
|---|---|
| `template_development.csv` | 31 claims (s01–s31) |
| `template_heldout.csv` | 16 claims (h01–h16) |
| `table_evidence.md` | For each table: the full LaTeX source, the text extracted from it, and the PDF page |
| `pdf_pages/*.png` | The rendered PDF page that holds each table |

1. Copy both templates to `eval/phase9b_human_review/reviews/<your_id>/development.csv` and `.../heldout.csv`. Use your
   own short ID, such as your initials.
2. Fill in the empty columns.
3. Do not edit any other column, and keep the two files separate.

## Evidence columns (read only)

| Column | Meaning |
|---|---|
| `claim` | The statement to check |
| `paper`, `table_id`, `latex_label`, `caption` | The cited table |
| `pdf_page`, `pdf_image` | Where the table is printed. **The printed page is the primary evidence** |
| `row_number`, `row_header`, `row_cells_c4`, `column` | The row and column the claim is about, and the whole row as the pipeline extracted it |
| `cell_c4` | The cell as extracted. `(blank)` means a blank written with a mark macro; `(empty)` means an empty LaTeX cell |
| `cell_latex`, `cell_latex_source` | The cell as written in the LaTeX source, with its file and line. `\multirow` cells are shared by several rows |
| `printed_on_page` | What the PDF prints in that cell, derived from the paper's macro definitions. Check it against the image |
| `mark_macros`, `source_notes` | How the paper defines its mark commands, and where prose refers to the table. These are pointers only; read the prose yourself |

**Note on MOSHI.** MOSHI defines its cross command as empty: `\newcommand{\xmark}{}` in `main.tex`. A cell the authors
wrote as `\xmark` therefore prints nothing, exactly like a cell left empty. HNET's cross prints a visible ✗. Treat the
two papers separately.

## Your columns

### `table_only_outcome`

Use only what is visible in the cited table: caption, headers, the row and its cells.

- `SUPPORTED`: the table itself establishes the claim.
- `UNSUPPORTED`: the table does not establish it. This covers a wrong mark, row, column, table or configuration, a
  conclusion the table does not state, and a claim it contradicts.
- `AMBIGUOUS`: the table allows more than one reasonable reading, or the claim cannot be pinned to one row, column or
  meaning.

`SUPPORTED_ELSEWHERE` is not allowed at this level.

### `whole_paper_outcome`

Use the table plus prose, definitions, notation, other sections and figures anywhere in the paper.

- `SUPPORTED`: the table itself establishes it. This requires `table_only_outcome = SUPPORTED`.
- `SUPPORTED_ELSEWHERE`: the table alone does not establish it, but other parts of the paper do. Name them in
  `source_page_or_section`.
- `UNSUPPORTED`: nothing in the paper establishes it, or the paper contradicts it.
- `AMBIGUOUS`: the paper leaves it open.

**Do not let prose upgrade the table-only outcome.** If prose is what makes the claim true, the table-only outcome is
not `SUPPORTED`.

### The other columns

| Column | What to enter |
|---|---|
| `supporting_evidence` | What decided it: the cell you saw, the caption phrase, or the prose sentence (quote it briefly) |
| `source_page_or_section` | PDF page(s) and/or section numbers you used |
| `legend_or_definition_needed` | `yes` if deciding needed a legend, definition or convention for what a mark or blank means; otherwise `no` |
| `confidence` | `high`, `medium` or `low` |
| `explanation` | One or two sentences on why. Required for every case |
| `reviewer_id` | Your ID, the same as your folder name |
| `review_date` | In the form YYYY-MM-DD |

## Rules of reading

1. **A blank cell is only a blank.** Do not read it as false, zero, absent or "not X" unless the paper says what a blank
   means. If the paper says so elsewhere, that is whole-paper evidence, not table evidence.
2. **A mark is only a mark.** A ✓ under a column does not by itself mean the row "has" or "is" that property, unless the
   paper defines the mark. The same applies to ✗.
3. **Establish meaning from the paper, not from habit.** If you rely on a common convention that the paper never states,
   set `legend_or_definition_needed = yes` and say so in `explanation`.
4. **Choose `AMBIGUOUS` rather than forcing a decision.** Do not reword a claim to make it decidable.
5. Take extra care with the claims about causality (MOSHI Table 4) and about components in the ablation tables (MOSHI
   Tables 3 and 13). Record the table-only and whole-paper judgements separately.

## Independence declaration (required)

The review counts only if it is your own human judgement. **Do not use ChatGPT, Claude or any other AI tool** to
decide, draft or check outcomes, and do not start from anyone else's answers.

When you hand in, send the project owner this statement, completed and signed with your reviewer ID and date:

> I, `<reviewer_id>`, confirm that:
>
> 1. I did not write, label or help prepare any of the 47 cases.
> 2. I reviewed every case myself, without AI tools and without seeing any earlier labels, audits, reviews or verifier
>    results.
> 3. I reviewed the development and held-out files separately.
>
> Date: YYYY-MM-DD

The project owner records this in the benchmark manifest, together with the SHA-256 hash of your two files. Editing a
file after that invalidates the record.

## How to hand in

If you have access to the project, save your files as:

- `eval/phase9b_human_review/reviews/<reviewer_id>/development.csv`
- `eval/phase9b_human_review/reviews/<reviewer_id>/heldout.csv`

Otherwise, send the two files to the project owner, who saves them there unchanged.

`<reviewer_id>` must be the same in the folder name and in every row of both files. Use a plain ID such as your
initials. IDs containing "assistant", "claude", "chatgpt", "developer", "self", "model-assisted" or "not-independent"
are rejected.

## Before you hand in

Every case needs both outcomes, evidence, a source, `legend_or_definition_needed`, confidence, an explanation, your ID
and a date. The project's check rejects incomplete files, edited claims, and a reviewer ID that does not match the
folder name.
