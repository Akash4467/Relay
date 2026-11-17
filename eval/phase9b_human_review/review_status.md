# Phase 9B review status (internal: do not give this file to reviewers)

**Status: AWAITING INDEPENDENT REVIEW.**

- The two-level protocol (v2) is ready as of 2026-11-14.
- No case has been reviewed.
- The benchmark is not frozen: `experiments/phase9b/benchmark_manifest.json` has status `awaiting_independent_review`.

## Folder layout

| Path | Who sees it | Content |
|---|---|---|
| `reviewer/` | **reviewers** | blind templates, the instructions, `table_evidence.md` and the PDF page images. No labels, categories, rationales or preparer flags |
| `reviews/<reviewer_id>/` | reviewers write here | `development.csv`, `heldout.csv` (filled-in copies of the templates) |
| `development_review.csv`, `heldout_review.csv` | internal | **the original 47 cases** with the Phase 8 labels, rationales and preparer flags. Unchanged: their hashes match `pre_review_sha256_v1`. Their old single-level `reviewer_*` columns are no longer used |
| `disagreement_log.csv` | adjudicator | written by the harness, filled in by the adjudicator |
| `pdf_pages/` | internal | the images referenced by the original CSVs (copied into `reviewer/`) |

## Procedure

1. **Before review starts:** add the ID of anyone who wrote or labelled the cases to `excluded_reviewer_ids` in the
   manifest. "assistant", "claude", "self", "developer" and "case-author" are always excluded.
2. **Each reviewer:**
   - gets only `reviewer/` and the two papers;
   - saves `reviews/<id>/development.csv` and `heldout.csv`;
   - records, for every case: a table-only outcome, a whole-paper outcome, evidence, page or section, whether a
     legend was needed, confidence, explanation, ID and date.
3. **Disagreements:** run `uv run python experiments/phase9b/evaluate_verifiers.py disagreements`. It validates every
   review file and refuses any that is:
   - incomplete;
   - by an excluded reviewer;
   - missing a case or mixing cases from the two splits;
   - carrying an edited claim;
   - using an invalid outcome or an inconsistent pair of levels.

   It then writes one row per disagreement:
   - `between_reviewers`, at either level;
   - `reviewer_vs_original`, at the table-only level against the Phase 8 label.

   Each row keeps every assessment, and every reviewer's evidence and explanation. Its working outcome is `AMBIGUOUS`
   and its status is `unresolved`.
4. **Adjudication:**
   - Someone not excluded fills in `adjudicated_outcome`, `adjudicator_id`, `adjudication_date`, `adjudication_reason`
     and `status=adjudicated`. `AMBIGUOUS` is a valid adjudicated outcome.
   - There is no majority vote: even a 2-to-1 split needs adjudication.
   - Re-running step 3 keeps adjudications. It refuses if the reviews behind an adjudicated row have changed.
5. **Freeze:** run `uv run python experiments/phase9b/evaluate_verifiers.py freeze`. It refuses unless:
   - all reviews are valid;
   - the log matches the reviews exactly;
   - every disagreement is adjudicated.

   It writes `experiments/phase9b/benchmark/{development,heldout}_v1.json`. Each case keeps:
   - its Phase 8 label;
   - every review;
   - the final table-only and whole-paper outcomes.

   It hashes them and marks the manifest frozen. A later version (v2, …) keeps v1 in `previous_versions`. A benchmark
   file is never overwritten.
6. **Evaluation**, only after the freeze and with your approval:
   `evaluate_verifiers.py --split development`, then `--split heldout`.
   - Each run checks the file hash and the split membership.
   - It scores both levels: table-only as `label`, whole-paper as `label_paper`, where `SUPPORTED_ELSEWHERE` counts as
     supported.

## Cases the preparer flagged (internal; not shown to reviewers; labels unchanged)

| Case | Original label | Why the label rests on the Phase 8 rule rather than the paper |
|---|---|---|
| s10 | unsupported | HNET ✗, printed visibly, in a capability table; the caption names H-Net as the first with dynamic multi-stage hierarchies |
| s14 | unsupported | 5.2 prose says RVQGAN is non-causal; the cell is `\xmark`, which prints blank |
| s23 | unsupported | ablation table: `\checkmark` marks an included component, `\xmark` (printed blank) one left out |
| s29 | unsupported | the caption treats deduplication as a training condition |
| h15 | unsupported | ✓ under causal; 5.2 prose says Mimi is fully causal |
| s13, s16, s22, h06, h08 | supported | "blank" is true of the page; the source is the authors' cross macro |

Under v2 these should come out as table-only UNSUPPORTED or AMBIGUOUS and whole-paper SUPPORTED_ELSEWHERE where the
prose decides. That is a prediction by the case author, to be checked, not a label.

## Change log

- 2026-11-14: v1 package. Single-level, with labels visible in the review CSVs.
- 2026-11-14: protocol v2.
  - Blind reviewer folder.
  - Two evidence levels with four outcomes.
  - Multi-reviewer disagreement log with mandatory adjudication.
  - Harness safeguards hardened.
  - The original 47 cases are byte-identical to v1.
