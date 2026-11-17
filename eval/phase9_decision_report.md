# Phase 9A — Decision report: experimental corpus c5

Offline only:

- no Groq or API calls, no generation;
- no change to `relay/`, `config/`, `results/`, c3, c4, v4 or v5;
- no commits.

All new code and data are under `experiments/phase9/`; the reports are under `eval/`.

c5 = c4 + e3 extraction + Table 3 section fix (s1) + table-chunk metadata (k1), built at
`experiments/phase9/corpus/c5`. Its manifest records:

- source hashes, equal to c4's;
- extraction, section and chunking versions;
- piece sizes and overlap;
- the embedding configuration;
- code and output hashes;
- the build time, the parent (c4) and the control result.

Detailed reports:

- `phase9_extraction_report.md`
- `phase9_section_metadata_report.md`
- `phase9_chunking_report.md`
- `phase9_retrieval_report.md`
- `phase9_ablation_report.md`

## 1. Does e3 correct all known extraction defects?

**It corrects every known `\phantom` defect, each verified on its PDF page:**

- the six table cells: MOSHI T5 Perplexity 142.2/140.3/136.8 → 42.2/40.3/36.8, and T6 Weight 001.0 → 1.0 ×3;
- **one more found in this phase:** the "x" from three `\phantom{x}` colour swatches in MOSHI Table 18's caption.

Phase 8 had missed the Table 18 caption and wrongly reported no caption change.

Nothing else changes: no other cell, header, row header, caption, paragraph or section, and no visible digit is lost.

e3 does not address the table-section placement problem, which is a document-structure issue rather than text
extraction.

## 2. Does the Table 3 section correction improve retrieval?

**No.**

- **Alone:** no effect in B or C. In A it is slightly worse: probe a1 loses the correct-first ranking and its context, a3
  loses its context, and t07/t08 table MRR falls.
- **Together with metadata:** it costs A two T3/T17 correct-first probes and one complete table question (t06). It
  removes three competing-table cases.

Table 3 is semantically misassigned: it is printed among 5.1's pages, but only 5.2 cites it. The same float placement
puts **seven more MOSHI tables** (T2, T4–T9) one subsection early. They were not corrected, per the phase scope.

## 3. Does section-and-label metadata improve table identification?

**Yes, at a cost.**

- **Gains:**
  - correct table in context: 24/24 answerable table questions in every setting (c4: 21/23/24);
  - table MRR up in every setting;
  - T3/T17 correct-first: A 9 → 14, C 9 → 13;
  - wrong table in context: C 11 → 4.
- **Costs:**
  - B: T3/T17 correct-first 14 → 13 (t08);
  - each table chunk grows by about 46 tokens (89 of 146 table chunks now exceed 256 tokens);
  - twice as many top passages are dropped by the 1600-token budget;
  - two previously complete questions (t32 A, t30 B) lose their gold row;
  - original-set MRR falls in A and B, and q13 B loses a quote;
  - competing same-paper tables enter the context more often in A and B.

## 4. Does c5 outperform c4 across all three retrieval settings?

**On the provisional table set, yes, in all three:** questions complete A 18 → 21, B 19 → 22, C 19 → 24.

**On the original 15-question set, no:**

- any gold and complete gold are equal in all settings;
- MRR is lower in A (.602 → .569) and B (.582 → .550), and equal in C;
- quote recall is lower in B (20 → 19).

Negation and exclusion probes (a8, t31) fail in every setting, in both corpora.

## 5. Which questions improve? (c4 → c5)

- **t13, t14, t15:** complete in A, B and C (e3).
- **Metadata:**
  - A: t07 complete; t06 and t10 table in context.
  - B: t10 complete (3 rows).
  - C: t29, t30 complete.
  - Unanswerable t08 (C) and t31 (A) now retrieve their table.
- **T3/T17 probes:** a3, a4, t07 (A) and a5, a6, t29, t30 (C) now rank the correct table first.
- **Small MRR gains:** q03 A, q11 B.

## 6. Which questions regress?

- **t32 (A)** and **t30 (B):** complete → incomplete. The gold row chunk is dropped by the token budget (metadata).
- **q13 (B):** 2 → 1 of 3 quotes; HNET Table 2 chunks crowd out the prose.
- **q10 (A):** MRR 1.0 → 0.5. q10 (B) is still 0 of 3, with a context of five HNET Table 6 chunks.
- **q12 (B):** MRR 1.0 → 0.5; both gold chunks are still in the context.
- **t08 (B, unanswerable):** T3 leaves the context, and the probe ranks T17 first.
- **t04 (A):** table MRR 1.0 → 0.5.
- **Competing table newly in context:** A t13–t16; B t01, t13, t14, t23; C t11.

All regressions trace to the metadata line; none to e3.

## 7. Are the gains consistent across both datasets?

**No.** Every gain is on the table set and the T3/T17 probes. The original set is flat or slightly worse. The table set's
labels are still provisional (assistant-made, independent review pending), and the probes were written by the assistant
after inspecting the tables.

## 8. Which changes are worth considering for the next phase?

1. **e3 extraction: yes.**
   - It is a pure correctness fix, verified against the PDF page by page.
   - Its effect is isolated and additive: exactly t13–t15 in every setting, no regression anywhere.
   - Recommended as the candidate for a corpus rebuild, as e3 alone (c4 + e3), pending your approval.
2. **Table-chunk metadata: promising, not ready.** It is the only change that improves table identification. Its costs
   come from the fixed budget. The next isolated experiment is one change only: a shorter line (table ID, label and row
   range, without the repeated paper title), or re-splitting rows so that chunks keep the piece size. Each should be
   measured as above.
3. **Section correction s1: no.** It brings no retrieval benefit. A systematic, rule-based float-section fix for all
   MOSHI tables would be a separate structural change, worth it only if section fields matter elsewhere, such as in
   citations or answers.

## 9. What remains untested?

- **Answer quality:** no generation was run. Whether the model answers better from c5 evidence is untested, and no
  answer-quality claim is made.
- **Labels:**
  - Independent human review of the 32-question table set and of the T3/T17 probes is still pending.
  - There is no held-out probe set.
- **Retrieval behaviour:**
  - Negation and exclusion queries (for example, "without MUSHRA").
  - Context crowding by one multi-chunk table (q10 B, q13 B). Unchanged, and slightly worsened by larger chunks.
- **Coverage:**
  - Sections for T2 and T4–T9 are uncorrected.
  - The effect of metadata on AIAYN/HNET-only questions beyond these sets.
  - Context caps other than L0 2500 / budget 1600.
- **Verifier:** C2g (Phase 9B) and the interaction of c5 evidence with v4/v5 verification were not run.

## Validation and audit (exact commands, exit codes, counts)

| Command | Exit | Result |
|---|---|---|
| `uv run python experiments/phase9/extraction_e3.py` | 0 | all checks passed |
| `uv run python experiments/phase9/section_correction.py` | 0 | all checks passed |
| `uv run python experiments/phase9/build_c5.py` | 0 | control = c4 byte for byte (10 files); c5 built |
| `uv run python experiments/phase9/evaluate_retrieval.py` | 0 | `retrieval_eval_2026-11-13T22-17-05Z.json` |
| `uv run python experiments/phase9/ablation.py` | 0 | `ablation_2026-11-13T22-39-25Z.json`; c4/c5 reproduction true; memory = disk true |
| `uv run pytest -q experiments/phase9` | 0 | 28 passed |
| `uv run pytest -q experiments/phase8` | 0 | 15 passed |
| `uv run pytest -q tests/unit tests/integration` | 0 | 358 passed |
| `go vet ./...` and `go test -count=1 ./...` in `standin/` and `loadgen/` | 0, 0, 0, 0 | ok |
| `sha256sum -c experiments/phase9/out/baseline_checksums.sha256` | 0 | 258 files unchanged (relay, config, results, data/processed, data/index, data/corpus/c4, eval, experiments/phase8, scripts, tests, standin, loadgen) |
| c3 (18 files) and c4 (19 files) checksum lists from Phase 8 | 0, 0 | unchanged |

**Git:**

- HEAD is still `4688baf`.
- Compared with the start of the phase, `git status` adds only the six `eval/phase9_*.md` reports. `experiments/` was
  already untracked.
- No new files appear in any protected tree.

**Credential scan** of `experiments/phase9/**` and `eval/phase9_*`: no matches.

Phase 9A stops here. Nothing is integrated or committed. Phase 9B (C2g), 9C or any production change awaits your approval.
