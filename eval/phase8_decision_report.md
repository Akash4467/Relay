# Phase 8 — Decision report: evidence semantics and retrieval stabilisation

Offline only. No Groq calls, no external inference, no change to `relay/`, c3, c4, production defaults or historical results.
v4 is frozen, with its 0.9 threshold. Nothing has been deployed or committed.

Detailed reports:

- `eval/phase8_extraction_report.md` (A)
- `eval/phase8_verifier_semantics_report.md` (B)
- `eval/phase8_table_retrieval_report.md` (C)
- `eval/phase8_context_report.md` (D)

All numbers come from `experiments/phase8/out/`.

## 1. Is the phantom extraction defect fixed?

**Yes, in the experimental extraction e3; it is not applied to any corpus.**

- `strip_phantoms` removes `\phantom`, `\hphantom` and `\vphantom` with their argument.
- It has 15 regression tests, all passing.
- Validated on all three papers: exactly 6 cells change, and each new value appears in the PDF text while no old one does.
  - MOSHI T5 Perplexity: 142.2/140.3/136.8 → 42.2/40.3/36.8.
  - MOSHI T6 Weight: 001.0 → 1.0 ×3. This defect was not previously known.
- No prose, caption or row-header changes.
- c4 still carries the wrong values. A versioned c5 rebuild needs approval.

## 2. Does textual symbol normalisation improve verifier reliability?

**No.**

| Representation | Dev FRR | Dev FAR | Held-out FAR |
|---|---|---|---|
| v4 on the original evidence | 46.2% | 11.1% | 25.0% |
| tokens (`[CHECKMARK]` …) | 23.1% | 16.7% | 0% |
| words (`checked/crossed/empty`) | 23.1% | 22.2% | 0% |

- Tokens halve false rejections but raise dev false acceptances. One new acceptance reads a blank as "causal" (s17).
- Words make blank-as-negative claims entailable: s14, s15, s23.
- v5 behaves identically to v4 on both representations.

## 3. Can the verifier distinguish checked, crossed and blank cells?

**v4 cannot.** It accepts "marked with a check" on ✗ cells (s04, s09, h04; entailment 0.98–0.99) and on a text cell (h16).

**The guarded deterministic checker C2g can, for literal mark claims it can resolve.** C2g is v4 plus a mark checker; an
unresolved mark claim is rejected, never passed to v4.

- False acceptances: 0 of 18 dev and 0 of 8 held-out unsupported claims.
- False rejections: 30.8% dev (tuned) and 37.5% held-out. They come from abstentions: the row is not unique from the claim,
  there are unexplained words, or the cell is not a mark.
- The unguarded fallback (C2+A) is unsafe: 12.5% held-out FAR.

## 4. Can retrieval distinguish MOSHI Tables 3 and 17?

**Partly, and only through the cross-encoder.**

- Correct table ranked first: 14/16 under B, 9/16 under A and C.
- Under B the gold table is always in the context, but the wrong table is also there in 13 of 16 queries.
- BM25 and dense, and therefore RRF, often favour T17 for T3 queries.
- Queries that mention MUSHRA in order to exclude it fail everywhere.
- Contributing defects:
  - T3 is indexed under the wrong section (5.1);
  - `short_caption` gives both tables the same caption text.

## 5. Does table-aware chunking improve retrieval without harming unrelated questions?

**No.**

- **whole_table (split only beyond 2 × piece size):**
  - Gains: duplicate table chunks fall (B: .453 → .203 on the table set), and q11 B and t10 are fixed.
  - Losses: q11 C, q13 A/B, t05 A, t30 B and t32 C regress, and T3/T17 discrimination drops (B 14 → 9).
- **Assembly limits** (max 2 or 1 chunk per table, table tokens ≤ 50%): each rescues one original question (q10 B, or q11 A)
  but removes the row chunk table questions need. Table completeness under B falls from 19 to 13, 8 and 15.
- **metadata line** (section + label on each table chunk): the only variant with a net gain on both sets, but small and mixed:
  - T3/T17 under C +4, table B +1 complete, original B +1 any-gold;
  - q13 A −1 quote, two T3/T17 losses, B MRR −.03.

## 6. Which candidate should proceed to the next phase?

**Recommended:**

1. **e3 extraction → a versioned c5 corpus.** This is a pure defect fix with a validated, collateral-free diff.

   The rebuild must:
   - re-run the Phase 7 checks;
   - correct t13–t15's expected values, which currently rely on `paper_cells`;
   - fix T3's section assignment, as a separate, separately-tested change.
2. **C2g mark checker as an experimental verifier candidate (v6-candidate), offline only.** It is the only variant with zero
   false acceptances on both case sets.

   Before any live use it needs:
   - independent human review of all 47 labels;
   - a larger held-out set;
   - an accepted cost of about 30–40% false rejections on mark claims, which become refusals.

**Not recommended:**

- token or word normalisation of marks;
- the unguarded C2+A;
- any per-table assembly cap;
- whole_table chunking.

**Optional:** the metadata variant, re-tested on c5 after the section fix, with held-out queries.

## 7. What remains unsafe or untested?

- **Labels and datasets:**
  - All 47 table-semantics labels, the 16 adversarial queries and the 32-question table set are assistant-made. They await
    independent review, so no final performance claim is made.
  - The original 15-question set and the table set are reported separately throughout.
- **No generation was run**, so nothing here shows how the model would answer from c4 or c5. Saved c3 answers were not used
  as evidence for c4.
- **v4 in production still accepts check claims on ✗ cells and on text cells.** This is a known false-acceptance path for
  table answers.
- **Blanks:** "blank" cells are never interpreted. A model answer that reads a blank as "no" is rejected by v4/C2g only when
  NLI disagrees. For blank interpretations, v4 rejected all five constructed cases, but that is a small sample.
- **Retrieval:**
  - Queries that use negation ("without MUSHRA") retrieve the wrong table.
  - Crowding by one multi-chunk table (q10 B) remains unsolved. A row-aware ranking inside a table was not tested.
- **Parser:** T3's section misassignment and MOSHI T6 `001.0` are present in c4.
- **Integrity, checked at the end of the phase:**
  - Checksums are unchanged: 49 protected files (including the `relay/verify/` sources, `config/app.yaml` and
    `results/results.csv`), the c3 corpus (18 files) and c4 (19 files).
  - No file under `relay/`, `config/` or `results/` was modified during the phase, and HEAD is still `4688baf`.
  - The only new git-visible files are the five reports; `experiments/` was already untracked.
  - Tests: `pytest experiments/phase8` 15 passed; `pytest tests/unit tests/integration` 358 passed; `go vet` and
    `go test` pass in standin and loadgen.
  - No credentials in the Phase 8 files.
  - The failed-patch duplicate output (`table_semantics_…21-03-16Z.json`) is kept.

Awaiting approval before any integration, corpus rebuild or live test.
