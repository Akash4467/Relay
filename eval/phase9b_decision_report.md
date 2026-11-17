# Phase 9B — Decision report: table-mark verifier C2g

## Outcome A: independent review unavailable

The review package is complete and the evaluation code is validated. **The benchmark is not frozen, and no performance
claim can be made.** Per the stopping condition, this phase stops here.

Offline only:

- no Groq, API or external inference calls;
- no change to `relay/`, `config/`, `results/`, c3, c4, c5, v4 or v5;
- no historical file overwritten;
- no commits.

## What exists

- **Review package:** `eval/phase9b_human_review/`, with 31 development and 16 held-out cases, each traced to its row,
  column, LaTeX cell and PDF page. Details are in `eval/phase9b_review_report.md`.
- **Evaluation code:** `experiments/phase9b/evaluate_verifiers.py`.
  - Variants: A (frozen v4), B1/B2 (normalised evidence), C (C2 with separate abstention) and D (C2g).
  - Metrics: TA/FA/TR/FR, FAR, FRR (with and without abstentions), coverage, abstention rate, ambiguous-case safety,
    per category, per case, and the C2g outcome lists.
  - A freeze gate.
- **Manifest:** `experiments/phase9b/benchmark_manifest.json`, with status `awaiting_independent_review` and pre-review
  hashes.
- **Infrastructure check:** on the development set with the unreviewed Phase 8 labels, every per-case decision of every
  variant equals the Phase 8 run. This reproduces old numbers; it is not a new result.

**Not produced, by design:**

- `phase9b_verifier_comparison.md` and `phase9b_error_analysis.md` need the frozen, reviewed labels.
- `adversarial_cases.json` and `phase9b_adversarial_report.md`: the brief schedules the adversarial set for after the
  freeze.
- No held-out evaluation was run.

## Answers to the nine questions

**1. Does C2g reduce false acceptance on independently reviewed claims?**

Cannot be answered yet: there are no reviewed claims. Phase 8's 0/26 false acceptances rest on assistant labels.

The review package shows that some of those labels may move.

- Five unsupported labels are arguably supported by the paper's prose or notation: s10, s14, s23, s29 and h15.
- If a reviewer relabels them as supported, they do not become false acceptances. Instead, every variant's score on them
  becomes a correct or false **rejection**.

**2. How many supported claims does it reject?**

On Phase 8 labels it rejects 4 of 13 dev and 3 of 8 held-out supported claims. It has no reviewed count.

**3. Is its abstention behaviour reliable?**

Untested on reviewed labels. Structurally, its abstentions are reasons it states: the row is not unique, a word is
unexplained, the cell is not a mark, there is no column phrase. The benchmark has no `ambiguous` case to test safe
abstention against.

**4. Does normalised evidence help or hurt?**

There is no reviewed answer. One package finding bears on it: MOSHI's `\xmark` prints blank. So an "empty" token on
MOSHI cells hides the authors' cross. Neither representation tested in Phase 8 is faithful to both the page and the
source.

**5. Does the v4 fallback introduce false acceptance?**

Untested on reviewed labels. On the Phase 8 sets, D sends only claims without a mark word to v4: s10–s12, s14, s15,
s17, s23, s29 and h15.

Four of the five flagged labels sit in this fallback path: s10, s14, s23, s29 and h15 are all fallback cases. So the
review decides most directly how the fallback is scored.

**6. Which claim categories remain unsafe?**

Not established. The current set has no wrong-table, multi-section or `ambiguous` cases. Those belong in the
post-freeze adversarial set.

**7. Is the improvement consistent across development and held-out?**

Not established.

**8. Is C2g suitable for another research experiment?**

It is suitable for the **reviewed-benchmark evaluation**, which is Outcome B of this phase. That needs:

1. an independent reviewer completing `eval/phase9b_human_review/`;
2. your decision on the label scope (`review_status.md`);
3. adjudication, then the freeze.

No recommendation about Phase 9C is made before those results.

**9. What remains untested?**

- Every reviewed-label metric, on development and held-out.
- The adversarial set.
- Category-level safety for wrong-table, multi-section and ambiguous claims.
- C2g on real model answers rather than constructed claims.
- C2g on c5 evidence. All cases cite c4 chunks; the cited tables are T3, T4, T6, T13 and T16 in MOSHI and T6 in HNET.
  - c5 changes MOSHI T6's Weight cells (e3) and T3's section, and it adds a metadata line to every table chunk.
  - The Phase 8 case locators are unaffected, but the evidence text differs.
- Whether the MOSHI `\xmark`-as-blank rendering should be represented differently in the extraction. This is an
  extraction question, outside this phase.

## Validation and audit

| Command | Exit | Result |
|---|---|---|
| `uv run python experiments/phase9b/build_review_package.py` | 0 | 31 + 16 cases; 7 + 3 flagged |
| `uv run python experiments/phase9b/evaluate_verifiers.py --provisional --split development --check-phase8 experiments/phase8/out/table_semantics_2026-11-13T21-09-22Z.json` | 0 | reproduction identical, 31 cases × 5 variants |
| `uv run python experiments/phase9b/evaluate_verifiers.py --split heldout` | 1 | refused: benchmark not frozen (intended) |
| `uv run pytest -q experiments/phase9b` | 0 | 15 passed |

| `uv run pytest -q experiments/phase9 experiments/phase8` | 0 | 43 passed |
| `uv run pytest -q tests/unit tests/integration` | 0 | 358 passed |
| `go vet ./...` and `go test -count=1 ./...` in `standin/` and `loadgen/` | 0, 0, 0, 0 | ok |
| `sha256sum --quiet -c experiments/phase9b/out/baseline_checksums.sha256` | 0 | 310 files unchanged (relay, config, results, data/processed, data/index, data/corpus, eval, experiments/phase8, experiments/phase9 including c5, scripts, tests, standin, loadgen) |
| credential pattern scan of `experiments/phase9b`, `eval/phase9b_*` | 1 (no match) | no credentials |

**Git:**

- HEAD is still `4688baf`.
- Compared with the start of the phase, `git status` adds only `eval/phase9b_review_report.md`,
  `eval/phase9b_decision_report.md` and `eval/phase9b_human_review/`. `experiments/` was already untracked.

## Benchmark hashes (pre-review, not frozen)

These come from `benchmark_manifest.json` → `pre_review_sha256`. They are the hashes of:

- the Phase 8 case files;
- the Phase 8 script;
- `review_targets.json`;
- the builder;
- the four package files;
- c4 `chunks_256.jsonl` and `tables.jsonl`.

Phase 9B stops here. Nothing is integrated or committed. The next step needs an independent reviewer and your
confirmation of the label scope.
