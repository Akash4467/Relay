# Phase 11F: report integrity and metric reconciliation

This is an audit; nothing in production changed. The Phase 11E report (`experiments/phase11e/phase11e_report.md`) is kept **unchanged** as the original record, and its corrections are listed in §7.

**Source of truth:** the raw artifacts and re-runs, not the prose.

**Evidence types used here:**
- **[R] directly reproduced in Phase 11F:** retrieval re-run and metrics recomputed with independent code; tests re-run.
- **[P] previously recorded:** taken from an earlier artifact, not re-measured.
- **[L] provisional or self-authored labels.**
- **[M] mocked:** a scripted provider with real retrieval and verification.
- **Real-model results:** none in Phase 11E or 11F.

## 1. Were the Phase 11E metrics correctly attributed? Yes.

`reconcile.py` [R] re-ran production retrieval for all 74 questions on each corpus:
- Configuration G from config, setting B;
- corpus paths through `relay.corpus` (each validated against its manifest);
- gold evidence matched by **new code** (`holds()`/`units_for()`, not `retrieval_lab.score`), using the unchanged gold definitions.

**All 148 per-question rows match Phase 11E exactly:** final evidence IDs, gold ranks, coverage and completeness. The table-set values .968/.767 belong to **c3** and .806/.667 to **c6**.

## 2. Recalculated metrics [R]

Unit-level metrics are computed per gold unit: one per evidence quote, or one per gold row.

| Set (labels) | Corpus | Recall@5 | Recall@8 | MRR | Gold units found in context | Gold units complete in context | Questions with complete gold |
|---|---|---|---|---|---|---|---|
| Original, 15 answerable (project labels, not independently reviewed) | c3 | 18/34 = .529 | 22/34 = .647 | .597 | 22/34 = .647 | — | 8/15 |
| | c6 | 18/34 = .529 | 22/34 = .647 | .596 | 22/34 = .647 | — | 8/15 |
| Original, non-table subset (12) | c3 / c6 | 16/27 = .593 | 16/27 = .593 | .565 / .565 | 16/27 | — | 6/12 |
| Table, 24 answerable **[L]** | c3 | 30/31 = .968 | 31/31 = 1.000 | .767 | 31/31 = 1.000 | see note | **3/24** |
| | c6 | 25/31 = .806 | 27/31 = .871 | .667 | 27/31 = .871 | see note | **22/24** |
| New, 10 answerable **[L]** | c3 / c6 | 11/12 = .917 | 11/12 = .917 | .562 / .562 | 11/12 = .917 | — | 9/10 |

- **Rounding:** three decimals throughout. The exact numerators and denominators are in `metric_reconciliation.json`.
- **"Complete" for quotes** is the same as "found".
- **Interpretation:**
  - For the table set, *found* means a row line containing the row's identifying strings. *Complete* means it also contains the cells that answer the question.
  - **c3's high table recall counts rows whose answering cells are missing** (c3 dropped the ✓/✗/blank marks). It overstates usable evidence.
  - The usable-evidence measure is complete gold: 3/24 on c3 against 22/24 on c6.
- **Same inputs for both corpora:**
  - the same question files, gold definitions, procedure and limits;
  - no case excluded;
  - the 25 unanswerable cases scored separately (§3).

## 3. Per question (`per_question_comparison.json`) [R]

**Original set (15 answerable):**
- Complete evidence and found units are **identical** for all 15.
- **q10's first gold rank is worse on c6.** It's the only rank change, giving MRR .597 against .596.
- Final evidence is identical for 12. It **differs for q04, q05 and q10**, without changing any gold outcome.

**Table set (24 answerable) [L]:**
- **Complete evidence:**
  - gained on c6: t01, t02, t03, t05, t06, t07, t11, t13, t14, t15, t17, t19, t21, t22, t23, t25, t28, t30, t32 (19);
  - lost: none;
  - same: t09, t10, t18, t26, t29.
- **Found rows:** fewer on c6 for t10 and t29; the same for the other 22.
- **First gold rank:**
  - better on c6: t03, t17, t23, t25;
  - worse on c6: t01, t06, t10, t18, t29, t30, t32;
  - same: the other 13.
- **Final evidence** is identical only for t13, t14, t15 and t19.

**New cases (10 answerable) [L]:** identical on every measure, including the final evidence.

**Unanswerable:**
- **Table (t04, t08, t12, t16, t20, t24, t27, t31) and new (n11, n12):** the target table is in context on both corpora, the same everywhere.
- **The 15 original unanswerable questions have no target table**, so retrieval doesn't score them. Only q18, q20 and q30 were exercised, in the mocked runs [M], and they were refused on both corpora.

**Excluded for unknown provenance:** none. No evaluated question depends on the 7 unknown-provenance artifacts (§6).

## 4. q11, t10, t29 (`q11_t10_t29_audit.json`) [R]

Production retrieval with Configuration G, setting B. The verifier is v4.2 with the real NLI model and a 0.9 threshold, run on a scripted draft that restates the gold answer and cites the chunks holding the gold rows.

| Case | Corpus | Gold chunks: fused rank → re-ranked → in final context | All gold in context | Complete | Verifier on the gold-restating draft |
|---|---|---|---|---|---|
| **q11**: "…single attention head… base configuration…" (retrieval query adds `[h = 1]`) | c3 | `AIAYN:T3:r02` (h=1, BLEU 24.9): 3 → 6 → yes; `AIAYN:T3:r01` (base, 25.8): 5 → 9 → **no** | no | no | rejected: "single attention head … 24.9" scores 0.79 against r02 (< 0.9); r01 "was not retrieved" |
| | c6 | identical | no | no | identical |
| **t10**: Table 4 rows with ✓ in the causal column | c3 | `MOSHI:T4:r03`: 8 → 2 → yes; `MOSHI:T4:r04`: 7 → 1 → yes | rows yes | **no**: c3's rows have no causal cell | rejected: entailment 0.30 |
| | c6 | `MOSHI:T4:r03`, `r04`: not in the fused top 20 | **no** | no | rejected: cited chunks "not retrieved" |
| **t29**: Table 17 row with ABX 31.3% | c3 | `MOSHI:T17:r01`: 1 → 2 → yes | row yes | **no**: c3's row lacks the marked cells | rejected: entailment 0.31 |
| | c6 | `MOSHI:T17:r01`: 5 → 10 → **no** | no | no | rejected: "not retrieved" |

**None of the three is fixed on either corpus.**
- **q11's root cause:** the cross-encoder ranks r01 at 9, and only 8 passages are kept.
- **A new finding:** the verifier also doesn't link "single attention head" to `h = 1`. The Phase 11C mocked draft passed only because it wrote "(h = 1)" explicitly.

## 5. Tests (`test_results.json`) [R]

`run_tests.py` ran every command below and recorded its exit status and the counts printed in its own output:

| Command | cwd | Exit | Result |
|---|---|---|---|
| `uv run pytest -q experiments/phase11e` | . | 0 | 11 passed |
| `uv run pytest -q experiments/phase11` | . | 0 | 24 passed |
| `uv run pytest -q experiments/phase11b` | . | 0 | 12 passed, 1 warning |
| `uv run pytest -q experiments/phase11c` | . | 0 | 6 passed |
| `uv run pytest -q experiments/phase10` | . | 0 | 31 passed |
| `uv run pytest -q experiments/phase9b` | . | 0 | 63 passed (Phase 9B data only read) |
| `uv run pytest -q tests/unit tests/integration` | . | 0 | 493 passed |
| `uv run python experiments/phase10/offline_integration.py` | . | 0 | 31 of 31 scenarios [M] |
| `go vet ./...` | standin | 0 | no findings |
| `go test -count=1 ./...` | standin | 0 | 1 package ok, 1 without tests |
| `go vet ./...` | loadgen | 0 | no findings |
| `go test -count=1 ./...` | loadgen | 0 | 1 package ok, 1 without tests |

These counts match the ones Phase 11E reported: its "94" was 31 + 63, and its "42" was 24 + 12 + 6.

The mocked per-corpus regressions (`experiments/phase11e/c3_regression.json`, `c6_regression.json`) are [P]+[M]: recorded in Phase 11E, not re-run here. Their pass status is re-checked by `experiments/phase11e/test_phase11e.py` above.

## 6. Fixture provenance (`fixture_provenance_audit.json`) [R]

Every classification was checked against the artifact itself, giving **15 recorded, 9 established and 7 unknown, all verified**:
- **Unknown (7):** no record carries any corpus field. They are excluded from comparisons.
  - `fixtures/perf/q05.json`, `q11.json`, `q16.json`;
  - `results/raw/real/2026-11-12.jsonl` and `2026-11-13.jsonl`;
  - `results/raw/e2e/2026-11-12.jsonl`;
  - `results/raw/traces/2026-11-12.jsonl`.
- **Established (9):** no corpus field, produced by code that read `data/processed` + `data/index`, and c3 is unchanged since the Phase 9A hash record (verified).
- **Recorded (15):** the stated corpus is present in each (details per artifact in the JSON).
- **Fixtures:** unchanged (sha256 recorded), none relabelled, none created for c6. The two fixture citations (`HNET:2.2.1:c023`, `AIAYN:T3:r02`) have identical chunk text in c3 and c6.

## 7. Corrections to the Phase 11E report

The original is preserved unchanged. Each correction below rests on the evidence cited.

| # | Phase 11E said | Correct statement | Evidence |
|---|---|---|---|
| 1 | Original set: "identical" (the chat summary) | Complete evidence and coverage are identical. **q10's first gold rank is worse on c6** (MRR .597 against .596), and the final evidence differs for q04, q05 and q10 | `per_question_comparison.json` |
| 2 | Table set: listed only rank losses | Rank is **better** on c6 for t03, t17, t23, t25 and worse for t01, t06, t10, t18, t29, t30, t32 | same |
| 3 | "The 15 original … unanswerable questions behave the same on both corpora" | Not measured at retrieval level (no target table). Only q18, q20 and q30 were exercised, mocked, refused on both | `metric_reconciliation.json`; `c*_regression.json` |
| 4 | Inventory evidence: answer-quality run has "corpus_version = c3 in every record" | Every **question/run** record (16) has c3; the 16 `started`/`end` bookkeeping records have no corpus field. Classification unchanged | `fixture_provenance_audit.json` |
| 5 | Inventory evidence for phase 11C outputs: "every row carries its corpus" | True for `compare_corpora` and `e2e_offline`. `corpus_audit_*` is a paired c3/c6 diff with explicit c3 and c6 values, not per-row corpus. The held-out phase 8 note says "real c4 rows", not "corpus c4". Classifications unchanged | same |
| 6 | "…cite HNET:2.2.1:c023 and AIAYN:T3:r02, which exist in both c3 and c6 (although AIAYN:T3:r02 names the same rows in both)" | Both chunks exist in both corpora with **identical text** | same |
| 7 | Table recall "lower on c6" presented without a caveat | c3's table recall (30/31) and coverage (31/31) count rows **without** their answering cells. Complete gold (3/24 against 22/24) is the usable-evidence measure | §2 |
| 8 | (not stated) | q11: even with r02 in context, the verifier rejects "single attention head … 24.9" (0.79) | §4 |

The latency figures (714.7 ms against 706.6 ms) and the mean token counts are [P]. They were measured once in Phase 11E and are not re-measured here; latency varies with machine load.

## 8. Remaining limitations

- **Labels:** the table set and the new cases are [L] (assistant-made, no independent review). The original set's gold quotes were located by the assistant.
- **Mocked only:** all integration results are [M]. There is **no real-model evaluation** of c6, Configuration G, v4.2 or p4.
- **Unresolved cases:** q11 (base row), t10 and t29.
- **Lower first-gold rank on c6:** 7 table questions and q10.
- **The verifier misses paraphrases of symbolic headers** (q11 wording).
- **Chunk ID meaning:** some table chunk IDs name different rows in c3 and c6 (MOSHI T3, T13, T16, T17; HNET T6).
- **Fixture provenance:** the 3 stand-in fixtures have unknown corpus provenance.

## 9–10. State and boundaries

- **c3 is still active and unchanged:**
  - `config/app.yaml` still has `corpus.active: c3` and `corpus_version: c3`;
  - c3 matches the Phase 9A hash record;
  - c6 matches its manifest.
- **Not done:** no Groq or API calls, no deployment, no commit, no push. HEAD is still `4688baf`.
- **Not modified:** fixtures, Phase 9B data, corpora and production code.

## 11. Ready for a controlled live-model evaluation?

**Technically yes, for designing one.**
- The offline results are reproducible.
- Metric direction and attribution are unambiguous.
- The integrity checks and fail-closed startup work for both corpora.
- The open questions (does c6's complete table evidence help real answers? the cost of q10 and the 7 table rank losses? q11 wording) can only be answered with real model output.

**Not yet authorisable.**
- The security precondition from Phase 10 is still unmet: no full scan, and the key hasn't been rotated after the trojan.
- A live run also needs your explicit approval of corpus, question set and call/token budget. c6 can be selected for that run in memory, without activating it.
