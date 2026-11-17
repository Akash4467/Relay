# Phase 11E: c6 fixture hygiene and offline regression

**Outcome:**
- c3 and c6 each pass their own **mocked offline regression**: scripted provider replies, with real retrieval and verification.
- Fixture provenance is explicit.
- c3 remains the active default and is unchanged. c6 is **not** activated.
- No Groq or API calls, deployment, commit or push. HEAD is still `4688baf`.

Mocked or stand-in results are not model performance.

## 1. Fixture inventory and provenance (`fixture_inventory.json`)

31 corpus-dependent artifacts, built by `fixture_inventory.py`, which only reads them:

| Provenance | Count | Artifacts |
|---|---|---|
| recorded (the artifact states its corpus) | 15 | answer-quality real run 2026-11-13 (`c3`); phase 10 live run 2 (`c3`); phase 11A ablations (`c6`); phase 11B/11C outputs (`c3` and `c6`, separate sections or rows); `eval/phase7/table_question_set.json` (written against `c4`); `eval/questions.jsonl` (verbatim quotes, c3-era; q09's quote matches only c3's unit-less text); `experiments/phase11/new_cases.json` (`c6`); the phase 8/9B claim cases (`c4`) |
| established (not stated, but provable) | 9 | phase 10 live run 1 and the phase 10 offline integration outputs: they loaded `data/processed` + `data/index`, whose files equal the c3 hashes recorded at the start of phase 9A and are unchanged (checked) |
| **unknown** | **7** | the 3 stand-in fixtures `fixtures/perf/q05.json`, `q11.json`, `q16.json` (they declare `prompt_version p1` and a 2026-11-12 recording time, but no corpus); `results/raw/real/2026-11-12.jsonl`, `results/raw/real/2026-11-13.jsonl`, `results/raw/e2e/2026-11-12.jsonl`, `results/raw/traces/2026-11-12.jsonl` (no corpus field) |

- **Unknown artifacts are excluded from all corpus-specific comparisons.** `select_for_comparison()` raises an error for unknown or mismatched provenance (tested).
- **Labels were not inferred.** The DESIGN change log introduces c2 on 2026-11-13, after the fixtures were recorded, but the artifacts themselves don't say which corpus was used.
- **Nothing was changed:** no fixture was relabelled, moved or modified (the `fixtures/` directory matches HEAD).

**c6-specific fixtures: none.** `scripts/record_fixtures.py` only converts **real** model answers into stand-in fixtures, and recording c6 answers needs live Groq calls, which aren't authorised. So nothing was re-recorded and no fixture was invented. The two fixtures with citations cite `HNET:2.2.1:c023` and `AIAYN:T3:r02`, which exist in both c3 and c6 (although `AIAYN:T3:r02` names the same rows in both).

## 2. Offline regression per corpus (`c3_regression.json`, `c6_regression.json`)

**Command:** `uv run python experiments/phase11e/regression.py c3` and `… c6`.

**How the corpus is selected:** through the production startup path. `main.State()` reads a config identical to `config/app.yaml` except `corpus.active`/`corpus_version`, held in memory only. The file on disk stayed `c3` throughout.

| Check | c3 | c6 |
|---|---|---|
| Startup: selected corpus, configured manifest/processed/index paths, 18 recorded hashes verified, indexes = selected corpus's chunks | pass | pass |
| Configuration G on; v4.2 (threshold 0.9); prompt p4 | pass | pass |
| Phase 10 offline suite (`offline_integration.py` SCENARIOS) | **31/31** | **31/31** |
| Phase 11C question scenarios (`e2e_offline.py` SCENARIOS) | **15/15** | **15/15** |
| Corpus identity: q09 evidence contains "230 ms" | no (expected) | yes (expected) |
| Fail closed (scratch copies only): corrupted index file, missing manifest, wrong `corpus_version` → startup refused; intact copy (control) starts | pass | pass |
| No fallback to c3 when the selected corpus fails | n/a | pass |

**Checked on every one of the 15 scenarios, for both corpora:**
- the trace records the selected `corpus_version`;
- caption sharing is on;
- the prompt starts with the **original** question, with no expansion terms;
- the verifier received exactly the retrieved chunks, and each equals the **selected** corpus's text (caption sharing changed nothing in the evidence);
- any answer is a verified draft.

**Refused on both corpora:**
- the q18/q20/q30 declines;
- fabricated answers;
- q12's false absence claim;
- invalid citations and unretrieved citations.

## 3. c3 vs c6 (`corpus_comparison.json`)

**Command:** `uv run python experiments/phase11e/compare.py`.

**Procedure:** identical for both corpora, with paths from `relay.corpus`:
- production `search()` with Configuration G, setting B;
- the Phase 11A metrics, unchanged.

| Set | Corpus | Recall@5 | Recall@8 | MRR | Coverage | Complete gold in context |
|---|---|---|---|---|---|---|
| Original (15 answerable; project labels, not independently reviewed) | c3 | .529 | .647 | .597 | .647 | 8 |
| | c6 | .529 | .647 | .596 | .647 | 8 |
| Table (24 answerable; **PROVISIONAL**) | c3 | .968 | 1.000 | .767 | 1.000 | 3 |
| | c6 | .806 | .871 | .667 | .871 | **22** |
| New (10 answerable; **PROVISIONAL**) | c3 | .917 | .917 | .562 | .917 | 9 |
| | c6 | .917 | .917 | .562 | .917 | 9 |

- **Complete evidence** (the row plus the cells that answer it): c6 gains it for 19 table questions and **loses it for none**.
- **Lower row ranking** on c6 for some table questions. That comes from longer row lines: tables split into more chunks, and the dense embeddings shift. t10 and t29 lose their rows; t01, t06, t18, t30 and t32 only rank lower.
- **Unanswerable cases:** the 15 original, 8 table and 2 new unanswerable questions behave the same on both corpora (whether the target table reaches the context, and refusal of mocked declines or fabrications).
- **Citation and evidence outcomes (mocked):** identical on both corpora, except t10's refusal reason (c3: verification failed; c6: the scripted model declined, because no Table 4 row was retrieved).
- **Latency** (interleaved, retrieval only): median 714.7 ms on c3 and 706.6 ms on c6.
- **Context tokens:** mean 1,489.1 on c3 and 1,502.8 on c6; maximum 1,599 on both (budget 1,600).

## 4. Known limitations (checked against the retrieved evidence; none is fixed)

- **q11 (both corpora):**
  - The h=1 row `AIAYN:T3:r02` (BLEU 24.9) is fused rank 3, re-ranked 6th, and **in context**.
  - The base row `AIAYN:T3:r01` (BLEU 25.8) is fused rank 5, re-ranked 9th, and **not in context**.
  - A draft citing r01 is refused as not retrieved.
- **t10:**
  - c3 retrieves Table 4, but its rows lack the causal column.
  - c6's rows have it (`causal = ✓/(blank)`), but they aren't retrieved.
  - Unanswerable from evidence on both corpora.
- **t29:** Table 17 splits into 7 chunks in c6 (3 in c3), so the gold row's chunk ranks 10th. Not complete on either corpus.
- **Table chunk IDs that name different rows in c3 and c6:**
  - MOSHI T3, T13, T16, T17 and HNET T6. At 256 tokens, c6 adds `MOSHI:T3:r03`, `MOSHI:T13:r03`, `MOSHI:T16:r07`–`r08`, `MOSHI:T17:r04`–`r07`, `HNET:T6:r04`–`r05`; at 512, five more.
  - Prose chunk IDs are identical.
  - Citations recorded under one corpus must not be read against the other; traces now record `corpus_version` and `evidence_ids`.

## 5. Tests

| Command | Result |
|---|---|
| `uv run pytest -q experiments/phase11e` | 11 passed |
| `uv run pytest -q experiments/phase11 experiments/phase11b experiments/phase11c` | 42 passed |
| `uv run pytest -q experiments/phase10 experiments/phase9b` | 94 passed (Phase 9B data only read) |
| `uv run pytest -q tests/unit tests/integration` | 493 passed |
| `uv run python experiments/phase10/offline_integration.py` (default config, c3) | 31 of 31 |
| `go vet ./... && go test -count=1 ./...` in `standin/` and `loadgen/` | pass (exit 0) |

## 6. Files

- **Created:** `experiments/phase11e/`:
  - code: `fixture_inventory.py`, `regression.py`, `compare.py`, `test_phase11e.py`;
  - results: `fixture_inventory.json`, `c3_regression.json`, `c6_regression.json`, `corpus_comparison.json`, `phase11e_report.md`;
  - raw outputs under `out/`.
- **No production file, config, corpus, fixture or Phase 9B file was changed.**
  - c3 matches the hashes recorded at phase 9A.
  - c6 matches its manifest.
  - `config/app.yaml` still has `corpus.active: c3` and `corpus_version: c3`.

## 7. Remaining blockers before a separate c6 activation decision

- Your approval to change `corpus.active` and `corpus_version` to c6.
- **Stand-in fixtures:**
  - the 3 have unknown provenance;
  - c6 fixtures need real answers (live calls), or the stand-in has to stay documented as corpus-agnostic replay.
- **Provisional labels:** the table set and the new cases have no independent review.
- **Retrieval limitations:** q11's base row, t10, t29 and the lower table-row ranking.
- **No real-model evaluation** of c6 with Configuration G, v4.2 and p4.
- **Security:** a full scan and key rotation are still outstanding.
