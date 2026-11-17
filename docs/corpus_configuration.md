# Corpus configuration (phase 11D)

The corpus the app serves is chosen in `config/app.yaml`, not by fixed paths in code. `relay/corpus.py` reads the choice, resolves the paths and checks the corpus before anything is served.

## Schema

```yaml
corpus_version: c3                 # cache and trace key; must equal the active corpus's corpus_version (checked)
corpus:
  active: c3                       # the corpus that is served; default, and the rollback baseline
  corpora:
    c3:
      corpus_version: c3
      manifest_path: config/corpus_manifests/c3.json
      processed_data_path: data/processed
      index_path: data/index
    c6:                            # candidate: selectable, NOT active
      corpus_version: c6
      manifest_path: experiments/phase10/corpus/c6/manifest.json
      processed_data_path: experiments/phase10/corpus/c6/processed
      index_path: experiments/phase10/corpus/c6/index
```

- **Paths:** all relative to the repository root, so the working directory doesn't matter. The config directory itself is also resolved from the repository root (`relay/config.py`).
- **Manifests:**
  - c4, c5 and c6 carry the manifest written when they were built (`corpus_version` plus `outputs_sha256`, keyed relative to the manifest's directory).
  - c3 predates manifests. Its manifest, `config/corpus_manifests/c3.json`, was written in phase 11D by `experiments/phase11d/make_c3_manifest.py` from hashes **recorded earlier**, at the start of phases 9A and 9B (the two lists are identical). It sets `"root": "data"`, so its keys are relative to `data/`.

## Selecting a corpus

- **Serve c3 (default):** `corpus.active: c3` and `corpus_version: c3`.
- **Serve c6, only after approval:** `corpus.active: c6` and `corpus_version: c6`. Both lines must change together, or startup refuses.
- **Use a corpus in a tool or test without changing the config:** `corpus.selected(app_config(), "c6")`, then `corpus.validate(...)`, then `corpus.load_indexes(...)`.

These callers load the active corpus through `relay.corpus`:
- the server (`relay/main.py`);
- the answer-quality runner (`relay/eval/answer_quality.py`);
- live validation (`relay/live_validation.py`);
- the retrieval-recall tool (`relay/eval/retrieval_recall.py`).

The trace records `corpus_version`, and `/health` reports the corpus check.

## Startup integrity checks (`corpus.validate`)

These run in order. Any failure stops startup with a message saying what to fix, and nothing is loaded:

1. The corpus is defined and has all four fields. The global `corpus_version` equals the active corpus's.
2. The manifest exists and parses as JSON.
3. The manifest declares the expected `corpus_version`.
4. `processed_data_path` and `index_path` equal the manifest's layout (`<root>/processed`, `<root>/index`).
5. Every file the configured piece sizes need exists: `chunks_{size}.jsonl`, `bm25_{size}.pkl`, `faiss_{size}.index`, `faiss_{size}.ids.json`.
6. Every file listed in `outputs_sha256` exists and matches its recorded SHA-256. A manifest without hashes is refused.
7. Every required file is covered by the manifest's hashes.

**There is no fallback.** If the selected corpus fails, the app does not quietly serve c3 instead.

## Rollback to c3

1. Set `corpus.active: c3` and `corpus_version: c3`.
2. Restart.

c3 (`data/processed`, `data/index`) is never moved or modified, and its integrity is checked at every start.

## Automated validations

`tests/unit/test_corpus_config.py` covers:
- c3 is the default;
- c6 can be selected explicitly;
- both real corpora pass the integrity check;
- paths resolve the same from another working directory;
- c3 is unchanged after selecting c6;
- each failure is reported clearly: a missing or unreadable manifest, the wrong version, version disagreement, a missing index or processed file, a hash mismatch, a missing listed file, a manifest without hashes, inconsistent paths, an undefined corpus;
- a broken corpus never falls back to c3;
- no consumer constructs an index from default paths;
- server startup serves the configured corpus and refuses a broken one.

## Limitations

- **Startup time:** hashing every listed file takes well under a second for these corpora. Much larger corpora would need a cheaper check.
- **Historical verifier tools still read c3 directly** (`relay/eval/stress_verifier.py`, `verifier_v2_check.py`, `verifier_v3_eval.py`, `relay/candidate_v5/evaluate.py`). They reproduce frozen c3 experiments and are deliberately not switched.
- **Selection is the only change.** Activating c6 still needs approval, and the stand-in fixtures recorded under c3 must be re-recorded or marked as c3 (stand-in only).
