# Phase 6: retrieval comparison, corpus c3 vs c4

Offline only; no model generation. Data: `experiments/phase6/out/retrieval_comparison_2026-11-13T20-27-24Z.json`;
query level: `eval/phase6_retrieval_comparison.csv` (one row per setting and question, c3 and c4 side by side).

## Corpus versions and configuration

| | c3 (production default) | c4 (candidate) |
|---|---|---|
| Location | `data/processed`, `data/index` | `data/corpus/c4/processed`, `data/corpus/c4/index` |
| Extraction | e1 | e2 (✓/✗ marks, blank marks, multirow spans, empty cells kept) |
| Chunks (256 / 512 tokens) | 455 / 255 | 465 / 260 |
| Everything else | identical: piece sizes, chunker, BAAI/bge-small-en-v1.5, BM25 + FAISS, RRF, cross-encoder, settings A/B/C, context cap 2500 (settings' 1,600-token budget applies) | |

**Control:** c3 was rebuilt from source with the same script and is byte-identical to the stored corpus, and the c3
numbers below reproduce the existing reports exactly (dev any-gold A 75.0 / B 75.0 / C 87.5; test A 100 / B 71.4 /
C 85.7). The comparison therefore isolates the extraction change.

## Method

- Questions: all 30 in `eval/questions.jsonl` (15 answerable: 8 dev, 7 test; 15 unanswerable). The test split was
  already re-measured once as a post-rebuild diagnostic; these test numbers are diagnostics too, not a new frozen result.
- Relevance: the only labels are the gold evidence quotes. A chunk is gold for a quote if it contains it (matched per
  corpus, so chunk-ID changes do not matter).
- Metrics: any/all gold and quote recall in the final context (existing definitions); Recall@1/5/10 and MRR over the
  20 ranked candidates. **Precision@5 and nDCG@10 use a proxy** (binary relevance = the chunk contains a gold quote);
  there are no graded relevance labels. Table-row success = gold quotes from tables reaching the context.
  False evidence = context chunks of HNET/MOSHI Table 6 that hold a 51/55 artifact.

## Results (all 15 answerable questions; * proxy relevance)

| Setting | Corpus | Any gold | All gold | Quote recall | MRR | R@5 | R@10 | P@5* | nDCG@10* | Table quotes in context | Artifact chunks in context | Missing evidence |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| A | c3 | 86.7% | 53.3% | 67.6% | 0.569 | 0.578 | 0.772 | 0.213 | 0.563 | 6/8 | 3 | q03, q14 |
| A | c4 | 86.7% | 46.7% | 64.7% | 0.602 | 0.578 | 0.750 | 0.213 | 0.563 | 5/8 | 0 | q03, q14 |
| B | c3 | 73.3% | 46.7% | 58.8% | 0.582 | 0.572 | 0.756 | 0.213 | 0.551 | 4/8 | 3 | q06, q10, q11, q14 |
| B | c4 | 73.3% | 46.7% | 58.8% | 0.582 | 0.572 | 0.756 | 0.213 | 0.551 | 4/8 | 0 | q06, q10, q11, q14 |
| C | c3 | 86.7% | 60.0% | 64.7% | 0.583 | 0.756 | 0.800 | 0.240 | 0.586 | 5/8 | 2 | q10, q14 |
| C | c4 | **93.3%** | 60.0% | 67.6% | **0.644** | 0.778 | 0.822 | 0.253 | 0.609 | 5/8 | 0 | q14 |

Mean context tokens: A 1,468 → 1,474, B 1,460 → 1,461, C 1,265 → 1,271.

## Query-level changes (answerable questions; everything else identical)

| Setting | Question | c3 → c4 | Why |
|---|---|---|---|
| C | q10 (test, multi-section) | gold rank 12 → **1**; 0 → 1 of 3 quotes in context | Improved. In c3 two Table 6 chunks with 51/55 cells were in q10's context; in c4 they are gone and a gold passage enters |
| A | q10 | gold rank 2 → 1; quotes 1 → 1 | Improved rank |
| B | q10 | gold rank 9 → 10; still 0 quotes | Slightly worse rank, same outcome (still missing) |
| B | q11 (dev, table) | gold rank 9 → 8; still 0 quotes | Slightly better rank, same outcome |
| A | q11 (dev, table) | 2 → 1 of 2 quotes in context; first rank 3 → 3 | Degraded. AIAYN chunks are textually identical in c3 and c4, so this is indirect: BM25's corpus size and term weights change with any rebuild (455 → 465 chunks, new tokens ✓/✗), and setting A has no re-ranker to absorb it |

**False evidence:** in c3, q10's context held corrupted Table 6 chunks under every setting (A 3, B 3, C 2 chunks);
in c4, 0 for every question and setting. q10 is the question whose live v5 draft quoted "a Multi-stage / Hierarchy
value of 51" (Phase 5).

**Earlier retrieval misses:** q06 unchanged (B rank 11: still missing; A and C retrieve it as before). q10 fixed under
C, unchanged under B. q14 still missing in all settings (no gold among the candidates: a vocabulary mismatch, not a
table problem).

**Affected tables:** no evaluation question has gold evidence in HNET Table 6 or MOSHI Tables 3, 4, 5, 6, 13, 16 or
17, so retrieval of those tables' own content cannot be measured with this set. What is measured: the corrupted chunks
no longer reach any context, and their structure survives chunking (caption and "Columns:" line in every chunk,
labelled row lines, 175 marks; see the extraction report).

## Conclusions and limitations

- c4 removes the false evidence entirely and improves setting C (any gold 86.7% → 93.3%, MRR 0.583 → 0.644), via q10.
  Setting B (the production default) is unchanged on every aggregate. Setting A loses one table quote (q11), an
  indirect effect of corpus-wide BM25 statistics.
- 15 answerable questions: one question moves an aggregate by 6.7 points, so none of these differences is
  statistically meaningful; they are per-question effects.
- Precision@5 and nDCG@10 rest on proxy labels; graded relevance labels do not exist.
- Answer quality on c4 is not measured (it needs generation). The verifier's handling of ✓/✗ cells is untested.
- Recommendation: c4 is a safe, isolated candidate. Adopting it would need approval, a `corpus_version` bump to c4,
  and a fresh answer-quality evaluation; until then c3 stays the default.
