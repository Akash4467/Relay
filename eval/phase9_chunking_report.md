# Phase 9A — Section-and-label metadata on table chunks (chunking k1)

Offline. Code: `experiments/phase9/table_chunk_metadata.py`, applied by `build_c5.chunk_corpus`.

## The change

Every table chunk (kind `table`) gets one line after its `Columns:` line. For example, `MOSHI:T3:r02` in c5:

```
Table 3: Ablation study on hyperparameters of the Mimi codec.
Columns: Quantization / Rate | Transformer / in encoder | … | MUSHRA (↑)
Source: Moshi: a speech-text foundation model for real-time dialogue (MOSHI); Table 3 (tab:mimi_ablations); section 5.2 Audio Tokenization; rows 3-4 of 6
Configuration Row 3: Quantization / Rate = ✓; …
```

The required elements are all present:

| Element | Where it is |
|---|---|
| paper identity | the metadata line (title and code) |
| table identifier | first line ("Table 3:") and the metadata line, with the LaTeX label |
| caption | first line: the existing short caption, unchanged. A full caption on every chunk was rejected in Phase 8 |
| section title | the metadata line: the chunk's own section, which is 5.2 for Table 3 only when s1 is applied |
| row range | the metadata line ("rows a-b of N", in table row order) |
| content | the existing row lines, unchanged |

## What is held constant

- **No whole-table chunking and no context caps.**
- **Unchanged:** piece sizes (256/512), overlap (none), the embedding model (BAAI/bge-small-en-v1.5), the index header,
  BM25, the re-ranker, settings A/B/C, candidates (20), final passages (8/8/4), the context budget (1600) and the pressure
  cap (2500).
- **Unchanged control:** before any variant, the unchanged pipeline was rebuilt into
  `experiments/phase9/corpus/c4_control/processed`, and all 10 files equal `data/corpus/c4/processed` byte for byte. In
  `ablation.py`, the `c4` variant equals the stored c4 chunks and the `c5` variant equals the stored c5 chunks.
- **Inserted after chunking:** chunk boundaries, IDs and order equal c4, and text, caption and figure chunks are
  untouched. The diff of c5 against c4 shows only the intended changes:
  - the metadata line on 146 table chunks at 256 tokens and 71 at 512;
  - T3's section fields;
  - the e3 cells and the T18 caption.

## The cost: chunks get longer

| Piece size | Table chunks | Tokens added per chunk | Chunks over the piece size (c4 → c5) | Largest c5 table chunk |
|---|---|---|---|---|
| 256 | 146 | mean 46.6 (41–54) | 0 → **89** | 306 |
| 512 | 71 | mean 46.2 (41–54) | 0 → **19** | 555 |

The context budget (1600 tokens) is unchanged, so **fewer chunks fit**. Passages that rank in the top `final_passages`
but are dropped by the token budget rise from c4 to c5:

| Dataset | A | B | C |
|---|---|---|---|
| table set (32 q.) | 32 → 64 | 42 → 77 | 24 → 29 |
| original (15 q.) | 28 → 34 | 25 → 36 | 8 → 9 |

Splitting into more chunks, so that each one fits the piece size, would be a second change (chunk boundaries). It was
deliberately not made, and it is the obvious next isolated experiment.

## Effect of the metadata line alone (ablation `metadata_only` vs c4)

| | A | B | C |
|---|---|---|---|
| table set: correct table in context | 21 → **24**/24 | 23 → **24** | 24 → 24 |
| table set: table MRR | .697 → **.899** | .938 → **.979** | .931 → **1.000** |
| table set: rows in context (31) | 26 → 28 | 27 → 29 | 29 → **31** |
| table set: questions complete (24) | 18 → 19 | 19 → 19 | 19 → 21 |
| table set: competing table of the same paper in context | 3 → **8** | 7 → **10** | 8 → 5 |
| T3/T17 probes: correct table first (16) | 9 → **14** | 14 → 13 | 9 → **13** |
| T3/T17 probes: wrong table in context | 9 → 10 | 13 → 11 | 11 → **4** |
| original: any gold / quote recall | 13/15 · 22 = | 11/15 · 20 → **19** | 14/15 · 23 = |
| original: MRR | .602 → .569 | .582 → .550 | .644 = |

### Per question

- **Gains:**
  - A: t06, t07 become complete; t10 table enters the context.
  - B: t10 becomes complete, with 3 rows.
  - C: t29, t30 become complete.
  - Probes: a2, a3, a4, t06, t07 (A) and a5, a6, t29, t30 (C) now rank the correct table first.
- **Losses:**
  - A: t32 incomplete. B: t30 incomplete. In both, the gold row chunk is in the top 8 but **dropped by the budget**.
  - q13 B loses a quote: HNET Table 2 chunks crowd it out.
  - t08 B (unanswerable): T3 leaves the context.
  - Original MRR falls in A and B: q10 A 1.0 → 0.5 (a Table 6 chunk now outranks the prose), q12 B 1.0 → 0.5.

## Answer

**Yes, metadata improves table identification:**

- the correct table is in the context for 24/24 answerable questions in every setting;
- table MRR rises in every setting;
- T3 vs T17 improves strongly in A (+5) and C (+4).

**But it is not free.** It costs about 46 tokens per table chunk, so more evidence is lost to the fixed budget. Two
previously complete table questions now fail for that reason. The original set gets a lower MRR in A and B and loses one
quote in B. Table chunks also crowd prose more: q10 B's context is all HNET Table 6.
