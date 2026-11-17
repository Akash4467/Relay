# Phase 5: root causes of the remaining false refusals and the table-extraction defect

Offline only: no Groq or other API calls. Verifier v4 and its config are unchanged, v5 is not deployed, and no
historical result, raw log, question set or grade was changed. All experiment code is in `experiments/phase5/`
(never imported by the pipeline); its outputs are in `experiments/phase5/out/` (new files only). Grades are
assistant-generated and preliminary. Scores come from the local NLI model (MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli).

## Summary

1. **The NLI rejections are mostly not noise.** One-change ablations against the exact premises show that most rejected
   claims add something the cited passage does not state: a qualifier taken from the question ("base", "with Inner
   Monologue"), an interpretive conclusion ("forming the basis of its hierarchical design", "confirming that adding a
   second stage improves performance"), or a changed word ("static" for the paper's "strong"). Removing that one
   element moves the score from ≤ 0.11 to ≥ 0.93 in four of six tests. The model is strict, and mostly right to be.
2. **One genuine model limitation:** claims listing several table values in prose. "Moshi achieves 26.6 on Web Q." scores
   0.986, but three values in one sentence score 0.40 (column names) to 0.76 (the table's own wording).
3. **A deterministic table check (experiment T) fixes q12 without touching NLI elsewhere:** it accepts q12's correct
   redraft, changes no labelled case, accepts no unsupported sentence, and rejects all 12 adversarial near-misses.
4. **Claim decomposition (experiment C)** rescues 2 supported sentences but no whole draft; its relation-error risk
   cannot be measured with the existing sets.
5. **The "51" is an extraction bug, not a model or verifier bug.** `\cmark` → `\ding{51}` → pylatexenc renders the
   argument "51" (and `\ding{55}` as "55"; `\checkmark` as nothing). It corrupts 8 tables (53 rows) in HNET and
   MOSHI; MOSHI's Mimi-codec ablation (Table 3) loses all five configuration columns. A one-line pre-mapping fixes it.

## A. NLI failures: exact pairs (live v5 run, `experiments/phase5/out/nli_pairs_2026-11-13T19-52-00Z.json`)

Every pair was rebuilt with the verifier's own functions (`premises`, `atomic_claims`, `plain_math`, context rule);
the replay reproduced every live decision before any pair was recorded. "Human-verifiable" means: can a reader confirm
the claim from the cited passage text alone.

| Draft / sentence | Claim (shortened) | Weakest part and score | Structural reason | Human-verifiable from the cited text? | Cause (from the ablation) |
|---|---|---|---|---|---|
| q02 d1 s1, d2 s1 | "The base Transformer uses eight parallel attention heads, and each head receives d_k = d_v = 64" | clause "The base Transformer uses eight parallel attention heads": 0.012 / 0.011 (whole 0.55 / 0.31; second clause 0.92) | none | Partly: the passage says "In this work we employ h=8 parallel attention layers"; "base" is not in it | Qualifier from the question: dropping "base" → 0.976 |
| q10 d1 s1 | "...a routing module … and a downsampler …, thereby creating a multi-stage representation" | whole 0.585 | none | Partly (added conclusion) | Interpretive tail |
| q10 d1 s2 | "described as 'dynamic, multi-stage hierarchies' in the ablation table, indicating …" | whole 0.332 | none | No (it is the related-architectures table, and the "indicating" part is inference) | Mislabel + inference |
| q10 d1 s3 | "In the 1-stage version, H-Net already learns context-dependent segmentations that improve over static heuristics" | whole 0.107 | none | Partly ("static" vs the paper's "strong") | Changed word: "strong" → 0.806 |
| q10 d1 s4 | "When a second dynamic chunking stage is added, performance is 'significantly better than H-Net (1-stage), validating …'" | whole 0.086 | none | Yes (quoted) | Framing clause around a quote |
| q10 d1 s5 | "Thus, the hierarchy is built from successive dynamic chunking operations…" | - | no citation | No | Uncited summary |
| q10 d2 s1 | "...a routing module … and a downsampler …, forming the basis of its hierarchical design" | whole 0.054 | none | Partly | Added conclusion: dropping it → 0.982 |
| q10 d2 s4 | "H-Net (2-stage) is significantly better than H-Net (1-stage), confirming that adding a second dynamic chunking stage improves performance" | whole 0.021 | none | First clause yes; tail is inference | Tail: first clause alone 0.928, tail alone 0.003 |
| q12 d1 s2 | "The retrieved passages do not state the … accuracies of Moshi with Inner Monologue." | - | P6: absence contradicted by row 'Moshi' (26.6 / 62.3 / 22.8) | No: false absence claim | Correctly rejected |
| q12 d2 s2 | "Moshi with Inner Monologue achieves 26.6 % on Web Questions, 62.3 % on Llama Questions, and 22.8 % on Audio TriviaQA" [T8:r01] | whole 0.062, contradiction 0.633 | none | Yes, with the sibling row "Moshi (no Inner Monologue)" (retrieved, but not cited) | Qualifier not in the cited row + multi-value list (below) |

Ablations (`ablations_2026-11-13T19-53-20Z.json`; same premises, one change at a time):

| Test | Variant | Entailment |
|---|---|---|
| q02 | original "The base Transformer uses eight parallel attention heads." | 0.012 |
| | drop "base" | 0.976 |
| | "8" instead of "eight" (keep "base") | 0.013 |
| | "The base model uses 8 parallel attention heads." | 0.132 |
| q12 | original (qualifier, %, prose names) | 0.062 |
| | drop "with Inner Monologue" | 0.550 |
| | table column names, keep qualifier | 0.025 |
| | table column names, no qualifier | 0.401 |
| | the table's own wording ("The configuration Moshi has Web Q. = 26.6, …") | 0.757 |
| | one value ("Moshi achieves 26.6 on Web Q.") | 0.986 |
| q10 | "…better than H-Net (1-stage), confirming that adding a second stage improves performance" | 0.021 |
| | first clause only | 0.928 |
| | the passage's own tail ("validating that … can potentially learn …") | 0.801 |
| q10 | caption claim + "forming the basis of its hierarchical design" | 0.054 |
| | without the added conclusion | 0.982 |

**Root causes, ranked by evidence:**
1. *Unstated qualifier or subject* (q02 ×2, q12): the model copies a qualifier from the question; the cited text does
   not contain it. Premise construction contributes for q12: only r01 was cited, so the "(no Inner Monologue)" sibling
   row that makes "Moshi" mean "with Inner Monologue" was not in the premise.
2. *Added interpretation* (q10 ×4): conclusions, framing and "confirming/indicating that" tails the passage does not
   state. Strict rejection is the intended behaviour; the fix belongs in generation (prompt), not in relaxing NLI.
3. *Multi-value table claims* (q12): an NLI-model limitation. Several numbers in one hypothesis lower entailment
   even when every value is right.
4. *Paraphrase that changes a word* (q10 d1 s3).
Not causes: the unicode hyphen and "%" (Phase 4), and negation (none of the rejected claims negates anything).

## B. Table-aware verification (experiment T, `experiments/phase5/table_check.py`)

For a sentence that cites only table rows and that v5 rejected *only* for low entailment, T accepts it without NLI when:
1. it names exactly one row variant of the cited table, using P6's sibling-qualifier rule over all retrieved rows of
   that table ("with Inner Monologue" = the plain "Moshi" row next to "Moshi (no Inner Monologue)");
2. every number in it equals a cell of that row, in a column the same clause names (column names match by compact
   text or a header abbreviation: "Web Q." ↔ "Web Questions");
3. it contains no comparative or negation (than, better, higher, not, no, without…; a row qualifier such as
   "without Inner Monologue" is exempt because rule 1 decides it);
4. every other content word is explained by the caption, row labels, column names or generic words.
The caption, row label, columns and cells are all used; nothing is accepted on a topic match alone.

Adversarial cases on the real Table 8 and HNET Table 2 rows (`table_adversarial_2026-11-13T19-59-14Z.json`): **15/15
as intended.** It accepted the 3 correct claims (q12's redraft, a plain-row claim, a "without Inner Monologue" claim)
and rejected all 12 near-misses: a wrong value, values swapped between columns, the wrong variant in both
directions, the wrong entity, a comparative, a negation, an extra condition, a number without its column, a claimed
missing cell, a multi-row comparison and the wrong-size row.

## C. Table extraction: where ✓ and ✗ became 51 and 55 (`extraction_fix_2026-11-13T19-56-22Z.json`)

| Stage | Evidence | Value |
|---|---|---|
| LaTeX source | `data/raw/HNET/src/tab/related.tex`: `\multirow{6}{*}{\cmark}`; `main.tex:94-95`: `\newcommand{\cmark}{\ding{51}}`, `\newcommand{\xmark}{\ding{55}}` | ✓ / ✗ |
| Macro expansion | `latex_parse.expand_macros` | `\ding{51}` / `\ding{55}` |
| **Text conversion (the fault)** | `latex_parse.to_text` → pylatexenc `latex_to_text`, which does not know `\ding` and keeps its argument | **"51" / "55"** |
| Intermediate representation | `data/processed/tables.jsonl`, HNET T6, H-Net row: `Autoregressive: '51'`, `Multi-stage / Hierarchy: '51'` | 51 |
| Chunk | `chunks_256.jsonl` HNET:T6:r03: "Configuration Hierarchical (dynamic): Autoregressive = 51; … Multi-stage / Hierarchy = 51; Example Architectures = H-Net" | 51 |
| Retrieved passage | live q10 run, HNET:T6:r03 | 51 |
| Model context | `prompts.user_message` for q10 contains "Multi-stage / Hierarchy = 51" | 51 |

Extent: HNET Table 6 (17 cells holding 51/55) and MOSHI Table 6 (6). MOSHI also writes ✓ as `\checkmark`, which
pylatexenc renders as an empty string, and defines `\xmark` as empty (its PDF shows a blank). So MOSHI Table 3 (the
Mimi codec ablation) lost all five configuration columns: its rows hold only the metric values.

**Proposed minimal fix:** in `latex_parse.to_text`, before pylatexenc, map `\ding{51}`/`\ding{52}`/`\checkmark` → "✓"
and `\ding{55}`/`\ding{56}` → "✗". Tried in memory on all three papers: it changes 8 tables (53 rows) and no prose
paragraph; the H-Net row becomes `Autoregressive: ✓, Multi-stage / Hierarchy: ✓`. Remaining: MOSHI's empty `\xmark`
cells are still dropped from rows, and a leading ✓ column becomes the row header; both need a small parse_tabular change
(keep blank cells as "–"; never use a mark cell as the row header).
**Regression tests:** `\ding{51}`, `\ding{55}`, `\checkmark`, `\cmark` via a paper macro inside `\multirow`; the
HNET T6 H-Net row; MOSHI T3 keeps its configuration columns; `tables.numbers` sees no 51/55 in those rows; no prose
change. **Cost:** a corpus rebuild (c3 → c4: chunks, both indexes), then the retrieval evaluation again; historical
results stay labelled c3.

## D. Candidates compared offline (`candidates_2026-11-13T20-01-56Z.json`)

Data: the four labelled verifier sets (unchanged), the 19 verified drafts of the 15-question run (46 graded
sentences) and the 6 live v5 drafts (15 sentences, graded for this report). Threshold 0.9 throughout.

| Variant | Labelled sets: supported / unsupported accepted | Real sentences accepted: supported (41) / partly supported (12) / unsupported or false (8) | Drafts accepted |
|---|---|---|---|
| Frozen v4 | v3_eval 16/18 · 2/24; targeted 7/8 · 2/10; calibration 5/6 · 1/6; held-out 1/4 · 0/4 | 16 / 0 / 3 | 3 (incl. two wrong q12 drafts) |
| v5 (P1+P6) | identical to v4 | 23 / 0 / 1 | 2 |
| v5 + T | identical | 23 / 1 / 1 | 3: + live q12 redraft |
| v5 + C | identical | 25 / 0 / 1 | 2 |
| v5 + T + C | identical | 25 / 1 / 1 | 3 |

- **Newly accepted vs v5.** T: live q12 redraft sentence 2 (correct numbers; "with Inner Monologue" rests on the
  sibling row, so graded partly supported). That makes the q12 redraft deliverable: both of its sentences are correct.
  C: q08 d1 s1 and q13 d2 s1 (both supported); neither makes its draft deliverable.
- **Unsupported claims newly accepted:** none by T or C. The one unsupported sentence every v5 variant still accepts
  is live q10 d2 s2 ("a Multi-stage / Hierarchy value of 51"): the extraction bug, which only fix C (Section C)
  removes.
- **Labelled sets:** no decision changed for any candidate. They contain no table-only multi-value claims and no
  participle tails, so they show that nothing else moves, not that the candidates are safe in general.

## False-acceptance risks

- **T** bypasses NLI for table-only claims. Mitigated by the single-row, named-column, no-comparative and
  no-unexplained-word rules (15/15 adversarial). Unmeasured: tables with ± or units inside cells, merged headers, two
  rows with the same label, or claims whose verb implies a relation ("triples").
- **C** drops the whole-sentence requirement: true clauses joined by a false relation ("X because Y") would pass.
  There is no labelled relation-error set, so this risk is unmeasured. Do not adopt C until one exists.
- **The extraction fix** changes corpus text, so retrieval scores and chunk boundaries can shift. Every retrieval
  measurement must be redone on c4.
- **Not recommended:** lowering the 0.9 threshold, or the sentence-window premises from Phase 4 (P3: a measured new
  false acceptance).

## Recommended implementation order (each needs separate approval)

1. **Extraction fix** (✓/✗ mapping, then blank-cell and row-header handling), with the listed tests; rebuild as corpus
   c4 and rerun the retrieval evaluation. Deterministic, and it removes the only false sentence v5 still accepts.
2. **Table check T** as a v5 extension, behind an explicit verifier flag, plus more adversarial cases (± and units,
   duplicate labels, relation verbs). It is the only candidate that turns a refusal into a correct answer here (q12).
   Also let table claims use sibling rows of the same table as evidence when they are retrieved.
3. **Generation-side change** (prompt; outside the verifier): keep the passage's own subject and qualifiers, one fact
   per sentence, no added conclusions. This targets the dominant cause (q02, q10) but needs live calls to evaluate.
4. **Claim decomposition C** only after a labelled relation-error set exists.
5. A new live evaluation only after 1-3, within a separately approved budget.

Three live questions and small labelled sets support no statistical claim; these results show mechanisms, not rates.
