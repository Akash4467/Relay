# Verifier v4: offline investigation of the real-run rejections and change proposals
> **Submission copy, corrected.** The original (eval/verifier_v4_report.md) said 17 verified drafts; the data contain 19 verified drafts and 5 "not in papers" replies. Only those counts were changed here. P1 and P6 were later evaluated offline in candidate_v5_offline_not_deployed/; their final-answer effect is unknown.


Status: **proposal only.** Verifier v4 (code, threshold 0.9, config) is unchanged. No Groq calls. All numbers come
from offline replays of the saved drafts in `results/raw/answer_quality/real_2026-11-13T18-06-56Z.jsonl` and of the
existing labelled sets, with the frozen v4 code (candidate changes were patched only inside a scratch process).

Replay note: the raw record stored retrieved passages without their table fields (`kind`, `table_id`, `rows`), so
replays restore passages from `data/processed/chunks_256.jsonl` by chunk_id (corpus c3, the run's corpus). With that,
the v4 replay reproduces every live decision (19 of 19 verified drafts). The runner now stores those fields.

## What decides acceptance in v4

- Level-1 checks reject outright: citations (parsed, retrieved), absence wording, table value owner, formulas.
- Then NLI entailment ≥ 0.9 on the whole sentence **and** on each clause (minimum).
- Level-2 flags (numbers, terms) need `entail_min_flagged`, which equals `entail_min` (0.9): **flags never change a
  decision**; they only appear in the rejection reason and the redraft feedback.

## Effect of candidates on the real drafts (19 verified drafts; labels are assistant-graded)

| Variant | Drafts accepted | Correct drafts (8) accepted | Wrong drafts (3) accepted |
|---|---|---|---|
| v4 (live = replay) | q12 d1 (factually_wrong), q26 d2 (correct_refusal) | 0 | 1 (q12) |
| P1 brackets | + q06 d1 (partially_correct) | 0 | 1 (q12) |
| P1 + P3 windows | + q06 d1, q06 d2 (partially_correct) | 0 | 1 (q12) |

## Effect of P3 on the labelled sets (v4 → v4 + windows)

| Set | Supported accepted | Unsupported accepted (false acceptance) |
|---|---|---|
| v3_eval, 42 | 16/18 → 17/18 (m4) | 2/24 → 2/24 |
| v3b_targeted, 18 | 7/8 → 8/8 (f2) | 2/10 → 2/10 |
| v2_stress calibration, 12 | 5/6 → 5/6 | 1/6 → **2/6 (l3: formula without the square root)** |
| held-out, 8 (evaluation only) | 1/4 → 2/4 (p2) | 0/4 → 0/4 |

## Proposals

### P1. Read full-width citation brackets 【…】 — recommended
- **Failing example:** q02 draft 1, "… (d_k = d_v = d_model/h = 64)【AIAYN:3.2.2:c014】" → "no citation". 21 of 46
  sentences in the run used 【…】; 5 of 10 answerable questions were affected.
- **Root cause:** `citations._BRACKET = \[([^\[\]]+)\]` reads only ASCII brackets; gpt-oss writes its native 【…】
  despite the prompt. The redraft feedback ("no citation") did not make it switch.
- **Minimal change:** map 【→[ and 】→] before parsing and stripping citations (one normalisation in citations.py).
- **Expected improvement:** +1 accepted draft on replay (q06 d1, partially correct); 0 correct drafts rescued. Its
  real value is that rejections then reflect content, and redraft feedback stops saying "no citation".
- **False-acceptance risk:** low. Every other check still applies (cited ID must be retrieved, NLI, tables,
  formulas). No labelled case uses 【…】, so their results cannot change.
- **Regression tests:** `citations_in` and `strip_citations` on 【ID】, mixed [ID]【ID】, an unretrieved ID in 【…】
  (still rejected), 【…】 holding text that is not an ID (ignored); the real-draft replay above.

### P2. Unicode hyphen (H‑Net) and "55.5 %" in level-2 flags — no change recommended
- **Failing examples:** q10 "term 'net' not in the cited passage" (H‑Net written with U+2011); q13 "number 55.5 % not
  in the cited passage" (the table says "= 55.5").
- **Root cause:** the term/number extraction does not normalise U+2011 or a spaced "%".
- **Expected improvement:** none on decisions (flags are diagnostic in v4). NLI is unaffected: `plain_math` already
  maps U+2011 (identical scores), and removing "%" moves scores by ≤ 0.04.
- **Option:** cosmetic fix for clearer redraft feedback only. **Risk:** none for acceptance.
- **Tests if done:** term extraction keeps "H‑Net" whole; "55.5 %" matches "55.5" for the % unit only.

### P3. Extra sentence-window premises — not recommended
- **Failing example:** q07 draft 2, "In encoder self‑attention all queries, keys and values come from the same previous
  encoder layer" (verbatim in AIAYN:3.2.3:c016) scores 0.476 against the whole passage, 0.984 against its sentence.
  q06 draft 2: 0.560 → 0.979. The same claim scored 0.99 and 0.13 in two q08 drafts.
- **Root cause:** whole multi-topic passages (up to ~340 words) dilute entailment; scores depend on the surrounding
  text.
- **Minimal change:** also score each sentence and adjacent pair of the cited non-table passages (max).
- **Expected improvement:** +1 real draft (q06 d2, partially correct); +1 supported case in three labelled sets.
- **False-acceptance risk: measured, real.** l3 (a formula without the square root) becomes accepted. More NLI pairs
  per claim also raise latency (not measured). Windows also lower some scores they replace, so they could only be
  an extra premise, never a substitute (q02: 0.80 whole vs 0.26 best window).
- **Tests required before any adoption:** all formula and number near-miss negatives; latency p50/p95.

### P4. Named subject vs. "we"/"this" in the passage — the largest cause; no minimal safe fix
- **Failing examples:** q04 "Mimi uses Q = 8 quantizers …" 0.016 vs "We use Q = 8 quantizers …" 0.981 (same
  passage); q02 "The base Transformer employs 8 parallel attention heads" 0.012 vs "The Transformer employs …" 0.960;
  q08 "The authors chose sinusoidal positional encodings because …" 0.020 vs "They hypothesized …" 0.956.
- **Root cause:** a limitation of passage-level NLI, not a bug: the passage names its subject implicitly ("we",
  "this function"); the claim names it ("Mimi", "base Transformer"), and the model treats the name as unsupported.
- **Options:** (a) a curated per-section subject note added to the premise ("§3.3.1 describes the Mimi codec");
  (b) a prompt change (keep the passage's own wording for subjects), which is not a verifier change and needs real
  calls to evaluate.
- **False-acceptance risk of (a):** medium: it can make wrong-component or wrong-paper attributions pass (the q20
  type). **Tests:** attribution negatives (e.g. "Moshi uses 8 heads" against the AIAYN passage) must stay rejected.

### P5. Compound sentences: whole sentence and every clause must pass — no verifier change recommended
- **Failing examples:** q10 "Adding a second stage improves performance: H‑Net (2‑stage) is significantly better than
  H‑Net (1‑stage) …" 0.005, but 0.928 without the prefix; q13 draft 2 whole 0.891 (clauses 0.980 and 0.945); q08
  draft 2 clause "They also chose the sinusoidal version" 0.038.
- **Root cause:** v4 scores the whole sentence and each clause and takes the minimum, so splitting never helps;
  framing prefixes and comparisons drag the whole-sentence score down.
- **Option:** accept when every clause passes even if the whole does not. **Risk:** high: relation errors between
  true clauses ("X because Y") would pass; there is no labelled relation-error set to measure it. A prompt rule of
  one fact per sentence is the safer route (needs real calls).

### P6. False acceptance of an absence claim (q12) — recommended, rejection-only
- **Failing example:** q12 delivered "The retrieved passages do not state the spoken question answering accuracy of
  Moshi with Inner Monologue." although MOSHI:T8:r01 ("Moshi: 26.6 / 62.3 / 22.8", table captioned "Evaluation of
  spoken question answering"), c090 and c091 were retrieved.
- **Root cause:** `absence.py` needs all content words of the missing part in one sentence of a non-table passage;
  `tables.absence_contradicted` needs the metric name among the columns. A row label ("Moshi") under a caption that
  matches the topic is checked by neither.
- **Minimal change:** reject an absence sentence when a retrieved table's caption shares its topic words and one of
  that table's row labels is named in the sentence.
- **Expected improvement:** the one delivered wrong answer (q12) becomes a refusal: wrong answers 1 → 0, false
  refusals 9 → 10.
- **Risk:** cannot raise false acceptance (it only rejects); may reject a true absence when the row exists but the
  requested metric does not. **Tests:** q12 rejected; a row present with the metric genuinely missing; the existing
  absence cases in the targeted set unchanged.

## Other rejections that are correct under the strict policy
- q07 draft 1: "the encoder stack contains only self‑attention and feed‑forward sub‑layers" is not in the cited c009.
- q09 draft 2: "230 ms … is higher than Moshi's practical latency" is an inference across passages, not a stated fact.
- q10: "can learn a nested hierarchy" drops the paper's "can potentially learn" (0.874 even without the prefix).
- q13: the GPT-3 XL claim is factually wrong (58.2 is H-Net's own XL row).

## Recommendation
Approve P1 and P6 as one small, rejection-neutral/rejection-only change (would become v5), with the regression tests
listed and a full re-run of the labelled sets and the real-draft replay. Do not adopt P3 (measured false acceptance).
P4 and P5 need either a curated design with attribution negatives or a prompt change evaluated with real calls; both
are out of scope without separate approval. P2 is cosmetic.
