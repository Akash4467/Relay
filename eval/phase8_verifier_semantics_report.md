# Phase 8 — Investigation B: how the verifier reads ✓, ✗ and blank table cells

Offline only. The verifier code is never modified: v4 is frozen (NLI `MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli`,
entailment ≥ **0.9**, unchanged). Only the evidence representation, or an added deterministic checker, differs.

Code: `experiments/phase8/table_semantics.py`. Runs (`experiments/phase8/out/`):

| Run | Output file |
|---|---|
| Dev | `table_semantics_2026-11-13T21-09-22Z.json` |
| Held-out | `table_semantics_2026-11-13T21-10-19Z.json` |
| Earlier dev runs, kept | `…20-59-23Z`, `…21-03-16Z` (duplicate run, see below), `…21-06-06Z` |

## Cases

All cases are **constructed** claims, written as test inputs and not model output, about real c4 table rows. Each claim cites
the c4 chunk that holds its row.

- **Labels:** relative to the cited table evidence only (caption + rows). They were assigned by the assistant from the LaTeX
  source and await **independent human review**.
- **Rule:** no paper defines its marks. A blank is never read as "no", false or zero, and a ✗ is never read as "does not
  support".
- **Dev:** `experiments/phase8/table_semantics_cases.json`, 31 cases (13 supported, 18 unsupported) over HNET:T6, MOSHI:T4,
  T3, T13 and T16.
- **Held-out:** `experiments/phase8/table_semantics_heldout.json`, 16 cases (8/8). They were written after the first dev run
  and **before** the checker was revised, then run once.

## Variants (all on the same cases and the same cited rows)

| Variant | What changes |
|---|---|
| A | v4 on the original c4 evidence (✓, ✗, `(blank)`; LaTeX-empty cells absent from the row line) |
| B1 | v4; marks as tokens `[CHECKMARK]` / `[CROSSMARK]` / `[EMPTY_CELL]`; empty cells restored from `tables.jsonl` |
| B2 | v4; marks as words `checked` / `crossed` / `empty`; same restoration |
| D1 / D2 | candidate v5 (P1 + P6) on the B1 / B2 evidence |
| C | first deterministic mark checker: literal "row R is marked X under column K" claims only; otherwise abstains (counts as a rejection) |
| C2 | C revised after the first dev run (general rules only: column phrase after the last "under", plural stemming, row found by distinctive printed values) |
| C+A, C2+A | the checker, falling back to v4 (A) when it abstains |
| C2g | guarded combination, registered before the held-out run. A claim that asserts a mark but that C2 cannot resolve is **rejected**; only claims without a mark word fall back to v4 |

## Results (FAR = unsupported claims accepted; FRR = supported claims rejected)

| Variant | Dev FAR (n=18) | Dev FRR (n=13) | Held-out FAR (n=8) | Held-out FRR (n=8) |
|---|---|---|---|---|
| A (v4) | 11.1% (2) | 46.2% (6) | 25.0% (2) | 50.0% (4) |
| B1 tokens | 16.7% (3) | 23.1% (3) | 0.0% | 25.0% (2) |
| B2 words | 22.2% (4) | 23.1% (3) | 0.0% | 50.0% (4) |
| D1 (v5 on B1) | 16.7% | 23.1% | 0.0% | 25.0% |
| D2 (v5 on B2) | 22.2% | 23.1% | 0.0% | 50.0% |
| C | 0.0% | 76.9% (27 of 31 abstain) | 0.0% | 100% (15 of 16 abstain) |
| C+A | 11.1% | 46.2% | 25.0% | 50.0% |
| C2 | 0.0% | 38.5% (16 abstain) | 0.0% | 37.5% (7 abstain) |
| C2+A | 0.0% | 15.4% | **12.5% (1)** | 12.5% |
| **C2g** | **0.0%** | 30.8% (4) | **0.0%** | 37.5% (3) |

C2's dev numbers are tuned; it was revised after seeing them. Only the held-out column is an unbiased estimate, and with 8 + 8
cases it is a very small one.

## Errors by case

### v4 (A) cannot tell the marks apart

- **s04, s09, h04:** it accepts "marked with a check" for a **✗** cell (entailment 0.98–0.99).
- **h16:** it accepts "H-Net is marked with a check under Chunking Mechanism", where the cell is the text "dynamic chunking"
  (0.973).
- **s03, s05, s06, h01, h05, h10, h14, s30:** it rejects correct ✓/✗ claims.

### Normalisation trades one error for another

- **B1:**
  - New false acceptances on dev: s02 (cross claimed on a ✓), s17 ("RVQGAN is causal" from an `[EMPTY_CELL]`) and s27
    (wrong column).
  - It fixes s04/s09/h04 and h16.
- **B2:** the word `empty` invites negative readings.
  - s14 "RVQGAN is not causal", s15 "causal value of 0" and s23 "does not use a split quantizer" are accepted.
  - That is exactly the blank-as-false or blank-as-zero error the rules forbid.

  Its held-out FAR of 0% does not offset this: the held-out set has no blank-interpretation case.
- **D1/D2:** they are identical to B1/B2 on every case. P1 and P6 do not touch table marks.

### The deterministic checker

- C2 rejects every mismatched mark it can resolve (s02, s04, s09, h04: "cell = ✗; claim says check").
- It abstains when:
  - the row is not unique from the claim's values (s26, s27, s28, h01, h12: 2–5 candidate rows);
  - there are unexplained words (s20, s22, h08: "score");
  - the cell is not a mark (h16).
- **C2+A is unsafe:** on an abstention it hands the claim to v4, which accepted h16 (held-out FAR 12.5%).
- **C2g** closes that path. Its remaining errors are all false **rejections**: s20, s22, s26, s28 on dev and h01, h08, h12
  on held-out. They come from abstentions.

### Blank cells

The blank is described as blank or unspecified in s13, s16, h06, h08 and s28.

- v4 accepts s13, s16 and h06, and rejects s28 (0.863).
- C2g accepts the cases it can resolve and rejects s28 and h08 (abstentions).
- Claims that interpret a blank (s14, s15, s17, s23, s29) are accepted only by B1 (s17) and B2 (s14, s15, s23). A, C2,
  C2+A and C2g reject all five.

## Answers

- **Does textual symbol normalisation improve verifier reliability?** **No, not reliably.**
  - Tokens (B1) halve FRR and fix the ✗-as-✓ errors.
  - But they add new false acceptances on dev: 16.7% vs 11.1%. One of them, s17, reads a blank as "causal".
  - Words (B2) are worse: they make blank-as-negative claims entailable (dev FAR 22.2%).
  - The held-out set (16 cases) is too small to overturn the dev regressions.
- **Can the verifier distinguish checked, crossed and blank cells?**
  - **v4: no.** It accepts check claims on ✗ cells and on a text cell.
  - **v4 + guarded deterministic checker (C2g): yes, for literal mark claims whose row and column it can resolve.**
    - It made 0 false acceptances in 47 cases (dev + held-out).
    - Its cost is false rejections: 30.8% dev (tuned) and 37.5% held-out.
    - Interpretive claims, such as "is causal" or "does not use", are rejected or left to v4. No variant may convert a mark
      into a property, because no paper defines its marks.

## Limitations and integrity

- Every label is the assistant's and awaits independent review. These are constructed claims, not real model answers.
- 47 cases over 5 tables is too few for a final claim.
- C2 was tuned on the dev set.
- An attempted script patch failed its own assertion and wrote nothing new to the script. The unchanged script re-ran,
  producing the duplicate dev output `…21-03-16Z`. It is kept, not deleted.
- **Not run:** live generation, real Groq, or any change to v4 or v5. The threshold stays at 0.9.
