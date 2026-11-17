# Phase 8 — Investigation A: the `\phantom` extraction defect

Offline only. No Groq calls, no change to `relay/`, c3, c4 or any result. The candidate is the experiment-only extraction **e3**,
which is e2 plus one change, in `experiments/phase8/phantom_fix.py`.

## Defect

pylatexenc renders `\phantom{X}` as `X`. In LaTeX, `\phantom{X}` reserves the width of X but prints nothing. It is used to
align numbers in a column. MOSHI writes `$\phantom{1}42.2$`, so the extracted cell reads `142.2` while the PDF shows `42.2`.
`\hphantom` and `\vphantom` were already dropped by pylatexenc. Phase 7 found this when checking expected answers against the
LaTeX source; table questions t13–t15 were marked incomplete because of it.

## Fix (e3)

`strip_phantoms(tex)` removes `\phantom`, `\hphantom` and `\vphantom` together with their brace-matched argument before the
existing converter runs. A nested phantom goes with its outer argument. An unbalanced brace leaves the text unchanged.
`e3()` swaps the converter in-process and always restores it. Parsing is otherwise e2 (`lp.parse_paper(..., extraction="e2")`).

## Regression tests

`experiments/phase8/test_phantom_fix.py`: 15 tests, all pass (`uv run pytest -q experiments/phase8/test_phantom_fix.py`).

- **Covered:**
  - `$\phantom{1}42.2$` → `42.2`; an unpadded `42.2` stays unchanged.
  - `1\phantom{0}` → `1`; `$\phantom{00}1.0$` → `1.0`.
  - `\hphantom` and `\vphantom` are removed, and so are nested phantoms.
  - `100.00\hphantom{00}`.
  - The old e2 path still gives `142.2`, which pins the defect so it cannot silently return.
  - Math and prose without phantoms are unchanged.
  - Unbalanced braces.
  - The converter is restored after use.

## Validation on the real corpus (`experiments/phase8/out/phantom_validation.json`)

e2 vs e3 on all three papers: same tables, same paragraph count. **Exactly 6 cells change**; no row header, paragraph or
caption (table or figure) changes.

| Table | Row | Column | e2 (c4) | e3 | e3 value in PDF text | e2 value in PDF text |
|---|---|---|---|---|---|---|
| MOSHI:T5 | [0, 1, 2, 3, 4, 5, 6, 7] | Perplexity | 142.2 | 42.2 | yes | no |
| MOSHI:T5 | [0, 1, 2, 3, 4, 5, 6, 7] | Perplexity | 140.3 | 40.3 | yes | no |
| MOSHI:T5 | [0, 2, 2, 2, 2, 2, 2, 2] | Perplexity | 136.8 | 36.8 | yes | no |
| MOSHI:T6 | [0, 0, 0, 0, 0, 0, 0, 0] | Semantic Token / Weight | 001.0 | 1.0 | yes | no |
| MOSHI:T6 | [0, 1, 1, 1, 1, 1, 1, 1] | Semantic Token / Weight | 001.0 | 1.0 | yes | no |
| MOSHI:T6 | [0, 2, 2, 2, 2, 2, 2, 2] | Semantic Token / Weight | 001.0 | 1.0 | yes | no |

Why only 6, although phantom macros occur 25 times in MOSHI's source (0 in AIAYN and HNET):

- 16 of them are `\hphantom`, which pylatexenc already dropped (e.g. `100.00\hphantom{00}` in `sections/safety.tex`).
- 6 are the `\phantom` cells above (`sections/evaluation.tex` lines 123–126 and 142–144).
- 3 are `\phantom{x}` colour swatches in `figures/safety.tex`, whose text is not in the corpus.

The MOSHI T6 `001.0` defect was not known before this validation; Phase 7 had only found T5.

## Answer

**Is the phantom extraction defect fixed?** Yes, in the experimental e3 extraction. The fix is validated against the PDF on
every affected cell and has no collateral change.

It is **not** applied to any corpus. c4 still contains `142.2`/`140.3`/`136.8` and `001.0`. A versioned c5 rebuild
(`corpus_version` bump, re-index, re-run of the Phase 7 checks) is needed and awaits approval.
