# Improvement loop history

All grades are assistant judgments, not independent review. Every rate is a raw count over its denominator. Live samples are small, so the results are directional, not significant.

## Baseline (2026-11-14)

**Configuration:** c3, prompt p4, verifier v4.2, gpt-oss-120b, retrieval B with Configuration G.

**Questions:** 25 fixed (`datasets/datasets.json`, sha256 in `datasets.sha256`). 12 were reused from the Phase 12 c3 run (copied); 13 were run live.

**Results:**

| | Count |
|---|---|
| Correct | 2/20 |
| Partially correct | 1/20 |
| Wrong (t05, accepted by the verifier) | 1/20 |
| False refusals | 16/20 |
| Correct refusals | 5/5 |
| Unsupported answers | 0/5 |
| Citations valid in shown answers | 5/5 |
| Complete evidence in context | 8/20 |
| Claim sentences accepted | 24/66 |
| Redraft success | 2/17 |
| Provider errors | 0 |

**Latency:**
- Time to first token: p50 799 ms, p95 1.37 s.
- Total: p50 28.6 s, p95 34.2 s.
- Governor wait: p50 22 s.

**Offline tests:** all suites passed (`baseline/offline_tests.json`).

## Iteration 001: verifier v4.3c (self-reference premise). Rejected offline

**Target:** F-VER-SUBJ. The NLI model scores near 0 (neutral) when a claim names the model but the passage says "we".

**Offline results:**
- Stress sets (160 pairs): false rejections 22 → 21; false acceptances 6 → 6.
- Live-draft replay (33 drafts; the replay reproduces the recorded verdicts 33/33): false rejections 9 → **10**. It rescued no live failure and flipped one correct answer (q03) to rejected.

**Live cost:** 0 tokens.

## Iteration 002: prompt p5r (redraft instruction). Accepted provisionally

**Target:** F-GEN-REDRAFT. The redraft resent the rejected sentence unchanged.

**Offline:** all suites passed (502 Python tests).

**Live:** 7 questions, 14 requests, 27,131 tokens.

| Same 7 questions | Baseline | p5r |
|---|---|---|
| Correct | 1/6 | 3/6 (q04 and q13 rescued) |
| Redraft success | 1/6 | 3/6 |
| Wrong answers | 0 | 0 |
| Unsupported answers | 0 | 0 |
| Correct refusal | 1/1 | 1/1 |

**Trade-off to approve:** redrafts read in the paper's voice ("We use …") or as raw table rows.

**New root cause found:** verbatim sentences are rejected against the whole-chunk premise (F-VER-PREMISE).

## Iteration 003: verifier v4.4c (sentence-level premises). Rejected offline

**Run:** a completed checkpointed comparison of 252 cases (160 stress cases plus 92 live drafts), after earlier memory stops (`iterations/iteration_003/comparison_rerun_2026-11-14/`).

| | v4.2 | v4.4c |
|---|---|---|
| False rejections, stress | 22 | 10 |
| False rejections, live drafts | 33 | 27 |
| Copied sentences still rejected | 5 of 9 | 0 of 9 |
| **False acceptances, stress** | **6** | **9** |
| **False acceptances, live drafts** | **2** | **3** |

**The new false acceptances are real factual errors:**
- base-model step time "1.0 s" (it is 0.4 s; 1.0 s is the big models);
- "twice as wide" (it is 4×);
- the attention formula without the square root.

**Decision:** the gate rejects any candidate whose false acceptances increase.

## Open (outside the loop's authority)

- **F-EXT-MARKS (t01, t05, t10, t13, t29):** c3 table rows lack check marks. c6 fixes t01, t05 and t13, but activating it needs the owner's approval.
- **F-RET-MISSING (q06, q10, q11, q14, q09):** no root cause per question yet.
