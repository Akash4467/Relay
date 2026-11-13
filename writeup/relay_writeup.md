# The Relay — Akash Bisht

## 1. What I built and why
Cited Q&A over the three papers: Python pipeline of plain functions, Go stand-in model service, Go open-loop load generator. **Retrieve** (the tool: BM25 + dense + cross-encoder re-rank, 1,600-token budget) → **structural check** (no usable passage: refuse, no model call) → **draft** (Groq gpt-oss-120b via a token-bucket governor) → **verify** (each sentence cites a retrieved passage that a local NLI model finds entails it at ≥ 0.9) → **one redraft** (the loop) → answer or **refuse**. **Shaping decision:** nothing is sent until the whole answer verifies, so first token = total time and no unverified sentence reaches a user. **Versions:** load tests and results.csv used verifier **v2**; the submitted verifier is **v4**, never load-tested; threshold 0.9 throughout.

## 2. The real service and my stand-in
7 real calls (10,147 tokens, no 429); limits in Table 1. Measured: a fixed 67-token preamble per request, reasoning tokens count against the budget, and daily requests refill continuously (one per 86.4 s). The stand-in replays the real answers and token counts, latency fitted to the 5 answer calls (first byte median 515 ms), with Groq-style limits and 429s, streaming and faults. **Least like the real thing:** 5 calibration calls; slowdown, faults and the 429 body are guesses; replies ignore prompt and retrieval, so it measures capacity, never quality.

## 3. Results
real_smoke is Groq; all other rows are the stand-in; all ran verifier v2.

**Table 1: Limits you found in the real service**

| Limit | Value you found | How you found it | Confidence (high, medium, low) |
|---|---|---|---|
| Requests per minute | 30 (documented) | Groq docs; no RPM header; not probed (5 real calls) | low |
| Tokens per minute | 8000 | x-ratelimit-limit-tokens header, all 5 responses | high |
| Requests per day | 1000 | x-ratelimit-limit-requests header, all 5 responses | high |
| Tokens per day | 200,000 (documented) | Groq docs; no header; not probed | low |
| Requests in flight at once | not measured | no concurrency test run | n/a |
| What it returns when a limit is hit | not observed (0 HTTP 429 in 5 real calls) | docs: HTTP 429 + retry-after + x-ratelimit-* headers; body never seen | medium (docs only) |
| Time until it recovers | daily: one request back every 86.4 s (continuous) | x-ratelimit-reset-requests after 1 of 1,000; per-minute recovery not observed | medium |

**Table 2a: Latency, one row per load scenario**

| Scenario | Real or stand-in | Requests sent | Duration (s) | Offered rate (per min) | First token p50 / p95 / p99 (ms) | Total time p50 / p95 / p99 (ms) |
|---|---|---|---|---|---|---|
| real_smoke | real | 3 | 99 | 1.82 | 4029 / 7928 / 8275* (N=3) | 4029 / 7928 / 8275* (N=3) |
| steady | stand-in | 21 | 600 | 2.10 | 2918 / 19452 / 27841* (N=21) | 2919 / 19455 / 27842* (N=21) |
| burst | stand-in | 25 | 360 | 4.17 | 2811 / 17998 / 31470* (N=25) | 2813 / 17999 / 31472* (N=25) |
| overload | stand-in | 43 | 480 | 5.38 | 2059 / 14757 / 22946* (N=42) | 2146 / 14675 / 22828* (N=43) |
| rpm_halved | stand-in | 20 | 480 | 2.50 | 1965 / 19064 / 34460* (N=20) | 1967 / 19066 / 34461* (N=20) |
| daily_exhausted | stand-in | 20 | 480 | 2.50 | 4 / 1538 / 1579* (N=20) | 4 / 1539 / 1580* (N=20) |

\* N < 100: p99 is not a stable percentile.

**Table 2b: Outcomes, same rows**

| Scenario | Succeeded % | Rate-limited % | Timed out or given up % | Retries per request | Peak queue length | Tokens per answer | Cost per 1,000 answers |
|---|---|---|---|---|---|---|---|
| real_smoke | 100.0 | 0.0 | 0.0 | 0.00 | not measured | 3320 (N=3) | $0.63 |
| steady | 71.4 | 28.6 | 0.0 | 0.00 | 0 | 3186 (N=15) | $0.63 |
| burst | 24.0 | 76.0 | 0.0 | 0.00 | 1 | 2030 (N=6) | $0.62 |
| overload | 37.2 | 60.5 | 2.3 | 0.02 | 0 | 2314 (N=16) | $0.69 |
| rpm_halved | 30.0 | 70.0 | 0.0 | 0.10 | 0 | 1889 (N=6) | $0.76 |
| daily_exhausted | 15.0 | 85.0 | 0.0 | 0.00 | 0 | 1524 (N=3) | $0.25 |

Succeeded includes refusals. Cost = 1,000 x list-price cost of hosted-call tokens (gpt-oss-120b, $0.15 / $0.60 per 1M in / out, read 2026-11-11) / succeeded requests; actual free-tier spend $0.

**Every completed stand-in request was a refusal**, so the load tests measure admission, limits, retries and latency, not answers served. **Answer quality is not established:** three real questions were run end to end and all three were refused (**zero verified answers**): q16 (unanswerable) correctly; q05 and q11 by verifier false rejections of correct, cited claims (q11's draft also held an unsupported absence sentence). **The required 15-question evaluation was not completed:** real calls were capped at an approved 8 (7 used), and the eighth was held because the verifier still refused q05 and q11 locally. The 30 questions in eval/questions.jsonl were evaluated for retrieval only; stand-in replies are replayed fixtures; neither is real-model answer quality. **Number I could not have guessed:** "steady", at 70% of computed capacity, was still 28.6% rate-limited.

**Verifier evaluation (separate from the load tests; local; no Groq calls during v3/v4 work).** v4 adds a formula-identity check and a lexical check for unsupported absence claims. 18-case targeted set, v3 → v4: accuracy 55.6% → 83.3%, false acceptance 70.0% → 20.0%, false rejection 12.5% both. 28-case set: false acceptance 13.3% → 6.7% but false rejection 7.7% → 15.4%; accuracy 89.3% both. Original 8-pair held-out: unchanged at 62.5% (FA 0/4, FR 3/4). Small sets drafted by the AI assistant, not evidence of general reliability; the formula check rejects some valid derived formulas, and the absence check is lexical, not semantic.

## 4. Where the time goes
steady-0019 (q11), 29.9 s, from its trace: **waiting 26.8 s** (governor 13.0 s before the draft, 13.8 s before the redraft, for per-minute tokens), **working 3.1 s** (retrieval 1.5 s, provider 0.77 + 0.82 s), **retrying 0 s**; all **before the first token**.

## 5. When a limit is hit
**Per-minute:** refused at once if the queue is full or the estimated wait exceeds the deadline; admitted requests wait FIFO for tokens; whoever would miss their deadline gets "busy"; a 429 freezes the bucket until retry-after. With the limit halved the governor kept its own cap and over-admitted. **Per-day:** the first per-day 429 freezes the budget; 17 later users got the capacity reply at once, the provider saw 2 calls. **Nobody gets a weaker answer:** no fallback model is built.

## Retrieval: quality versus cost
A = 256-token pieces, 8 fetched, no re-rank; B = 256, 8, re-rank; C = 512, 4, re-rank; same budget.

**Table 3: Retrieval settings**

| Setting (piece size, pieces fetched, re-rank yes or no) | Questions asked | Correct and cited % | Wrong % | Correctly refused % | Tokens per answer | First token p50 / p95 (ms) | Total time p50 / p95 (ms) |
|---|---|---|---|---|---|---|---|
| A: 256 tokens, 8 pieces, re-rank no | 13 | not measured* | not measured* | not measured* | 2866 | 3599 / 9825 (N=13) | 3600 / 9827 (N=13) |
| B: 256 tokens, 8 pieces, re-rank yes | 13 | not measured* | not measured* | not measured* | 3078 | 2037 / 12275 (N=12) | 2073 / 11618 (N=13) |
| C: 512 tokens, 4 pieces, re-rank yes | 13 | not measured* | not measured* | not measured* | 2708 | 3003 / 15520 (N=13) | 3004 / 15521 (N=13) |

\* Not measured: stand-in replies are fixtures, so answer quality needs real-model runs.

Retrieval-only coverage per setting is supplementary (results/tables.md, Table 3b). One question moves coverage 12–14 points: not separable. I keep B as the default without evidence that it is better; near the limit retrieval keeps its full budget.

## How this library will mislead a naive retriever
1. **Different words:** "How is the input segmented in the Transformer compared with H-Net?" (byte-pair, word-piece): gold never found. 2. **Related work outranks results:** "At GPT-3 Large compute, what average accuracy does the 2-stage H-Net reach?". 3. **Rows without context:** "What dev BLEU does the base configuration get?": sibling rows outrank the base row.

## 6. Retries
At most 2 per call, on 5xx, timeouts and per-minute 429s; wait = max(retry-after, jittered backoff); never past the deadline or on a per-day 429; ≤ 10% of first attempts per minute; through the governor. 4 of 168 retried. **Worse:** rpm_halved-0019 waited 19.2 s, hit a 429, waited 3 s more, was rejected: "busy" after 38.3 s.

## 7. What the person sees
At once, before any model call: "The research assistant has used today's model capacity, so I can't write a new answer right now. Full answers resume in about N minutes" (N from the 429's retry-after). No guess, no cheaper model.

## 8. What I'm unsure about, what I left out, what I learned
**Least certain:** that a small NLI model can judge claims about formulas and tables; it refuses correct answers, and if it errs the other way the guarantee fails. **Not built:** the 20B fallback (no quality gate) and a cache (it would hide the load behaviour measured). **Learned:** with reasoning models a rate limit is a token budget, and a daily limit trickles back rather than resetting at midnight. **In short:** limits are handled predictably and no unverified sentence was sent, but correct answering is not shown.

## Your feedback on Vaani
[TO BE WRITTEN BY AKASH (50–60 words, replace this whole bracket): 3–5 lines from your own visit to https://www.vaani.dev/. Say what stood out to you, what confused you, and one thing you would change. There is no right answer; it must be your honest view. This placeholder is sized to the space reserved in the two pages.]

## How I worked with the assistant
[TO BE WRITTEN BY AKASH (140–160 words, replace this whole bracket), in your own words. The brief asks for: the three prompts that mattered most and why each mattered; one thing the assistant got wrong that you caught, and how you caught it; one decision where you overruled the assistant, and why; and what in this submission is distinctly yours. Candidate examples from the project record are listed in writeup/PERSONAL_SECTIONS_TODO.md; they are suggestions only and must be confirmed or replaced by you. End with one line saying that the unedited conversation export is submitted alongside this report, and give its file name. This placeholder is sized to the space reserved in the two pages, so keep the final text about this long or the report will exceed the two-page limit; run the build and render steps afterwards and check that the PDF is still exactly two pages.]
