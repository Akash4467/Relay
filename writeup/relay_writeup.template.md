# The Relay — Akash Bisht

## 1. What I built and why
Cited Q&A over the three papers: Python pipeline of plain functions, Go stand-in model service, Go open-loop load generator. **Retrieve** (the tool: BM25 + dense + cross-encoder re-rank, 1,600-token budget) → **structural check** (no usable passage: refuse, no model call) → **draft** (Groq gpt-oss-120b via a token-bucket governor) → **verify** (each sentence cites a retrieved passage that a local NLI model finds entails it at ≥ 0.9) → **one redraft** (the loop) → answer or **refuse**. **Shaping decision:** nothing is sent until the whole answer verifies, so first token = total time and no unverified sentence reaches a user. **Versions:** load tests and results.csv used verifier **v2**; the submitted verifier is **v4**, never load-tested; threshold 0.9 throughout.

## 2. The real service and my stand-in
7 real calls (10,147 tokens, no 429); limits in Table 1. Measured: a fixed 67-token preamble per request, reasoning tokens count against the budget, and daily requests refill continuously (one per 86.4 s). The stand-in replays the real answers and token counts, latency fitted to the 5 answer calls (first byte median 515 ms), with Groq-style limits and 429s, streaming and faults. **Least like the real thing:** 5 calibration calls; slowdown, faults and the 429 body are guesses; replies ignore prompt and retrieval, so it measures capacity, never quality.

## 3. Results
real_smoke is Groq; all other rows are the stand-in; all ran verifier v2.

{{TABLE1}}

{{TABLE2A}}

{{TABLE2B}}

**Every completed stand-in request was a refusal**, so the load tests measure admission, limits, retries and latency, not answers served. **Answer quality is not established:** three real questions were run end to end and all three were refused (**zero verified answers**): q16 (unanswerable) correctly; q05 and q11 by verifier false rejections of correct, cited claims (q11's draft also held an unsupported absence sentence). **The required 15-question evaluation was not completed:** real calls were capped at an approved 8 (7 used), and the eighth was held because the verifier still refused q05 and q11 locally. The 30 questions in eval/questions.jsonl were evaluated for retrieval only; stand-in replies are replayed fixtures; neither is real-model answer quality. **Number I could not have guessed:** "steady", at 70% of computed capacity, was still 28.6% rate-limited.

**Verifier evaluation (separate from the load tests; local; no Groq calls during v3/v4 work).** v4 adds a formula-identity check and a lexical check for unsupported absence claims. 18-case targeted set, v3 → v4: accuracy 55.6% → 83.3%, false acceptance 70.0% → 20.0%, false rejection 12.5% both. 28-case set: false acceptance 13.3% → 6.7% but false rejection 7.7% → 15.4%; accuracy 89.3% both. Original 8-pair held-out: unchanged at 62.5% (FA 0/4, FR 3/4). Small sets drafted by the AI assistant, not evidence of general reliability; the formula check rejects some valid derived formulas, and the absence check is lexical, not semantic.

## 4. Where the time goes
steady-0019 (q11), 29.9 s, from its trace: **waiting 26.8 s** (governor 13.0 s before the draft, 13.8 s before the redraft, for per-minute tokens), **working 3.1 s** (retrieval 1.5 s, provider 0.77 + 0.82 s), **retrying 0 s**; all **before the first token**.

## 5. When a limit is hit
**Per-minute:** refused at once if the queue is full or the estimated wait exceeds the deadline; admitted requests wait FIFO for tokens; whoever would miss their deadline gets "busy"; a 429 freezes the bucket until retry-after. With the limit halved the governor kept its own cap and over-admitted. **Per-day:** the first per-day 429 freezes the budget; 17 later users got the capacity reply at once, the provider saw 2 calls. **Nobody gets a weaker answer:** no fallback model is built.

## Retrieval: quality versus cost
A = 256-token pieces, 8 fetched, no re-rank; B = 256, 8, re-rank; C = 512, 4, re-rank; same budget.

{{TABLE3}}

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
