# The Relay — Akash Bisht

## 1. What I built and why
Cited Q&A over the three papers: Python pipeline of plain functions, Go stand-in model service, Go open-loop load generator. **Retrieve** (the tool: BM25 + dense + cross-encoder re-rank, 1,600-token budget) → **structural check** (no usable passage: refuse, no model call) → **draft** (Groq gpt-oss-120b via a token-bucket governor) → **verify** (each sentence cites a retrieved passage that a local NLI model finds entails it at ≥ 0.9) → **one redraft** (the loop) → answer or **refuse**. **Shaping decision:** nothing is sent until the whole answer verifies, so first token = total time and no unverified sentence reaches a user. **Versions:** load tests and results.csv used verifier **v2**; production runs **v4.2**, the live-measured candidate **v4.8c** (§3); neither was load-tested; threshold 0.9.

## 2. The real service and my stand-in
First: 31 real calls (58,617 tokens, no 429): 7 to calibrate (10,147 tokens), 24 for the 15-question evaluation; limits in Table 1. Measured: a 67-token preamble per request, reasoning tokens count against the budget, and daily requests refill continuously (one per 86.4 s). The stand-in replays the real answers and token counts, latency fitted to the 5 answer calls (first byte median 515 ms), with Groq-style limits and 429s, streaming and faults. **Least like the real thing:** 5 calibration calls; slowdown, faults and the 429 body are guesses; replies ignore the prompt, so it measures capacity, never quality.

## 3. Results
real_smoke is Groq; all other rows are the stand-in; all ran verifier v2.

{{TABLE1}}

{{TABLE2A}}

{{TABLE2B}}

**Every completed stand-in request was a refusal**, so the load tests measure operations, not answers served. **Number I could not have guessed:** "steady", at 70% of computed capacity, was still 28.6% rate-limited.

**Earlier (production, verifier v4, my 15 questions):** correct 0.0%, wrong 6.7% (q12), correctly refused 33.3%.

{{E2E}}

## 4. Where the time goes
steady-0019 (q11), 29.9 s: **waiting 26.8 s** (governor 13.0 s before the draft, 13.8 s before the redraft, for per-minute tokens), **working 3.1 s** (retrieval 1.5 s, provider 0.77 + 0.82 s), **retrying 0 s**; all **before the first token**.

## 5. When a limit is hit
**Per-minute:** refused if the queue is full or the estimated wait exceeds the deadline; admitted requests wait FIFO for tokens; whoever would miss their deadline gets "busy"; a 429 freezes the bucket until retry-after. With the limit halved the governor kept its own cap and over-admitted. **Per-day:** the first per-day 429 freezes the budget; 17 later users got the capacity reply, the provider saw 2 calls. **Nobody gets a weaker answer:** no fallback model is built.

## Retrieval: quality versus cost
A = 256-token pieces, 8 fetched, no re-rank; B = 256, 8, re-rank; C = 512, 4, re-rank; same budget.

{{TABLE3}}

Retrieval-only coverage: results/tables.md, Table 3b; one question moves it 12–14 points. I keep B as default (not shown better); live quality was measured for B only; near the limit retrieval keeps its full budget.

## How this library will mislead a naive retriever
1. **Different words:** "How is the input segmented in the Transformer compared with H-Net?": gold never found. 2. **Related work outranks results:** "At GPT-3 Large compute, what average accuracy does the 2-stage H-Net reach?". 3. **Rows without context:** "What dev BLEU does the base configuration get?": sibling rows outrank the base row.

## 6. Retries
At most 2 per call, on 5xx, timeouts and per-minute 429s; wait = max(retry-after, jittered backoff); never past the deadline or on a per-day 429; ≤ 10% of first attempts per minute; through the governor. 4 of 168 retried. **Worse:** rpm_halved-0019 waited 19.2 s, hit a 429, waited 3 s more, was rejected: "busy" after 38.3 s.

## 7. What the person sees
At once, before any model call: "The research assistant has used today's model capacity, so I can't write a new answer right now. Full answers resume in about N minutes" (N from the 429's retry-after). No cheaper model.

## 8. What I'm unsure about, what I left out, what I learned
**Least certain:** that a small NLI model can judge claims about formulas and tables; it refused correct answers, and when it erred the other way (q12) a false sentence was sent. **Not built:** the 20B fallback (no quality gate) and a cache (it would hide the load behaviour measured). **Learned:** with reasoning models a rate limit is a token budget, and a daily limit trickles back rather than resetting at midnight. **Candidates (not deployed):** c6 corpus; R3b retrieval (paper terms; fixes q14's retrieval); p7; v4.8c (exact quotes, cells; check-mark, "not stated" checks). §3's v4.8c row rests on the recorded verifier module; its label wrongly reads v4.2 (assessment/e2e/live/ERRATA.md); no v4.8c-only safeguard fired, so it cannot separate v4.8c from v4.7c. **Open:** refused q01, q07, q08, q12; partial q06, q09–q11, q14 (omits word-piece; ambiguous tokenizer wording); t10, t29 not retrieved; copied-row style; a Moshi / Inner Monologue gap. **In short:** limits are handled predictably, but grounding as built refuses too much to be useful.

## Your feedback on Vaani
I found Vaani's website easy to navigate, convenient to use, and well-designed from a UI perspective. It explains the product clearly and makes the overall experience straightforward. I haven't explored the backend or technical implementation, so I cannot suggest improvements in that area. Based on my experience navigating the website, the overall experience was excellent.

## How I worked with the assistant
I used AI throughout the project, beginning with research into the problem, reviewing similar projects, and exploring alternative approaches. My first important prompts asked it to investigate the topic, identify existing solutions, and present different implementation options without prematurely ranking them. I then made the architectural and technical decisions based on my own research and understanding. My subsequent prompts focused on implementing my designs, debugging, testing, evaluating the system, and preparing documentation. I also reviewed the assistant's output rather than accepting it blindly. For example, I caught outdated documentation describing a LangGraph architecture and a 20B fallback that were not part of the implemented system. I chose to document the actual implementation and preserve the historical v2 measurements rather than reconstruct an old implementation. The research direction, architecture, design choices, evaluation decisions, and final technical work were my own. I used AI as an implementation and analysis assistant, not as the decision-maker.
