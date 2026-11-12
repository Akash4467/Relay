# The Relay — Decisions & Architecture

Nov 11, 2026 · @Akash Bisht

## Stack

Python runs the RAG application; Go runs the stand-in model service and the open-loop load generator, where concurrency and timing precision matter.

| Component | Technology |
| --- | --- |
| Main application | Python + FastAPI, single worker (one event loop, so in-memory limiter state is global); embeddings and cross-encoder run in a bounded thread pool, never on the event loop; ingestion runs offline |
| Orchestration | LangGraph (explicit state machine), every node wrapped by a timing span |
| Admission and rate control | In-process: response cache, single-flight, deadline queue, per-model token buckets |
| Real LLM | Groq (OpenAI-compatible API): openai/gpt-oss-120b primary, openai/gpt-oss-20b fallback |
| HTTP client | httpx |
| Streaming | SSE |
| Stand-in server | Go (net/http, goroutine per stream) |
| Load generator | Go, open-loop Poisson arrivals with simulated user patience |
| Keyword retrieval | BM25 |
| Embeddings | Sentence Transformers, BGE-small |
| Vector store | FAISS |
| Reranker | Local cross-encoder (kept only if Table 3 justifies it) |
| PDF validation | PyMuPDF |
| Chunk metadata | JSON |
| Traces | JSONL, one span per node / call / wait / retry |
| Results | Pandas: traces → results.csv → tables |

Provider switch by environment variables only; the client code is identical for both:

| Variable | Real | Stand-in |
| --- | --- | --- |
| LLM\_PROVIDER | groq | standin |
| GROQ\_BASE\_URL | https://api.groq.com/openai/v1 | http://localhost:8001/v1 |
| GROQ\_API\_KEY | own key (never committed) | dummy |
| MODEL\_NAME | openai/gpt-oss-120b | openai/gpt-oss-120b |

## Request path and admission

A system-wide admission layer sits in front of the graph and every request carries a deadline. System pressure is one global signal; the model and context for each request are decided per request at dispatch.

1. Arrival: request stamped with arrival time and deadline (= simulated user patience).
2. Normalize conservatively: lowercase and collapse whitespace only; hyphens, apostrophes and technical symbols are kept.
3. Cache (built after the baseline): key = hash(normalized question, model, retrieval config version, corpus version, prompt version, verification policy version, generation config version). Stores the answer with its citations and verification metadata. Hit → return immediately, 0 hosted tokens. A bypass flag is on for every load and quality experiment.
4. Single-flight (built after the baseline): one shared task per key with independent waiters. A waiter that leaves (disconnect or its own deadline) never cancels the task; the task is cancelled only when no waiters remain or its own maximum deadline passes.
5. Admission queue: bounded FIFO. Rejected on arrival when the queue is full or the estimated wait exceeds the remaining deadline; expired requests are removed before dispatch. Queue rejection and deadline expiry are recorded separately.
6. Concurrency limit: semaphore on in-flight hosted calls.
7. Graph runs (see Pipeline architecture); the model is chosen per request and the call goes through that model's rate governor.

### Rate governor

- Two levels of buckets. Organization level: ITPM and OTPM, created only if the limits page shows an "X in / Y out" breakdown, and any other organization-wide limit confirmed for the account. Model level: RPM, RPD, TPM, TPD per model. A request must acquire every applicable bucket before dispatch.
- Groq documents limits as organization-wide (shared by all keys and users) and per model; the shared-capacity check in Limit probing confirms whether the 120B and 20B token buckets are really independent. If not, a shared organization TPM bucket is added.
- Reserve before calling: input tokens from a local tokenizer (checked against reported usage), plus p95 output tokens including reasoning, plus a 10% safety margin. Input and output reserve against ITPM and OTPM separately when those exist. Settled to actual usage when the response ends; overruns (actual > reservation) are counted and logged.
- Local buckets are an admission estimate, not the provider's quota. A 429 is authoritative: that bucket freezes until retry-after or reset, and its effective limit is lowered when the 429 arrived while the local estimate said the request fit.
- Headers expose only RPD (requests) and TPM (tokens); RPM, TPD, ITPM and OTPM are tracked locally and learned from 429s. A missing or malformed header keeps the local estimate and is logged.
- Estimated wait for a slot is computed from the binding bucket (in-flight slots, token availability, queued reservations ahead), not queue length × average latency, and is exposed to the admission queue and to the model decision.

### System pressure and per-request decisions

System pressure is one global signal; it sets the context cap, expansion and redraft. Values are initial and get recalibrated from measured TPM.

| Pressure | Trigger | Context sent to Groq | Expansion | Redraft |
| --- | --- | --- | --- | --- |
| L0 Healthy | 120B buckets healthy | ≤ 2,500 tokens | Paragraph, neighbours if needed | Allowed (120B only) |
| L1 Constrained | 120B TPM bucket < 50% or queue growing | ≤ 1,500 tokens | Containing paragraph only | Off |
| L2 Critical | 120B cannot serve within typical deadlines | ≤ 1,000 tokens | Off | Off |
| L3 Exhausted | Both models' daily quota gone | None | Off | Off |

### Model decision (per request, at dispatch)

| Situation | Decision |
| --- | --- |
| 120B has capacity now | 120B |
| 120B estimated wait ≤ 2 s and fits the request's deadline | Wait for 120B |
| 120B wait > 2 s or past the deadline, 20B has capacity, category passes the 20B gate | 20B |
| 120B daily quota gone, category passes the 20B gate | 20B |
| Category fails the 20B gate | Honest "reduced service" refusal for that question type |
| 20B also cannot serve within the deadline | Shed with an honest unavailability reply |
| 20B answer fails verification | Refuse, no redraft |

20B suitability gate: classify\_query tags each question with a category (direct, multi-section, numerical/table, cross-paper) by rules. 20B fallback is allowed only for categories where the model benchmark shows 20B correct-and-cited within 10 points of 120B and its false acceptance no higher; an unvalidated fallback is never used just to raise availability.

### Answer tiers

Every reply carries a tier, shown to the user and recorded in results.csv: normal (120B), reduced (20B; the reply says so), refused (insufficient evidence or gated category), unavailable (quota or shed). A reduced-tier answer is never presented as equal in quality because it passed evidence checks.

L3 reply: cached verified answer if one exists; otherwise an honest unavailability message with the quota reset time.

### Retry policy

- At most 2 transport retries per hosted call, exponential backoff with full jitter, never shorter than retry-after.
- Never retry a per-day 429: go to the model decision (20B or unavailable).
- Never retry once the request's deadline has passed.
- Retry budget: transport retries in any minute ≤ 10% of initial hosted attempts in that minute (100 initial → at most 10 retries); over budget → fail fast.
- Every retry re-reserves through the governor, so retries cannot bypass the limits.

| Counter | Meaning |
| --- | --- |
| Initial attempts | First hosted call of a generation |
| Transport retries | Re-sending a failed HTTP call |
| Redraft attempts | New generation to correct an answer |
| Fallback generations | Generation on another model |

Redrafts and fallbacks are not transport retries; each counter has its own column.

## Pipeline architecture

LangGraph with an explicit state machine: fixed nodes and conditional edges; the LLM never chooses the route.

&#91;embedded content: LangGraph state machine · 11 nodes, level-aware routing, 1 bounded loop\]

| Node | Responsibility | LLM call |
| --- | --- | --- |
| classify\_query | Identify greeting, research question or unsupported request (rules) | No |
| budget\_check | Copy system pressure (L0–L3) and scope result into state; route L3 to unavailable | No |
| canned\_response | Reply to greetings / out-of-scope requests | No |
| unavailable | L3 reply: cached verified answer, else unavailability message with reset time | No |
| retrieve | Hybrid search, rerank, expansion, trimmed to the level's context cap (tool call) | No |
| relevance\_check | Cross-encoder threshold + evidence sufficiency | No |
| draft | Generate the complete cited answer; model chosen per request (120B or 20B) | Yes, 1 |
| verify | Evidence consistency checks on the complete answer | No |
| redraft | Correct unsupported answer; only at L0 and only before anything is released | Yes, max 1 |
| refuse | Justified refusal (insufficient evidence, failed verification when degraded, second failure) | No |

Every node runs inside a timing wrapper that records start, end, governor wait and retries to the request trace.

Requirements met: more than four steps, a retrieval tool call, conditional branches, a bounded retry loop.

## Streaming and verification

The production path is Option 1: generate the complete answer, verify it, redraft once if allowed, then stream the verified answer to the user. One verification system, one release guarantee. Buffered sentence release (Option 4 + 5) is a separate experiment only, run if time allows and reported against Option 1; it never becomes the production path in this submission, because released sentences cannot be qualified or retracted by later ones.

Experiment design (Option 4 + 5), if run:

- Per-sentence checks run inside the draft node as tokens arrive; the verify node keeps only whole-answer checks.
- Redraft is allowed only before the first sentence is released.
- A sentence that fails after release is dropped and replaced with "The papers do not state this."
- The prompt forces one citation per sentence; the sentence splitter protects e.g., et al., §, decimals and Fig.

Flow:

1. Retrieve evidence (top-k chunks).
2. Validate retrieval: relevance, source IDs, evidence sufficiency.
3. Generate a streaming draft; tokens received internally, not exposed.
4. Sentence buffer: verify each completed claim unit.
   - Verified → release sentence to the user.
   - Failed → hold, correct or refuse.
5. Complete response.

Verification is an evidence consistency check, not proof of factual correctness:

| Level | Check | Technique |
| --- | --- | --- |
| 1 | Citation integrity | Deterministic: cited chunk ID and section exist and were retrieved for this request |
| 2 | Evidence matching | Deterministic: every number, unit and key term in a sentence appears in its cited chunk |
| 3 | Semantic support | Optional local NLI model (cited chunk entails the sentence); kept only if it lowers false acceptance at acceptable latency |

Levels 1–2 cannot catch a claim that reuses the evidence's words with a different meaning (evaluation hardware reported as training hardware). The verifier's false acceptance rate (incorrect claims it approved) is measured on the eval set and reported.

| Metric | Definition |
| --- | --- |
| Model TTFT | Time until Groq returns its first token |
| User-visible answer TTFT | Time until the first verified answer sentence reaches the user |
| Transport TTFT | Reported separately, if measured |
| Total latency | Time until the complete response is delivered |

Progress messages ("Searching papers…") never count as the first answer token.

## Ingestion

Primary source is the exact-version arXiv LaTeX; the PDF is kept for page-level validation and as a citation fallback.

| Paper | arXiv ID | Version |
| --- | --- | --- |
| Attention Is All You Need | 1706.03762 | v7 |
| Moshi: a speech-text foundation model for real-time dialogue | 2410.00037 | v1 |
| Dynamic Chunking for End-to-End Hierarchical Sequence Modeling | 2507.07955 | v2 |

1. Download versioned source archives + PDFs.
2. Extract text with a custom Python parser, preserving section hierarchy.
3. Normalize equations, references and LaTeX macros (never blanket-strip commands; keep mathematical meaning).
4. Structure as section → subsection → paragraph.
5. Chunk with stable IDs.
6. Store section and source metadata with every chunk (JSON).
7. Validate against the PDF with PyMuPDF.

Citation format: Paper + Section + Chunk ID. Parser scope is limited to what the three papers use: sections and subsections, paragraph boundaries, equations, tables, figure captions, references, \\input files and the papers' own macros. No general LaTeX handling. A paper whose extraction stays broken switches to Docling or GROBID for that paper only.

## Chunking and retrieval

Hybrid search (BM25 + dense) fused with Reciprocal Rank Fusion, over section-aware chunks, with small-to-big expansion and tables stored as their own records.

| Layer | Choice |
| --- | --- |
| Chunking | Section-aware |
| Keyword retrieval | BM25 |
| Semantic retrieval | BGE-small |
| Fusion | RRF (rank-based, no raw-score comparison) |
| Reranking | Local cross-encoder, kept only if Table 3 shows it is worth its latency; bounded thread pool |
| Evidence expansion | Containing paragraph → neighbouring paragraphs if needed → whole section only for broad questions; context cap enforced after expansion; table rows keep their headers and units |
| Tables | Structured table records with their own metadata |
| Vector storage | FAISS |
| Citation | Paper + Section + Chunk ID |

Retrieval flow: question → BM25 + dense → RRF → cross-encoder rerank → evidence expansion → top-k evidence → LLM.

### Unanswerable questions

Refuse when evidence is insufficient; similarity scores alone are not proof.

1. Relevance threshold on the cross-encoder score when reranking is on, otherwise on the top dense cosine similarity (RRF scores are rank-based and uncalibrated, never thresholded).
2. Evidence sufficiency check, deterministic: the question's key entities must co-occur in the evidence.
3. Citation-grounded answer verification.

Thresholds are fitted on a dev split of the question set and reported on a held-out test split.

### Retrieval budget policy

| Level | Context sent to Groq | Retrieval behaviour |
| --- | --- | --- |
| L0 Healthy | ≤ 2,500 tokens | Ranked chunks + bounded expansion + table rows, filled best-first to the cap |
| L1 Constrained | ≤ 1,500 tokens | Ranked chunks, containing paragraph only |
| L2 Critical | ≤ 1,000 tokens | Strongest ranked chunks only |
| L3 Exhausted | None | Retrieval runs only to serve the unavailable reply; no generation |

"Ranked" means cross-encoder order when reranking is enabled, RRF order otherwise; the policy works either way.

The budget is set in context tokens, not chunk counts, because input tokens set throughput: answers per minute ≈ measured TPM ÷ tokens per answer. Caps are initial and recalibrated from the measured TPM. Retrieval is never cut so far that answers become unsupported; below the cap's minimum useful evidence, the request refuses.

Table 3 settings (initial), all on the same questions:

| Setting | Piece size | Candidates after RRF | Re-rank | Final passages | Expansion |
| --- | --- | --- | --- | --- | --- |
| A | 256 tokens | 20 | No | 4 | No |
| B | 256 tokens | 20 | Yes | 4 | No |
| C | 512 tokens | 20 | Yes | 4 | No |
| D (optional, if quota remains) | 256 tokens | 20 | Yes | 4 | Yes |

Each pair changes one variable: A vs B isolates reranking, B vs C isolates piece size, B vs D isolates expansion. The amount fetched is varied separately by the context caps of the pressure levels.

## Models and Groq

Model cascade with one hosted LLM call per normal request: openai/gpt-oss-120b primary ("120B"), openai/gpt-oss-20b fallback ("20B"). Groq shut down llama-3.3-70b-versatile and llama-3.1-8b-instant for free and developer-tier accounts on 16 Aug 2026 and recommends these two as replacements; both are listed as production models ([deprecations](https://console.groq.com/docs/deprecations.md), [models](https://console.groq.com/docs/models.md)). Both IDs are confirmed with a one-token call on the account before any experiment.

Both are reasoning models, which changes three things:

- Hidden reasoning tokens are generated before the first answer token, so model TTFT is split into first-byte and first-content-token; user-visible TTFT counts from first content.
- Reasoning tokens are output tokens: they count toward TPM (and OTPM if enabled) and must be in the reservation. Measured as total output tokens minus visible answer tokens, since reported reasoning-token counts may read 0.
- reasoning\_effort is a cost lever: low by default (citation-grounded extraction needs little reasoning); low vs medium compared on the dev split if quota allows. Reasoning is hidden from the response.

Groq's sample limits table lists 30 RPM, 1K RPD, 8K TPM and 200K TPD for both gpt-oss models ([rate limits](https://console.groq.com/docs/rate-limits.md)); the account's limits page is authoritative. At 8K TPM and about 3.5K tokens per answer, 120B serves about 2 answers per minute: TPM, not RPM, is the binding limit.

| Item | Decision |
| --- | --- |
| Primary model | openai/gpt-oss-120b, reasoning\_effort low |
| Fallback model | openai/gpt-oss-20b, reasoning\_effort low; per category, only where the 20B gate passes |
| Normal hosted calls | 1 per request |
| Verification | Evidence consistency checks (local) |
| Retry policy | Bounded backoff with jitter (see Request path and admission) |
| Quota tracking | Per-model RPM, TPM, RPD, TPD; ITPM/OTPM only if the limits page shows them |
| Load experiments | Real + stand-in |

| Stage | Implementation | Hosted LLM calls |
| --- | --- | --- |
| Query understanding | Local rules (also tags the question category) | 0 |
| Retrieval | BM25 + dense | 0 |
| Evidence validation | Local relevance checks | 0 |
| Answer generation | 120B, or 20B per the model decision | 1 |
| Citation verification | Evidence consistency checks | 0 |
| Correction | 120B, only at L0 | max 1 |

Model choice per request comes from the governor's degradation level and the fallback rule (Request path and admission), not from a per-request quota check. On the 20B path, redraft is off.

### Model benchmark (both models, same 15–20 questions)

- Correct answer %
- Correct citation %
- Correct refusal %
- Input and output tokens
- TTFT and total latency

### Rate-limit headers to log

| Header | Meaning |
| --- | --- |
| x-ratelimit-limit-requests | Requests per day limit |
| x-ratelimit-remaining-requests | Remaining daily requests |
| x-ratelimit-limit-tokens | Tokens per minute limit |
| x-ratelimit-remaining-tokens | Remaining tokens per minute |
| x-ratelimit-reset-requests | Daily request quota reset |
| x-ratelimit-reset-tokens | Token quota reset |
| retry-after | Wait after a 429, when provided |

Request headers are not treated as a complete RPM measurement. Also recorded per call: HTTP status, response body, model ID, request timestamp, input/output tokens, TTFT, total duration.

### Concurrency probing

No assumed fixed concurrency limit. Controlled tests at 1, 2, 4, 8 and 16 concurrent requests, each measuring successes, 429s, timeouts, TTFT and total latency. Rising latency with continued success is not treated as a hard limit.

### Limit probing

Results are reported as an observed operating boundary under the test conditions, never as the definitive limit.

1. Read the account's limits page first (RPM, RPD, TPM, TPD, any ITPM/OTPM breakdown); nothing else runs on the organization during probes, since limits are organization-wide.
2. RPM probe on 20B: documented RPM + 5 tiny requests (max\_tokens 1) spread over one minute, so TPM cannot be the cause. Record each timestamp, status and 429 body, which names the limit that was hit.
3. TPM probe separately: few requests with large prompts, so RPM cannot be the cause.
4. Recovery: after the first 429, one probe per retry-after interval until success, not a continuous stream.
5. Shared-capacity check: one 120B call, then compare 20B's x-ratelimit-remaining-tokens before and after, to confirm that per-model buckets are independent.
6. Repeat once at another time if results are inconsistent.

| Report | Value |
| --- | --- |
| Documented limit | From the limits page |
| Successful requests before the first 429 | Observed |
| Number of 429s | Observed |
| Time until the next success | Observed |
| Could another limit explain the 429? | From the 429 body |

### Real-call budget

| Priority | Use | Model | Calls |
| --- | --- | --- | --- |
| 0 | Model ID check, one token each | 120B + 20B | 2 |
| 1 | Latency and token baseline | 120B + 20B | 15 |
| 2 | Limit probing (RPM, TPM, recovery, shared-capacity check) | 20B first, then 120B | ≈ 45 |
| 3 | Retrieval comparison: 3 settings × 30 questions | 120B | 90 |
| 4 | Model comparison: chosen setting × 30 questions | 20B | 30 |
| 5 | Integration and steady real run | 120B | 15 |
| — | Reserve: redrafts, retries, surprises (≈ 10%) | Both | 20 |
| — | Optional setting D × 30 | 120B | 30 |

TPD, not RPD, is the planning limit: ≈ 140 120B calls × ≈ 3.5K tokens ≈ 490K tokens, about 2.5 days of a 200K TPD budget. The retrieval comparison starts by day 2; setting A's smaller context costs less.

The plan is a ceiling, not a spending target: if quota runs short, calls are spent in priority order, and limit probing never eats into the quality evaluation. Calls are spread over 2–3 days to fit the measured per-model daily limits; every real answer is recorded as a fixture for the stand-in. Every hosted request counts, including retries, redrafts and fallback calls. Daily exhaustion is simulated in the stand-in, not forced on the real quota; the observed real daily limit is documented separately. Burst and sustained overload run on the stand-in.

## Stand-in service

A separate Go server that mimics Groq's OpenAI-compatible API, so the pipeline cannot tell which provider it is calling; Go keeps its overhead low, and the server logs its own processing time so client-observed latency splits into simulated delay, server overhead and transport. Only the endpoints and fields the app uses are implemented: POST /v1/chat/completions, with SSE when stream=true.

Two modes, never mixed:

| Mode | Purpose | Reply content |
| --- | --- | --- |
| A Performance | Load scenarios: latency, limits, failures, queueing | Deterministic fixtures matched by question ID: the recorded real answer with its real token counts. The load generator only sends questions that have fixtures; no random replay |
| B Functional | Correctness of the pipeline: verification, refusal, redraft, fallback | Fixed question-answer-evidence fixtures, including deliberately unsupported and wrong-meaning answers to measure the verifier |

Answer quality is measured only on real-service runs, never from stand-in replies.

| Capability | Implementation |
| --- | --- |
| Streaming | SSE chunks, text/event-stream, at measured tokens/second; a hidden reasoning phase of measured length precedes the first content token |
| Latency | TTFT sampled from a distribution fitted to real measurements, scaled by input tokens; generation time from output tokens ÷ measured rate; per model |
| Load degradation | TTFT and token rate worsen with in-flight requests, curve fitted from the real concurrency tests (1–16) |
| Rate limits | RPM, TPM, RPD, TPD per model, independent; window type matches what the real service showed |
| 429 responses | retry-after, x-ratelimit-\* headers, OpenAI-compatible error body, copied from real 429s |
| Random failures | Configurable HTTP 500 rate |
| Slow success | Configurable share of calls that complete correctly but far slower |
| Mid-stream stall | Stream pauses after some tokens, then resumes or drops |
| Timeouts | Responses delayed past the client timeout |
| Self-timing | Server logs its processing time per request |

Known gaps to report: no network or region effects; Groq's real scheduling is unknown; one concurrency curve cannot capture model-specific or time-of-day behaviour; fixtures do not change when prompts or retrieval change, so stand-in runs say nothing about answer quality.

## Evaluation

One fixed set of 30 self-written questions with gold answers and gold citations, reused across every retrieval setting and both models: 15 answerable, 15 unanswerable or misleading. Split 15 dev (threshold fitting) / 15 test, stratified by category; the test split is not looked at during tuning.

| Question category | Count | Example |
| --- | --- | --- |
| Direct factual, exact terminology | 4 | Single-passage fact using a rare term (Mimi, dechunking) |
| Multi-section | 5 | Needs evidence from more than one section |
| Numerical / table-based | 6 | Value read from a results table |
| Unanswerable, near-miss | 10 | What GPU was used to train Moshi? (retrieves plausible but insufficient chunks) |
| Misleading / cross-paper | 5 | Term defined differently across papers, or a claim attributed to the wrong paper |

Grades per request (results.csv): correct, wrong, refused\_correctly, refused\_wrongly.

| Metric | Meaning |
| --- | --- |
| False acceptance rate | Answered although the papers do not support it (most important) |
| False refusal rate | Refused although the papers support it |
| Answer precision / recall | Of accepted questions, share truly answerable / of answerable questions, share accepted |
| Verifier false acceptance | Incorrect claims the verifier approved (Mode B fixtures + real answers) |

On 15 test questions one question = 6.7 points; smaller differences are reported as noise, settings are then chosen on tokens and latency, and no threshold is claimed to be production-grade.

### Verifier stress test (reported separately)

About 30 hand-built claim/evidence pairs, run locally with no hosted calls and never mixed into the retrieval-quality score. Each targets one failure mode:

| Evidence says | Claim says | Expected |
| --- | --- | --- |
| GPU used for evaluation | GPU used for training | Reject |
| Model A reaches 90% | Model B reaches 90% | Reject |
| Latency is 100 ms | Latency is 100 seconds | Reject |
| Paper discusses acoustic tokens | Paper defines semantic tokens | Reject |
| Value in Table 2 | Same value attributed to Table 3 | Reject |
| Exact supported sentence | Same claim, paraphrased | Accept |

False acceptance rate (FAR) = unsupported claims accepted ÷ unsupported claims tested, reported per verifier level (1–2 deterministic, + level 3 NLI if run).

## Measurement and results.csv

Every figure in the tables is computed by script from results.csv, which is derived from the per-request JSONL traces.

- Load generator: Go, open-loop Poisson arrivals at the scenario's offered rate; each simulated user gives up at its patience deadline and is recorded as given up.
- Scenarios: steady, sudden burst, sustained overload, per-minute limit halved, daily budget exhausted.

| Column | Definition |
| --- | --- |
| request\_id | One row per user request, not per hosted call |
| scenario / source | Scenario name; real or stand-in |
| start\_time | Arrival at the front door |
| first\_token\_ms | User-visible answer TTFT: arrival → first answer content sent, including queue and governor wait |
| total\_ms | Arrival → last byte sent |
| status | ok (any reply sent in time: answer, refusal, canned or cached), rate\_limited (shed or final 429), timeout (deadline passed, user gave up, upstream timeout), error |
| retries | Transport retries only, summed across the request's hosted calls; redrafts and fallbacks have their own columns |
| tokens\_in / tokens\_out | Summed across draft + redraft; 0 for cache hits |
| retrieval\_setting | A, B or C |
| answer\_grade | correct, wrong, refused\_correctly, refused\_wrongly |

Extra columns: failure\_reason, model, answer\_tier, degradation\_level, cache\_hit, queue\_wait\_ms, governor\_wait\_ms, retrieval\_ms, generation\_ms, verification\_ms, model\_ttft\_ms, redraft\_attempts, fallback\_generations, reasoning\_tokens, first\_byte\_ms, reservation\_overrun\_tokens, question\_category. tokens\_out includes reasoning tokens.

| failure\_reason | Who failed | status |
| --- | --- | --- |
| queue\_rejected | Server: queue full or wait > deadline at arrival | rate\_limited |
| deadline\_expired | Server: expired while queued | timeout |
| local\_governor\_reject | Server: no capacity on any model within deadline | rate\_limited |
| upstream\_429 | Provider: 429 after retries | rate\_limited |
| internal\_timeout | Upstream call or stream exceeded its timeout | timeout |
| gave\_up | User: patience ran out before the reply | timeout |
| upstream\_error | Provider: 5xx after retries | error |

User gave up, server rejected, provider rejected and internal timeout are never merged.

Cost per 1,000 answers = 1,000 × (estimated cost of all hosted calls, including retries, redrafts and fallbacks) ÷ successful answers, at the published per-model input/output prices with their date. Actual free-tier spend ($0) is reported separately, and so is cost per attempted request.

Prices used: gpt-oss-120b $0.15 input / $0.60 output, gpt-oss-20b $0.075 / $0.30 per 1M tokens ([Groq models page](https://console.groq.com/docs/models.md), read 11 Nov 2026).

Cache and single-flight are measured in isolation, on the stand-in, after all main runs (which keep both off):

1. Baseline: cache and single-flight off.
2. Cache on: repeated questions arriving apart in time.
3. Single-flight on, cache off: identical questions arriving concurrently.
4. Both on.

## Build order

A working end-to-end request comes before any control-plane work.

| Phase | Deliverable | Done when |
| --- | --- | --- |
| 1 | Exact-version ingestion and hybrid retrieval | Gold chunks retrieved for the dev questions |
| 2 | One request end to end: retrieve, cited answer, evidence checks, refusal | An unsupported claim is caught and an unanswerable question refused |
| 3 | LangGraph orchestration and tracing | Per-node spans in JSONL |
| 4 | Groq client and rate governor | Real baseline and limit probes recorded |
| 5 | Go stand-in calibrated from phase 4 data | Mode A and Mode B fixtures running |
| 6 | Admission queue and overload handling | All five scenarios run |
| 7 | Load tests, evaluation, tables from results.csv | Every table recomputed by script |
| 8 | Cache, single-flight, optional experiments | Only if time remains |

## Open decisions

- [ ] "Fast" target (user-visible TTFT percentile and value) and the simulated user patience distribution.
- [ ] Recalibrate every initial number after real measurements: context caps, pressure triggers, 2 s fallback wait, expected output tokens for reservations, 10% retry budget, refusal thresholds.
- [ ] L3 reply: add top cited passages (no generation) to the unavailability message, or not.
- [ ] reasoning\_effort: keep low, or run the low vs medium comparison on the dev split.
- [ ] Which optional experiments to run if time allows: setting D, local NLI verifier, Option 4 + 5 streaming (FIFO stays; no scheduler experiment unless measurements show a problem).
- [ ] LangGraph kept with timing wrappers; switch to a plain asyncio state machine if it gets in the way.

## Sources

- [Groq rate limits](https://console.groq.com/docs/rate-limits.md): limit types incl. ITPM/OTPM, organization-level scope, headers (RPD and TPM only), sample limits.
- [Groq models](https://console.groq.com/docs/models.md): production models, Enterprise-only Llama entries, prices.
- [Groq deprecations](https://console.groq.com/docs/deprecations.md): Llama 3.3 70B and 3.1 8B shut down for free and developer tiers on 16 Aug 2026.
- [Groq reasoning](https://console.groq.com/docs/reasoning.md): reasoning\_effort and reasoning visibility for gpt-oss.
