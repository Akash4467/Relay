# The Relay — Full Design Specification

Status: approved for implementation (after two architecture reviews), 2026-11-11.
Author: Akash Bisht. This file is the single source of truth for the build. If code and this file disagree, fix one of them deliberately and note it in §20 (Change log).

Groq facts in this file were checked against Groq's docs on 2026-11-11 (links in §21). Re-check the account's limits page before any real experiment.

---

## 0. Contents

1. The assignment (what is being graded)
2. Goals, non-goals, and the trade-offs I accept
3. System overview
4. Repository layout
5. Configuration
6. Ingestion (papers → chunks)
7. Retrieval
8. The request API
9. Admission layer (cache, single-flight, queue)
10. Rate governor, system pressure, model decision
11. LLM client (Groq and stand-in)
12. LangGraph pipeline (state, nodes, edges)
13. Prompting and answer format
14. Verification
15. Retry policy and counters
16. Stand-in model service (Go)
17. Load generator (Go)
18. Tracing, results.csv, tables
19. Experiments and evaluation
20. Build order, testing, change log
21. Sources
22. Open decisions

---

## 1. The assignment (what is being graded)

Build a small working pipeline that answers engineers' questions about three arXiv papers, with citations, and behaves well under bursts, rate limits and overload. Graders care whether choices follow from measurements and whether I can say what each choice costs. A live session follows where I explain results and change the system.

### 1.1 Hard requirements from the brief

- Language: Python, C++ or Go. **Chosen: Python (app) + Go (stand-in, load generator).**
- Pipeline: ≥ 4 steps, ≥ 1 tool call, ≥ 1 branch or loop; one step retrieves passages from the library.
- Library, exact versions only:
  | Code | Paper | arXiv | Version |
  | --- | --- | --- | --- |
  | AIAYN | Attention Is All You Need | 1706.03762 | v7 |
  | MOSHI | Moshi: a speech-text foundation model for real-time dialogue | 2410.00037 | v1 |
  | HNET | Dynamic Chunking for End-to-End Hierarchical Sequence Modeling | 2507.07955 | v2 |
- Every answer cites where in the papers it came from. If the papers do not say, the assistant says so instead of guessing.
- Real hosted model: Groq free tier, my own account and key. Find its limits from docs AND from what it returns at a limit.
- Stand-in model service I control: can be slow, fail, refuse at a limit, get slower under load, stream piece by piece. Calibrated from real measurements. Say where it is least like the real thing.
- Runs: a modest number of real requests to learn latency spread, limits, behaviour at a limit. Then ≥ 3 load shapes (required rows: steady, sudden burst, sustained overload, per-minute limit halved, daily budget exhausted). Stand-in for loads the real service cannot take. Keep raw results for both.
- Answer quality on ≥ 15 self-written questions; tokens per answer.
- Retrieval: ≥ 3 settings (piece size, pieces fetched, re-ranking) with quality, tokens per answer, TTFT.
- `results.csv`: one row per request, columns exactly: `request_id, scenario, source (real|stand-in), start_time, first_token_ms, total_ms, status (ok|rate_limited|timeout|error), retries, tokens_in, tokens_out, retrieval_setting, answer_grade (correct|wrong|refused_correctly|refused_wrongly)`. Every table figure must be computable from it. Extra columns are allowed (§18.3).
- Tables 1, 2a, 2b, 3 (templates in §18.4) completely filled; "not measured" where not measured; ms for time; one decimal for percentages.
- Write-up: max 2 pages, headings in this order:
  1. What I built and why (design + the one decision that shaped it most)
  2. The real service and my stand-in
  3. Results (TTFT and total at median and slow end, say percentiles and N; failures; given-up; real vs stand-in; a number they could not have guessed; quality on ≥ 15 questions; tokens per answer)
  4. Where the time goes (one slow request split into waiting / working / retrying; before or after first token; show how I know)
  5. When a limit is hit (per-minute and per-day: who is delayed, dropped, gets a weaker answer, why)
  - Retrieval: quality versus cost
  - How this library will mislead a naive retriever (3 ways, each with an example question)
  6. Retries (policy + one case where retrying made things worse)
  7. What the person sees (3pm, daily budget gone)
  8. What I'm unsure about, what I left out, what I learned
  - Your feedback on Vaani (https://www.vaani.dev/) — written by me personally
  - How I worked with the assistant (real conversation history; 3 key prompts; one thing it got wrong that I caught; one decision where I overruled it; what is distinctly mine)
- Ground rules: submit within 4 days; ~6–8 h of work; never include the API key; stay within Groq's terms; **no extra accounts or tricks around limits** — the daily limit is shared by everything I run.

### 1.2 What graders will recompute

They read the tables first and recompute them from `results.csv`. Therefore: tables are generated by script from `results.csv` only (§18.4). Never hand-edit numbers.

---

## 2. Goals, non-goals, trade-offs

### 2.1 Definition of "fast" (OPEN — see §22)
User-visible TTFT = arrival at the front door → first byte of verified answer content sent to the user. Target to be fixed after the real baseline; working placeholder: p95 user-visible TTFT ≤ 8 s under steady load on the stand-in.

### 2.2 Priorities (in order)
1. Never send a wrong or embarrassing answer to stay fast. Correct refusal beats unsupported answer.
2. Nobody waits forever: every request has a deadline and gets an honest reply (answer, refusal, busy, unavailable).
3. Stay inside the real quota; spend it on measurement that matters.
4. Low TTFT.
5. Low cost per answer.

### 2.3 Trade-offs accepted (say these in the write-up)
- Whole-answer verification before release costs TTFT (user waits for full generation) in exchange for a single, defensible correctness guarantee.
- One hosted LLM call per normal request: retrieval, classification, sufficiency and verification are local. Costs some semantic checking power; saves 2–3× requests and tokens.
- Under pressure, context shrinks and redraft is disabled: lower answer recall, higher refusal rate, in exchange for throughput.
- 20B fallback only for question categories where it was validated; other categories get an honest "reduced service" refusal.
- FIFO queue (no priority scheduler).

### 2.4 Non-goals (deliberately not built — candidates for heading 8)
- No Redis, Celery, Kubernetes, database-backed queue, or multi-process workers.
- No semantic (embedding-similarity) answer cache — only exact normalized-question cache.
- No sentence-level streaming in the production path (only an optional experiment).
- No LLM-based verifier call (would double hosted requests).
- No general LaTeX parser.

---

## 3. System overview

```
Go loadgen (open-loop Poisson, patience deadlines)
        │  HTTP POST /ask  (SSE response)
        ▼
FastAPI app (Python, single uvicorn worker, one event loop)
  ├─ Admission: normalize → cache (bypassable) → single-flight → bounded FIFO deadline queue → semaphore
  ├─ LangGraph state machine (11 nodes)
  │     classify_query → budget_check → retrieve → relevance_check → draft → verify → return
  │     side exits: canned_response, unavailable, refuse, redraft (L0 only, max 1)
  ├─ Retrieval (local): BM25 + BGE-small (FAISS) → RRF → [cross-encoder] → bounded expansion → context cap
  ├─ Rate governor: org-level buckets (ITPM/OTPM if present) + per-model RPM/RPD/TPM/TPD; reservations
  ├─ Pressure (global L0–L3) + model decision (per request, 120B/20B/shed)
  ├─ LLM client (httpx, SSE) ──► Groq (real)  or  Go stand-in (same OpenAI-compatible API)
  └─ Tracing: JSONL spans per request → scripts → results.csv → tables
```

Only `draft` and `redraft` make hosted LLM calls.

Models (Groq, verified 2026-11-11):
- Primary: `openai/gpt-oss-120b` ("120B"), `reasoning_effort: "low"`.
- Fallback: `openai/gpt-oss-20b` ("20B"), `reasoning_effort: "low"`.
- NOT usable: `llama-3.3-70b-versatile`, `llama-3.1-8b-instant` — shut down for free/developer tiers on 2026-08-16 (Enterprise only).
- Both gpt-oss models are reasoning models: hidden reasoning tokens precede the answer and count as output tokens.

Groq sample limits for both gpt-oss models (docs table; the account limits page is authoritative): 30 RPM, 1K RPD, 8K TPM, 200K TPD. Limits are organization-wide (shared by all keys and users) and per model. Some orgs also have ITPM/OTPM.

Consequence: at ~3.5K tokens per answer, 120B serves ~2 answers/min and ~57 answers/day. **TPM and TPD are the binding limits, not RPM/RPD.**

---

## 4. Repository layout

```
relay/
  CLAUDE.md                      # rules for Claude Code (short)
  docs/DESIGN.md                 # this file
  .env.example                   # GROQ_API_KEY=, LLM_PROVIDER=, GROQ_BASE_URL=
  pyproject.toml                 # python deps (see §4.1)
  config/
    app.yaml                     # models, caps, pressure triggers, timeouts, retry, queue
    limits.yaml                  # values copied from the Groq limits page (+ observed)
    retrieval_settings.yaml      # settings A, B, C, D
    standin.yaml                 # stand-in calibration (fitted from real data)
    scenarios/
      steady.yaml  burst.yaml  overload.yaml  rpm_halved.yaml  daily_exhausted.yaml
      real_steady.yaml
  data/
    raw/{AIAYN,MOSHI,HNET}/      # source.tar.gz, paper.pdf, extracted tex files (git-ignored if large)
    processed/
      sections.jsonl             # section tree per paper
      paragraphs.jsonl           # paragraph records
      tables.jsonl               # table records
      chunks_{256,512}.jsonl     # chunk records per piece size
    index/
      bm25_{256,512}.pkl
      faiss_{256,512}.index
      faiss_{256,512}.ids.json
  eval/
    questions.jsonl              # 30 questions with gold answers/citations, split dev/test
    verifier_stress.jsonl        # ~30 claim/evidence pairs
    grades/                      # manual grade sheets per run
  fixtures/
    perf/                        # Mode A: recorded real answers by question_id
    functional/                  # Mode B: crafted answers (correct, unsupported, wrong-meaning)
  relay/                         # Python package
    __init__.py
    main.py                      # FastAPI app, /ask, /health, startup (load indexes, warm models)
    config.py                    # pydantic settings loading config/*.yaml and .env
    admission/
      normalize.py  cache.py  singleflight.py  queue.py
    governor/
      buckets.py  governor.py  pressure.py  model_decision.py  wait_estimate.py
    llm/
      client.py                  # OpenAI-compatible SSE client (httpx), timeouts, header capture
      tokenizer.py               # local token counting
      prompts.py                 # system/user prompt templates, versioned
    graph/
      state.py  build.py  tracing.py
      nodes/
        classify_query.py  budget_check.py  canned_response.py  unavailable.py
        retrieve.py  relevance_check.py  draft.py  verify.py  redraft.py  refuse.py  return_answer.py
    retrieval/
      fetch_papers.py  latex_parse.py  pdf_validate.py  chunker.py  index_build.py
      search.py  rrf.py  rerank.py  expand.py  context.py
    verify/
      sentences.py  citations.py  evidence.py  nli.py  sufficiency.py
    trace/
      writer.py  schema.py
    eval/
      run_eval.py  grade.py  stress_verifier.py
  standin/                       # Go module
    go.mod
    cmd/standin/main.go
    internal/{config,limits,latency,fixtures,sse,failures,metrics}/
  loadgen/                       # Go module
    go.mod
    cmd/loadgen/main.go
    internal/{scenario,arrivals,client,output}/
  scripts/
    check_models.py              # 1-token call per model (2 real calls)
    baseline_latency.py          # real latency/token baseline
    probe_limits.py              # RPM/TPM/recovery/shared-capacity probes
    concurrency_test.py          # 1,2,4,8,16 concurrent real requests
    record_fixtures.py           # turn real answers into fixtures/perf
    fit_standin.py               # fit latency distributions → config/standin.yaml
    make_results_csv.py          # traces + loadgen output → results/results.csv
    make_tables.py               # results.csv → results/tables.md
  tests/
    unit/                        # one test file per module, no network
    integration/                 # units wired together; stand-in or local fake only, never real Groq
    fixtures/                    # tiny synthetic corpus, CSVs and fixtures for tests
  results/
    raw/real/*.jsonl             # every real call: request, headers, status, body, timings
    raw/standin/*.jsonl
    raw/traces/*.jsonl           # app spans
    raw/loadgen/*.jsonl          # client-side timings
    results.csv
    tables.md
  writeup/
    relay_writeup.md
    conversation_history/        # exported assistant conversations (submission requirement)
```

### 4.1 Python dependencies
`fastapi`, `uvicorn`, `httpx`, `langgraph`, `pydantic`, `pyyaml`, `rank-bm25`, `sentence-transformers`, `faiss-cpu`, `pymupdf`, `pylatexenc`, `tiktoken`, `pandas`, `numpy`, `orjson`. Optional: an NLI cross-encoder model for verifier level 3.

Models used locally:
- Embeddings: `BAAI/bge-small-en-v1.5` (query prefix per model card).
- Reranker (conditional): `cross-encoder/ms-marco-MiniLM-L-6-v2`.
- NLI (optional experiment): a small NLI cross-encoder (e.g. DeBERTa-v3-small NLI).

### 4.2 Go
Go ≥ 1.22, standard library only where possible (`net/http`, `encoding/json`, `math/rand/v2`, `time`). No frameworks.

---

## 5. Configuration

### 5.1 Environment (.env, never committed)
```
LLM_PROVIDER=groq            # groq | standin
GROQ_API_KEY=...             # real key; "dummy" for stand-in
GROQ_BASE_URL=https://api.groq.com/openai/v1     # stand-in: http://localhost:8001/v1
PRIMARY_MODEL=openai/gpt-oss-120b
FALLBACK_MODEL=openai/gpt-oss-20b
TRACE_DIR=results/raw/traces
SCENARIO=dev
```
The client code is identical for both providers; only the base URL and key differ.

### 5.2 config/app.yaml (initial values — all recalibrated after real measurements)
```yaml
prompt_version: p1
verification_policy_version: v1
generation_config_version: g1
corpus_version: c1                     # bump when chunks are rebuilt

models:
  primary:  {id: openai/gpt-oss-120b, reasoning_effort: low, max_tokens: 900, temperature: 0.2}
  fallback: {id: openai/gpt-oss-20b,  reasoning_effort: low, max_tokens: 900, temperature: 0.2}

reservation:
  output_p95_tokens: {openai/gpt-oss-120b: 700, openai/gpt-oss-20b: 700}   # includes reasoning; refit from baseline
  safety_margin: 0.10

pressure:                               # global, recomputed every 1 s
  L0: {context_cap: 2500, expansion: bounded, redraft: true}
  L1: {context_cap: 1500, expansion: paragraph_only, redraft: false,
       trigger: {primary_tpm_remaining_frac_lt: 0.5, or_queue_growing_for_s: 10}}
  L2: {context_cap: 1000, expansion: off, redraft: false,
       trigger: {primary_est_wait_ms_gt: 2000}}
  L3: {trigger: both_daily_quotas_exhausted}

model_decision:
  max_wait_for_primary_ms: 2000
  fallback_gate_categories: []          # filled from model benchmark (§19.4)

queue:
  max_length: 50
  semaphore_inflight_hosted_calls: 4

timeouts:
  connect_ms: 3000
  first_byte_ms: 15000
  stream_idle_ms: 8000                  # max gap between SSE chunks
  total_call_ms: 45000

retry:
  max_transport_retries_per_call: 2
  backoff_base_ms: 500
  backoff_cap_ms: 8000
  budget_frac_of_initial_per_min: 0.10

deadlines:
  default_patience_ms: 20000            # used when the client sends none

cache:
  enabled: false                        # main experiments run with cache off
singleflight:
  enabled: false
```

### 5.3 config/limits.yaml
Copied from the Groq limits page before experiments, then updated with observed values.
```yaml
source: "console limits page, read YYYY-MM-DD"
organization:
  itpm: null      # set only if the page shows an "X in / Y out" breakdown
  otpm: null
  shared_tpm: null   # set only if the shared-capacity check (§19.2 step 5) shows shared buckets
models:
  openai/gpt-oss-120b: {rpm: 30, rpd: 1000, tpm: 8000, tpd: 200000}
  openai/gpt-oss-20b:  {rpm: 30, rpd: 1000, tpm: 8000, tpd: 200000}
window: unknown   # fixed | sliding — set from probe observations
```

### 5.4 config/retrieval_settings.yaml
```yaml
A: {piece_tokens: 256, candidates: 20, rerank: false, final_passages: 4, expansion: false}
B: {piece_tokens: 256, candidates: 20, rerank: true,  final_passages: 4, expansion: false}
C: {piece_tokens: 512, candidates: 20, rerank: true,  final_passages: 4, expansion: false}
D: {piece_tokens: 256, candidates: 20, rerank: true,  final_passages: 4, expansion: true}   # optional
```
Each pair changes one variable: A↔B reranking, B↔C piece size, B↔D expansion. The amount fetched is varied by the pressure context caps.

---

## 6. Ingestion (papers → chunks)

Offline, run once: `python -m relay.retrieval.fetch_papers && python -m relay.retrieval.latex_parse && python -m relay.retrieval.chunker && python -m relay.retrieval.index_build`.

### 6.1 Fetch exact versions
- Source: `https://arxiv.org/e-print/{id}v{n}` (tar.gz or single gz tex).
- PDF: `https://arxiv.org/pdf/{id}v{n}`.
- Store SHA-256 of each file in `data/raw/{CODE}/manifest.json` with the URL and download time. Refuse to proceed if the version suffix is missing.
- If the network is blocked where code runs, download manually and place files in `data/raw/{CODE}/`.

### 6.2 LaTeX parse — limited scope parser
Support only what the three papers use:
1. Find the main `.tex` (contains `\documentclass` / `\begin{document}`).
2. Resolve `\input{}` / `\include{}` recursively (relative paths, add `.tex`).
3. Collect the papers' own `\newcommand` / `\def` macros with no arguments or simple arguments; expand them in text (model names must survive).
4. Walk structure: `\section`, `\subsection`, `\subsubsection`, `\paragraph`; `\appendix` switches numbering to letters.
5. Paragraphs: split on blank lines inside a section.
6. Text conversion: `pylatexenc` LatexNodes2Text for prose; keep inline math as plain text (e.g. `d_model`, `O(n^2)`); display equations kept as one-line normalized TeX inside the paragraph, tagged `[EQ]`.
7. Tables: `table`/`tabular` → table record: caption, label, header row(s), data rows; each row converted to `"<row header>: <col header> = <value>; ..."`; units kept with headers. `\multicolumn` expanded by repeating header text.
8. Figures: keep caption text only (figure record, type `figure_caption`).
9. References/bibliography: excluded from retrieval chunks (kept in a separate file for the "bibliography pollution" write-up example).
10. Section numbers: assign from document order exactly as the PDF shows them (validate in 6.3).

Output records:
```jsonc
// sections.jsonl
{"paper":"MOSHI","section_id":"4.2","title":"...","parent":"4","order":17}
// paragraphs.jsonl
{"paper":"MOSHI","section_id":"4.2","para_idx":3,"text":"...","has_equation":false}
// tables.jsonl
{"paper":"MOSHI","table_id":"T3","label":"tab:results","section_id":"5.1","caption":"...",
 "headers":["Model","..."],"rows":[{"row_header":"Moshi","cells":{"Latency (ms)":"200"}}],
 "row_texts":["Moshi: Latency (ms) = 200; ..."]}
```
If a paper's extraction stays broken after reasonable effort, switch that paper only to Docling or GROBID; note it in §20.

### 6.3 PDF validation (PyMuPDF)
- For each paragraph, fuzzy-find its first ~12 words in the PDF text; record `page` when found (best effort; equations/tables may fail).
- Report coverage: % paragraphs located per paper. Section headings must all be found; mismatch → fix numbering.
- Page is an optional extra in citations, never required.

### 6.4 Chunking
- Section-aware: a chunk never crosses a section boundary.
- Piece sizes 256 and 512 tokens (tokenizer: §11.4), built by packing whole sentences of consecutive paragraphs; a paragraph longer than the size is split on sentence boundaries; overlap 0 (paragraph neighbours are reachable through expansion instead).
- Each table row group becomes its own chunk (`kind: table`) carrying the caption + header line + up to N rows that fit; headers and units are repeated in every table chunk.
- Figure captions become `kind: caption` chunks.
- Chunk ID format: `{PAPER}:{section_id}:c{NNN}` e.g. `MOSHI:4.2:c017` (NNN = order within the paper, zero-padded). Table chunks: `MOSHI:T3:r01`.
- Chunk record:
```jsonc
{"chunk_id":"MOSHI:4.2:c017","paper":"MOSHI","section_id":"4.2","section_title":"...",
 "kind":"text","piece_tokens":256,"para_span":[3,4],"text":"...","n_tokens":241,
 "page":7,"corpus_version":"c1"}
```

### 6.5 Indexes
- BM25 (`rank_bm25.BM25Okapi`) over lowercased tokens; keep hyphenated terms whole ("rq-transformer") and also add split parts.
- Dense: BGE-small embeddings, normalized, FAISS `IndexFlatIP` (corpus is tiny).
- One index pair per piece size.
- Also store the BM25 vocabulary IDF table (used by the sufficiency check, §14.4).

---

## 7. Retrieval

`search(question, setting, context_cap) -> Evidence`

1. BM25 top `candidates` and dense top `candidates`.
2. Reciprocal Rank Fusion: `score(d) = Σ 1/(60 + rank_i(d))`; take top `candidates`.
3. If `setting.rerank`: cross-encoder scores (question, chunk text) for the candidates; sort by score. Runs in the bounded thread pool (§8.3).
4. Take `final_passages` best ("ranked" = cross-encoder order if reranking, else RRF order).
5. Expansion (if `setting.expansion` and pressure allows):
   1. containing paragraph(s) of the chunk;
   2. neighbouring paragraph(s) only if the chunk sits at a paragraph edge or the question is tagged multi-section;
   3. whole section only for broad questions (classifier tag `broad`) and only if it fits the cap.
   Table chunks expand to the table caption + header + the matching rows, never the whole table if it would break the cap.
6. Context cap: fill best-first until `context_cap` tokens; never cut a chunk mid-sentence; a table chunk keeps its header line.
7. Return `Evidence{passages:[{chunk_id, text, score_ce?, score_rrf, dense_cos}], top_dense_cos, top_ce, tokens}`.

Relevance threshold input (§14.4): top cross-encoder score when reranking is on; otherwise top dense cosine. Never threshold RRF scores (rank-based, uncalibrated).

Ways this library misleads a naive retriever (collect concrete failing questions during eval for the write-up):
- Same word, different meaning across papers ("tokens": text tokens in AIAYN, audio/semantic tokens in MOSHI, bytes/chunks in HNET; "chunking", "attention").
- Shared ancestry: MOSHI and HNET describe/compare against Transformers, so Transformer questions retrieve the wrong paper.
- Bibliography and related-work text is keyword-dense but not the paper's own claim.
- Numbers live in tables; naive PDF parsing mangles them.
- Questions that sound answerable but aren't (e.g. "What GPU was used to train Moshi?") still retrieve plausible chunks.

---

## 8. The request API

### 8.1 Endpoint
`POST /ask`
```jsonc
{"question":"...", "request_id":"optional-uuid", "deadline_ms":20000,
 "retrieval_setting":"B", "scenario":"steady", "bypass_cache":true, "question_id":"q07"}
```
Response: `text/event-stream`:
```
event: meta     data: {"request_id":"...","admitted":true}
event: content  data: {"text":"..."}            # verified answer streamed in chunks
event: done     data: {"tier":"normal","status":"ok","citations":["MOSHI:4.2:c017"],
                       "model":"openai/gpt-oss-120b","reason":null}
```
Rejected / refused / unavailable replies are also sent as `content` + `done` (with `tier` and `reason`) so the user always reads a sentence, never a bare error. HTTP status is 200 for all handled outcomes; 503 only if the app itself is down.

`GET /health` → indexes loaded, models warm, governor snapshot, pressure level, queue length.

`question_id` is optional and only used by the stand-in fixture lookup and grading; the app must not use it to answer.

### 8.2 Process model
Single uvicorn worker, one asyncio event loop. All admission, governor and pressure state is in-memory and global to the process.

### 8.3 Off-loop work
A `ThreadPoolExecutor(max_workers=2)` runs embedding, cross-encoder and NLI inference (torch releases the GIL). BM25 runs inline (benchmark it; move to the pool if > 20 ms). Ingestion never runs in the server. Record `retrieval_ms` separately from model time.

---

## 9. Admission layer

Order for each request:

1. **Stamp**: `arrival_ns` (monotonic) + wall `start_time`; `deadline = arrival + deadline_ms`.
2. **Normalize** (conservative): Unicode NFKC, lowercase, collapse whitespace, strip leading/trailing whitespace. Keep hyphens, apostrophes, symbols, digits, punctuation.
3. **Cache** (built after the baseline; off in main runs; `bypass_cache` respected):
   - key = SHA-256 of `(normalized_question, model_id, retrieval_setting, corpus_version, prompt_version, verification_policy_version, generation_config_version)`.
   - stores only verified answers (tier normal or reduced) with citations and verification metadata.
   - no TTL (papers are static); invalidated by any version bump.
   - lookup tries the primary-model key first, then fallback key.
4. **Single-flight** (built after the baseline; off in main runs):
   - one shared `asyncio.Task` per cache key; waiters `await asyncio.shield(task)` with their own deadline.
   - a waiter leaving (client disconnect or own deadline) never cancels the task.
   - the task is cancelled only when its waiter count reaches 0 or its own max deadline (max of current waiters' deadlines, capped at `timeouts.total_call_ms`) passes.
5. **Queue** (bounded FIFO):
   - reject on arrival if `len(queue) >= max_length` → reason `queue_rejected`.
   - reject on arrival if `estimated_wait_ms > remaining_deadline_ms` → reason `queue_rejected`.
   - before dispatch, drop expired entries → reason `deadline_expired`.
   - rejected/dropped requests get an honest busy reply immediately.
6. **Semaphore** on in-flight hosted calls (`semaphore_inflight_hosted_calls`).
7. Run the graph.

Estimated wait (`wait_estimate.py`) is computed from the binding resource, not queue length × mean latency:
`est = max(slot_wait, token_wait)` where
- `slot_wait` = time until enough semaphore slots free for this request's position, using expected service time per queued request (input tokens known, output from p95);
- `token_wait` = time until the primary model's TPM bucket (and ITPM/OTPM if present) has the summed reservations of the requests ahead + this one, at the bucket's refill rate.

---

## 10. Rate governor, system pressure, model decision

### 10.1 Buckets (`governor/buckets.py`)
- Token bucket with capacity `C` and refill rate `C/60` per second for per-minute limits; `C/86400` for per-day, or fixed reset at the header's `x-ratelimit-reset-requests` time if observed to be a fixed window. Window type is configured from probe observations.
- Operations: `try_reserve(n) -> Reservation|None`, `settle(reservation, actual_n)`, `freeze_until(t)`, `lower_effective_limit(frac)`.

### 10.2 Bucket structure
- Organization level (only if confirmed): `ITPM`, `OTPM`, `shared_TPM`.
- Model level, per model: `RPM`, `RPD`, `TPM`, `TPD`.
- A call must acquire **all** applicable buckets atomically (all-or-nothing) before dispatch.

### 10.3 Reservation
```
input_tokens   = local tokenizer count of the full prompt (system + user + evidence)
output_reserve = output_p95_tokens[model] * (1 + safety_margin)      # includes reasoning tokens
TPM, TPD       ← input_tokens + output_reserve
ITPM           ← input_tokens          (if present)
OTPM           ← output_reserve        (if present)
RPM, RPD       ← 1
```
After the response: settle every bucket to actual usage (`prompt_tokens`, `completion_tokens` from the final stream chunk). If actual > reserved: count `reservation_overrun_tokens` and immediately deduct the difference.

### 10.4 Provider feedback
- Response headers on every call: `x-ratelimit-limit-requests`, `x-ratelimit-remaining-requests`, `x-ratelimit-reset-requests` (these are **per day**); `x-ratelimit-limit-tokens`, `x-ratelimit-remaining-tokens`, `x-ratelimit-reset-tokens` (these are **per minute**); `retry-after` (seconds, only on 429). There is **no** RPM header.
- Headers resync only RPD and TPM buckets (lower the local remaining to the header value if lower).
- Missing or malformed header → keep local estimate, log `header_anomaly`.
- A 429 is authoritative: freeze the implicated bucket until `retry-after` (or reset); if the local estimate said the call fit, lower that bucket's effective limit by 10% for 5 minutes. Parse the 429 body to learn which limit was hit and log it.

### 10.5 System pressure (global, every 1 s) — `pressure.py`
| Level | Trigger | Context cap | Expansion | Redraft |
| --- | --- | --- | --- | --- |
| L0 Healthy | primary buckets healthy | 2,500 | bounded | allowed (120B only) |
| L1 Constrained | primary TPM remaining < 50% or queue growing for 10 s | 1,500 | containing paragraph only | off |
| L2 Critical | primary estimated wait > 2 s | 1,000 | off | off |
| L3 Exhausted | both models' daily quota (RPD or TPD) gone | — | — | — |

Pressure is copied into the request state by `budget_check`; it never chooses the model.

### 10.6 Model decision (per request, at dispatch) — `model_decision.py`
```
if primary can reserve now:                       → 120B
elif primary_est_wait <= 2000 ms and fits deadline: → wait, then 120B
elif category in fallback_gate_categories and fallback can serve within deadline: → 20B
elif primary daily quota gone and category in gate and fallback ok:             → 20B
elif category not in gate:                        → refuse(reason="reduced_service_category")
else:                                             → shed(reason="local_governor_reject")
```
- 20B answers are tier `reduced` and the reply says so in one sentence.
- 20B failing verification → refuse (no redraft).
- `fallback_gate_categories` comes from the model benchmark (§19.4): a category passes if 20B correct-and-cited is within 10 points of 120B and its false acceptance is not higher.

### 10.7 Answer tiers
`normal` (120B) · `reduced` (20B) · `refused` (insufficient evidence, failed verification, gated category) · `unavailable` (quota exhausted, shed, queue rejected, expired).

### 10.8 What the person sees (write-up heading 7: 3pm, daily budget gone)
L3 reply: if a cached verified answer exists → return it (tier normal/reduced, marked "cached"). Otherwise:
"The research assistant has used today's model capacity, so I can't write a new answer right now. Full answers resume at {reset time, local}. [Optional, OPEN §22: the most relevant passages: …citations…]". Never a guessed answer.

---

## 11. LLM client

### 11.1 Request
`POST {GROQ_BASE_URL}/chat/completions`
```json
{"model":"openai/gpt-oss-120b","messages":[...],"stream":true,
 "max_tokens":900,"temperature":0.2,"reasoning_effort":"low","include_reasoning":false}
```
(`include_reasoning` and `reasoning_format` are mutually exclusive; use one. Verify parameter names against Groq's reasoning docs at build time.)

### 11.2 Streaming and timing
Record per call: `t_send`, `t_first_byte` (first SSE line), `t_first_content` (first non-empty `delta.content`), `t_last`, inter-chunk gaps, HTTP status, all response headers, final `usage` (Groq may put it in the last chunk, e.g. under `x_groq.usage`; verify), `model`, error body.
- `reasoning_tokens = completion_tokens − tokens(visible content)` (reported reasoning counts may read 0).
- Timeouts: connect, first byte, stream idle gap, total (config `timeouts`). A stalled stream (idle > `stream_idle_ms`) aborts with `internal_timeout`.

### 11.3 Raw logging of real calls
Every real Groq call appends one line to `results/raw/real/{date}.jsonl`: request (without key), status, headers, body or error body, all timestamps, usage. These raw files are submitted.

### 11.4 Tokenizer
`tiktoken` encoding for gpt-oss (`o200k_harmony` if available, else `o200k_base`). Validate on the baseline: compare local count with `usage.prompt_tokens`; store the ratio and correct reservations by it.

---

## 12. LangGraph pipeline

### 12.1 State (`graph/state.py`, TypedDict)
```python
class RelayState(TypedDict, total=False):
    request_id: str; question: str; question_id: str | None
    scenario: str; source: str                    # "real" | "stand-in"
    retrieval_setting: str
    deadline_ns: int; arrival_ns: int
    intent: str                                   # greeting | out_of_scope | research
    category: str                                 # direct | multi_section | numerical_table | cross_paper
    broad: bool
    pressure: str                                 # L0..L3
    context_cap: int; expansion: str; redraft_allowed: bool
    evidence: dict                                # Evidence (passages, scores, tokens)
    sufficient: bool; refusal_reason: str | None
    model: str | None; tier: str | None
    draft_text: str; verified: bool; verify_report: dict
    redrafts: int                                 # 0 or 1
    answer_text: str; citations: list[str]
    counters: dict                                # initial_attempts, transport_retries, redraft_attempts, fallback_generations
    tokens_in: int; tokens_out: int; reasoning_tokens: int
    status: str; failure_reason: str | None
```

### 12.2 Nodes
| Node | Does | LLM |
| --- | --- | --- |
| classify_query | rules: greeting / out_of_scope / research; category tag; broad tag | no |
| budget_check | copy pressure (L0–L3) + scope into state; route L3 → unavailable | no |
| canned_response | reply to greetings / out-of-scope | no |
| unavailable | L3 reply (§10.8) | no |
| retrieve | §7 with the setting and the pressure's cap/expansion (tool call) | no |
| relevance_check | threshold + sufficiency (§14.4) | no |
| draft | model decision → reserve → stream complete answer internally | yes, 1 |
| verify | levels 1–2 (+3 if enabled) on the complete answer | no |
| redraft | only if `pressure == L0` and model is 120B and `redrafts == 0`; prompt includes the failed sentences | yes, max 1 |
| refuse | honest refusal sentence with reason | no |
| return_answer | stream verified answer to the user; write cache if enabled | no |

### 12.3 Edges
```
START → classify_query
classify_query → canned_response   if intent != research
classify_query → budget_check      otherwise
budget_check → unavailable         if pressure == L3
budget_check → retrieve            otherwise
retrieve → relevance_check
relevance_check → refuse           if not sufficient
relevance_check → draft            otherwise
draft → refuse                     if model decision = refuse/shed or draft returned NOT_IN_PAPERS
draft → verify
verify → return_answer             if verified
verify → redraft                   if not verified and redraft_allowed and model==120B and redrafts==0
verify → refuse                    otherwise
redraft → verify
canned_response, unavailable, refuse, return_answer → END
```

### 12.4 classify_query rules (initial)
- greeting: whole message matches `^(hi|hello|hey|thanks|thank you|good (morning|evening))\b.{0,20}$`.
- out_of_scope: no token from the corpus vocabulary with IDF above the median AND no research cue words (what, how, why, which, does, compare, explain, value, table, result); otherwise research. (Retrieval + relevance_check is the real gate; this rule only catches obvious chit-chat.)
- category: `numerical_table` if the question asks for a number/metric (how many, what value, score, BLEU, latency, %, table, parameters, ms); `cross_paper` if it names ≥ 2 papers/models or uses compare/versus/difference between; `multi_section` if it asks how/why X relates to Y or for a process; else `direct`.
- broad: "overview", "summarize", "explain the architecture", "how does X work".

### 12.5 Timing wrapper (`graph/tracing.py`)
Every node is wrapped: span `node.<name>` with `start_ns`, `end_ns`, and attributes; the draft node also emits sub-spans `queue_wait`, `governor_wait`, `llm.attempt[k]` (with `first_byte`, `first_content`, `status`), `retry_backoff[k]`.

---

## 13. Prompting and answer format

### 13.1 System prompt (prompt_version p1)
```
You answer questions about exactly three papers using ONLY the passages provided.
Rules:
1. Every sentence must end with one or more citations in square brackets using the passage IDs given, e.g. [MOSHI:4.2:c017].
2. Use only facts stated in the cited passage. Copy numbers and units exactly.
3. If the passages do not contain the answer, reply with exactly: NOT_IN_PAPERS
4. If only part is answered, answer that part and add one sentence: "The papers do not state <missing part>." with no citation.
5. Do not use outside knowledge. Do not mention passages that were not provided.
6. Be concise: at most 6 sentences.
```

### 13.2 User message
```
Question: {question}

Passages:
[{chunk_id}] ({paper} §{section_id} {section_title})
{text}

[{chunk_id}] ...
```

### 13.3 Redraft message (appended)
```
Your previous answer had unsupported sentences:
- "{sentence}" — {reason}
Rewrite the answer following the rules. Remove or fix those sentences.
```

### 13.4 Reply wording
- Reduced tier prefix: "Answered by a smaller backup model while capacity is limited."
- Refusal: "The three papers don't state this." + one line on what was searched (paper/section names of the top passages), no guessing.
- Busy: "The assistant is at capacity right now and couldn't start your question in time. Please try again in about {s} seconds."

---

## 14. Verification

Verification is an **evidence consistency check, not proof of factual correctness**. Say this in the write-up.

### 14.1 Sentence split (`verify/sentences.py`)
Split on `. ? !` followed by space + capital/`[`, but protect: `e.g.`, `i.e.`, `et al.`, `Fig.`, `Eq.`, `Sec.`, `§`, `vs.`, decimals (`0.1`), version numbers, citation brackets. Citations at sentence end stay with the sentence.

### 14.2 Level 1 — citation integrity (deterministic)
- Regex `\[([A-Z]+:[0-9A-Za-z.]+:[cr]\d+)(?:[,;]\s*[A-Z]+:[0-9A-Za-z.]+:[cr]\d+)*\]`.
- Every sentence (except the allowed "The papers do not state …" sentence) has ≥ 1 citation.
- Every cited ID exists in the corpus AND was in this request's evidence.

### 14.3 Level 2 — evidence signals (deterministic; flags only since 2026-11-12, see §14.5)
Key terms for the checks below = technical-looking terms (digit, hyphen, inner or mid-sentence capital) plus words rare in the corpus (IDF ≥ 80th percentile); ordinary words are never key, so a paraphrase is not flagged.
For each sentence, against the union of its cited chunks' text (normalized: lowercase, unicode minus/dashes unified, thousands separators removed):
- every number (with its unit if present: ms, s, %, M, B, k, Hz, kHz, tokens, layers, heads) appears in a cited chunk; "8 heads" requires 8 and "head" near each other (window 5 tokens);
- every key term (tokens with IDF above the 80th percentile of the corpus vocabulary, and capitalized/technical tokens like `RQ-Transformer`, `Mimi`, `H-Net`) appears in a cited chunk;
- paper attribution: if the sentence names a paper/model, at least one cited chunk belongs to that paper.
Failures are reported per sentence with a reason.

### 14.4 Sufficiency (relevance_check node) — revised 2026-11-12: structural, not lexical
Refuse before calling the model only when:
- no usable passage (non-empty text) is in the context after the token budget; or
- the question explicitly names several papers and one of them has no passage in the context.
Otherwise the model attempts an answer. No similarity threshold (unanswerable dev/test questions score as high as answerable ones) and no key-term co-occurrence rule: on dev it refused 2–3 of 8 answerable questions over words like "go", "matter", "segmented". **Lexical rarity is not evidence of answerability.** Refusal of unanswerable questions then rests on the model (`NOT_IN_PAPERS`) and the verifier; its accuracy, the false-refusal rate on answerable questions and the extra tokens spent on unanswerable questions are measured on real-model runs.

(Superseded: relevance thresholds `t_ce`/`t_cos` and the 70% key-entity co-occurrence rule. The example "What GPU was used to train Moshi?" is in fact answerable: MOSHI §4.4, H100.)

### 14.5 Semantic support — the deciding check (revised 2026-11-12, no longer optional)
Local NLI cross-encoder (`verification.nli_model`, run in the bounded thread pool; never an LLM call). Premise = each cited passage as indexed (paper + section header + text), and all cited passages joined; hypothesis = the sentence without its citations; support = the best entailment probability. A sentence is accepted if support ≥ `entail_min`, or ≥ `entail_min_flagged` when level 2 flagged it. **Level 2 terminology and number checks are diagnostic signals, not standalone rejection rules; semantic evidence support decides.** A paraphrase in ordinary words is never rejected for differing from the paper's wording, and a claim made of the paper's own words is not accepted unless it is entailed. Thresholds are fitted on the verifier stress set (§19.6) and Mode B fixtures, never on test questions; latency per sentence is measured.

Flow: retrieve → structural sufficiency (refuse, no LLM call) → draft → level 1 citations (reject) → level 2 signals + semantic support (reject) → redraft once if allowed → answer or refuse.

### 14.6 Known blind spot
Levels 1–2 cannot catch a claim that reuses the evidence's words with a different meaning (evaluation hardware reported as training hardware). Measured by the stress test (§19.6).

---

## 15. Retry policy and counters

- At most 2 transport retries per hosted call; exponential backoff with full jitter: `sleep = random(0, min(cap, base * 2^k))`, and never shorter than `retry-after`.
- Retry on: connect errors, 5xx, per-minute 429, stream stall. Never on: per-day 429 (→ model decision), 4xx other than 429.
- Never retry once the request's deadline has passed (or can't fit `retry wait + expected service`).
- Retry budget: transport retries in any rolling minute ≤ 10% of initial hosted attempts in that minute (100 initial → ≤ 10 retries). Over budget → fail fast with `upstream_429`/`upstream_error`.
- Every retry re-reserves through the governor.
- Counters (all in results.csv extra columns):
  | Counter | Meaning |
  | --- | --- |
  | initial_attempts | first hosted call of a generation |
  | transport_retries | re-sending a failed HTTP call (this is the `retries` column) |
  | redraft_attempts | new generation to correct an answer |
  | fallback_generations | generation on the fallback model |

Planned "retry made things worse" experiment (heading 6): on the stand-in, run the burst scenario with (a) naive retries (no jitter, ignore retry-after, no budget) vs (b) the policy; compare 429 count, total hosted calls, p95 TTFT, given-up %.

---

## 16. Stand-in model service (Go)

### 16.1 API (OpenAI-compatible subset)
`POST /v1/chat/completions` with the same JSON as §11.1. `stream=true` → SSE `data: {...}\n\n` chunks in OpenAI chunk format, final chunk with `usage` (mirror where real Groq puts it), then `data: [DONE]`. `stream=false` → single JSON completion.
`GET /v1/models` (lists configured models). `GET /metrics` (in-flight, bucket levels, counters). `POST /admin/config` (hot-swap scenario knobs, e.g. halve RPM, exhaust daily).

### 16.2 Modes (never mixed)
| Mode | Purpose | Reply content |
| --- | --- | --- |
| A performance | load scenarios | fixture by `question_id` from `fixtures/perf/` (recorded real answer + its real token counts incl. reasoning). Loadgen only sends questions with fixtures. No random replay. Unknown question → HTTP 400 (bug, not data). |
| B functional | pipeline correctness | fixtures from `fixtures/functional/` keyed by `question_id` + `variant` (correct, unsupported, wrong_meaning, not_in_papers) |

Answer quality is measured only on real-service runs.

Fixture format:
```jsonc
{"question_id":"q07","model":"openai/gpt-oss-120b","content":"... [MOSHI:4.2:c017]",
 "prompt_tokens":2140,"completion_tokens":420,"reasoning_tokens":180,
 "recorded_at":"2026-11-12T10:03:00Z","retrieval_setting":"B","prompt_version":"p1"}
```
The question_id reaches the stand-in via a request header `X-Relay-Question-Id` set by the app only when `LLM_PROVIDER=standin`.

### 16.3 Latency model (fitted by `scripts/fit_standin.py` → `config/standin.yaml`)
```
ttft_ms      = LogNormal(mu, sigma) * (1 + a * prompt_tokens/1000) * (1 + b * inflight)
reasoning_ms = reasoning_tokens / rate_tps * (1 + c * inflight)     # hidden phase before first content
token_gap_ms = 1000 / rate_tps * (1 + c * inflight)                 # per streamed content token chunk
```
`mu, sigma` from real TTFT; `a` from TTFT vs prompt size; `rate_tps` from real streams; `b, c` from the real concurrency tests (1–16). Per model.

### 16.4 Limits
Per model RPM/RPD/TPM/TPD (+ optional ITPM/OTPM), independent; window type matches observed. 429 response copies the real one captured in probes: same headers (`retry-after`, `x-ratelimit-*` with the per-day/per-minute meanings) and the same error JSON body shape.

### 16.5 Faults (configurable probabilities per scenario)
- `p_500`: HTTP 500 immediately.
- `p_slow_success`: completes correctly but ttft × `slow_factor` (e.g. 5–10×).
- `p_stall`: stream pauses `stall_ms` after k tokens, then resumes (or drops with probability `p_drop`).
- `p_timeout`: never sends the first byte before the client timeout.

### 16.6 Self-timing
Logs per request: received, first byte, last byte, simulated delays, server processing overhead → `results/raw/standin/*.jsonl`. Used to split observed latency = simulated + server overhead + transport.

### 16.7 Known gaps (write-up heading 2)
No network/region effects; Groq's real scheduling unknown; one concurrency curve can't capture model-specific or time-of-day behaviour; fixtures don't change with prompt/retrieval changes, so stand-in runs say nothing about answer quality.

---

## 17. Load generator (Go)

`loadgen -scenario config/scenarios/burst.yaml -target http://localhost:8000/ask -out results/raw/loadgen/burst_{ts}.jsonl`

Scenario YAML:
```yaml
name: burst
source: stand-in            # or real
questions: eval/questions.jsonl     # only ids with perf fixtures when stand-in
retrieval_setting: B
duration_s: 300
arrivals:                   # open loop: piecewise Poisson rates (req/min)
  - {from_s: 0,   to_s: 60,  rate_per_min: 6}
  - {from_s: 60,  to_s: 90,  rate_per_min: 120}
  - {from_s: 90,  to_s: 300, rate_per_min: 6}
patience_ms: {dist: lognormal, median: 15000, sigma: 0.5, min: 3000, max: 60000}
seed: 42
standin_admin: {}           # e.g. {rpm_scale: 0.5} for rpm_halved, {rpd_remaining: 0} for daily_exhausted
```
- Open loop: arrival times drawn up front from the Poisson process; requests fire at those times regardless of responses (no coordinated omission).
- Each simulated user cancels the request at its patience deadline → recorded `gave_up=true`.
- Client-side record per request: `request_id, question_id, scheduled_ns, sent_ns, first_content_ns, done_ns, gave_up, http_status, done_event`.

Scenario set (rates relative to measured capacity `K` answers/min on the stand-in, calibrated to real):
| Scenario | Source | Shape |
| --- | --- | --- |
| real_steady | real | ~1.5 req/min for ~10 min (≈ 15 calls) |
| steady | stand-in | 0.7 K for 10 min |
| burst | stand-in | 0.5 K baseline, 20 K for 30 s, back to 0.5 K |
| overload | stand-in | 2 K sustained for 10 min |
| rpm_halved | stand-in | steady shape with stand-in RPM × 0.5 |
| daily_exhausted | stand-in | steady shape, both models' daily quota at 0 after 2 min |

---

## 18. Tracing, results.csv, tables

### 18.1 Trace span schema (`results/raw/traces/*.jsonl`)
```jsonc
{"request_id":"...","span":"node.retrieve","start_ns":...,"end_ns":...,
 "attrs":{"setting":"B","n_passages":4,"context_tokens":1620,"retrieval_ms":38}}
```
Spans: `request`, `admission.queue`, `node.*`, `governor.wait`, `llm.attempt`, `retry.backoff`, `verify.level1/2/3`.

### 18.2 Where-the-time-goes (heading 4)
For one slow request, from spans: waiting = admission.queue + governor.wait + retry.backoff; working = retrieval + llm attempts (reasoning phase + generation) + verification; retrying = failed llm.attempt durations + backoff. Before/after first token: compare each span's end with `first_content` of the final successful attempt. Show the span timeline.

### 18.3 results.csv (`scripts/make_results_csv.py`)
Required columns (exact names, this order) then extras:
| Column | Definition |
| --- | --- |
| request_id | one row per user request |
| scenario | scenario name |
| source | `real` or `stand-in` |
| start_time | ISO-8601 wall time of arrival at the front door |
| first_token_ms | user-visible TTFT: arrival → first verified answer content sent (includes queue and governor wait); empty if none sent |
| total_ms | arrival → last byte sent (or give-up time) |
| status | `ok` (any reply sent in time: answer, refusal, canned, cached), `rate_limited` (queue_rejected, local_governor_reject, upstream_429), `timeout` (deadline_expired, gave_up, internal_timeout), `error` (upstream_error, bugs) |
| retries | transport retries only, summed over the request's hosted calls |
| tokens_in | prompt tokens summed over draft + redraft (0 for cache hits / no call) |
| tokens_out | completion tokens incl. reasoning, summed over draft + redraft |
| retrieval_setting | A/B/C/D |
| answer_grade | correct / wrong / refused_correctly / refused_wrongly (graded rows only; empty otherwise) |
| failure_reason | queue_rejected, deadline_expired, local_governor_reject, upstream_429, internal_timeout, gave_up, upstream_error, reduced_service_category, null |
| model, answer_tier, degradation_level, cache_hit, question_id, question_category | |
| queue_wait_ms, governor_wait_ms, retrieval_ms, generation_ms, verification_ms, model_ttft_ms, first_byte_ms | |
| redraft_attempts, fallback_generations, initial_attempts, reasoning_tokens, reservation_overrun_tokens | |

Grading join: `eval/grades/*.csv` (`request_id, answer_grade, notes`) merged by request_id.

### 18.4 Tables (`scripts/make_tables.py`, from results.csv only)
- Percentiles: p50, p95, p99 with numpy `method="linear"`; report N per row. p99 with N < 100 is flagged "(N small)".
- Table 1 — limits found: Requests per minute, Tokens per minute, Requests per day, Tokens per day, Requests in flight at once, What it returns when a limit is hit, Time until it recovers; each with value, how found, confidence. (Filled from probe raw files + `config/limits.yaml`; probe rows are also in results.csv with scenario `probe_*`.)
- Table 2a — per scenario: Real or stand-in, Requests sent, Duration (s), Offered rate (per min), First token p50/p95/p99 (ms), Total p50/p95/p99 (ms).
- Table 2b — per scenario: Succeeded %, Rate-limited %, Timed out or given up %, Retries per request, Peak queue length (from `admission.queue` spans), Tokens per answer (tokens_in+tokens_out over ok answers with a hosted call), Cost per 1,000 answers.
- Table 3 — per retrieval setting: Questions asked, Correct and cited %, Wrong %, Correctly refused %, Tokens per answer, First token p50/p95, Total p50/p95.

Cost per 1,000 answers = 1000 × Σ(cost of all hosted calls incl. retries, redrafts, fallbacks) ÷ successful answers, at gpt-oss-120b $0.15 in / $0.60 out and gpt-oss-20b $0.075 / $0.30 per 1M tokens (Groq models page, read 2026-11-11). Actual free-tier spend ($0) and cost per attempted request reported separately.

---

## 19. Experiments and evaluation

### 19.1 Real-call budget (a ceiling, spent in priority order; limit probing never eats into quality evaluation)
| Priority | Use | Model | Calls |
| --- | --- | --- | --- |
| 0 | Model ID check, 1 token each | 120B + 20B | 2 |
| 1 | Latency and token baseline | 120B + 20B | 15 |
| 2 | Limit probing (RPM, TPM, recovery, shared-capacity) | 20B first, then 120B | ≈ 45 |
| 3 | Retrieval comparison: 3 settings × 30 questions | 120B | 90 |
| 4 | Model comparison: chosen setting × 30 questions | 20B | 30 |
| 5 | Integration + real steady run | 120B | 15 |
| — | Reserve (≈ 10%) | both | 20 |
| — | Optional setting D × 30 | 120B | 30 |

TPD is the planning limit: ≈ 140 120B calls × ≈ 3.5K tokens ≈ 490K tokens ≈ 2.5 days of 200K TPD. Start the retrieval comparison by day 2. Every real answer is recorded as a perf fixture. Daily exhaustion is simulated in the stand-in, never forced on the real quota.

### 19.2 Limit probing (report "observed operating boundary", never "the limit")
1. Read the limits page; nothing else runs on the organization during probes.
2. RPM probe on 20B: documented RPM + 5 tiny requests (max_tokens 1) over one minute (TPM can't be the cause). Record timestamps, status, 429 body (names the limit).
3. TPM probe: few large-prompt requests (RPM can't be the cause).
4. Recovery: after the first 429, one probe per retry-after interval until success.
5. Shared-capacity check: one 120B call, compare 20B's `x-ratelimit-remaining-tokens` before/after.
6. Repeat once at another time if inconsistent.
Report: documented limit · successes before first 429 · number of 429s · time to next success · could another limit explain it.
Concurrency tests: 1, 2, 4, 8, 16 concurrent small requests → successes, 429s, timeouts, TTFT, total. Rising latency with continued success is not a hard limit.

### 19.3 Eval question set (`eval/questions.jsonl`)
30 questions, 15 answerable + 15 unanswerable/misleading, stratified 15 dev / 15 test; test untouched during tuning.
| Category | Count |
| --- | --- |
| Direct factual, exact terminology | 4 |
| Multi-section | 5 |
| Numerical / table | 6 |
| Unanswerable, near-miss | 10 |
| Misleading / cross-paper | 5 |
```jsonc
{"question_id":"q07","split":"dev","category":"numerical_table","answerable":true,
 "question":"...","gold_answer":"...","gold_citations":["MOSHI:T3:r01"],"notes":"..."}
```
Grading (manual, with rules): correct = answer matches gold and every claim cites a chunk that supports it; wrong = any unsupported/incorrect claim; refused_correctly = unanswerable and refused; refused_wrongly = answerable and refused.
Metrics: false acceptance rate (answered although unsupported — most important), false refusal rate, answer precision/recall, verifier FAR. One test question = 6.7 points; smaller differences are noise → choose on tokens and latency; no threshold claimed production-grade.

### 19.4 Model benchmark
Same 30 questions, chosen retrieval setting, 120B vs 20B: correct %, correct citation %, correct refusal %, tokens in/out, reasoning tokens, TTFT, total. Per category → `fallback_gate_categories`.

### 19.5 Retrieval comparison (Table 3)
Settings A, B, C (+ D optional) on the same 30 questions with 120B. Report quality, tokens per answer, TTFT. Choose one and say why. Under pressure, retrieval follows the context caps (§10.5).

### 19.6 Verifier stress test (`eval/verifier_stress.jsonl`, local only, reported separately)
~30 claim/evidence pairs, e.g.:
| Evidence | Claim | Expected |
| --- | --- | --- |
| GPU used for evaluation | GPU used for training | reject |
| Model A reaches 90% | Model B reaches 90% | reject |
| Latency is 100 ms | Latency is 100 seconds | reject |
| Discusses acoustic tokens | Defines semantic tokens | reject |
| Value in Table 2 | Same value attributed to Table 3 | reject |
| Supported sentence | Same claim paraphrased | accept |
FAR = unsupported claims accepted ÷ unsupported claims tested, per verifier level.

### 19.7 Optional experiments (only if time remains)
Setting D; NLI level 3; reasoning_effort low vs medium (dev split); Option 4+5 sentence-buffered streaming (experiment only); cache and single-flight in isolation (baseline → cache → single-flight → both, stand-in).

---

## 20. Build order, testing, change log

### 20.1 Build order
| Phase | Deliverable | Done when |
| --- | --- | --- |
| 1 | Ingestion + hybrid retrieval | gold chunks retrieved for the dev questions |
| 2 | One request end to end: retrieve → cited answer → evidence checks → refusal | an unsupported claim is caught and an unanswerable question refused |
| 3 | LangGraph orchestration + tracing | per-node spans in JSONL |
| 4 | Groq client + rate governor | real baseline and probes recorded |
| 5 | Go stand-in calibrated from phase 4 | Mode A and B fixtures running |
| 6 | Admission queue + overload handling | all scenarios run |
| 7 | Load tests, evaluation, tables | every table recomputed by script |
| 8 | Cache, single-flight, optional experiments | only if time remains |

Do not build the admission controller before a working request path exists. Do not build the simulator before real measurements exist.

### 20.2 Tests (pytest; no real Groq calls in tests)
Rule: after every unit, write its unit tests (`tests/unit/`) and extend the integration tests (`tests/integration/`) that wire it to every unit built so far; run the full suite and keep it green before starting the next unit (see CLAUDE.md "Testing after every unit").

Minimum coverage:
- Chunk IDs stable across rebuilds with the same corpus_version.
- Sentence splitter on tricky strings (e.g., et al., §3.2, 0.1, Fig. 2).
- Level 1/2 verifier on Mode B fixtures and the stress set.
- Governor: concurrent reservations never exceed capacity; settle/overrun; 429 freeze; header resync.
- Model decision table, every row.
- Single-flight: one waiter cancelling doesn't cancel the task.
- Queue: rejection on full/too-long wait; expiry before dispatch.
- results.csv columns and status mapping; tables recompute from a tiny synthetic CSV.
Go: unit tests for buckets, SSE framing, latency sampling determinism with a seed.

### 20.3 Change log
- 2026-11-11: v1 architecture; review 1 (verification levels, Option 1 production path, governor as estimate, deterministic fixtures, per-request model decision, single-flight semantics, counters); review 2 (models switched to gpt-oss after Llama shutdown, org-level buckets, probing method, Table 3 isolation, 20B gate, stress test, reservation margin).
- 2026-11-11: environment managed with uv (`uv sync`, `uv run ...`) instead of venv + pip; Python pinned to 3.12 in pyproject.toml. Repo root is the project folder itself (it plays the role of `relay/` in §4).
- 2026-11-11: phase 1 started. `fetch_papers` downloads the e-print and PDF; arXiv redirects `/e-print/{id}vN` to `/src/{id}vN` (same version), recorded as `final_url` in each manifest.
- 2026-11-12: `latex_parse` deviations from §6.2: `\paragraph{}` and unnumbered `\subsection*`/`\subsubsection*` are run-in titles inside their parent section (the PDFs give them no number, so they get no section_id); unnumbered top-level sections (AIAYN "Attention Visualizations", HNET "Acknowledgements") get an alphanumeric id from their title. Figure captions go to `figures.jsonl`; bibliographies to `bibliography_{CODE}.txt`. Output text is NFKC-normalized (math alphabets → ASCII). Handled paper quirks: MOSHI puts `\section` inside `table` floats (hoisted), MOSHI `\ensuremathmode` uses `\ifmmode` (math branch kept), HNET uses `\captionof{table}` inside figures. Checked against the PDFs: tables AIAYN 4 / MOSHI 19 / HNET 8, figures 5 / 11 / 16, all numbered section headings found.
- 2026-11-12: `pdf_validate` (§6.3) matches a paragraph's first 12 words, else any 8-word window of its first 40, against the PDF's normalized words; writes `page` into paragraphs.jsonl and `pdf_validation.json`. Coverage: AIAYN 100.0%, MOSHI 100.0%, HNET 98.8% (2 misses: a footnote-URL paragraph and a math-heavy one); every section heading found. Fix found by it: display equations set off by blank lines (and the "where …" text after them) are now merged into the paragraph that introduces them, as §6.2 intends.
- 2026-11-12: `chunker` (§6.4) details. Packing unit = whole paragraph when it fits the piece, else its sentences (`verify/sentences.py`, built now and shared with the verifier), else token cuts for a single over-long sentence. Table row chunks repeat a header line `Columns: …` and the caption; when the caption is longer than ¼ of the piece, row chunks repeat only its first sentence and the full caption becomes its own `caption` chunk (without this, 256-token table chunks held 1.7 rows and 55 overflowed). Table chunk `rNN` = row-group number. Caption chunks share the `cNNN` counter. Result (corpus c1): 256 → 455 chunks (268 text, 51 caption, 136 table), 512 → 256 chunks (152 / 37 / 67); none over its piece size. Tokenizer: tiktoken `o200k_harmony`. Config files created from §5.2/§5.4 with `relay/config.py` (pydantic `RetrievalSetting`).
- 2026-11-12: eval set `eval/questions.jsonl` written from Akash's 30 question ideas, rewritten to be gradable and not answerable from memory. Deviations from §19.3 (Akash's decision): distribution is direct 6 / multi_section 4 / numerical_table 3 / cross_paper 2 / unanswerable 15 (refusal_type near_miss 6, misleading 4, out_of_scope 5); fields follow his naming (`expected_answer`, `paper_id`, `evidence` = [{section, page, quote}]) instead of `gold_answer`/`gold_citations`. Gold chunk IDs are not stored: each evidence `quote` is an exact text substring resolved to chunk IDs at eval time, so gold survives re-chunking and works for both piece sizes (a test checks every quote resolves at 256 and 512, and that `must_not_appear` terms are absent from the target paper). The §14.4/§19.3 example "What GPU was used to train Moshi?" is answerable (MOSHI §4.4: H100); q21 asks for the GPU count instead, which the paper does not give. Parser fixes found while writing gold: `group()` mis-read a trailing backslash as escaping the opening brace; `\rule` struts and `\multirow{n}*{…}` leaked into AIAYN Table 3.
- 2026-11-12: `index_build` + retrieval (`rrf`, `rerank`, `context`, `search`) built. Indexed text = header "{paper title} | {section_id} {section_title}" + chunk text (for BM25 and BGE alike), so questions naming a paper or section topic can match. `expand.py` not built yet: only optional setting D uses expansion. First done-check (`relay/eval/retrieval_recall.py`, dev split, 8 answerable, cap 2500, `eval/reports/retrieval_dev.json`): any gold quote retrieved A 50.0% / B 87.5% / C 87.5%; all gold quotes A 25.0% / B 37.5% / C 62.5%; context tokens A 837 / B 796 / C 1426. Findings: reranking is the biggest single gain; 4 passages of 256 use only ~800 of the 2,500-token cap, so multi-evidence questions lose quotes; the cross-paper question q14 never retrieves its AIAYN evidence in any setting; unanswerable dev questions score as high as answerable ones (top cosine up to 0.81, cross-encoder up to 3.9), so a score threshold alone cannot refuse.
- 2026-11-12: two retrieval policies (Akash's decisions), explicit and configurable. (1) Depth and budget are separate: `final_passages` = passages offered to the context builder (A/B/D 8, C 4), `context_budget` = 1,600 tokens per setting; the builder applies min(budget, pressure cap), so Table 3 compares settings at the same token budget and reports actual context tokens. (2) Per-paper coverage: only when `paper_mentions.named_papers()` finds ≥ 2 explicitly named papers (a bare "Transformer" does not name AIAYN), candidates are retrieved within each named paper, reranked globally, and `retrieval_policy.cross_paper_reserve_per_paper` (2) passages per paper are pulled to the front round-robin before the token cap. Every other question keeps global retrieval. No similarity threshold for refusal (q16/q24 show high scores without support); refusal is decided by the phase 2 evidence checks. Dev rerun at 1,600 tokens: any gold A 75.0% / B 87.5% / C 87.5%; all gold A 50.0% / B 37.5% / C 62.5%; quote recall A 63.2% / B 63.2% / C 68.4%; mean context tokens A 1,465 / B 1,460 / C 1,316. q14 now gets both papers' passages but still no gold chunk: the question says "segmented", the papers say "byte-pair encoding" / "word-piece vocabulary" / "GPT2 tokenizer" (vocabulary mismatch, a write-up example). Test split not run.
- 2026-11-12: **Phase 1 closed with known retrieval limitations** (Akash's decision), not "retrieval solved". Known failures kept as evidence, questions not rewritten: q14 vocabulary mismatch ("segmented" vs the papers' tokenizer terms); q03 retrieves one of its two evidence passages. Reranking is not ruled out: its early lead came partly from shallow depth, and it may still matter at larger candidate depths. Setting D removed from the active comparison until `expand.py` exists (it would duplicate B). Retrieval-only test split run once, frozen, on a committed tree: `eval/reports/retrieval_test_frozen.json` (the script refuses to overwrite it or to run on a dirty tree); test results are not used to tune chunking, depth, reranking or query processing. The full-pipeline test (answer quality, citations, refusal, latency with the real model) runs once after the setting is chosen.
- 2026-11-12: frozen retrieval-only test run done on commit a0a6a8c (report committed in 4df3575). Test split, 7 answerable: any gold A 100.0% / B 71.4% / C 85.7%; all gold 57.1% for all three; quote recall A 66.7% / B 60.0% / C 60.0%; mean context tokens A 1,501 / B 1,437 / C 1,213. The dev ranking (B, C ahead of A) does not hold on test. All test misses are HNET questions (q06, q10, q13): HNET's related-work appendix (§A, A.1, A.3) and Table 6 ("related architectures") outrank its own method and results text, an example of related-work text misleading a naive retriever. Unanswerable test questions reach cross-encoder scores up to 7.75 (q17), above most answerable ones. Not used for tuning.
- 2026-11-12: **verification redesign** (Akash's decision; §14.3–14.5 rewritten). (1) Pre-model sufficiency is structural, not lexical: refuse only with no usable passage or a named paper missing from a multi-paper question. Dev measurement that drove it: every key-term rule tried refused 1–3 of 8 answerable dev questions over phrasing words ("go", "matter", "prevents", "segmented"). (2) Lexical rarity is not evidence of answerability. (3) Level 2 (numbers, technical/rare terms, paper attribution) only flags. (4) Semantic support decides: local NLI entailment ≥ `entail_min`, or ≥ `entail_min_flagged` when flagged; the NLI check (former optional level 3) is now part of the production verifier. Built: `llm/prompts.py`, `verify/citations.py`, `verify/evidence.py`, `verify/nli.py`, `verify/verifier.py`, `verify/sufficiency.py`, `pipeline.py` (the phase 2 path that phase 3's graph nodes will wrap). To measure on real-model runs: refusal accuracy on unanswerable questions, false-refusal rate on answerable ones, and tokens spent (structural-only checking sends more unanswerable questions to the model).
- 2026-11-12: verifier stress test run once (`eval/verifier_stress.jsonl`: 40 pairs from the papers, 20 supported / 20 unsupported, 25 calibration / 15 held-out; `relay/eval/stress_verifier.py`; raw scores and decisions in `eval/reports/verifier_stress.json`, never overwritten). Thresholds chosen on calibration only (cost = 2 × false acceptances + false rejections), then frozen. Held-out (8 supported, 7 unsupported): small P1 (t 0.25) FAR 42.9% / FRR 25.0%; small P2 (tc 0.25, tf 0.25) FAR 57.1% / FRR 25.0%; base P1 (t 0.95) FAR 14.3% / FRR 37.5% (calibration FRR 66.7%); base P2 (tc 0.45, tf 0.02) FAR 42.9% / FRR 0.0%. Latency per claim p50/p95: small 286/437 ms, base 553/808 ms. Base P2's held-out false acceptances are the predicted blind spot (u06 reversed cosine rule, u17 supported fact + added claim: both neutral, unflagged) plus one NLI error (u08 entailment 0.996 for a reversed causal/non-causal claim, which also defeats P1). Flag-rule defects seen on calibration rows (not fixed: fixing and re-evaluating on the same held-out pairs would tune on them): "one ... one" read as the number 1 (s20); ordinary words rare in the corpus ("lets", "parallelized", "generalize") flagged. No model, policy or threshold adopted yet; config still holds placeholders.
- 2026-11-12: stress test round 2 prepared (Akash: option 3). Level 2 flag fixes, diagnosed on round-1 calibration pairs only: "one" is no longer read as the number 1 (s20); rarity no longer flags anything, so only name-like terms (digit, hyphen/underscore, inner or mid-sentence capital), numbers and paper attribution raise flags (s04 "parallelized", s06 "lets", u19 "generalize" were plain paraphrase); the IDF `Vocabulary` was removed from the verifier. Atomic claim decomposition added (`verify/claims.py`): the sentence plus each clause split at connectors, every part must pass; the full sentence stays because "X because Y" can invent the link between two true facts. 40 fresh pairs (`eval/verifier_stress_v2.jsonl`, no claim reused from round 1, same category mix, 25/15 split) with new causal reversals (u16, u18, u19) and compounds (u10, u17, u20 mixed; s04, s19 fully supported). Compared: `cross-encoder/nli-deberta-v3-base` vs `MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli`, policies P1/P2, whole vs atomic.
- 2026-11-12: pre-run fixes from review: lists ("A, B, and C") and "since" are no longer split into clauses; NLI models load in float32 (the ANLI checkpoint defaults to float16 on CPU). Round 2 run once (`eval/reports/verifier_stress_v2.json`). Held-out (8 supported / 7 unsupported) FAR / FRR: base P1-whole (t 0.98) 0.0% / 62.5%; base P1-atomic (t 0.6) 0.0% / 75.0%; base P2-whole 42.9% / 25.0%; base P2-atomic 28.6% / 37.5%; ANLI P1-whole (t 0.9) 14.3% / 25.0% (accuracy 80.0%); ANLI P2-whole 14.3% / 62.5%; ANLI P1-atomic (t 0.9) 14.3% / 50.0%; ANLI P2-atomic 28.6% / 50.0%. Latency per sentence p50/p95: whole ≈ 445/580 ms, atomic ≈ 516/1,618 ms. Atomic decomposition caught no extra unsupported claim on held-out and added false rejections of fully supported compounds (s09, s19). The one ANLI P1 false acceptance is u08 ("all four DNA models were H-Nets"); its false rejections include s02, a verbatim sentence (entailment 0.563). No passage exceeded the 512-token NLI input (max 328 for 256-token chunks). Not adopted: decision pending.
- 2026-11-12: real request path built (no real call made yet). `llm/client.py`: streaming OpenAI-compatible client; per Groq docs (read 2026-11-12) it sends `max_completion_tokens` (max_tokens is deprecated), `reasoning_effort`, `include_reasoning: false` (reasoning_format is unsupported for gpt-oss); usage is read from `usage` or `x_groq.usage` because the docs do not say where streaming usage appears; raw calls appended to results/raw/{real,standin}/ without request headers; `from_env` defaults to the stand-in unless LLM_PROVIDER=groq. `governor/`: per-model RPM/RPD/TPM/TPD buckets, all-or-nothing reservation, settle to actual usage, resync from headers (requests = per day, tokens = per minute), 429 freezes the per-minute buckets; no retries yet. `llm/hosted.py`: the pipeline's LLM callable, reserve -> call -> settle. `config/limits.yaml` from the docs' sample table. `scripts/check_models.py`, `scripts/ask.py` (refuses test-split questions).
- 2026-11-13: provisional verifier configured (Akash): `MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli`, P1 whole-sentence, entailment ≥ 0.9, no atomic split. A stricter bar for flagged claims (0.95–0.99) was checked on round-2 calibration pairs only: no extra catch, one extra false rejection, so flags stay diagnostic at 0.9.
- 2026-11-13: **first real Groq calls** (approved budget 8, used 7; raw: `results/raw/real/2026-11-12.jsonl` (UTC date), summaries: `results/raw/e2e/2026-11-12.jsonl`). All HTTP 200, no 429, 10,147 tokens in total, key absent from every log. Model checks: both IDs exist; gpt-oss adds a fixed 67-token preamble to every request (78 reported prompt tokens vs 11 counted locally, both models) → `reservation.request_overhead_tokens: 67`; afterwards local prompt counts were within ±1 token of Groq's, with no reservation overrun. Usage arrives in the top-level `usage` field of the stream and includes `completion_tokens_details.reasoning_tokens` (14–73 per call). Headers: RPD reset after one request read 1m26.4s = 86,400 s / 1,000, i.e. the daily request limit refills continuously; per-minute tokens remaining read 7,906 for each model after its own 94-token call, suggesting separate per-model TPM (to confirm in limit probing); `x-ratelimit-reset-tokens` sometimes implies a few hundred tokens more than reported usage (first q05 call: 17.48 s ≈ 2,331 tokens vs 1,930 reported), unexplained; the governor resyncs to the header, so it stays conservative. Model latency (120b, answers): first byte 394–983 ms, first content 648–1,145 ms, generation done 757–1,189 ms; the first call of the session queued 1.36 s. Outcomes (dev questions, setting B): q05 answerable → **refused** (false refusal): all three draft-1 sentences were correct, NLI accepted one (0.94) and rejected two (0.84, 0.60); the redraft's restatement of the accepted claim scored 0.63; LaTeX in answers (`\tfrac12`, `\hat{x}_t`) also tripped a level-2 term flag. q11 answerable → **refused** (false refusal): the correct claim "single-head dev BLEU 24.9 [AIAYN:T3:r02]" scored 0.03/0.09 against the table-row text; the model correctly said the base BLEU was not stated (base row not retrieved, the known q11 retrieval miss). q16 unanswerable → **correctly refused** by the model (`NOT_IN_PAPERS`, 1 call). User-visible time (whole answer verified before release): 8.4 s, 4.0 s, 2.9 s. Not tuned on these questions.
- 2026-11-13: **operational findings from the local re-run of those drafts** (no API call; recomputed entailments equal the logged ones; these are observations, not an evaluation set). Every rejection was the P1 condition "best entailment < 0.9"; no level-1 failure, no contradiction rule (P1 has none), and flags never changed a threshold (flagged = unflagged = 0.9). q05: the gold passage HNET:2.2.1:c023 was retrieved and cited; all five sentences in the two drafts are correct. Scores: 0.935 (accepted), 0.844, 0.601 / redraft 0.631, 0.611. Causes: (a) generalization: the same claim scored 0.935 and 0.631 depending on LaTeX spelling (`\bigl(` vs `\!\left(`); (b) sentence-level NLI loses context: draft-2 S2 starts "It then forms…"; (c) retrieval: draft-1 S3 ("lower similarity → boundary") is stated almost verbatim in HNET:2.2.1:c024, which was not retrieved (the known q05 miss), so it was cited to a passage that supports it only indirectly (neutral 0.363). Implementation defect seen: the LaTeX macro `tfrac12` was flagged as a technical term (harmless here because the thresholds are equal). q11: AIAYN:T3:r02 was retrieved and contains "(A): h = 1; … BLEU / (dev) = 24.9"; the claim is correct but scored entailment 0.028/0.093 with contradiction 0.542/0.547: generalization failure on table-row premises, likely because the same chunk lists sibling rows "(A): h = 4 … 25.5", "(A): h = 16 … 25.8". No flags. Second defect: the exempt sentence "The papers do not state the development BLEU of the base configuration" is false (Table 3 base row: 25.8); the row (AIAYN:T3:r01) was not retrieved, so the model could not see it, and the verifier never checks exempt sentences. q16: correct refusal (`NOT_IN_PAPERS`, 1 call, 1,836 tokens spent on an unanswerable question that the structural check rightly let through).
- 2026-11-13: **fixes for the real-call findings** (Akash's P0–P3; ANLI + P1 at 0.9 kept; nothing tuned on q05/q11; no Groq call; versions bumped: prompt p2, verification policy v2, corpus c2). P0 refusal explanations: the exemption is gone. The prompt (p2) asks for "The retrieved passages do not state <missing part>." with no citation and no number; level 1 checks every such absence sentence (no citation, no value) and rejects any unscoped claim that the papers/authors do not state something (the model only saw the retrieved passages). An answer made only of absence sentences is a refusal. The refusal reply no longer claims the papers lack the answer: "I couldn't find this in the passages I retrieved from the three papers, so I won't guess. I searched: …". `NOT_IN_PAPERS` stays a control signal. Limit: whether the retrieved passages really lack the missing part is not provable by entailment; absence sentences are checked for scope and content, not for truth. P1 context: a claim opening with a reference ("It", "They", "This", …) is also scored with the previous sentence placed before each premise, only if that sentence was accepted; the best entailment decides, so context resolves the reference but cannot supply missing support; the context used is recorded per sentence. P2 tables: table chunks carry structured rows (configuration; per cell column, metric, split, value; "BLEU / (dev)" → metric BLEU, split dev); row lines read "Configuration <name>: …" (existing gold quotes stay substrings); the verifier scores a table claim against each row alone (rendered "Configuration (A): h is 1; BLEU (dev split) is 24.9.") and against all rows as separate labelled lines (for comparisons), so a sibling row is never beside the claimed value. Retrieval: with rows naming their configuration, AIAYN:T3:r01 ("Configuration base") ranks 2–3 by BM25 for base-configuration queries at both piece sizes (regression test). P3 flags: claims and passages are de-LaTeXed (delimiters, \commands incl. digit shorthand like \tfrac12, escapes, braces) before number/term extraction; numbers inside LaTeX are still checked. Tests: 229 passed (regressions for all four). Not re-measured: the frozen retrieval reports and both stress-test reports describe corpus c1 and the earlier verifier; the c2 corpus and verifier v2 need fresh evidence pairs before any claim about their accuracy.
- 2026-11-13: corpus c3: `\specialrule{…}{…}{…}` leaked into AIAYN Table 4 row headers ("1pt-1pt0pt Transformer (4 layers)"); now stripped with the other table rules (regression test). **Verifier v2 local check** (`relay/eval/verifier_v2_check.py`, run once, `eval/reports/verifier_v2_check.json`, no LLM call): 20 fresh answers (`eval/verifier_v2_stress.jsonl`: table rows, comparisons, LaTeX, pronoun resolution, scoped absence, unsupported additions; 12 calibration / 8 held-out; no q05/q11/q16 evidence, no earlier claim), last sentence judged by the full production verifier. At the provisional 0.9: held-out FAR 0.0% (0/4), FRR 75.0% (3/4: comparison c2 0.128, LaTeX l2 0.067, pronoun p2 0.592), accuracy 62.5%; calibration FAR 16.7% (p3), FRR 33.3% (c1 0.638, l1 0.663). The calibration split alone would choose 0.6; at 0.6 held-out FAR rises to 25.0% (c3 accepted) with the same FRR, so 0.9 stays. Latency per answer (verify only): mean 1,305 ms, p50 1,325 ms, p95 3,100 ms (table chunks add one premise per row). Scoped absence worked (a1, a2 accepted; a3 unscoped rejected); single table rows worked (t1 0.973, t2 0.993; t3 sibling value rejected 0.051); unsupported additions all rejected. Rejected-context check: p3's rejected first sentence gave no context to "It …" (as designed), but "It keeps continuous gradients…" was accepted on its own (0.996) because the NLI model resolves "It" to the passage's subject (the STE), not the answer's (the smoothing module); the answer as a whole still fails because its first sentence is rejected. Replays (observations only): q05 drafts unchanged (0.935 / 0.844 / 0.601; 0.631 / 0.611; draft-2 "It then forms…" got no context because its previous sentence was rejected): the correct explanation still does not pass; LaTeX normalization changed flags only, the NLI input is still raw LaTeX. q11: the table claim scored 0.015 / 0.311 even against its own row alone, so per-row premises did not fix it; the old drafts' "The papers do not state …" is now rejected as unscoped, as intended. q16: still refused correctly through `NOT_IN_PAPERS` (1 call; the refusal is worded about the retrieved passages). Local checks for q05/q11 did not pass, so the eighth approved Groq call stays unspent.
- 2026-11-13: **verifier v2 frozen** (Akash): ANLI + P1 whole-sentence at 0.9, not lowered. Its measured quality is reported as measured: held-out 0/4 false acceptances and 3/4 false rejections on the v2 check (8 held-out answers), so no claim of high verifier accuracy. Prompt p3: rule 7 asks for plain text, no LaTeX, formulas in ordinary characters with mathematical meaning and every number kept. No further NLI experiments.
- 2026-11-13: **stand-in, load generator, front door** (phases 5–6, minimal). Stand-in (`standin/`, Go stdlib): OpenAI-compatible SSE; Mode A only (Mode B functional fixtures not built); replies from `fixtures/perf/` recorded from the real calls (`scripts/record_fixtures.py`: q05, q11 with their real redrafts, q16), so load runs cycle three questions; latency fitted from the 5 real answer calls (first byte log-normal median 515 ms, σ 0.368; hidden reasoning 280 tok/s; content 530 tok/s), load slowdown and fault rates are configured guesses (no concurrency or failures observed on the real service); RPM/RPD/TPM/TPD buckets with continuous refill (as the real RPD header showed), charged prompt + recorded completion at admission; 429 with retry-after, Groq's x-ratelimit-* headers and Groq's public error format (no real 429 body observed); prompt tokens estimated as characters/4 + 67; self-timing log per request in results/raw/standin/. Front door (`relay/main.py`): FastAPI, one worker, bounded FIFO queue (max 50) and 4 work slots (config), rejection on arrival when full or when the estimated wait (max of slot wait and primary-model token wait, refill-rate based) exceeds the deadline, expiry while queued, immediate unavailable reply while a per-day 429 is in force, work in a thread pool, one trace line per request with queue, retrieval, governor wait, provider first byte, provider first token, generation, NLI verification, user-visible first token and total; `/admin/reset` (stand-in only) gives every scenario a fresh day. Retry policy built (§15) with one deviation: the 10% budget has a floor of one retry per minute. Not built: pressure levels L1–L3 (requests run at L0 caps), 20B fallback (no validated gate), cache, single-flight, LangGraph (the pipeline is plain functions; tracing is per request, not per node). Load generator (`loadgen/`, Go stdlib): open-loop seeded Poisson arrivals, latency from the scheduled time, log-normal patience (median 15 s, σ 0.5, 3–60 s), admin steps at set times, summary of offered / admitted / completed / rejected / timed out / gave up / errors. Scenarios are JSON (Go's stdlib has no YAML) in `config/scenarios/`, rates relative to K ≈ 2.3 requests/min (8,000 TPM at ≈ 3,400 tokens per request with the recorded fixtures and redrafts): steady 0.7K for 600 s; burst 0.5K with 20K for 30 s; overload 2K for 480 s; rpm_halved = steady shape with per-minute requests and tokens halved; daily_exhausted = steady shape with the daily budget emptied at 120 s; retrieval_A/B/C = steady shape per setting for Table 3 latency and tokens (quality is not measured on the stand-in). `scripts/make_results_csv.py` and `scripts/make_tables.py` build results.csv and Tables 1, 2a, 2b, 3 (+ component medians).
- 2026-11-13: **load runs completed and validated** (2026-11-12 19:50–20:45 UTC; 168 stand-in requests in 8 scenarios + 3 real requests + 2 real model checks = 173 rows). Every scenario reconciles (completed + rejected + timed out + gave up + errors = offered), every request has exactly one trace, ids are unique, sends were at most 6.3 ms late. Daily exhaustion: after the budget was emptied, 17 user arrivals produced only 2 provider calls (the per-day 429 that revealed it, and one probe after retry-after) and 17 "today's capacity" replies. Halved limit: every successful call carried `x-ratelimit-limit-tokens: 4000` (steady: 8000), 2 TPM 429s. Retries: 4 of 168 requests; backoff = retry-after where given; per-day 429s never retried. Stand-in token accounting: completions equal the fixtures exactly; prompt estimate (chars/4 + 67) a median 10.5% below the local tiktoken count (−25.7% to +8.2%); no reservation overrun. Findings that limit the results: (1) every completed request was a refusal: zero verified answers in any scenario (p1 fixtures); (2) NLI verification ran only when a fixture's citations matched this run's retrieval: the fixtures cite c1 / 256-token chunk ids, so q11 under B (20/20) and everything under C never reached NLI (level 1 rejects in < 1 ms) and their latency understates verification; NLI time where it ran: ≈ 2.4–4.2 s per request; (3) "steady" at 0.7 K still had 28.6% rate-limited: Poisson clustering, two calls per q05/q11 request and per-call reservations (prompt + 770) exceed 8,000 TPM within users' patience, so K ≈ 2.3/min was optimistic; (4) the governor learns a halved limit only through remaining-token headers and 429 freezes, not the new cap (effective-limit lowering not built), which made the retry in rpm_halved-0019 cost 3 s for nothing; (5) a retry attempt rejected locally has no call record, so its governor wait is missing from the trace. Script fixes during validation (no raw data changed): a user who gave up gets no reply tier even if the server finished later (it had inherited the server's tier); supporting tables added to results/tables.md: reply tiers and NLI time where NLI ran.
- 2026-11-13: **why q05/q11 were refused, and submission trimming** (local replay of the recorded drafts through the current corpus c3 + verifier v2 for A/B/C; no LLM call; threshold 0.9 unchanged; nothing written to results/). Real run (p1, corpus c1, setting B): both refusals were the NLI threshold, with the evidence present: q05 cited the retrieved gold HNET:2.2.1:c023, all sentences correct, entailment 0.935 / 0.844 / 0.601, redraft 0.631 / 0.611; q11 cited the retrieved AIAYN:T3:r02 for the correct 24.9, entailment 0.03 / 0.09, and its absence sentence was false (base row r01 not retrieved). No citation, structural or level-2 rejection. Replay today: q05 A and B give the same scores (NLI); q05 C and q11 B/C fail level 1 because the fixtures cite chunk ids that c3 / 512-token pieces do not retrieve (c023 does not exist at 512; under c3 setting B retrieves neither T3:r01 nor T3:r02 for q11, a retrieval regression from c1 not covered by the frozen c1 retrieval reports); q11 A retrieves r01 and r02 but the table claim still scores 0.015 / 0.311; the old "The papers do not state …" sentence is rejected as unscoped. Conclusion: the root cause is the verifier (NLI over-strict on LaTeX, pronouns and table rows) with correct evidence present; the p1-fixture / c3 mismatch adds citation failures in the load runs but did not cause the real-run refusals. Whether a p3 draft would pass is not measured (needs Groq calls). Write-up shortened; Table 1 wording and the Table 2b note in scripts/make_tables.py shortened (text only, no values).
- 2026-11-13: **retrieval-only evaluation re-run on the current index (corpus c3)** (Akash's request; no LLM; questions, settings A/B/C and the 1,600-token budget unchanged; nothing tuned). The c1 reports stay as they were (`eval/reports/retrieval_dev.json`, `retrieval_test_frozen.json`); the new ones are `retrieval_dev_c3.json` and `retrieval_test_c3.json`. Deviation: the test split was measured a second time, on an uncommitted tree (`--allow-dirty`, recorded as git_dirty with the sha256 of both chunk files), because committing was not authorized; the frozen file still cannot be overwritten. Any gold retrieved, dev / test: A 75.0 / 100.0, B 75.0 / 71.4, C 87.5 / 85.7 (c1: B dev 87.5; the drop is q11, whose two table rows B no longer retrieves); quote recall A 63.2 / 73.3, B 57.9 / 60.0, C 68.4 / 60.0. On c3, B, the configured default, covers no more gold than A or C on either split; the setting was not changed (decision left to Akash). Deviation from CLAUDE.md rule 4: Table 3b (coverage) is generated by `scripts/make_tables.py` from these two reports, not from results.csv, because coverage is not per-request data; Table 3 itself still comes only from results.csv.
- 2026-11-13: **verifier policy v3** (Akash's controlled pass; threshold 0.9 unchanged; no LLM call; ANLI model unchanged; v2 reports untouched). Changes: (A) `verify/mathnorm.py` rewrites claim and premise into one plain form before NLI (sizing/spacing/font commands dropped, rac/	frac12 -> a / b, accents kept as x-hat / z-bar, Unicode narrow spaces and non-breaking hyphens normalized; displayed text unchanged); (B) context rule unchanged (only from an accepted previous sentence), now also applied per clause; (C) table rows rendered as one sentence ("The configuration X has m on the dev set = v, ..."), the all-sibling-rows premise replaced by a premise of only the relevant rows (named or value-matched, across all cited tables), and `verify/tables.py` rejects a value owned only by a row of an unnamed configuration when the claim names another, and an absence sentence when a retrieved row of the named configuration reports the whole named metric (rejection-only; configurations without a two-letter name, e.g. "(A)", are never used); (D) every clause from `claims.atomic_claims` must be entailed as well as the whole sentence. The row rendering was chosen on the v3 calibration table items plus q11 (disclosed tuning on a known failure). Evaluation: new set `eval/verifier_v3_eval.jsonl` (42 answers written by the assistant before any v3 change, sha256 836fe67c…, 7 categories, 14 calibration / 28 evaluation, no q05/q11/q16 evidence, no AIAYN Table 3); v2 scored on it before the change (`eval/reports/verifier_v3_eval_v2_baseline.json`), v3 once after (`…_v3.json`). Evaluation split, v2 -> v3: FAR 26.7% (4/15) -> 13.3% (2/15), FRR 7.7% (1/13) -> 7.7% (1/13), accuracy 82.1% -> 89.3%; the whole difference is two absence claims contradicted by a retrieved table row (a3, a4); direct, math, table, compound and numeric items were already all correct under v2, so this set does not show an effect of (A), (C)-rendering or (D). Remaining errors: p4 (supported pronoun claim, 0.70) and a5, a6 (absence claims contradicted by text passages: no check exists). Original 8 held-out pairs re-scored under v3 into `eval/reports/verifier_v2_check_policy_v3.json`: unchanged, FAR 0/4, FRR 3/4 (c2 0.097, l2 0.054, p2 0.597), accuracy 62.5%. Latency per answer on the new set: mean 545 -> 697 ms, p95 967 -> 1,934 ms (more pairs: clauses, relevant-row premises). Replays (observations): q05 draft 1 now accepts 2 of 3 sentences (0.950, 0.924; the third, 0.560, is supported only by the unretrieved c024) and is still refused; draft 2 0.735 / 0.138 (the formula now matches exactly, the prose does not; no context because sentence 1 was rejected); q11 draft 2's BLEU claim passes (0.979; draft 1 0.459) but both drafts stay refused by their unscoped "The papers do not state" sentence, as intended; under c3 setting B and C q11 fails citation checks (rows not retrieved) and q05 under C cites a chunk that does not exist at 512 tokens: refusals, as designed; q16 still refused by NOT_IN_PAPERS with one call. The load runs and results.csv were produced under v2 and are not re-run.
- 2026-11-13: **verifier policy v4 (safety pass on v3)** (threshold 0.9 unchanged; no LLM call; v2/v3 reports untouched; v3 code snapshot kept outside the repo; note: the v2 code itself was replaced in place by the v3 pass and was never committed, so v2 exists only as its reports). Three changes, all rejection-only: (1) `mathnorm` keeps sub/superscript groups (`k_{t-1}` -> `k_(t - 1)`; v3 wrongly mapped it to the same text as `k_t - 1`); (2) `verify/formulas.py`: every LaTeX-delimited formula in a claim that states a relation must occur in a cited passage up to notation (plain_math, spaces removed); algebra is not checked, so rearranged or derived formulas are rejected (conservative), and plain-text formulas are not checked; (3) `verify/absence.py`: an absence sentence is rejected when all content words of its missing part (at least 2; question/framing words dropped) occur in one sentence of one retrieved text passage; lexical, retrieved evidence only. New targeted set `eval/verifier_v3b_targeted.jsonl` (18 cases written before v4, sha256 7b35a808…, never tuned on): v3 -> v4 FAR 70.0% (7/10) -> 20.0% (2/10), FRR 12.5% (1/8) -> 12.5% (1/8), accuracy 55.6% -> 83.3%; wrong formulas rejected 3/5 -> 5/5, supported formulas accepted 3/4 -> 3/4 (f2 NLI 0.896), supported absences accepted 4/4 -> 4/4, text-contradicted absences rejected 0/5 -> 3/5, of which two on weak evidence (u3 matched the heading "Label Smoothing.", u1 the learning-rate formula line), u2 missed because "used" is a framing word, u5 missed (other words). v3 evaluation split: FAR 13.3% -> 6.7% (a5 now rejected), FRR 7.7% -> 15.4% (m4 newly rejected, NLI 0.932 -> 0.874 because the kept subscript groups change the NLI input), accuracy 89.3% -> 89.3%. Original 8 held-out: unchanged, FAR 0/4, FRR 3/4, 62.5% (`eval/reports/verifier_v2_check_policy_v4.json`; l2 and l4 now fail the formula check instead of NLI: l2 is supported but derived from a case definition, the stated cost of the rule). Replays: q05 draft 1 0.952 / 0.937 / 0.538 (refused), draft 2 0.757 / 0.175 (refused); q11 draft 2 BLEU claim 0.979 but refused by its unscoped absence sentence, draft 1 0.459; q16 refused by NOT_IN_PAPERS, 1 call. Still zero verified end-to-end answers.
- 2026-11-13: **submission audit (documentation only; verifier v4 frozen, threshold 0.9; no Groq call, no load run, no behaviour change).** Version boundaries: results/results.csv, Tables 1/2a/2b/3 and every load-test number = verifier v2; original 8-pair held-out = v2 report plus v3/v4 re-scores in separate files; 28-case set = v2 baseline, v3, v4; 18-case targeted set = v3 vs v4; submitted code = v4. v4 was never load-tested, and v2 load latency is never combined with v4 verifier latency. Provenance: Table 3b now reads `results/retrieval_eval.csv` (`scripts/make_retrieval_csv.py`, one row per report and setting from the four retrieval reports, which are unchanged); the c3 test run is labelled a post-rebuild diagnostic and shown beside the frozen c1 test result. results/results.csv regenerates byte-for-byte from results/raw/ (sha256 84fc38ff…; 173 rows; the 12 required columns in order). v3 code archived in `archive/verifier_v3_snapshot/` (copied from the temporary session directory, SHA256SUMS inside); v2 code is not recoverable (edited in place, never committed). Write-up: version statement in §1, v2 labels in §3, a verifier-evaluation paragraph separate from the load results, v4 limits in §8, personal sections left as placeholders (no conversation export in the repository yet). Known stale text not changed: CLAUDE.md still describes a LangGraph state machine and a 20B fallback that were not built (see the not-built list above).
- 2026-11-13: **submission readiness (documentation only; verifier v4, threshold 0.9 and all experiment data unchanged; no Groq call, no load run).** Report cut to 2 rendered A4 pages with the personal sections as placeholders (`scripts/render_writeup.py`: headless Chrome, 10 pt body, 7.5 pt tables, 1.4/1.5 cm margins; latency tables keep cells on one line). Generated tables made more compact without changing a value: p99 with N < 100 marked "*" with a caption instead of "(N small)" in every cell; Table 3's unmeasured quality cells "not measured*" with a caption; Table 3b shows any-gold coverage only (dev c3, test c3 post-rebuild diagnostic, test c1 frozen), quote recall and context tokens stay in results/retrieval_eval.csv; shorter Table 2b note. CLAUDE.md corrected: it described a LangGraph state machine and a 20B fallback as the app; it now lists the function pipeline as built and the planned-but-not-built items, consistent with this changelog. Personal-section guidance and candidate examples (labelled as suggestions) in writeup/PERSONAL_SECTIONS_TODO.md; no conversation export exists in the repository.
- 2026-11-13: **strict two-page report** (Akash's decision; documentation only). Table 3b moved out of the report (kept in results/tables.md and results/retrieval_eval.csv; one pointer sentence in the report); personal-section placeholders sized to the reserved space (57 and 147 words; reserve 50–60 and 140–160); Table 2a footnote shortened. Rendered with scripts/render_writeup.py: exactly 2 A4 pages, page 2 about 95% full, typography unchanged.

---

## 21. Sources
- Groq rate limits: https://console.groq.com/docs/rate-limits.md (limit types incl. ITPM/OTPM, organization-level scope, headers = RPD + TPM only, sample limits)
- Groq models: https://console.groq.com/docs/models.md (production models, Enterprise-only Llama entries, prices)
- Groq deprecations: https://console.groq.com/docs/deprecations.md (Llama 3.3 70B / 3.1 8B shut down for free and developer tiers on 2026-08-16)
- Groq reasoning: https://console.groq.com/docs/reasoning.md (reasoning_effort, reasoning visibility)
- Papers: https://arxiv.org/abs/1706.03762v7 · https://arxiv.org/abs/2410.00037v1 · https://arxiv.org/abs/2507.07955v2

---

## 22. Open decisions (decide, then move to §20.3)
- [ ] "Fast" target (user-visible TTFT percentile + value) and the patience distribution.
- [ ] Recalibrate every initial number after real measurements: context caps, pressure triggers, 2 s fallback wait, output p95 reservations, 10% retry budget, refusal thresholds.
- [ ] L3 reply: include top cited passages (no generation) or not.
- [ ] reasoning_effort: keep low, or run low vs medium on the dev split.
- [ ] Which optional experiments to run (D, NLI, Option 4+5). FIFO stays.
- [ ] LangGraph kept with timing wrappers; switch to a plain asyncio state machine if it gets in the way.
