# The Relay: architecture and what was built

The Relay answers questions about exactly three research papers. Every sentence of an answer must cite a passage
from those papers that a checker has confirmed supports it; otherwise the person gets an honest refusal. It was
built to stay predictable when the model provider is busy: bursts of users, per-minute and per-day rate limits,
slow or failing calls.

| Paper | Code | arXiv version |
|---|---|---|
| Attention Is All You Need | AIAYN | 1706.03762v7 |
| Moshi: a speech-text foundation model | MOSHI | 2410.00037v1 |
| H-Net: dynamic chunking | HNET | 2507.07955v2 |

The model is `openai/gpt-oss-120b` on Groq's free tier (`reasoning_effort: low`, at most 900 output tokens). Real
calls are scarce (a shared daily token budget), so all load tests ran against a Go stand-in that imitates Groq.

## 1. The big picture

<div class="fig"><svg viewBox="0 0 720 384" width="100%" xmlns="http://www.w3.org/2000/svg" font-family="Arial, sans-serif"><defs><marker id="a" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="#333"/></marker></defs><rect x="10" y="55" width="115" height="74" rx="6" fill="#f3f4f6" stroke="#555" stroke-width="1"/><text x="67.5" y="71" text-anchor="middle" font-weight="bold" font-size="11">Client</text><text x="67.5" y="84" text-anchor="middle" font-size="9.5">POST /ask</text><text x="67.5" y="97" text-anchor="middle" font-size="9.5">waits for one</text><text x="67.5" y="110" text-anchor="middle" font-size="9.5">SSE reply</text><rect x="156" y="55" width="115" height="74" rx="6" fill="#f3f4f6" stroke="#555" stroke-width="1"/><text x="213.5" y="71" text-anchor="middle" font-weight="bold" font-size="11">Admission</text><text x="213.5" y="84" text-anchor="middle" font-size="9.5">FIFO queue, max 50</text><text x="213.5" y="97" text-anchor="middle" font-size="9.5">deadline per request</text><text x="213.5" y="110" text-anchor="middle" font-size="9.5">reject early if late</text><rect x="302" y="55" width="115" height="74" rx="6" fill="#f3f4f6" stroke="#555" stroke-width="1"/><text x="359.5" y="71" text-anchor="middle" font-weight="bold" font-size="11">Retrieve</text><text x="359.5" y="84" text-anchor="middle" font-size="9.5">BM25 + BGE-small</text><text x="359.5" y="97" text-anchor="middle" font-size="9.5">RRF fusion</text><text x="359.5" y="110" text-anchor="middle" font-size="9.5">re-rank, 1,600 tokens</text><rect x="448" y="55" width="115" height="74" rx="6" fill="#f3f4f6" stroke="#555" stroke-width="1"/><text x="505.5" y="71" text-anchor="middle" font-weight="bold" font-size="11">Sufficiency</text><text x="505.5" y="84" text-anchor="middle" font-size="9.5">structural check:</text><text x="505.5" y="97" text-anchor="middle" font-size="9.5">any usable passage?</text><text x="505.5" y="110" text-anchor="middle" font-size="9.5">no model call</text><rect x="594" y="55" width="115" height="74" rx="6" fill="#fde7c8" stroke="#555" stroke-width="1"/><text x="651.5" y="71" text-anchor="middle" font-weight="bold" font-size="11">Draft</text><text x="651.5" y="84" text-anchor="middle" font-size="9.5">gpt-oss-120b</text><text x="651.5" y="97" text-anchor="middle" font-size="9.5">every sentence</text><text x="651.5" y="110" text-anchor="middle" font-size="9.5">cites a passage</text><rect x="156" y="8" width="115" height="30" rx="6" fill="#f8d7d7" stroke="#555" stroke-width="1"/><text x="213.5" y="24" text-anchor="middle" font-weight="bold" font-size="11">busy / capacity reply</text><rect x="448" y="8" width="115" height="30" rx="6" fill="#f8d7d7" stroke="#555" stroke-width="1"/><text x="505.5" y="24" text-anchor="middle" font-weight="bold" font-size="11">refuse (no model)</text><rect x="594" y="190" width="115" height="74" rx="6" fill="#f3f4f6" stroke="#555" stroke-width="1"/><text x="651.5" y="206" text-anchor="middle" font-weight="bold" font-size="11">Verify (v4)</text><text x="651.5" y="219" text-anchor="middle" font-size="9.5">citations retrieved</text><text x="651.5" y="232" text-anchor="middle" font-size="9.5">NLI entails ≥ 0.9</text><text x="651.5" y="245" text-anchor="middle" font-size="9.5">tables, formulas,</text><text x="651.5" y="258" text-anchor="middle" font-size="9.5">absence claims</text><rect x="448" y="190" width="115" height="74" rx="6" fill="#d9f2df" stroke="#555" stroke-width="1"/><text x="505.5" y="206" text-anchor="middle" font-weight="bold" font-size="11">Answer</text><text x="505.5" y="219" text-anchor="middle" font-size="9.5">every sentence</text><text x="505.5" y="232" text-anchor="middle" font-size="9.5">verified; sent</text><text x="505.5" y="245" text-anchor="middle" font-size="9.5">in one piece</text><rect x="594" y="300" width="115" height="74" rx="6" fill="#fde7c8" stroke="#555" stroke-width="1"/><text x="651.5" y="316" text-anchor="middle" font-weight="bold" font-size="11">Redraft (once)</text><text x="651.5" y="329" text-anchor="middle" font-size="9.5">model gets the</text><text x="651.5" y="342" text-anchor="middle" font-size="9.5">verifier's</text><text x="651.5" y="355" text-anchor="middle" font-size="9.5">objections</text><rect x="448" y="300" width="115" height="74" rx="6" fill="#f8d7d7" stroke="#555" stroke-width="1"/><text x="505.5" y="316" text-anchor="middle" font-weight="bold" font-size="11">Refuse</text><text x="505.5" y="329" text-anchor="middle" font-size="9.5">honest refusal</text><text x="505.5" y="342" text-anchor="middle" font-size="9.5">with reason</text><rect x="156" y="190" width="261" height="74" rx="6" fill="#dbe8fb" stroke="#555" stroke-width="1"/><text x="286.5" y="206" text-anchor="middle" font-weight="bold" font-size="11">Rate governor</text><text x="286.5" y="219" text-anchor="middle" font-size="9.5">RPM · RPD · TPM · TPD token buckets</text><text x="286.5" y="232" text-anchor="middle" font-size="9.5">reserve → call → settle (every call, retries too)</text><text x="286.5" y="245" text-anchor="middle" font-size="9.5">a 429 freezes the bucket until retry-after</text><rect x="156" y="300" width="261" height="74" rx="6" fill="#dbe8fb" stroke="#555" stroke-width="1"/><text x="286.5" y="316" text-anchor="middle" font-weight="bold" font-size="11">Model provider</text><text x="286.5" y="329" text-anchor="middle" font-size="9.5">Groq free tier (real)</text><text x="286.5" y="342" text-anchor="middle" font-size="9.5">or the Go stand-in (load tests)</text><text x="286.5" y="355" text-anchor="middle" font-size="9.5">same OpenAI-compatible streaming client</text><rect x="10" y="190" width="115" height="184" rx="6" fill="#ffffff" stroke="#555" stroke-width="1" stroke-dasharray="4 3"/><text x="67.5" y="206" text-anchor="middle" font-weight="bold" font-size="11">Trace</text><text x="67.5" y="219" text-anchor="middle" font-size="9.5">one record per</text><text x="67.5" y="232" text-anchor="middle" font-size="9.5">request: queue,</text><text x="67.5" y="245" text-anchor="middle" font-size="9.5">governor, retrieval,</text><text x="67.5" y="258" text-anchor="middle" font-size="9.5">provider, retries,</text><text x="67.5" y="271" text-anchor="middle" font-size="9.5">verification times</text><line x1="125" y1="92" x2="156" y2="92" stroke="#333" stroke-width="1.2" marker-end="url(#a)"/><line x1="271" y1="92" x2="302" y2="92" stroke="#333" stroke-width="1.2" marker-end="url(#a)"/><line x1="417" y1="92" x2="448" y2="92" stroke="#333" stroke-width="1.2" marker-end="url(#a)"/><line x1="563" y1="92" x2="594" y2="92" stroke="#333" stroke-width="1.2" marker-end="url(#a)"/><line x1="213.5" y1="55" x2="213.5" y2="38" stroke="#333" stroke-width="1.2" marker-end="url(#a)"/><line x1="505.5" y1="55" x2="505.5" y2="38" stroke="#333" stroke-width="1.2" marker-end="url(#a)"/><line x1="651.5" y1="129" x2="651.5" y2="190" stroke="#333" stroke-width="1.2" marker-end="url(#a)"/><line x1="594" y1="227" x2="563" y2="227" stroke="#333" stroke-width="1.2" marker-end="url(#a)"/><text x="578.5" y="221" text-anchor="middle" font-size="9" font-style="italic" fill="#333">pass</text><line x1="625" y1="264" x2="625" y2="300" stroke="#333" stroke-width="1.2" marker-end="url(#a)"/><text x="620" y="286" text-anchor="end" font-size="9" font-style="italic" fill="#333">fail</text><line x1="680" y1="300" x2="680" y2="264" stroke="#333" stroke-width="1.2" marker-end="url(#a)"/><text x="675" y="286" text-anchor="end" font-size="9" font-style="italic" fill="#333">re-verify</text><line x1="594" y1="337" x2="563" y2="337" stroke="#333" stroke-width="1.2" marker-end="url(#a)"/><text x="578.5" y="331" text-anchor="middle" font-size="9" font-style="italic" fill="#333">fails</text><line x1="594" y1="129" x2="405" y2="190" stroke="#333" stroke-width="1.2" marker-end="url(#a)" stroke-dasharray="4 3"/><text x="500" y="169" text-anchor="middle" font-size="9" font-style="italic" fill="#333">reserve tokens</text><line x1="286" y1="264" x2="286" y2="300" stroke="#333" stroke-width="1.2" marker-end="url(#a)"/></svg></div>

One request, left to right. Grey steps are plain Python functions (`relay/pipeline.py`); the two orange steps are
the only ones that call the model; the blue blocks protect the provider's limits. Nothing is sent to the person
until the whole answer has passed verification, so the first token arrives together with the complete answer.

## 2. Building the library (offline, once)

<div class="fig"><svg viewBox="0 0 720 232" width="100%" xmlns="http://www.w3.org/2000/svg" font-family="Arial, sans-serif"><defs><marker id="a" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="#333"/></marker></defs><text x="10" y="14" font-size="10" font-weight="bold">Offline, once: build the library</text><text x="10" y="138" font-size="10" font-weight="bold">Experiments: load, measure, report</text><rect x="10" y="24" width="115" height="74" rx="6" fill="#f3f4f6" stroke="#555" stroke-width="1"/><text x="67.5" y="40" text-anchor="middle" font-weight="bold" font-size="11">Fetch</text><text x="67.5" y="53" text-anchor="middle" font-size="9.5">exact arXiv</text><text x="67.5" y="66" text-anchor="middle" font-size="9.5">versions:</text><text x="67.5" y="79" text-anchor="middle" font-size="9.5">LaTeX source + PDF</text><rect x="156" y="24" width="115" height="74" rx="6" fill="#f3f4f6" stroke="#555" stroke-width="1"/><text x="213.5" y="40" text-anchor="middle" font-weight="bold" font-size="11">Parse LaTeX</text><text x="213.5" y="53" text-anchor="middle" font-size="9.5">sections, paragraphs,</text><text x="213.5" y="66" text-anchor="middle" font-size="9.5">tables, figures</text><text x="213.5" y="79" text-anchor="middle" font-size="9.5">(no bibliography)</text><rect x="302" y="24" width="115" height="74" rx="6" fill="#f3f4f6" stroke="#555" stroke-width="1"/><text x="359.5" y="40" text-anchor="middle" font-weight="bold" font-size="11">Validate</text><text x="359.5" y="53" text-anchor="middle" font-size="9.5">parsed text</text><text x="359.5" y="66" text-anchor="middle" font-size="9.5">checked against PDF,</text><text x="359.5" y="79" text-anchor="middle" font-size="9.5">page numbers</text><rect x="448" y="24" width="115" height="74" rx="6" fill="#f3f4f6" stroke="#555" stroke-width="1"/><text x="505.5" y="40" text-anchor="middle" font-weight="bold" font-size="11">Chunk</text><text x="505.5" y="53" text-anchor="middle" font-size="9.5">256 and 512 tokens</text><text x="505.5" y="66" text-anchor="middle" font-size="9.5">455 / 255 pieces</text><text x="505.5" y="79" text-anchor="middle" font-size="9.5">table rows kept</text><rect x="594" y="24" width="115" height="74" rx="6" fill="#f3f4f6" stroke="#555" stroke-width="1"/><text x="651.5" y="40" text-anchor="middle" font-weight="bold" font-size="11">Index</text><text x="651.5" y="53" text-anchor="middle" font-size="9.5">BM25 + BGE-small</text><text x="651.5" y="66" text-anchor="middle" font-size="9.5">FAISS vectors</text><text x="651.5" y="79" text-anchor="middle" font-size="9.5">one per piece size</text><rect x="10" y="148" width="115" height="74" rx="6" fill="#f3f4f6" stroke="#555" stroke-width="1"/><text x="67.5" y="164" text-anchor="middle" font-weight="bold" font-size="11">Load generator</text><text x="67.5" y="177" text-anchor="middle" font-size="9.5">Go, open loop</text><text x="67.5" y="190" text-anchor="middle" font-size="9.5">Poisson arrivals</text><text x="67.5" y="203" text-anchor="middle" font-size="9.5">9 scenarios</text><rect x="156" y="148" width="115" height="74" rx="6" fill="#f3f4f6" stroke="#555" stroke-width="1"/><text x="213.5" y="164" text-anchor="middle" font-weight="bold" font-size="11">The Relay app</text><text x="213.5" y="177" text-anchor="middle" font-size="9.5">FastAPI, 1 worker</text><text x="213.5" y="190" text-anchor="middle" font-size="9.5">pipeline above</text><rect x="302" y="148" width="115" height="74" rx="6" fill="#dbe8fb" stroke="#555" stroke-width="1"/><text x="359.5" y="164" text-anchor="middle" font-weight="bold" font-size="11">Model</text><text x="359.5" y="177" text-anchor="middle" font-size="9.5">Go stand-in</text><text x="359.5" y="190" text-anchor="middle" font-size="9.5">(or Groq: 7 real</text><text x="359.5" y="203" text-anchor="middle" font-size="9.5">calls in total)</text><rect x="448" y="148" width="115" height="74" rx="6" fill="#f3f4f6" stroke="#555" stroke-width="1"/><text x="505.5" y="164" text-anchor="middle" font-weight="bold" font-size="11">results/raw/</text><text x="505.5" y="177" text-anchor="middle" font-size="9.5">append-only logs:</text><text x="505.5" y="190" text-anchor="middle" font-size="9.5">client, server</text><text x="505.5" y="203" text-anchor="middle" font-size="9.5">traces, provider</text><rect x="594" y="148" width="115" height="74" rx="6" fill="#d9f2df" stroke="#555" stroke-width="1"/><text x="651.5" y="164" text-anchor="middle" font-weight="bold" font-size="11">Report</text><text x="651.5" y="177" text-anchor="middle" font-size="9.5">results.csv →</text><text x="651.5" y="190" text-anchor="middle" font-size="9.5">tables.md →</text><text x="651.5" y="203" text-anchor="middle" font-size="9.5">write-up + PDF</text><line x1="125" y1="61" x2="156" y2="61" stroke="#333" stroke-width="1.2" marker-end="url(#a)"/><line x1="271" y1="61" x2="302" y2="61" stroke="#333" stroke-width="1.2" marker-end="url(#a)"/><line x1="417" y1="61" x2="448" y2="61" stroke="#333" stroke-width="1.2" marker-end="url(#a)"/><line x1="563" y1="61" x2="594" y2="61" stroke="#333" stroke-width="1.2" marker-end="url(#a)"/><line x1="125" y1="185" x2="156" y2="185" stroke="#333" stroke-width="1.2" marker-end="url(#a)"/><line x1="271" y1="185" x2="302" y2="185" stroke="#333" stroke-width="1.2" marker-end="url(#a)"/><line x1="417" y1="185" x2="448" y2="185" stroke="#333" stroke-width="1.2" marker-end="url(#a)"/><line x1="563" y1="185" x2="594" y2="185" stroke="#333" stroke-width="1.2" marker-end="url(#a)"/></svg></div>

- **Fetch** (`relay/retrieval/fetch_papers.py`): downloads each paper's LaTeX source and PDF at the exact
  version above. A URL without a version suffix is never used.
- **Parse** (`latex_parse.py`): turns the LaTeX into sections, paragraphs, tables and figure captions. The
  bibliography is kept out of retrieval.
- **Validate** (`pdf_validate.py`): checks the parsed text against the PDF and attaches page numbers.
- **Chunk** (`chunker.py`): packs whole paragraphs (falling back to whole sentences) into pieces of about 256 or
  512 tokens, never crossing a section. Each table row becomes its own piece under its caption and header, so a
  number is never separated from what it measures. IDs look like `MOSHI:4.2:c017` (text) and `AIAYN:T3:r01`
  (table row). Result: 455 pieces at 256 tokens, 255 at 512.
- **Index** (`index_build.py`): a BM25 keyword index and a dense index (`BAAI/bge-small-en-v1.5` embeddings in
  FAISS) for each piece size. The server only loads these files; it never rebuilds them.

## 3. One request, step by step

**Front door and admission** (`relay/main.py`). FastAPI with one worker. `POST /ask` takes a question and an
optional deadline (default 20 s) and replies as a server-sent event stream (`meta`, `content`, `done`). The
request joins a FIFO queue of at most 50. It is turned away at once if the queue is full or its estimated wait
already exceeds its deadline, and it is dropped with a "busy" reply if the deadline passes while it waits. This
fails fast instead of making people wait for an answer that cannot arrive in time.

**Retrieve** (`relay/retrieval/search.py`). The retrieval tool of the pipeline:

1. BM25 and the dense index each return their top 20 candidates.
2. Reciprocal rank fusion merges the two lists by rank, so keyword scores and cosine scores are never compared.
3. Optionally, a cross-encoder (`cross-encoder/ms-marco-MiniLM-L-6-v2`) re-ranks the merged list.
4. Whole passages, best first, fill a 1,600-token context.

When a question names two or more papers, each named paper gets its own candidates and reserved places in the context,
so a comparison question never loses one side. CPU-heavy steps run in a bounded thread pool, never on the event
loop. Three settings were compared at the same token budget:

| Setting | Piece size | Passages offered | Re-rank |
|---|---|---|---|
| A | 256 tokens | 8 | no |
| B (default) | 256 tokens | 8 | yes |
| C | 512 tokens | 4 | yes |

**Structural sufficiency** (`relay/verify/sufficiency.py`). If no usable passage reached the context, or the
question names a paper with no passage in it, the request is refused before any model call. The check is
structural on purpose: a rare word missing from the passages is not proof the papers lack the answer.

**Draft** (`relay/llm/`). One model call with the question and the numbered passages. The prompt asks for plain
sentences, each ending with the ID of the passage it relies on, e.g. `[MOSHI:4.2:c017]`. A sentence saying
something is missing must be scoped to the retrieved passages.

**Verify** (`relay/verify/`, policy v4, frozen). No model is called here: everything is deterministic code or a
local classifier. For every sentence:

1. **Citations:** every factual sentence cites at least one passage, and every cited ID exists and was actually
   retrieved for this request.
2. **Entailment:** a local natural-language-inference model (`MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli`)
   must judge that the cited passage entails the sentence with probability at least 0.9. Sentences with several
   clauses are split, and every clause must pass. LaTeX is normalised first (e.g. `\frac{a}{b}` to `a/b`), and
   table rows are read as sentences ("The configuration X has m on the dev set = v").
3. **Tables:** a number must belong to the row the sentence names, not a neighbouring row.
4. **Formulas:** a formula in the answer must appear in a cited passage, up to notation.
5. **Absence claims:** "the retrieved passages do not state X" is rejected when a retrieved sentence contains
   all of X's content words.

**Redraft, at most once.** If any sentence fails, the model gets the verifier's objections and one more try. If
the second draft also fails, the person gets an honest refusal with the reason. There is no third try.

**Answer.** Only a fully verified answer is sent. The design choice behind this: sending text as it is
generated would be faster to the first word, but it could show a sentence that later fails verification.

## 4. Staying inside the provider's limits

**Rate governor** (`relay/governor/`). Groq's free tier limits gpt-oss-120b to 30 requests and 8,000 tokens per
minute, and to 1,000 requests and 200,000 tokens per day (`config/limits.yaml`). The governor keeps a token
bucket for each of the four limits; a bucket refills continuously. Every hosted call follows three steps:

- **reserve:** prompt tokens plus 67 fixed preamble tokens plus an output estimate of 700 tokens, plus a
  10% margin;
- **call:** the call goes out only if all four buckets can cover it at once;
- **settle:** afterwards the reservation is corrected to the real usage, and the buckets are re-synced to the
  provider's rate-limit headers.

A 429 (rate-limit) reply freezes the bucket until the `retry-after` time. Waiting for the governor is bounded by
the request's deadline.

**Retries** (`relay/llm/hosted.py`):

- At most 2 per call, only on server errors (5xx), timeouts and per-minute 429s.
- The wait is the longer of `retry-after` and a random ("jittered") exponential backoff: 0.5 s base, capped at
  8 s.
- A shared budget: retries in any minute are at most 10% of first attempts in that minute.
- Never past the request's deadline, never after a per-day 429, and always through the governor.
- Call timeouts: connect 3 s, first byte 15 s, gap between chunks 8 s, whole call 45 s.

**When the daily quota is gone.** The first per-day 429 freezes the daily buckets. Every later request is
answered at once, before any model call: "The research assistant has used today's model capacity, so I can't
write a new answer right now. Full answers resume in about N minutes", where N comes from the 429's
`retry-after`. No guessed answer and no cheaper model.

**Tracing.** Every request writes one trace record (`results/raw/traces/`) with the time it spent queued,
retrieving, waiting for the governor, waiting for the provider, generating, retrying and verifying. All timings
use monotonic clocks.

## 5. The stand-in and the load generator

**Stand-in model service** (`standin/`, Go). An OpenAI-compatible server the app talks to with the same
client code it uses for Groq. What it imitates:

- **Replies:** the recorded answers and token counts of the real model, chosen by question ID.
- **Limits:** Groq's four limits and 429 replies.
- **Latency:** fitted to the 5 real answer calls. First byte is log-normal with a median of about 515 ms; then
  280 reasoning tokens/s and 530 answer tokens/s, streamed in 8-token chunks.
- **Faults, which are guesses:** 1% server errors, 2% slow responses (6 times slower), 1% 4-second stalls,
  0.5% timeouts.

An admin endpoint resets the service, scales its limits or exhausts the daily quota on cue. Its replies ignore
the prompt and the retrieved passages, so it measures capacity, never answer quality.

**Load generator** (`loadgen/`, Go). Open-loop: requests arrive on a Poisson schedule whether or not earlier ones
finished, as real users do. Each simulated user has a patience (log-normal, median 15 s) and gives up after it.
Scenarios (`config/scenarios/`, questions q05, q11, q16, fixed seed):

| Scenario | What it tests |
|---|---|
| smoke | 1-minute check that everything is wired up |
| steady | 1.61 requests/min for 10 min, about 70% of computed capacity |
| burst | 1.15/min with a 30-second burst at 46/min |
| overload | 4.6/min for 8 min, well past capacity |
| rpm_halved | provider's per-minute limits halved at the start |
| daily_exhausted | daily quota exhausted 2 minutes in |
| retrieval_A / B / C | same load under each retrieval setting |

## 6. From raw logs to the report

- `results/raw/` holds every log, append-only: the load generator's view of each request, the server traces,
  the stand-in's logs and the real Groq calls (stored without the Authorization header).
- `scripts/make_results_csv.py` joins them by request ID into `results/results.csv`, one row per request.
- `scripts/make_tables.py` turns that into `results/tables.md`.
- `scripts/build_writeup.py` inserts those tables into the report and this document, so no number is retyped.

**Selected results** (generated from results.csv; verifier v2):

{{TABLE2A}}

{{TABLE2B}}

{{TIERS}}

{{TABLE3}}

{{TABLE3B}}

**What they show:**

- The limit handling worked as designed. Requests that could not be served in time were turned away quickly
  rather than left hanging. With the daily quota gone, 17 users got the capacity reply at once, while the
  provider saw only 2 calls.
- Even "steady", at about 70% of computed capacity, was 28.6% rate-limited. With a reasoning model, the
  per-minute token budget is the real limit, not the request count.
- **Every completed stand-in request was a refusal.** The load tests therefore measure admission, limits,
  retries and latency, not answers served.

## 7. The verifier's history

All load tests and `results.csv` used verifier v2. Its weakness (refusing correct answers about formulas and
tables) led to two local revisions, each evaluated on hand-built claim sets with no Groq calls:

- **v2:** citation checks plus whole-sentence entailment at 0.9.
- **v3:**
  - LaTeX normalisation;
  - clause-by-clause checks;
  - table rows as sentences, with only the rows relevant to a sentence used as its evidence;
  - table value-owner and absence checks.
- **v4 (submitted, frozen):**
  - fixes a v3 bug that read `k_{t-1}` as `k_t - 1`;
  - adds the formula check and the lexical absence check.

| Claim set | v2 | v3 | v4 |
|---|---|---|---|
| 28 cases: accuracy | 82.1% | 89.3% | 89.3% |
| 28 cases: false acceptance | 26.7% | 13.3% | 6.7% |
| 28 cases: false rejection | 7.7% | 7.7% | 15.4% |
| 18 targeted cases: accuracy | n/a | 55.6% | 83.3% |
| 18 targeted cases: false acceptance | n/a | 70.0% | 20.0% |
| 8 held-out pairs: accuracy | 62.5% | 62.5% | 62.5% |

v4 accepts fewer wrong claims but rejects more correct ones. These sets are small and were drafted with the AI
assistant, so they are not evidence of general reliability. v4 was never load-tested. Reports:
`eval/reports/verifier_*`.

## 8. Honest status

- **Real model use:** 31 real Groq calls in total, no 429. 7 calibration and smoke calls: three questions ran end to
  end and were refused (q16 correctly; q05 and q11 by verifier false rejections). Then 24 calls for the
  15-question evaluation (setting B, verifier v4, 48,470 tokens).
- **Answer quality (preliminary, graded by the AI assistant, not yet confirmed by the owner):** 0 of 10 answerable
  questions answered correctly and cited, 9 falsely refused, 1 wrong answer delivered (q12: a false claim that
  retrieved table values were not stated); all 5 unanswerable questions correctly refused. All 8 drafts graded
  correct (of 19 verified drafts) were rejected: 【ID】 citations the parser ignores, a named subject the passage
  calls "we", and whole-sentence plus per-clause entailment. Details: eval/answer_quality_report.md.
- **Offline candidate v5 (not deployed):** P1 reads 【ID】 citations, P6 rejects absence claims a retrieved table
  contradicts. No labelled verifier case changes; replayed on the saved drafts it removes the 21 parsing failures
  and q12's wrong answer, but 5 answerable questions would need new live redrafts, so its effect on answers is
  unknown. v4 stays frozen.
- **Retrieval** was evaluated alone on the 30-question set (15 answerable). Setting B stays the default, though
  it was not shown to be better: one question moves coverage by 12 to 14 points.
- **Planned but not built** (`docs/DESIGN.md` §20.3 records every deviation): the LangGraph state machine, the
  gpt-oss-20b fallback (no quality gate for it), pressure levels, a response cache, single-flight, and stand-in
  replies that react to the prompt.
- **Least certain:** whether a small entailment model can judge claims about formulas and tables. It refused
  correct answers, and when it erred the other way (q12) a false sentence was sent.
