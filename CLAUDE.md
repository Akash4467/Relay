# The Relay — instructions for Claude Code

Read `docs/DESIGN.md` before changing anything. It is the single source of truth: architecture, schemas, algorithms, config values, experiments and the build order. If you need to deviate from it, say so first and record the change in DESIGN.md §20.3.

## What this project is
A citation-grounded Q&A pipeline over exactly three arXiv papers (AIAYN 1706.03762v7, MOSHI 2410.00037v1, HNET 2507.07955v2), built to behave well under bursts, rate limits and overload. It runs against Groq's free tier (real) and a Go stand-in (simulated). This is a graded assignment: every number in the write-up must be recomputable from `results/results.csv`.

- Python app (`relay/`): FastAPI single worker + LangGraph state machine (11 nodes), local hybrid retrieval (BM25 + BGE-small/FAISS + RRF + optional cross-encoder), admission layer, rate governor, deterministic verification.
- Go (`standin/`, `loadgen/`): OpenAI-compatible stand-in model service; open-loop Poisson load generator.
- Models: `openai/gpt-oss-120b` (primary), `openai/gpt-oss-20b` (fallback), both `reasoning_effort: low`. Do NOT use `llama-3.3-70b-versatile` or `llama-3.1-8b-instant`: they were shut down for free tier on 2026-08-16.

## Hard rules
1. **Never spend real Groq calls without asking me first.** The daily quota is shared by everything and is the scarcest resource (TPD ~200K, ≈ 57 answers/day). Tests, dev runs and load tests use the stand-in (`LLM_PROVIDER=standin`). Before any real run, state how many calls and tokens it will use.
2. **Never put the API key in code, logs, fixtures, results or commits.** It lives only in `.env` (git-ignored). Raw logs store requests without the Authorization header.
3. No extra accounts, keys or tricks to get around Groq limits.
4. Never hand-edit numbers in tables or the write-up. Tables come from `scripts/make_tables.py`, which reads only `results/results.csv`.
5. Never delete or overwrite anything in `results/raw/`. Raw results are a deliverable; new runs append new files.
6. Exact paper versions only. Never fetch a paper URL without the version suffix.
7. Only `draft` and `redraft` nodes call the LLM. No LLM calls for classification, retrieval checks or verification.
8. The production path verifies the complete answer before anything is sent (Option 1). Sentence-level streaming is an optional experiment only, behind a flag, default off.
9. Every hosted call goes through the rate governor (reserve → call → settle). No direct calls that bypass it, including retries.
10. Answer quality is measured only on real-service runs. The stand-in replies from fixtures matched by question_id; never random replay.
11. Don't add Redis, Celery, databases, Docker orchestration or extra services.

## Testing after every unit (mandatory)
A "unit" is one module or component (e.g. `latex_parse`, `chunker`, `rrf`, `buckets`, `model_decision`, a graph node, the stand-in's SSE writer). After finishing each unit, before starting the next one:

1. **Unit tests**: write tests for the unit you just built in `tests/unit/test_<module>.py` (Go: `<pkg>_test.go` next to the code). Cover the normal case, edge cases and failure paths. Pure logic is tested with no network, no models loaded where avoidable (use small fakes), and fixed random seeds.
2. **Integration tests**: write or extend tests in `tests/integration/` that run the new unit together with every unit already built (e.g. after `rerank`: question → BM25 + dense → RRF → rerank → evidence; after `verify`: retrieve → draft from a Mode B fixture → verify → refuse/return). Integration tests that need a model use the Go stand-in or a local fake server, **never real Groq**.
3. **Run the whole suite**: `uv run pytest -q tests/unit tests/integration` plus `go test ./...` in `standin/` and `loadgen/`. Everything must pass before moving on: not only the new tests, but all old ones too.
4. **Report**: tell me what was tested, the pass/fail count, and anything not covered or skipped (with the reason). Don't mark a unit done with failing or skipped tests unless I agree.
5. If a later change breaks an earlier unit's tests, fix it before continuing; never delete or weaken a test just to make it pass without telling me.

Tests that would hit real Groq do not exist in the suite; real calls happen only in `scripts/` and only after I approve them (Hard rule 1).

## Conventions
- Chunk IDs: `{PAPER}:{section_id}:c{NNN}` (tables `{PAPER}:T{n}:r{NN}`); papers `AIAYN`, `MOSHI`, `HNET`. Citations in answers: `[MOSHI:4.2:c017]`.
- Bump `prompt_version`, `corpus_version`, `verification_policy_version` or `generation_config_version` in `config/app.yaml` whenever the corresponding thing changes; they are part of the cache key and recorded with fixtures.
- All timing with monotonic nanoseconds (`time.perf_counter_ns()` / Go `time.Now()` monotonic); wall time only for `start_time`.
- Every node is wrapped by the tracing wrapper; new code paths that wait (queue, governor, backoff) must emit a span.
- `results.csv` required columns are fixed by the brief (DESIGN.md §18.3). Add extra columns only at the end.
- CPU-bound inference (embeddings, cross-encoder, NLI) runs in the bounded thread pool, never on the event loop.
- Config values live in `config/*.yaml`, not in code. Initial values are placeholders to be recalibrated from measurements.
- Python: type hints, pydantic models for config and API, `pytest` for tests. Go: standard library, `go test`.

## Commands (target; create them as the build reaches each phase)
```
# setup
uv sync                     # creates .venv (Python 3.12) and installs deps + dev group
# ingestion (offline)
uv run python -m relay.retrieval.fetch_papers
uv run python -m relay.retrieval.latex_parse
uv run python -m relay.retrieval.pdf_validate
uv run python -m relay.retrieval.chunker
uv run python -m relay.retrieval.index_build
# app
uv run uvicorn relay.main:app --port 8000 --workers 1
# stand-in
cd standin && go run ./cmd/standin -config ../config/standin.yaml -port 8001
# load test
cd loadgen && go run ./cmd/loadgen -scenario ../config/scenarios/burst.yaml -target http://localhost:8000/ask
# results
uv run python scripts/make_results_csv.py && uv run python scripts/make_tables.py
# tests (run after every unit)
uv run pytest -q tests/unit tests/integration && (cd standin && go test ./...) && (cd loadgen && go test ./...)
```

## Current phase
Follow the build order in DESIGN.md §20.1. Phase 2 (one request end to end, with a caught unsupported claim and a correct refusal) must work before any admission, governor-tuning or simulator work.

## Open decisions
See DESIGN.md §22. Ask me rather than choosing silently.
