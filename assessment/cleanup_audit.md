# Repository cleanup audit (2026-11-14)

Conservative, behaviour-preserving cleanup. Production is unchanged (c3 + p4 + v4.2). The candidate (c6 + R3b + p7 + v4.8c) and every module it depends on are kept.

## Inventory

| | Files |
|---|---|
| Tracked by git | 192 |
| Untracked, not ignored | 1,451 |
| Ignored (`.venv`, `data/raw`, `data/index`, caches, `.env`) | 32,556 |

- The untracked files include almost all work since the last commit: `relay/candidate_*`, `assessment/`, `experiments/phase8`–`12`, `submission/`, and the new tests.
- Git cannot restore an untracked file, so **no untracked work was deleted**. The one exception is a byte-identical duplicate whose content is kept elsewhere (see below).

## Method

The audit was read-only.

1. **Every top-level function and class:** each of the 1,319 definitions in 233 Python files was checked for references in all other code, tests, docs and config (340 text files outside `.venv` and raw data). **None is unreferenced.**
2. **Every Python module:** 18 module names appear nowhere else. All 18 are standalone scripts run as `python path/to/script.py` (experiments, evaluation tools), and each produced an evidence artifact. They are kept as evidence tooling.
3. **Exact duplicates by SHA-256:** all are intentional.
   - Identical raw request bodies across live runs (each run directory is self-contained).
   - Phase 12 records copied into the improvement-loop baseline.
   - `archive/verifier_v3_snapshot` copies of modules that did not change.
   - Corpus files shared across corpus versions.
   - **Exception:** the root transcript (below).
4. **Leftover files:** no zero-byte, temporary, backup, debug or stray render files were found.
5. **Dependencies:** every entry in `pyproject.toml` was checked for direct imports, dynamic loading (`import_module`), CLI use and documentation.

## Changes made

| Item | Kind | Reason |
|---|---|---|
| 35 `__pycache__` / `.pytest_cache` directories outside `.venv` | Generated cache (ignored) | Regenerated automatically; not part of the project |
| `2026-11-13-124510-local-command-caveatthe-command-below-was-run-d.txt` (repository root) | Duplicate | Byte-identical to `submission/final/conversation_history/claude_code_conversation_export_2026-11-13.txt` (moved on 2026-11-14 to `docs/conversation_history/`) (sha256 `84723782a90cc1d3…`). The only reference is a recorded `git status` snapshot inside `eval/phase7/conditions.json`, which is evidence, not a dependency |
| `langgraph` dependency | Unused | Never imported, directly or dynamically, anywhere. CLAUDE.md: "the LangGraph state machine … not built". Removed with `uv remove`, which updates `pyproject.toml` and `uv.lock` consistently |
| `orjson` dependency | Unused | Never imported or referenced anywhere. Removed with `uv remove` |

## Kept on purpose

**Active and required support:**
- `relay/` production modules, `config/`, `standin/`, `loadgen/`, `scripts/`, `tests/`, `fixtures/`, and `data/corpus` plus the processed corpus.
- Every `relay/candidate_*` module on the candidate's chain:
  - prompts: p7 → p6 → p5r;
  - verifier: v4.8c → v4.7c → v4.6c → v4.5c;
  - retrieval: R3b → R2.

**Historical evidence (category C), kept in place:**
- `results/` (including `results/raw/`), `eval/`, `experiments/`, `assessment/`, `archive/verifier_v3_snapshot`, `submission/`, and the write-ups.
- The rejected candidates `relay/candidate_v4_3` and `relay/candidate_v4_4` (iterations 001 and 003). Their iteration records, replay tools and tests name them by module path, so moving them would break reproducibility.
- `relay/candidate_v5` (the Phase 9 candidate, documented in `eval/verifier_v5_offline_comparison_corrected.md` and `results/candidate_v5/`).

**Dependencies kept although not imported directly:**
- `uvicorn`: the documented server command, `uv run uvicorn relay.main:app`.
- `pylatexenc`: imported by `relay/retrieval/latex_parse.py`.

## Uncertain, kept

- **Production modules contain no unreferenced top-level code.** Checking methods inside classes for dead branches would mean editing frozen production modules (verifier v4.2, retrieval), which this cleanup does not do.
- `writeup/relay_writeup.md` is out of step with its own template: the template was edited after the last build. That is the owner's call, so it is left as is.
- `markdown-it-py` is imported by `scripts/render_writeup.py` but not declared in `pyproject.toml`; it is present only as a transitive dependency. It is left as is, because adding dependencies is outside this cleanup.
