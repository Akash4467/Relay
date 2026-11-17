# Errata: verifier label in live-run metadata

**Affected:** runs whose verifier was a question-aware candidate (one exposing `verify_for`, not `verify`).

**What is wrong:** `assessment/improvement_loop/tools/live_eval.py` labelled these runs `verifier: "v4.2"` in `run.json`, and `verifier_version: "v4.2"` in each record. The **verifier actually used** is the module named in `run.json` → `verifier_module`: the harness called that module's `verify_for(question)`.

**Evidence:** for example, E25_V48 contains 5 accepted sentences scored below the 0.9 NLI threshold (q07, q08, q11, q14, q15). Only the candidate's exact-support rule can accept those; v4.2 cannot.

**Not affected:**
- outcomes, grades, tokens, latency, and the raw requests and responses;
- the answers were verified by the candidate module.

**Affected:**
- the label only;
- the claim-level statistic in the `*_metrics.json` of these runs (`claim_sentences_accepted`). `metrics.py` replays each sentence with the labelled verifier, so this statistic reflects v4.2, not the candidate.

**Fix:** the label is now taken from the module's `VERSION` (2026-11-14). Raw records are not edited (CLAUDE.md rule 5 spirit: raw results are never overwritten).

| Run | Verifier actually used |
|---|---|
| `E25_R3b_p7_v47_c6/` | relay.candidate_v4_7.verifier |
| `E25_V48/` | relay.candidate_v4_8.verifier |
| `G4_p7_v47_c6/` | relay.candidate_v4_7.verifier |
| `R2_p7_v47_c6/` | relay.candidate_v4_7.verifier |
| `R3b_p7_v47_c6/` | relay.candidate_v4_7.verifier |
| `R3b_p7_v47_c6_resume/` | relay.candidate_v4_7.verifier |
| `V48_R3b_p7_c6/` | relay.candidate_v4_8.verifier |
