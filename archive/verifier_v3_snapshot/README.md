# Verifier v3 snapshot (archival, not imported)

Copy of `relay/verify/*.py` as it stood under `verification_policy_version: v3`, taken on 2026-11-13 just before the
v4 changes. It exists so the v3 results in `eval/reports/` can be traced to code. Nothing imports it, and the tests
do not collect it.

- Original location (temporary, outside the repository, may disappear):
  `C:\Users\akash\AppData\Local\Temp\claude\a25d8a997f81f9f8\454fedfe-fb69-498c-8073-c84553604e09\scratchpad\verify_v3_snapshot\`
- Copied into the repository on 2026-11-13 with `cp -n` (nothing overwritten). Hashes: `SHA256SUMS`.
- Reports produced with this code: `eval/reports/verifier_v3_eval_v3.json`, `eval/reports/verifier_v2_check_policy_v3.json`,
  `eval/reports/verifier_v3b_targeted_v3.json`.
- Not archived: the v2 code. It was edited in place by the v3 pass and was never committed, so v2 exists only as its
  reports (`eval/reports/verifier_v2_check.json`, `eval/reports/verifier_v3_eval_v2_baseline.json`) and as the
  behaviour recorded in `results/` (all load runs used v2).
- Limitation: the snapshot covers `relay/verify/` only. To run it, it would have to replace `relay/verify/` in a copy of
  the tree as it stood at v3 (`relay/eval/verifier_v3_eval.py` and the config were identical apart from the policy
  version string).
