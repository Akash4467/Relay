"""Phase 9B diagnostic only: store the ChatGPT model-assisted audit (NOT an independent human review).

Copies the submitted files byte for byte to model_audit/as_submitted/, then writes model_audit/{development,heldout}.csv
with legend_or_definition_needed reduced to yes/no. Any free-text value is kept, verbatim, appended to `explanation`.
Rule: text starting with "yes" -> yes, "no" -> no; anything else must be listed in MANUAL (with a reason) or the script
stops. Validates with the Phase 9B review validator. Never writes under eval/phase9b_human_review/reviews/.
"""
from __future__ import annotations

import csv
import hashlib
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, "experiments/phase9b")
import evaluate_verifiers as ev  # noqa: E402
from build_review_package import REVIEWER_EVIDENCE, REVIEWER_INPUT  # noqa: E402

SRC = Path("eval/phase9b_human_review/reviewer/Akash")
OUT = Path("experiments/phase9b/model_audit")
AUDIT_ID = "chatgpt_model_assisted_audit_NOT_INDEPENDENT"
MANUAL = {"s13": ("no", "the claim only describes the cell as blank; the audit's own explanation says only the literal "
                        "blank-cell observation is supported, so no legend was needed to decide this claim")}


def yes_no(case_id: str, text: str) -> str:
    t = text.strip().lower()
    if t.startswith("yes"):
        return "yes"
    if t.startswith("no"):
        return "no"
    if case_id in MANUAL:
        return MANUAL[case_id][0]
    raise SystemExit(f"{case_id}: cannot map {text!r} to yes/no; add it to MANUAL with a reason")


def main() -> int:
    (OUT / "as_submitted").mkdir(parents=True, exist_ok=True)
    record = {"status": "DIAGNOSTIC MODEL AUDIT - NOT INDEPENDENT HUMAN REVIEW; never to be frozen or placed under "
                        "eval/phase9b_human_review/reviews/", "reviewer_id": AUDIT_ID, "source": SRC.as_posix(),
              "mapping_rule": "starts with 'yes' -> yes; starts with 'no' -> no; otherwise MANUAL",
              "manual": {k: {"value": v, "reason": r} for k, (v, r) in MANUAL.items()}, "files": {}, "changed": {}}
    ok = True
    for split in ("development", "heldout"):
        src, kept = SRC / f"{split}.csv", OUT / "as_submitted" / f"{split}.csv"
        if not kept.exists():
            shutil.copyfile(src, kept)
        assert kept.read_bytes() == src.read_bytes(), kept
        with kept.open(encoding="utf-8", newline="") as f:
            rows = list(csv.DictReader(f))
        assert list(rows[0]) == REVIEWER_EVIDENCE + REVIEWER_INPUT, "column names differ from the template"
        changed = {}
        for r in rows:
            original = r["legend_or_definition_needed"]
            r["legend_or_definition_needed"] = yes_no(r["case_id"], original)
            if original.strip() not in ("Yes", "No", "yes", "no"):
                r["explanation"] = f"{r['explanation'].rstrip()} [legend_or_definition_needed, original text: {original}]"
                changed[r["case_id"]] = {"from": original, "to": r["legend_or_definition_needed"]}
            elif original != r["legend_or_definition_needed"]:
                changed[r["case_id"]] = {"from": original, "to": r["legend_or_definition_needed"]}
        out = OUT / f"{split}.csv"
        with out.open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        errs = ev.validate_review(rows, split, AUDIT_ID, {})
        same_ids = [r["case_id"] for r in rows] == [r["case_id"] for r in csv.DictReader(
            open(Path("eval/phase9b_human_review/reviewer") / f"template_{split}.csv", encoding="utf-8"))]
        record["files"][split] = {"rows": len(rows), "as_submitted_sha256": hashlib.sha256(kept.read_bytes()).hexdigest(),
                                  "normalized_sha256": hashlib.sha256(out.read_bytes()).hexdigest(),
                                  "case_order_matches_template": same_ids, "validation_errors": errs}
        record["changed"][split] = changed
        ok &= not errs and same_ids
        print(split, len(rows), "rows;", len(changed), "legend values normalised;", len(errs), "validation errors")
    (OUT / "audit_record.json").write_text(json.dumps(record, indent=1, ensure_ascii=False), encoding="utf-8")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
