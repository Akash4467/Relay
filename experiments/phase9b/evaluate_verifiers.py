"""Phase 9B: compare verifier variants on table-mark claims (offline; local NLI only, no API).

Variants (registered before any reviewed-label run; Phase 8 names in brackets):
  A   frozen v4 on the original c4 evidence                                    [A]
  B1  frozen v4, marks as [CHECKMARK]/[CROSSMARK]/[EMPTY_CELL]                  [B1]
  B2  frozen v4, marks as checked/crossed/empty                                 [B2]
  C   the guarded deterministic checker alone: resolves literal mark claims,
      otherwise ABSTAINS (never passes anything to v4)                         [C2]
  D   C, plus: an unresolved claim containing a mark word is REJECTED; only
      claims without a mark word fall back to v4 (A)                           [C2g]
The Phase 8 code is loaded unchanged from experiments/phase8/table_semantics.py (functions and constants only, via the
AST), so C and D are the original checker. v4 is called exactly as in Phase 8 (threshold 0.9 from config/app.yaml).

Scoring. Labels: supported / unsupported / ambiguous. Decisions: accept / reject / abstain.
  TA = supported accepted, FR = supported rejected, FA = unsupported accepted, TR = unsupported rejected.
  Abstentions are counted on their own and are neither acceptances nor rejections:
  FAR = FA / unsupported; FRR = FR / supported; coverage = TA / supported; abstention rate = abstain / all.
  FRR_incl_abstain = (FR + supported abstained) / supported (Phase 8 counted abstention as rejection).
  Ambiguous cases: abstain = safe abstention, reject = conservative, accept = unsafe acceptance.

Modes:
  --provisional   Phase 8 (unreviewed) labels, DEVELOPMENT set only, output tagged provisional. Infrastructure check:
                  with --check-phase8 FILE it asserts every per-case decision equals the stored Phase 8 run.
  (default)       frozen, independently reviewed labels from benchmark_manifest.json; refuses to run unless the
                  manifest is frozen and every benchmark file hash matches.
  disagreements   validate eval/phase9b_human_review/reviews/<reviewer_id>/{development,heldout}.csv and (re)write
                  disagreement_log.csv (reviewer vs reviewer, and reviewer vs the original Phase 8 label).
  freeze          build the frozen benchmark from the validated reviews and adjudications (see freeze()).

Two evidence levels: every reviewed case has a table-only outcome (scored as `label`) and a whole-paper outcome
(`label_paper`; SUPPORTED_ELSEWHERE counts as supported). A frozen run reports both.
"""
from __future__ import annotations

import ast
import copy
import csv
import hashlib
import json
import re
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, ".")

ROOT = Path(".")
HERE = ROOT / "experiments/phase9b"
P8_SCRIPT = ROOT / "experiments/phase8/table_semantics.py"
P8_CASES = {"development": ROOT / "experiments/phase8/table_semantics_cases.json",
            "heldout": ROOT / "experiments/phase8/table_semantics_heldout.json"}
REVIEW = ROOT / "eval/phase9b_human_review"
MANIFEST = HERE / "benchmark_manifest.json"
BENCH = HERE / "benchmark"
C4 = ROOT / "data/corpus/c4/processed"
P8_NAMES = {"REPR", "MARK", "NEG", "FRAME", "full_rows", "represent", "_stem", "words2", "deterministic_v2", "run_verifier"}
VARIANTS = ("A", "B1", "B2", "C", "D")
LABELS = ("supported", "unsupported", "ambiguous")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# ---- the original Phase 8 code, loaded without running its top-level experiment
def load_phase8(globals_: dict) -> dict:
    tree = ast.parse(P8_SCRIPT.read_text(encoding="utf-8"))
    keep = [n for n in tree.body if (isinstance(n, ast.FunctionDef) and n.name in P8_NAMES) or
            (isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id in P8_NAMES for t in n.targets))]
    found = {n.name if isinstance(n, ast.FunctionDef) else n.targets[0].id for n in keep}
    assert found == P8_NAMES, P8_NAMES - found
    ns = {"re": re, "copy": copy, **globals_}
    exec(compile(ast.Module(body=keep, type_ignores=[]), str(P8_SCRIPT), "exec"), ns)
    return ns


# ---- scoring (pure)
def decide(variant: str, r: dict) -> str:
    if variant == "C":
        return "abstain" if r["C"]["abstained"] else ("accept" if r["C"]["accepted"] else "reject")
    return "accept" if r[variant]["accepted"] else "reject"


def score(rows: list[dict], variant: str, key: str = "label") -> dict:
    c = Counter((r[key], decide(variant, r)) for r in rows)
    n = {lab: sum(v for (l, _), v in c.items() if l == lab) for lab in LABELS}
    ta, fr, sab = c["supported", "accept"], c["supported", "reject"], c["supported", "abstain"]
    fa, tr, uab = c["unsupported", "accept"], c["unsupported", "reject"], c["unsupported", "abstain"]
    rate = lambda a, b: round(a / b, 3) if b else None
    return {"n": len(rows), "n_supported": n["supported"], "n_unsupported": n["unsupported"],
            "n_ambiguous": n["ambiguous"], "TA": ta, "FR": fr, "FA": fa, "TR": tr,
            "abstain_supported": sab, "abstain_unsupported": uab,
            "FAR": rate(fa, n["unsupported"]), "FRR": rate(fr, n["supported"]),
            "FRR_incl_abstain": rate(fr + sab, n["supported"]), "coverage": rate(ta, n["supported"]),
            "abstention_rate": rate(sum(v for (_, d), v in c.items() if d == "abstain"), len(rows)),
            "ambiguous_safe_abstain": c["ambiguous", "abstain"], "ambiguous_reject": c["ambiguous", "reject"],
            "ambiguous_unsafe_accept": c["ambiguous", "accept"]}


def by_category(rows: list[dict], variant: str, key: str = "label") -> dict:
    return {cat: score([r for r in rows if r["category"] == cat], variant, key) for cat in sorted({r["category"] for r in rows})}


def c2g_outcomes(rows: list[dict], key: str = "label") -> dict:
    """Section 7's C2g lists. Ambiguous cases are listed apart: no decision on them is 'correct'."""
    out = {k: [] for k in ("resolved_correct", "resolved_wrong", "resolved_on_ambiguous", "rejected_safely_unresolved",
                           "rejected_wrongly_unresolved", "fallback_v4", "unresolved")}
    for r in rows:
        src, acc, lab = r["D"]["from"], r["D"]["accepted"], r[key]
        if src == "deterministic":
            out["resolved_on_ambiguous" if lab == "ambiguous" else
                "resolved_correct" if acc == (lab == "supported") else "resolved_wrong"].append(r["id"])
        else:
            out["unresolved"].append(r["id"])
            if src == "v4":
                out["fallback_v4"].append(r["id"])
            elif lab == "supported":
                out["rejected_wrongly_unresolved"].append(r["id"])
            else:
                out["rejected_safely_unresolved"].append(r["id"])
    return out


# ---- benchmark loading and the freeze gate
def provisional_cases(split: str) -> list[dict]:
    if split != "development":
        raise SystemExit("provisional mode runs the development set only; held-out waits for the frozen, reviewed benchmark")
    return json.loads(P8_CASES[split].read_text(encoding="utf-8"))["cases"]


def originals() -> dict[str, dict[str, dict]]:
    return {s: {c["id"]: c for c in json.loads(p.read_text(encoding="utf-8"))["cases"]} for s, p in P8_CASES.items()}


def frozen_cases(split: str, manifest_path: Path = MANIFEST) -> list[dict]:
    m = json.loads(manifest_path.read_text(encoding="utf-8"))
    if m.get("status") != "frozen":
        raise SystemExit(f"benchmark not frozen (status: {m.get('status')}); independent review is required first")
    entry = m["benchmark_files"][split]
    path = manifest_path.parent / entry["path"]
    if sha256(path) != entry["sha256"]:
        raise SystemExit(f"{path} does not match its frozen hash")
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("split") != split or {c["id"] for c in data["cases"]} != set(originals()[split]):
        raise SystemExit(f"{path} does not hold exactly the {split} cases")
    return data["cases"]


# ---- the two-level review protocol
TABLE_OUTCOMES = ("SUPPORTED", "UNSUPPORTED", "AMBIGUOUS")
PAPER_OUTCOMES = ("SUPPORTED", "SUPPORTED_ELSEWHERE", "UNSUPPORTED", "AMBIGUOUS")
LEVELS = {"table_only": "table_only_outcome", "whole_paper": "whole_paper_outcome"}
REQUIRED = ("table_only_outcome", "whole_paper_outcome", "supporting_evidence", "source_page_or_section",
            "legend_or_definition_needed", "confidence", "explanation", "reviewer_id", "review_date")
ALWAYS_EXCLUDED = ("assistant", "claude", "self", "developer", "case-author", "case_author",
                   # model-generated reviews are never independent review (the Phase 9B ChatGPT audit and any variant)
                   "chatgpt_model_assisted_audit_not_independent", "chatgpt", "model_assisted", "model-assisted",
                   "not_independent", "not-independent")
LOG_FIELDS = ["case_id", "set", "level", "kind", "assessments", "evidence", "working_outcome", "status",
              "adjudicated_outcome", "adjudicator_id", "adjudication_date", "adjudication_reason"]
SCORE_LABEL = {"SUPPORTED": "supported", "SUPPORTED_ELSEWHERE": "supported", "UNSUPPORTED": "unsupported",
               "AMBIGUOUS": "ambiguous"}


def excluded(reviewer_id: str, manifest: dict) -> bool:
    rid = reviewer_id.strip().lower()
    names = [n.lower() for n in (*ALWAYS_EXCLUDED, *manifest.get("excluded_reviewer_ids", []))]
    return any(n == rid or n in rid for n in names) if rid else True


def validate_review(rows: list[dict], split: str, reviewer_id: str, manifest: dict) -> list[str]:
    """Every problem with one reviewer's file for one split (empty list = complete and valid)."""
    errs, cases = [], originals()[split]
    ids = [r.get("case_id", "") for r in rows]
    if sorted(ids) != sorted(cases) or len(set(ids)) != len(ids):
        errs.append(f"{split}: the file must hold each {split} case exactly once (got {len(ids)} rows)")
    if excluded(reviewer_id, manifest):
        errs.append(f"reviewer '{reviewer_id}' is excluded: the case author cannot be the independent reviewer")
    # the IDs written in the rows are checked too, so renaming the folder cannot launder an excluded reviewer
    for rid in sorted({(r.get("reviewer_id") or "").strip() for r in rows} - {reviewer_id, ""}):
        if excluded(rid, manifest):
            errs.append(f"reviewer '{rid}' in the file is excluded (folder name '{reviewer_id}' does not override it)")
    for r in rows:
        cid = r.get("case_id", "?")
        if cid in cases and r.get("claim") != cases[cid]["claim"]:
            errs.append(f"{cid}: claim text was edited")
        missing = [k for k in REQUIRED if not (r.get(k) or "").strip()]
        if missing:
            errs.append(f"{cid}: incomplete review, empty {missing}")
            continue
        t, p = r["table_only_outcome"].strip(), r["whole_paper_outcome"].strip()
        if t not in TABLE_OUTCOMES:
            errs.append(f"{cid}: table_only_outcome must be one of {TABLE_OUTCOMES}")
        if p not in PAPER_OUTCOMES:
            errs.append(f"{cid}: whole_paper_outcome must be one of {PAPER_OUTCOMES}")
        if p == "SUPPORTED" and t != "SUPPORTED":
            errs.append(f"{cid}: whole-paper SUPPORTED means the table supports it; use SUPPORTED_ELSEWHERE")
        if p == "SUPPORTED_ELSEWHERE" and t == "SUPPORTED":
            errs.append(f"{cid}: SUPPORTED_ELSEWHERE contradicts a table-only SUPPORTED")
        if r["legend_or_definition_needed"].strip().lower() not in ("yes", "no"):
            errs.append(f"{cid}: legend_or_definition_needed must be yes or no")
        if r["confidence"].strip().lower() not in ("high", "medium", "low"):
            errs.append(f"{cid}: confidence must be high, medium or low")
        if r["reviewer_id"].strip() != reviewer_id:
            errs.append(f"{cid}: reviewer_id '{r['reviewer_id']}' differs from the folder name '{reviewer_id}'")
        if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", r["review_date"].strip()):
            errs.append(f"{cid}: review_date must be YYYY-MM-DD")
    return errs


def load_reviews(review_dir: Path, manifest: dict) -> dict[str, dict[str, list[dict]]]:
    """reviews/<reviewer_id>/{development,heldout}.csv, all validated; refuses on any problem."""
    reviews, errs = {}, []
    for d in sorted(p for p in (review_dir / "reviews").glob("*") if p.is_dir()):
        reviews[d.name] = {}
        for split in ("development", "heldout"):
            f = d / f"{split}.csv"
            if not f.exists():
                errs.append(f"{d.name}: {f.name} missing")
                continue
            with f.open(encoding="utf-8", newline="") as fh:
                rows = list(csv.DictReader(fh))
            errs += [f"{d.name}: {e}" for e in validate_review(rows, split, d.name, manifest)]
            reviews[d.name][split] = rows
    if len(reviews) < manifest.get("min_reviewers", 1):
        errs.append(f"{len(reviews)} reviewer(s); at least {manifest.get('min_reviewers', 1)} required")
    if errs:
        raise SystemExit("review not usable:\n  " + "\n  ".join(errs))
    return reviews


def compute_disagreements(reviews: dict[str, dict[str, list[dict]]]) -> list[dict]:
    """One row per (case, level, kind): reviewers who differ, and a table-only outcome that differs from the original
    Phase 8 label. Both sides' assessments and evidence are kept; the working outcome is AMBIGUOUS until adjudicated."""
    orig, out = originals(), []
    for split in ("development", "heldout"):
        by_case = {cid: {rid: next(r for r in reviews[rid][split] if r["case_id"] == cid) for rid in reviews}
                   for cid in orig[split]}
        for cid, per in by_case.items():
            for level, col in LEVELS.items():
                outcomes = {rid: r[col].strip() for rid, r in per.items()}
                evidence = "; ".join(f"{rid}: {r['supporting_evidence'].strip()} [{r['source_page_or_section'].strip()}]"
                                     f" ({r['explanation'].strip()})" for rid, r in per.items())
                base = {"case_id": cid, "set": split, "level": level, "evidence": evidence,
                        "working_outcome": "AMBIGUOUS", "status": "unresolved", "adjudicated_outcome": "",
                        "adjudicator_id": "", "adjudication_date": "", "adjudication_reason": ""}
                if len(set(outcomes.values())) > 1:
                    out.append(base | {"kind": "between_reviewers",
                                       "assessments": "; ".join(f"{k}={v}" for k, v in outcomes.items())})
                original = orig[split][cid]["label"].upper()
                if level == "table_only" and any(v != original for v in outcomes.values()):
                    out.append(base | {"kind": "reviewer_vs_original", "assessments": "; ".join(
                        [f"original(phase8, assistant)={original}"] + [f"{k}={v}" for k, v in outcomes.items()])})
    return out


def write_disagreement_log(review_dir: Path = REVIEW, manifest_path: Path = MANIFEST) -> list[dict]:
    """(Re)write disagreement_log.csv. Adjudications already entered are kept; if the reviews behind an adjudicated row
    changed, it refuses rather than silently dropping or reusing that adjudication."""
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    rows = compute_disagreements(load_reviews(review_dir, manifest))
    path = review_dir / "disagreement_log.csv"
    with path.open(encoding="utf-8", newline="") as f:
        old = {(r["case_id"], r["level"], r["kind"]): r for r in csv.DictReader(f) if r.get("level")}
    new_keys = {(r["case_id"], r["level"], r["kind"]) for r in rows}
    for key, o in old.items():
        if o.get("adjudicated_outcome") and key not in new_keys:
            raise SystemExit(f"{key}: was adjudicated but the disagreement no longer exists; reviews changed")
    for r in rows:
        o = old.get((r["case_id"], r["level"], r["kind"]))
        if o and o.get("adjudicated_outcome"):
            if o["assessments"] != r["assessments"]:
                raise SystemExit(f"{r['case_id']} {r['level']}: reviews changed after adjudication")
            r |= {k: o[k] for k in ("working_outcome", "status", "adjudicated_outcome", "adjudicator_id",
                                    "adjudication_date", "adjudication_reason")}
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=LOG_FIELDS)
        w.writeheader()
        w.writerows(rows)
    return rows


MODEL_AUDIT = HERE / "model_audit"
ASSESSMENT = [k for k in REQUIRED if k not in ("reviewer_id", "review_date")]


def is_model_audit_copy(rows: list[dict], split: str, audit_dir: Path = MODEL_AUDIT) -> bool:
    """True when every assessment field equals the stored model audit (an audit copied in under another name)."""
    path = audit_dir / f"{split}.csv"
    if not path.exists():
        return False
    with path.open(encoding="utf-8", newline="") as f:
        audit = {r["case_id"]: [r[k].strip() for k in ASSESSMENT] for r in csv.DictReader(f)}
    return all(audit.get(r["case_id"]) == [(r.get(k) or "").strip() for k in ASSESSMENT] for r in rows)


def independent_reviewers(review_dir: Path, reviews: dict, manifest: dict,
                          audit_dir: Path = MODEL_AUDIT) -> tuple[list[str], dict[str, str]]:
    """Reviewers that count as genuinely independent human review. A reviewer qualifies only if the manifest's
    review_provenance[<id>] says independent: true, human: true, model_assisted: false, owner: false and
    authored_cases_or_labels: false, records the sha256 of both submitted files and those still match, the reviewer's
    own signed independence declaration is on file (signed_declaration_on_file: true), the id is not
    excluded, and the submission is not a copy of the model audit. Missing fields never default to qualifying."""
    ok, why_not = [], {}
    for rid, splits in reviews.items():
        p = manifest.get("review_provenance", {}).get(rid, {})
        flags = {"independent": True, "human": True, "model_assisted": False, "owner": False,
                 "authored_cases_or_labels": False}
        bad = [k for k, want in flags.items() if p.get(k) is not want]
        files = p.get("files", {})
        stale = [s for s in ("development", "heldout") if files.get(s, {}).get("sha256") !=
                 sha256(review_dir / "reviews" / rid / f"{s}.csv")]
        if excluded(rid, manifest):
            why_not[rid] = "excluded reviewer id"
        elif not p:
            why_not[rid] = "no review_provenance entry in the manifest"
        elif bad:
            why_not[rid] = f"provenance does not establish independent human review ({', '.join(bad)})"
        elif stale:
            why_not[rid] = f"provenance file hashes missing or do not match the submitted files ({', '.join(stale)})"
        elif p.get("signed_declaration_on_file") is not True:
            why_not[rid] = "the reviewer's own signed independence declaration is not on file"
        elif any(is_model_audit_copy(splits[s], s, audit_dir) for s in splits):
            why_not[rid] = "submission is a copy of the model-assisted audit"
        else:
            ok.append(rid)
    return ok, why_not


def freeze(review_dir: Path = REVIEW, bench_dir: Path = BENCH, manifest_path: Path = MANIFEST, version: str = "v1") -> dict:
    """Final two-level outcomes -> benchmark/{split}_{version}.json, hashed, manifest frozen.
    Refuses unless every reviewer file is complete and valid, no reviewer is excluded, the disagreement log matches the
    reviews exactly, and every disagreement is adjudicated (by someone not excluded). No majority vote: a case whose
    reviewers disagree takes its adjudicated outcome. At least one reviewer must qualify as genuinely independent
    human review (independent_reviewers()). Nothing is written until everything is validated."""
    m = json.loads(manifest_path.read_text(encoding="utf-8"))
    if any(v.get("version") == version for v in [m, *m.get("previous_versions", [])] if v.get("status") == "frozen"):
        raise SystemExit(f"benchmark {version} is already frozen; create a new version instead")
    reviews = load_reviews(review_dir, m)
    independent, why_not = independent_reviewers(review_dir, reviews, m)
    if len(independent) < max(1, int(m.get("min_independent_reviewers", 1))):
        raise SystemExit("cannot freeze: at least one genuinely independent human reviewer is required.\n  " +
                         "\n  ".join(f"{rid}: {why}" for rid, why in why_not.items()))
    expected = compute_disagreements(reviews)
    with (review_dir / "disagreement_log.csv").open(encoding="utf-8", newline="") as f:
        log = {(r["case_id"], r["level"], r["kind"]): r for r in csv.DictReader(f) if r.get("level")}
    errs = []
    for e in expected:
        key = (e["case_id"], e["level"], e["kind"])
        a = log.get(key)
        if a is None or a["assessments"] != e["assessments"]:
            errs.append(f"{key}: missing or stale in disagreement_log.csv (run the 'disagreements' command)")
            continue
        allowed = TABLE_OUTCOMES if e["level"] == "table_only" else PAPER_OUTCOMES
        if a["adjudicated_outcome"] not in allowed or not a["adjudicator_id"].strip() or not a["adjudication_date"].strip():
            errs.append(f"{key}: not adjudicated (outcome in {allowed}, adjudicator and date required)")
        elif excluded(a["adjudicator_id"], m):
            errs.append(f"{key}: adjudicator '{a['adjudicator_id']}' is excluded")
    if errs:
        raise SystemExit("cannot freeze:\n  " + "\n  ".join(errs))
    built = {}
    for split, cases in originals().items():
        out = []
        for cid, case in cases.items():
            per = {rid: next(r for r in reviews[rid][split] if r["case_id"] == cid) for rid in reviews}
            final = {}
            for level, col in LEVELS.items():
                adj = {log[k]["adjudicated_outcome"] for k in log if k[0] == cid and k[1] == level
                       and any(e["case_id"] == cid and e["level"] == level and e["kind"] == k[2] for e in expected)}
                if len(adj) > 1:
                    raise SystemExit(f"{cid} {level}: conflicting adjudicated outcomes {sorted(adj)}")
                final[level] = adj.pop() if adj else next(iter(per.values()))[col].strip()
            if final["whole_paper"] == "SUPPORTED" and final["table_only"] != "SUPPORTED":
                raise SystemExit(f"{cid}: final whole-paper SUPPORTED needs table-only SUPPORTED")
            out.append(dict(case) | {
                "label_phase8": case["label"], "table_only_outcome": final["table_only"],
                "whole_paper_outcome": final["whole_paper"], "label": SCORE_LABEL[final["table_only"]],
                "label_paper": SCORE_LABEL[final["whole_paper"]],
                "reviews": {rid: {k: r[k] for k in REQUIRED} for rid, r in per.items()}})
        built[split] = out
    paths = {s: bench_dir / f"{s}_{version}.json" for s in built}
    for p in paths.values():
        if p.exists():
            raise SystemExit(f"{p} exists; never overwrite a benchmark version")
    bench_dir.mkdir(parents=True, exist_ok=True)
    files = {}
    for split, cases in built.items():
        paths[split].write_text(json.dumps({"version": version, "split": split, "cases": cases}, indent=1,
                                           ensure_ascii=False), encoding="utf-8")
        files[split] = {"path": paths[split].relative_to(manifest_path.parent).as_posix(),
                        "sha256": sha256(paths[split]), "n": len(cases),
                        "table_only": dict(Counter(c["table_only_outcome"] for c in cases)),
                        "whole_paper": dict(Counter(c["whole_paper_outcome"] for c in cases))}
    if m.get("status") == "frozen":
        m.setdefault("previous_versions", []).append(
            {k: m[k] for k in ("status", "version", "frozen_at", "benchmark_files", "review_files_at_freeze")})
    m |= {"status": "frozen", "version": version, "frozen_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
          "benchmark_files": files, "reviewers": sorted(reviews), "independent_reviewers": sorted(independent),
          "review_files_at_freeze": {p.relative_to(review_dir).as_posix(): sha256(p)
                                     for p in sorted([*(review_dir / "reviews").rglob("*.csv"),
                                                      review_dir / "disagreement_log.csv"])}}
    manifest_path.write_text(json.dumps(m, indent=1), encoding="utf-8")
    return m


# ---- running the variants
def run(cases: list[dict]) -> list[dict]:
    from relay.config import app_config
    from relay.eval.verifier_v2_check import CachedNLI
    from relay.retrieval import latex_parse as lp
    from relay.retrieval.chunker import structured_row
    from relay.verify import nli
    from relay.verify.verifier import verify as verify_v4

    sys.path.insert(0, str(HERE))
    from build_review_package import cited_chunk

    cfg = app_config()["verification"]
    chunks = [json.loads(l) for l in open(C4 / "chunks_256.jsonl", encoding="utf-8")]
    g = {"re": re, "copy": copy, "json": json, "lp": lp, "structured_row": structured_row, "CHUNKS": chunks,
         "IDS": {c["chunk_id"] for c in chunks}, "T": cfg["entail_min"],
         "TABLES": {(t["paper"], t["table_id"]): t for t in map(json.loads, open(C4 / "tables.jsonl", encoding="utf-8"))},
         "NLI": CachedNLI(nli.load_nli(cfg["nli_model"]))}
    p8 = load_phase8(g)
    out = []
    for case in cases:
        chunk = cited_chunk(case, chunks)
        b1, b2 = p8["represent"](chunk, p8["REPR"]["B1"]), p8["represent"](chunk, p8["REPR"]["B2"])
        det, why = p8["deterministic_v2"](case, chunk)
        r = {"id": case["id"], "label": case["label"], "label_paper": case.get("label_paper"), "category": case["category"], "claim": case["claim"],
             "chunk": chunk["chunk_id"], "A": p8["run_verifier"](verify_v4, case, chunk),
             "B1": p8["run_verifier"](verify_v4, case, b1), "B2": p8["run_verifier"](verify_v4, case, b2),
             "C": {"accepted": bool(det), "abstained": det is None, "why": why}}
        mark_claim = any(rx.search(case["claim"]) for rx in p8["MARK"].values())
        r["D"] = {"accepted": det if det is not None else (False if mark_claim else r["A"]["accepted"]),
                  "from": "deterministic" if det is not None else ("rejected: unresolved mark claim" if mark_claim else "v4"),
                  "why": why}
        out.append(r)
    return out


def check_phase8(rows: list[dict], phase8_file: Path) -> dict:
    """Every per-case decision must equal the stored Phase 8 run (A, B1, B2, C2 -> C, C2g -> D)."""
    old = {r["id"]: r for r in json.loads(phase8_file.read_text(encoding="utf-8"))["results"]}
    pairs = {"A": "A", "B1": "B1", "B2": "B2", "C": "C2", "D": "C2g"}
    diffs = [(r["id"], v) for r in rows for v, o in pairs.items()
             if r[v]["accepted"] != old[r["id"]][o]["accepted"]
             or (v == "C" and r["C"]["abstained"] != old[r["id"]]["C2"]["abstained"])]
    return {"phase8_file": phase8_file.as_posix(), "cases_compared": len(rows), "differences": diffs,
            "identical": not diffs and set(old) == {r["id"] for r in rows}}


def main(argv: list[str]) -> int:
    if argv[:1] == ["freeze"]:
        print(json.dumps(freeze(), indent=1))
        return 0
    if argv[:1] == ["disagreements"]:
        rows = write_disagreement_log()
        print(f"{len(rows)} disagreement row(s) in disagreement_log.csv;",
              sum(not r["adjudicated_outcome"] for r in rows), "awaiting adjudication")
        return 0
    provisional = "--provisional" in argv
    split = argv[argv.index("--split") + 1] if "--split" in argv else "development"
    cases = provisional_cases(split) if provisional else frozen_cases(split)
    rows = run(cases)
    result = {"mode": "provisional (Phase 8 labels, unreviewed; not a performance claim)" if provisional else "frozen",
              "split": split, "variants": {v: score(rows, v) for v in VARIANTS},
              "by_category": {v: by_category(rows, v) for v in VARIANTS}, "c2g": c2g_outcomes(rows),
              "phase8_script_sha256": sha256(P8_SCRIPT), "results": rows}
    if not provisional:  # both evidence levels, and the exact benchmark the numbers come from
        m = json.loads(MANIFEST.read_text(encoding="utf-8"))
        result["benchmark"] = {"version": m["version"], "file": m["benchmark_files"][split]}
        result["whole_paper"] = {"variants": {v: score(rows, v, "label_paper") for v in VARIANTS},
                                 "by_category": {v: by_category(rows, v, "label_paper") for v in VARIANTS},
                                 "c2g": c2g_outcomes(rows, "label_paper")}
    if "--check-phase8" in argv:
        result["reproduction"] = check_phase8(rows, Path(argv[argv.index("--check-phase8") + 1]))
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ")
    out = HERE / "out" / f"verifiers_{'provisional_' if provisional else ''}{split}_{stamp}.json"
    with out.open("x", encoding="utf-8") as f:
        json.dump(result, f, indent=1, ensure_ascii=False)
    for v in VARIANTS:
        print(v, {k: result["variants"][v][k] for k in ("TA", "FR", "FA", "TR", "abstain_supported",
                                                          "abstain_unsupported", "FAR", "FRR", "FRR_incl_abstain")})
    if "reproduction" in result:
        print("reproduction identical:", result["reproduction"]["identical"], result["reproduction"]["differences"])
    print(out.as_posix())
    return 0 if result.get("reproduction", {}).get("identical", True) else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
