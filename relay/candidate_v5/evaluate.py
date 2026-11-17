"""Offline comparison of frozen v4 with candidate v5 (P1 only; P1 + P6). No LLM or API calls.

Inputs (read only): the labelled verifier sets, the saved real drafts in results/raw/answer_quality/, the assistant
grades in eval/answer_quality_sentences.csv, corpus c3 chunks. Outputs (new files only):
results/candidate_v5/comparison_{UTC}.json and eval/verifier_v5_offline_comparison.md.

Real drafts: passages are restored from data/processed/chunks_256.jsonl by the stored chunk_id (the run's corpus;
the stored text must match exactly), because the raw record lacks the table fields v4 needs. Before any candidate
is scored, the v4 replay must reproduce every live decision. Question outcomes follow the pipeline: draft accepted ->
delivered; else the saved redraft decides; if a variant rejects a draft that v4 accepted there is no saved redraft,
and the outcome is "needs a live redraft" (never guessed).

    uv run python -m relay.candidate_v5.evaluate
"""
from __future__ import annotations

import csv
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from relay.candidate_v5.verifier import verify as verify_v5
from relay.config import app_config
from relay.eval.verifier_v2_check import CachedNLI
from relay.llm import prompts
from relay.verify import nli
from relay.verify.citations import CHUNK_ID
from relay.verify.verifier import verify as verify_v4

OUT_DIR = Path("results/candidate_v5")
REPORT = Path("eval/verifier_v5_offline_comparison.md")
RAW = Path("results/raw/answer_quality/real_2026-11-13T18-06-56Z.jsonl")
SETS = [("v3_eval (calibration + evaluation)", "eval/verifier_v3_eval.jsonl", None),
        ("v3b_targeted", "eval/verifier_v3b_targeted.jsonl", None),
        ("v2_stress calibration", "eval/verifier_v2_stress.jsonl", "calibration"),
        ("held-out (evaluation only)", "eval/verifier_v2_stress.jsonl", "heldout")]
VARIANTS = {"v4": None, "v5 P1": {"p1": True, "p6": False}, "v5 P1+P6": {"p1": True, "p6": True}}
CORRECT = {"correct_full_cited", "correct_partially_cited"}


def run_variant(name, answer, passages, ids, model, t):
    kw = VARIANTS[name]
    return verify_v4(answer, passages, ids, model, t, t) if kw is None else \
        verify_v5(answer, passages, ids, model, t, t, **kw)


def labelled(model, t, chunks) -> dict:
    out = {}
    for set_name, path, split in SETS:
        pairs = [json.loads(l) for l in Path(path).read_text(encoding="utf-8").splitlines()]
        pairs = [p for p in pairs if split is None or p.get("split") == split]
        res = {}
        for v in VARIANTS:
            dec = {p["pair_id"]: run_variant(v, p["answer"], [chunks[c] for c in p["evidence"]], set(chunks), model, t)
                   .sentences[-1].ok for p in pairs}  # same rule as verifier_v2_check.judge: the last sentence
            sup = [p for p in pairs if p["supported"]]
            uns = [p for p in pairs if not p["supported"]]
            res[v] = {"supported_accepted": sum(dec[p["pair_id"]] for p in sup), "supported": len(sup),
                      "unsupported_accepted": sum(dec[p["pair_id"]] for p in uns), "unsupported": len(uns),
                      "decisions": dec}
        for v in list(VARIANTS)[1:]:
            res[v]["changed_vs_v4"] = [pid for pid in res[v]["decisions"]
                                       if res[v]["decisions"][pid] != res["v4"]["decisions"][pid]]
        out[set_name] = res
    return out


def real(model, t, chunks) -> dict:
    recs = [json.loads(l) for l in RAW.read_text(encoding="utf-8").splitlines()]
    head = next(r for r in recs if r["kind"] == "run")
    assert head["corpus_version"] == app_config()["corpus_version"] == "c3", "replay needs the run's corpus (c3)"
    qs = [r for r in recs if r["kind"] == "question"]
    grades = {(g["question_id"], g["draft_n"], g["sentence_n"]): g
              for g in csv.DictReader(Path("eval/answer_quality_sentences.csv").open(encoding="utf-8"))}
    drafts, questions = [], []
    for r in qs:
        passages = []
        for p in r["retrieval"]["passages"]:
            c = chunks[p["chunk_id"]]
            assert c["text"] == p["text"], f"{p['chunk_id']}: stored text differs from corpus c3"
            passages.append(c)
        per_variant, v4_reasons = {}, []
        for v in VARIANTS:
            accepted, reasons = [], []
            for d in r["drafts"]:
                if d["verifier"] is None:
                    accepted.append(None)  # NOT_IN_PAPERS: the model declined, nothing to verify
                    reasons.append(None)
                    continue
                rep = run_variant(v, d["text"], passages, set(chunks), model, t)
                if v == "v4" and rep.ok != d["verifier"]["accepted"]:
                    raise SystemExit(f"v4 replay differs from the live run: {r['question_id']} draft {d['n']}")
                label = next((g["draft_assistant_label"] for k, g in grades.items()
                              if k[:2] == (r["question_id"], str(d["n"]))), "")
                sents = []
                for i, s in enumerate(rep.sentences, 1):
                    g = grades.get((r["question_id"], str(d["n"]), str(i)), {})
                    sents.append({"n": i, "accepted": s.ok, "reasons": s.reasons,
                                  "support": g.get("assistant_citation_support"), "factual": g.get("assistant_factual"),
                                  "parse_failure": "no citation" in s.reasons and bool(CHUNK_ID.search(s.sentence))})
                drafts.append({"question_id": r["question_id"], "draft": d["n"], "variant": v, "label": label,
                               "accepted": rep.ok, "sentences": sents, "sentence_count_matches":
                               len(rep.sentences) == len(d["verifier"]["sentences"])})
                accepted.append(rep.ok)
                reasons.append([s.reasons for s in rep.sentences])
            if v == "v4":
                v4_reasons = reasons
            per_variant[v] = outcome(r, accepted, reasons, v4_reasons, grades)
        questions.append({"question_id": r["question_id"], "answerable": r["answerable"], **per_variant})
    return {"drafts": drafts, "questions": questions}


def outcome(r: dict, accepted: list, reasons: list, v4_reasons: list, grades: dict) -> dict:
    """The final delivered response under one variant, from each saved draft's decision. A saved redraft stands
    for the candidate's redraft only if the candidate rejected draft 1 for exactly v4's reasons (the redraft prompt
    carries those reasons); otherwise, or when no redraft was saved, the outcome is unknown."""
    refusal = "false_refusal" if r["answerable"] else "correct_refusal"
    for n, ok in enumerate(accepted, 1):
        if ok is None:
            return {"result": "refused (model declined)", "grade": refusal, "drafts": accepted}
        if ok:
            label = next(g["draft_assistant_label"] for k, g in grades.items() if k[:2] == (r["question_id"], str(n)))
            return {"result": f"delivered draft {n}", "grade": label, "drafts": accepted}
        if n == 1 and len(accepted) == 1:
            if r["final"]["status"] == "answered":  # v4 accepted draft 1, so no redraft was ever made
                return {"result": "draft 1 rejected; no saved redraft", "grade": "unknown_needs_live_redraft",
                        "drafts": accepted}
        elif n == 1 and reasons[0] != v4_reasons[0]:
            return {"result": "draft 1 rejected for other reasons than v4; saved redraft not valid",
                    "grade": "unknown_needs_live_redraft", "drafts": accepted}
        if n == len(accepted):
            return {"result": "refused (verification failed)", "grade": refusal, "drafts": accepted}
    raise AssertionError("no drafts")


def summarise(res: dict) -> dict:
    s = {}
    for v in VARIANTS:
        ds = [d for d in res["drafts"] if d["variant"] == v]
        sents = [x for d in ds for x in d["sentences"]]
        qs = Counter(q[v]["grade"] for q in res["questions"])
        s[v] = {
            "drafts_accepted": sum(d["accepted"] for d in ds), "drafts_rejected": sum(not d["accepted"] for d in ds),
            "responses_delivered": sum(q[v]["result"].startswith("delivered") for q in res["questions"]),
            "correct_drafts_accepted": sum(d["accepted"] for d in ds if d["label"] in CORRECT),
            "correct_drafts": sum(d["label"] in CORRECT for d in ds),
            "wrong_drafts_accepted": sum(d["accepted"] for d in ds if d["label"] == "factually_wrong"),
            "wrong_drafts": sum(d["label"] == "factually_wrong" for d in ds),
            "partially_correct_drafts_accepted": sum(d["accepted"] for d in ds if d["label"] == "partially_correct"),
            "supported_true_sentences_accepted": sum(x["accepted"] for x in sents
                                                     if x["support"] == "supported" and x["factual"] == "true"),
            "supported_true_sentences": sum(x["support"] == "supported" and x["factual"] == "true" for x in sents),
            "unsupported_or_false_sentences_accepted": sum(x["accepted"] for x in sents
                                                           if x["support"] == "unsupported" or x["factual"] == "false"),
            "unsupported_or_false_sentences": sum(x["support"] == "unsupported" or x["factual"] == "false" for x in sents),
            "citation_parse_failures": sum(x["parse_failure"] for x in sents),
            "sentence_count_mismatches": sum(not d["sentence_count_matches"] for d in ds),
            "correct_refusals": qs["correct_refusal"], "false_refusals": qs["false_refusal"],
            "wrong_delivered": qs["factually_wrong"],
            "correct_delivered": qs["correct_full_cited"] + qs["correct_partially_cited"],
            "partially_correct_delivered": qs["partially_correct"],
            "needs_live_redraft": qs["unknown_needs_live_redraft"],
        }
    return s


def write_report(lab: dict, res: dict, summ: dict, meta: dict) -> str:
    V = list(VARIANTS)
    L = ["# Verifier v5 candidate: offline comparison with frozen v4", "",
         f"Generated by relay/candidate_v5/evaluate.py at {meta['at']} (results: `{meta['json']}`). Offline only: "
         "no LLM or API calls. v4 (relay/verify, threshold "
         f"{meta['threshold']}) is unchanged; v5 is relay/candidate_v5/verifier.py and is not used by the pipeline. "
         "Real-draft grades are assistant-graded and not yet confirmed by the owner.", "",
         "## 1-3. Labelled verifier sets (decision on the judged sentence)", "",
         "| Set | " + " | ".join(f"{v}: supported accepted / unsupported accepted" for v in V) + " |",
         "|---|" + "---|" * len(V)]
    for name, r in lab.items():
        L.append(f"| {name} | " + " | ".join(
            f"{r[v]['supported_accepted']}/{r[v]['supported']} / {r[v]['unsupported_accepted']}/{r[v]['unsupported']}"
            for v in V) + " |")
    changed = {f"{name} ({v})": r[v]["changed_vs_v4"] for name, r in lab.items() for v in V[1:] if r[v]["changed_vs_v4"]}
    L += ["", "Pairs whose decision differs from v4: " + (json.dumps(changed) if changed else "none") + ".", "",
          "## 1-3. Saved real drafts (17 verified drafts, 6 model refusals; passages restored from corpus c3)", "",
          "v4 replay reproduced every live decision (checked before scoring the candidates).", "",
          "| Measure | " + " | ".join(V) + " |", "|---|" + "---|" * len(V)]
    rows = [("Drafts accepted by the verifier (of 17)", "drafts_accepted", None),
            ("Drafts rejected by the verifier (of 17)", "drafts_rejected", None),
            ("Final responses delivered (of 15)", "responses_delivered", None),
            ("Correct drafts accepted", "correct_drafts_accepted", "correct_drafts"),
            ("Factually wrong drafts accepted", "wrong_drafts_accepted", "wrong_drafts"),
            ("Partially correct drafts accepted", "partially_correct_drafts_accepted", None),
            ("Supported, true sentences accepted", "supported_true_sentences_accepted", "supported_true_sentences"),
            ("Unsupported or false sentences accepted", "unsupported_or_false_sentences_accepted",
             "unsupported_or_false_sentences"),
            ("Citation parsing failures (sentences)", "citation_parse_failures", None),
            ("Questions: correct answers delivered", "correct_delivered", None),
            ("Questions: partially correct delivered", "partially_correct_delivered", None),
            ("Questions: wrong answers delivered", "wrong_delivered", None),
            ("Questions: correct refusals", "correct_refusals", None),
            ("Questions: false refusals", "false_refusals", None),
            ("Questions: needs a live redraft (unknown)", "needs_live_redraft", None)]
    for label, k, tot in rows:
        L.append(f"| {label} | " + " | ".join(f"{summ[v][k]}" + (f"/{summ[v][tot]}" if tot else "") for v in V) + " |")
    L += ["", "## 4. Per-question decisions (real drafts)", "",
          "| Question | Answerable | " + " | ".join(V) + " |", "|---|---|" + "---|" * len(V)]
    for q in res["questions"]:
        L.append(f"| {q['question_id']} | {'yes' if q['answerable'] else 'no'} | " +
                 " | ".join(f"{q[v]['result']} → {q[v]['grade']}" for v in V) + " |")
    L += ["", "Sentence-count mismatches between variants and the live record: " +
          ", ".join(f"{v} {summ[v]['sentence_count_mismatches']}" for v in V) + ".", ""]
    return "\n".join(L)


def main() -> dict:
    cfg = app_config()
    t = cfg["verification"]["entail_min"]
    model = CachedNLI(nli.load_nli(cfg["verification"]["nli_model"]))
    chunks = {json.loads(l)["chunk_id"]: json.loads(l) for l in
              Path("data/processed/chunks_256.jsonl").read_text(encoding="utf-8").splitlines()}
    lab = labelled(model, t, chunks)
    res = real(model, t, chunks)
    summ = summarise(res)
    at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"comparison_{at}.json"
    meta = {"at": at, "json": out.as_posix(), "threshold": t, "verification_policy_version_frozen":
            cfg["verification_policy_version"], "raw": RAW.as_posix()}
    with out.open("x", encoding="utf-8") as f:
        json.dump({"meta": meta, "labelled": lab, "real": res, "summary": summ}, f, indent=1, ensure_ascii=False)
    REPORT.write_text(write_report(lab, res, summ, meta), encoding="utf-8")
    return summ


if __name__ == "__main__":
    print(json.dumps(main(), indent=1))
