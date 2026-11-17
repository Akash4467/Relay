"""Checkpointed offline verifier comparison (no API): production v4.2 vs a candidate verifier module.

Cases: (1) the 160 labelled verifier stress pairs in eval/ (judged as relay.eval.verifier_v2_check.judge does: the
last sentence; a 'claim' pair gets its evidence IDs as its citation); (2) every recorded live draft (not
NOT_IN_PAPERS) in the given run dirs, judged as a whole answer (Report.ok), with the run's draft_labels.json label.

Each case is appended to <out>/cases.jsonl as soon as it is done; a rerun skips cases already there (resume).
Before each case the free physical memory is read; below --min-free-gb the run stops cleanly (exit 3), keeping
everything done so far. Writes <out>/summary.json when every case is done.

usage: verifier_compare_checkpointed.py <candidate module> <out dir> [--min-free-gb 1.0] <run dir> ...
"""
from __future__ import annotations

import ctypes
import importlib
import json
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, ".")
from relay.config import app_config  # noqa: E402
from relay.verify import nli  # noqa: E402
from relay.verify import verifier as v42  # noqa: E402
from relay.verify.citations import strip_citations  # noqa: E402

STRESS = ["eval/verifier_stress.jsonl", "eval/verifier_stress_v2.jsonl", "eval/verifier_v2_stress.jsonl",
          "eval/verifier_v3_eval.jsonl", "eval/verifier_v3b_targeted.jsonl"]
SHOULD_REJECT = ("mixed", "wrong", "unsupported")
TYPO = str.maketrans({"‑": "-", "‐": "-", "–": "-", " ": " ", " ": " ", " ": " ",
                      "’": "'", "‘": "'", "“": '"', "”": '"'})


def free_gb() -> float:
    class MS(ctypes.Structure):
        _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong), ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong), ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong), ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong), ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]
    m = MS()
    m.dwLength = ctypes.sizeof(MS)
    ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
    return m.ullAvailPhys / 2**30


def norm(t: str) -> str:
    return re.sub(r"\s+", " ", t.translate(TYPO)).strip().rstrip(".").strip()


def answer_of(p: dict) -> str:
    if "answer" in p:
        return p["answer"]
    return p["claim"].rstrip().rstrip(".") + " " + "".join(f"[{e}]" for e in p["evidence"]) + "."


def sentence_rows(rep, by_id: dict) -> list[dict]:
    rows = []
    for s in rep.sentences:
        body = norm(strip_citations(s.sentence))
        exact = bool(body) and bool(s.citations) and any(body in norm(by_id[c]["text"]) for c in s.citations if c in by_id)
        rows.append({"sentence": s.sentence, "ok": s.ok, "entailment": s.entailment, "absence": s.absence,
                     "exact_copy_of_cited": exact, "reasons": s.reasons})
    return rows


def cases(runs: list[str]) -> list[dict]:
    out = []
    for f in STRESS:
        for p in (json.loads(l) for l in open(f, encoding="utf-8") if l.strip()):
            out.append({"case_id": f"stress:{Path(f).stem}:{p['pair_id']}", "kind": "stress", "file": f,
                        "pair_id": p["pair_id"], "category": p.get("category"), "split": p.get("split"),
                        "label": "supported" if p["supported"] else "unsupported", "text": answer_of(p),
                        "evidence": p["evidence"], "note": p.get("note")})
    for run in runs:
        lp = Path(run) / "draft_labels.json"
        labels = json.loads(lp.read_text(encoding="utf-8"))["labels"] if lp.exists() else {}
        for f in sorted(Path(run).glob("records/*.json")):
            r = json.loads(f.read_text(encoding="utf-8"))
            for i, d in enumerate(r.get("drafts") or []):
                if d.strip() == "NOT_IN_PAPERS" or i >= len(r.get("verification") or []):
                    continue
                out.append({"case_id": f"draft:{run}:{r['question_id']}#{i}", "kind": "draft", "run": run,
                            "corpus": r.get("corpus_version", "c3"),
                            "question_id": r["question_id"], "draft": i, "label": labels.get(f"{r['question_id']}#{i}"),
                            "recorded_ok": r["verification"][i]["ok"], "text": d, "evidence": r["evidence_ids"]})
    return out


def read_cases(path: Path) -> list[dict]:
    """Complete lines only: a line cut by an interruption is ignored (that case is redone on resume)."""
    rows = []
    for l in path.open(encoding="utf-8"):
        try:
            rows.append(json.loads(l))
        except json.JSONDecodeError:
            pass
    return list({r["case_id"]: r for r in rows}.values())


def main(cand_mod: str, out: Path, runs: list[str], min_free: float) -> int:
    cand = importlib.import_module(cand_mod)
    out.mkdir(parents=True, exist_ok=True)
    done_path = out / "cases.jsonl"
    done = {r["case_id"] for r in read_cases(done_path)} if done_path.exists() else set()
    todo = [c for c in cases(runs) if c["case_id"] not in done]
    print(f"cases: {len(done)} done, {len(todo)} to do; free memory {free_gb():.2f} GB", flush=True)
    if todo:
        t = app_config()["verification"]["entail_min"]
        model = nli.load_nli(app_config()["verification"]["nli_model"])
        proc = {"c3": "data/processed", "c6": "experiments/phase10/corpus/c6/processed"}
        CC = {k: {json.loads(l)["chunk_id"]: json.loads(l) for l in open(f"{d}/chunks_256.jsonl", encoding="utf-8")}
              for k, d in proc.items()}
        print(f"model loaded; free memory {free_gb():.2f} GB", flush=True)
        vs = {"v4.2": v42.verify, cand.VERSION: cand.verify}
        for n, c in enumerate(todo, 1):
            if free_gb() < min_free:
                print(f"STOP: free memory {free_gb():.2f} GB < {min_free} GB after {n - 1} cases; rerun to resume", flush=True)
                return 3
            C = CC[c.get("corpus", "c3")]  # each draft against its own corpus's chunks
            ev = [C[e] for e in c["evidence"]]
            by_id = {p["chunk_id"]: p for p in ev}
            res = {}
            for name, fn in vs.items():
                t0 = time.perf_counter_ns()
                rep = fn(c["text"], ev, set(C), model, t, t)
                ms = (time.perf_counter_ns() - t0) / 1e6
                rows = sentence_rows(rep, by_id)
                ok = rows[-1]["ok"] if c["kind"] == "stress" else rep.ok
                res[name] = {"ok": ok, "ms": round(ms, 1), "sentences": rows}
            row = {**c, **res, "candidate": cand.VERSION, "at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
            with done_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
            flip = "" if res["v4.2"]["ok"] == res[cand.VERSION]["ok"] else f"  FLIP -> {res[cand.VERSION]['ok']}"
            print(f"{len(done) + n:3d} {c['case_id'][:70]:70s} {c['label']} v4.2={res['v4.2']['ok']} "
                  f"{cand.VERSION}={res[cand.VERSION]['ok']} ({res[cand.VERSION]['ms']:.0f} ms){flip}", flush=True)
    summarize(out, cand.VERSION)
    return 0


def summarize(out: Path, cv: str) -> None:
    rows = read_cases(out / "cases.jsonl")
    good = lambda r: r["label"] in ("supported", "correct")  # noqa: E731
    bad = lambda r: r["label"] in ("unsupported",) + SHOULD_REJECT  # noqa: E731
    def ids(sel):  # noqa: E306
        return [r["case_id"] for r in sel]
    s = {}
    for kind in ("stress", "draft"):
        k = [r for r in rows if r["kind"] == kind]
        s[kind] = {
            "n": len(k), "n_should_accept": sum(map(good, k)), "n_should_reject": sum(map(bad, k)),
            "n_unlabelled_or_uncertain": sum(not good(r) and not bad(r) for r in k),
            "v4.2": {"false_rejections": ids(r for r in k if good(r) and not r["v4.2"]["ok"]),
                     "false_acceptances": ids(r for r in k if bad(r) and r["v4.2"]["ok"])},
            cv: {"false_rejections": ids(r for r in k if good(r) and not r[cv]["ok"]),
                 "false_acceptances": ids(r for r in k if bad(r) and r[cv]["ok"])},
            "newly_rescued_correct": ids(r for r in k if good(r) and not r["v4.2"]["ok"] and r[cv]["ok"]),
            "newly_rejected_correct": ids(r for r in k if good(r) and r["v4.2"]["ok"] and not r[cv]["ok"]),
            "new_false_acceptances": ids(r for r in k if bad(r) and not r["v4.2"]["ok"] and r[cv]["ok"]),
            "flips_on_unlabelled_or_uncertain": ids(r for r in k if not good(r) and not bad(r)
                                                   and r["v4.2"]["ok"] != r[cv]["ok"]),
            "ms_p50": {n: sorted(r[n]["ms"] for r in k)[len(k) // 2] if k else None for n in ("v4.2", cv)}}
        if kind == "draft":
            s[kind]["recorded_equals_v4.2_replay"] = f"{sum(r['recorded_ok'] == r['v4.2']['ok'] for r in k)}/{len(k)}"
    exact = []
    for r in rows:
        for n in ("v4.2", cv):
            for x in r[n]["sentences"]:
                if x["exact_copy_of_cited"] and not x["ok"]:
                    exact.append({"case_id": r["case_id"], "verifier": n, "sentence": x["sentence"],
                                  "entailment": x["entailment"], "reasons": x["reasons"]})
    s["exact_copy_sentences_rejected"] = {n: [e for e in exact if e["verifier"] == n] for n in ("v4.2", cv)}
    s["exact_copy_sentences_total"] = sum(x["exact_copy_of_cited"] for r in rows for x in r["v4.2"]["sentences"])
    (out / "summary.json").write_text(json.dumps(s, indent=1, ensure_ascii=False), encoding="utf-8")
    for kind in ("stress", "draft"):
        k = s[kind]
        print(kind, {"n": k["n"], "FR v4.2": len(k["v4.2"]["false_rejections"]), f"FR {cv}": len(k[cv]["false_rejections"]),
                     "FA v4.2": len(k["v4.2"]["false_acceptances"]), f"FA {cv}": len(k[cv]["false_acceptances"]),
                     "rescued": k["newly_rescued_correct"], "newly_rejected": k["newly_rejected_correct"],
                     "NEW FA": k["new_false_acceptances"], "unlabelled flips": k["flips_on_unlabelled_or_uncertain"]})
    print("exact copies:", s["exact_copy_sentences_total"], "| still rejected:",
          {n: len(v) for n, v in s["exact_copy_sentences_rejected"].items()})


if __name__ == "__main__":
    args = sys.argv[1:]
    mf = 1.0
    if "--min-free-gb" in args:
        i = args.index("--min-free-gb")
        mf = float(args[i + 1])
        del args[i:i + 2]
    sys.exit(main(args[0], Path(args[1]), args[2:], mf))
