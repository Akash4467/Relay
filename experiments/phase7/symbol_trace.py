"""Phase 7, stage 4 (offline): trace ✓, ✗, (blank), an empty LaTeX cell and [symbol] from the extracted table through
the chunk, retrieval and model context, then probe frozen verifier v4 with CONSTRUCTED claims (test inputs written for
this probe; they are not model output). The generated-answer step is pending: no model output exists for these
questions and none was produced. Writes eval/phase7/raw/symbol_trace.json (new file)."""
import json
import sys
from pathlib import Path

sys.path.insert(0, ".")
from relay.config import app_config, retrieval_settings
from relay.llm import prompts
from relay.retrieval.index_build import load_model
from relay.retrieval.rerank import load_reranker
from relay.retrieval.search import Index, search
from relay.verify import nli
from relay.verify.verifier import verify

cfg = app_config()
T = cfg["verification"]["entail_min"]
NLI = nli.load_nli(cfg["verification"]["nli_model"])
CORPORA = {"c3": (Path("data/index"), Path("data/processed")),
           "c4": (Path("data/corpus/c4/index"), Path("data/corpus/c4/processed"))}
QS = {q["question_id"]: q for q in json.loads(Path("eval/phase7/table_question_set.json").read_text(encoding="utf-8"))["questions"]}
B = retrieval_settings()["B"]
embedder, reranker = load_model(), load_reranker()
CASES = [  # symbol kind, table, row locator, column, related question, constructed probe claims (text, intended label)
    ("checked ✓", ("HNET", "T6"), "Example Architectures = H-Net", "Multi-stage / Hierarchy", "t02",
     [("In the related-architectures table, the H-Net row is marked with a check under Multi-stage / Hierarchy.", "literal, correct"),
      ("In the related-architectures table, the H-Net row is marked with a cross under Multi-stage / Hierarchy.", "literal, wrong"),
      ("H-Net supports multi-stage hierarchies.", "interpretation (stated in the caption)")]),
    ("crossed ✗", ("HNET", "T6"), "Example Architectures = MANTa", "Multi-stage / Hierarchy", "t03",
     [("In the related-architectures table, MANTa is marked with a cross under Multi-stage / Hierarchy.", "literal, correct"),
      ("In the related-architectures table, MANTa is marked with a check under Multi-stage / Hierarchy.", "literal, wrong"),
      ("MANTa does not support multi-stage hierarchies.", "interpretation (meaning of ✗ not defined)")]),
    ("blank via empty macro (blank)", ("MOSHI", "T4"), "RVQGAN", "causal", "t11",
     [("In the audio quality table, the causal cell for RVQGAN is blank.", "literal, correct"),
      ("RVQGAN is not causal.", "interpretation of a blank (table alone does not say)"),
      ("RVQGAN is causal.", "wrong")]),
    ("blank, empty LaTeX cell", ("MOSHI", "T6"), "Transcript / Length (↑) = 538", "Depthwise / Parametrization", "t18",
     [("In the delay-pattern ablation, the row with transcript length 538 has no mark under Depthwise Parametrization.", "literal, correct"),
      ("The row with transcript length 538 does not use depthwise parametrization.", "interpretation of a blank")]),
]


def chunk_with(index, key, locator):
    return next(c for c in index.chunks.values() if c.get("rows") and (c["paper"], c.get("table_id")) == key
                and any(locator in l for l in c["text"].split("\n")[2:]))


out = {"symbol_[symbol]_occurrences": {}, "cases": []}
for corpus, (i, p) in CORPORA.items():
    text = "".join(open(p / f, encoding="utf-8").read() for f in ("chunks_256.jsonl", "tables.jsonl", "paragraphs.jsonl"))
    out["symbol_[symbol]_occurrences"][corpus] = text.count("[symbol]")
indexes = {c: Index(B.piece_tokens, embedder, *v) for c, v in CORPORA.items()}
for kind, key, locator, column, qid, probes in CASES:
    case = {"symbol": kind, "table": f"{key[0]}:{key[1]}", "row": locator, "column": column, "question": qid, "corpora": {}}
    for corpus, index in indexes.items():
        tables = [t for t in map(json.loads, open(CORPORA[corpus][1] / "tables.jsonl", encoding="utf-8"))
                  if (t["paper"], t["table_id"]) == key]
        row = next((r for r in tables[0]["rows"] if any(locator in f"{k} = {v}" or locator == r["row_header"]
                                                        for k, v in list(r["cells"].items()) + [("", "")])
                    or locator in r["row_header"]), None)
        chunk = chunk_with(index, key, locator)
        line = next(l for l in chunk["text"].split("\n")[2:] if locator in l)
        ev = search(QS[qid]["question"], B, cfg["pressure"]["L0"]["context_cap"], index, reranker)
        ctx_ids = [x["chunk_id"] for x in ev.passages]
        context = prompts.user_message(QS[qid]["question"], ev.passages)
        probe_res = []
        for claim, label in probes:
            rep = verify(f"{claim} [{chunk['chunk_id']}]", [chunk], set(index.chunks), NLI, T, T)
            s = rep.sentences[0]
            probe_res.append({"claim": claim, "label": label, "accepted": rep.ok,
                              "entailment": None if s.entailment is None else round(s.entailment, 3), "reasons": s.reasons})
        case["corpora"][corpus] = {
            "extracted_cell": None if row is None else row["cells"].get(column, "<column absent>"),
            "chunk": chunk["chunk_id"], "chunk_line": line[:260], "row_chunk_in_context_B": chunk["chunk_id"] in ctx_ids,
            "line_in_model_context": line in context, "generated_answer": "PENDING: no model output (no API calls in this phase)",
            "verifier_probes_constructed": probe_res}
    out["cases"].append(case)
Path("eval/phase7/raw/symbol_trace.json").open("x", encoding="utf-8").write(json.dumps(out, indent=1, ensure_ascii=False))
print("[symbol] occurrences:", out["symbol_[symbol]_occurrences"])
for c in out["cases"]:
    print(f"\n== {c['symbol']} | {c['table']} {c['row']} / {c['column']}")
    for corpus, r in c["corpora"].items():
        print(f"  {corpus}: cell {r['extracted_cell']!r} | in B context {r['row_chunk_in_context_B']} | in model context {r['line_in_model_context']}")
        print(f"      line: {r['chunk_line'][:150]}")
        for pr in r["verifier_probes_constructed"]:
            print(f"      v4 {'ACCEPT' if pr['accepted'] else 'reject'} e={pr['entailment']} [{pr['label']}] {pr['claim'][:80]}")
