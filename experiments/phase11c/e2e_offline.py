"""Phase 11C MOCKED END-TO-END (offline; no API): the real FastAPI app (/ask), real retrieval with Configuration G
on c3 and on c6 (real embedder, cross-encoder, glossary), real verifier v4.2 (real NLI, threshold 0.9); only the
model's replies are scripted. Mocked drafts are NOT model performance: "faithful" drafts copy or restate the evidence
to show what the evidence can support; adversarial drafts must be refused.

Checked for every request: the prompt holds the ORIGINAL question, every evidence chunk ID, and each table caption at
most once (caption sharing); the verifier received exactly the retrieved chunks, complete and identical to the corpus
text (spy on relay.pipeline.verify); refusal/answer outcomes. Writes experiments/phase11c/out/e2e_offline_{UTC}.json.
"""
from __future__ import annotations

import copy
import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

os.environ["LLM_PROVIDER"] = "standin"
os.environ["GROQ_API_KEY"] = "dummy-offline-key"
os.environ.pop("ALLOW_REAL", None)
sys.path.insert(0, ".")
import httpx  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

import relay.pipeline as pipeline  # noqa: E402
from relay import main  # noqa: E402
from relay.governor.governor import Governor  # noqa: E402
from relay.llm.client import ChatClient  # noqa: E402
from relay.llm.hosted import RetryBudget  # noqa: E402
from relay.retrieval.search import Index  # noqa: E402

OUT = Path("experiments/phase11c/out")
CORPORA = {"c3": (Path("data/index"), Path("data/processed")),
           "c6": (Path("experiments/phase10/corpus/c6/index"), Path("experiments/phase10/corpus/c6/processed"))}
QS = {json.loads(l)["question_id"]: json.loads(l)["question"] for l in open("eval/questions.jsonl", encoding="utf-8")}
QS |= {q["question_id"]: q["question"] for q in json.load(open("eval/phase7/table_question_set.json", encoding="utf-8"))["questions"]}
USAGE = {"choices": [], "x_groq": {"usage": {"prompt_tokens": 1500, "completion_tokens": 100, "total_tokens": 1600}}}


def sse(text: str) -> bytes:
    chunks = [{"choices": [{"delta": {"content": text}, "finish_reason": None}]},
              {"choices": [{"delta": {"content": ""}, "finish_reason": "stop"}]}, USAGE]
    return b"".join(f"data: {json.dumps(c)}\n\n".encode() for c in chunks) + b"data: [DONE]\n\n"


def blocks(req_json: dict) -> list[tuple[str, str]]:
    user = req_json["messages"][1]["content"]
    out = []
    for b in re.split(r"\n\n(?=\[[A-Z]+:[^\]]+\] \()", user.split("Passages:\n", 1)[1]):
        m = re.match(r"\[([^\]]+)\] \([^\n]*\)\n(.*)", b, re.S)
        if m:
            out.append((m.group(1), m.group(2)))
    return out


def cite_with(needle: str, template: str):
    """A draft citing the first passage whose (model-facing) text contains `needle`; NOT_IN_PAPERS if none does."""
    def f(req_json):
        cid = next((c for c, t in blocks(req_json) if needle in t), None)
        return template.format(cid=cid) if cid else "NOT_IN_PAPERS"
    return f


FIRST = lambda tmpl: (lambda r: tmpl.format(cid=blocks(r)[0][0]))  # noqa: E731
Q12_FACT = "Moshi without Inner Monologue gets 9.2 on Web Questions [{cid}]."
SCENARIOS = [  # (name, question id, drafts, expected outcome or None = record only)
    ("q02_faithful", "q02", [cite_with("d_k=d_v=d_model/h=64", "In this work we employ h=8 parallel attention layers, or heads [{cid}]. "
                                       "For each of these we use d_k=d_v=d_model/h=64 [{cid}].")] * 2, None),
    ("q09_units", "q09", [cite_with("230", "Moshi's theoretical latency of 160 ms is lower than the 230 ms average in "
                                    "natural conversations measured over 10 languages [{cid}].")] * 2, None),
    ("q11_partial", "q11", [cite_with("h = 1; d_k = 512", "The variant with a single attention head (h = 1) reaches a "
                                      "development BLEU of 24.9 [{cid}]. The retrieved passages do not state the "
                                      "development BLEU of the base configuration.")] * 2, None),
    ("q11_unretrieved_base", "q11", [lambda r: "The base configuration reaches a development BLEU of 25.8 [AIAYN:T3:r01]."] * 2,
     "refused_unless_r01_retrieved"),
    ("q12_both_rows", "q12", [lambda r: "Moshi reaches 26.6 on Web Questions, 62.3 on LlaMA Questions and 22.8 on Audio "
                              "Trivia QA [MOSHI:T8:r01]. Without Inner Monologue, Moshi reaches 9.2, 21.0 and 7.3 "
                              "[MOSHI:T8:r02]."] * 2, None),
    ("t10_marks", "t10", [cite_with("Mimi, adv. loss only", "In the audio quality evaluation table, the causal column of "
                                    "Mimi, adv. loss only is ✓ [{cid}].")] * 2, None),
    ("q18_refusal", "q18", [lambda r: "NOT_IN_PAPERS"], "refused"),
    ("q20_refusal", "q20", [lambda r: "NOT_IN_PAPERS"], "refused"),
    ("q30_refusal", "q30", [lambda r: "NOT_IN_PAPERS"], "refused"),
    ("q20_fabricated", "q20", [FIRST("The Transformer scores 86.4 on MMLU, below GPT-4 [{cid}].")] * 2, "refused"),
    ("q30_fabricated", "q30", [FIRST("Moshi's dynamic chunking layer uses a chunk size of 4 [{cid}].")] * 2, "refused"),
    ("q12_false_absence", "q12", [cite_with("9.2", Q12_FACT + " The retrieved passages do not state the spoken question "
                                            "answering accuracy of Moshi with Inner Monologue.")] * 2, "refused"),
    ("invalid_citation", "q02", [lambda r: "The base Transformer uses 8 attention heads [AIAYN:9.9:c999]."] * 2, "refused"),
    ("unretrieved_citation", "q02", [lambda r: "The base Transformer uses 8 attention heads [HNET:1:c001]."] * 2, "refused"),
    ("redraft_then_faithful", "q02", [lambda r: "The base Transformer uses 8 attention heads [AIAYN:9.9:c999].",
                                      cite_with("d_k=d_v=d_model/h=64", "In this work we employ h=8 parallel attention "
                                                "layers, or heads [{cid}].")], None),
]


class Provider:
    def __init__(self, drafts):
        self.drafts, self.requests, self.sent = list(drafts), [], []

    def __call__(self, req):
        body = json.loads(req.content)
        self.requests.append(body)
        text = self.drafts.pop(0)(body) if self.drafts else "NOT_IN_PAPERS"
        self.sent.append(text)
        return httpx.Response(200, content=sse(text))


def main_() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    main.TRACE_DIR = OUT / "traces"
    s = main.State()  # real models; production config (G on, v4.2, p4)
    main.STATE = s
    seen: list = []
    real_verify = pipeline.verify

    def spy(draft, passages, *a, **k):
        seen.append(copy.deepcopy(passages))
        return real_verify(draft, passages, *a, **k)
    pipeline.verify = spy
    http = TestClient(main.app)
    results = []
    for corpus, (idx, proc) in CORPORA.items():
        s.indexes = {size: Index(size, s.indexes[256].embedder if 256 in s.indexes else None, idx, proc) for size in (256, 512)}
        corpus_text = {cid: c["text"] for cid, c in s.indexes[256].chunks.items()}
        for name, qid, drafts, expect in SCENARIOS:
            provider = Provider(drafts)
            s.client = ChatClient("https://fake-provider.invalid/v1", "dummy-offline-key", s.cfg["timeouts"], "standin",
                                  OUT / "provider_raw", httpx.MockTransport(provider))
            s.governor, s.budget = Governor.from_config(), RetryBudget.from_config(s.cfg["retry"])
            seen.clear()
            question = QS[qid]
            events, ev_name = [], None
            with http.stream("POST", "/ask", json={"question": question, "request_id": f"p11c-{corpus}-{name}",
                                                   "scenario": "phase11c_offline", "deadline_ms": 120000}) as r:
                for line in r.iter_lines():
                    if line.startswith("event: "):
                        ev_name = line[7:]
                    elif line.startswith("data: "):
                        events.append((ev_name, json.loads(line[6:])))
            tr = json.loads(sorted((OUT / "traces").glob("*.jsonl"))[-1].read_text(encoding="utf-8").splitlines()[-1])
            done = next((d for n, d in events if n == "done"), {})
            content = next((d["text"] for n, d in events if n == "content"), "")
            ev_ids = tr.get("evidence_ids") or []
            checks = {}
            for req in provider.requests:
                user = req["messages"][1]["content"]
                # caption sharing applies to TABLE chunks; a separate full-caption chunk (kind "caption") is its own evidence
                captions = [t.split("\n", 1)[0].split(":")[0] for c, t in blocks(req)
                            if re.search(r":T\d+:r\d+$", c) and t.startswith("Table ")]
                checks["prompt_has_original_question"] = checks.get("prompt_has_original_question", True) and \
                    user.startswith(f"Question: {question}\n") and "[" + "; ".join(tr.get("query_expansion") or ["~"]) + "]" not in user
                checks["prompt_has_all_evidence_ids"] = checks.get("prompt_has_all_evidence_ids", True) and \
                    all(f"[{c}]" in user for c in ev_ids)
                checks["each_caption_at_most_once"] = checks.get("each_caption_at_most_once", True) and \
                    len(captions) == len(set(captions))
            checks["verifier_got_the_retrieved_chunks"] = all([p["chunk_id"] for p in ps] == ev_ids for ps in seen)
            checks["verifier_chunks_complete_and_unmodified"] = all(p["text"] == corpus_text[p["chunk_id"]] for ps in seen for p in ps)
            if expect == "refused":
                checks["outcome"] = done.get("tier") == "refused"
            elif expect == "refused_unless_r01_retrieved":
                checks["outcome"] = done.get("tier") == "refused" or "AIAYN:T3:r01" in ev_ids
            if done.get("tier") == "normal":  # an answer is only ever a verified draft
                checks["answer_is_a_verified_draft"] = any(content == pipeline.citations.normalize_brackets(d).strip()
                                                           for d in provider.sent) and bool(done.get("citations"))
            results.append({"corpus": corpus, "scenario": name, "question_id": qid, "expansion": tr.get("query_expansion"),
                            "evidence_ids": ev_ids, "drafts": provider.sent, "llm_calls": tr.get("llm_calls"),
                            "outcome": done.get("tier"), "reason": done.get("reason"),
                            "failures": None, "checks": checks, "passed": all(checks.values()), "expected": expect})
            print(f"{'PASS' if all(checks.values()) else 'FAIL'} {corpus} {name:24} {done.get('tier')}/{done.get('reason')} "
                  f"calls={tr.get('llm_calls')} {[k for k, v in checks.items() if not v]}")
    pipeline.verify = real_verify
    p = OUT / f"e2e_offline_{datetime.now(timezone.utc).strftime('%Y-%m-%dT%H-%M-%SZ')}.json"
    p.write_text(json.dumps(results, indent=1, ensure_ascii=False), encoding="utf-8")
    print(p.as_posix(), "| passed", sum(r["passed"] for r in results), "of", len(results))
    return 0 if all(r["passed"] for r in results) else 1


if __name__ == "__main__":
    raise SystemExit(main_())
