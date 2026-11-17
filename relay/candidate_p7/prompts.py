"""Candidate prompt p7 = p6 + evidence-guided redraft feedback. For evaluation only (production stays p4).

Root cause (assessment/e2e/live/B12_fix_*; report §5): with p6 the model still writes rejected sentences in its own
words, against rules 8-9: "26.6 % accuracy on Web Questions" where the cited row says "Web Q." (0.01), "the original
Transformer" where the passage says "We trained our models" (0.00). Its redraft gets only the verifier's reason, so
it guesses again. Probe (scratch p6.py): the same facts in the row's own terms are accepted ("Its development BLEU
score is 25.8": 0.98) and wrong values still fail.

Change: the redraft message shows, for each rejected sentence, deterministic evidence from the passages it cited:
  - a table citation: the cited row(s) whose values the sentence mentions (or every row of a chunk with <= 3 rows),
    written "<row label>: <column> = <value>; ..." exactly as extracted;
  - a text citation: the passage sentence with the most words in common;
  - the content words of the sentence that the cited passages do not contain (e.g. "original").
No LLM call, no new facts: everything shown is already in the passages the model was given. The system prompt and
first-draft messages are p6's; the verifier is unchanged.
"""
from __future__ import annotations

import re

from relay.candidate_p5r import prompts as p5r
from relay.candidate_p6 import prompts as p6
from relay.llm import prompts as p4
from relay.verify.absence import _STOP
from relay.verify.citations import citations_in, strip_citations
from relay.verify.sentences import split_sentences

PROMPT_VERSION = "p7"
SYSTEM = p6.SYSTEM
messages = p6.messages
_NUM = re.compile(r"\d+(?:\.\d+)?")
_WORD = re.compile(r"[A-Za-z][A-Za-z\-]{3,}")
# words p6 rule 8/9 tells the model to use ("the authors report", "In Table N, the ... configuration has"): never
# listed as missing, so the hint does not contradict the rules
_ALLOWED = {"authors", "author", "paper", "papers", "report", "reports", "reported", "also", "table", "configuration",
            "configurations", "states", "state", "stated"}


def _row_text(row: dict) -> str:
    cells = "; ".join(f"{v['column'] if v.get('column') else v['metric']} = {v['value']}" for v in row["values"])
    return f"{row['configuration']}: {cells}"


def _words(text: str) -> set[str]:
    return {w.lower() for w in _WORD.findall(text)}


def hints(sentence: str, passages: list[dict]) -> list[str]:
    by_id = {p["chunk_id"]: p for p in passages}
    cited = [by_id[c] for c in citations_in(sentence) if c in by_id]
    claim = strip_citations(sentence)
    out = []
    nums = set(_NUM.findall(claim))
    for p in cited:
        if p.get("kind") == "table" and p.get("rows"):
            rows = [r for r in p["rows"] if nums & {n for v in r["values"] for n in _NUM.findall(v["value"])}]
            rows = rows or (p["rows"] if len(p["rows"]) <= 3 else [])
            out += [f'Row in [{p["chunk_id"]}]: "{_row_text(r)}"' for r in rows]
        else:
            cw = _words(claim) - _STOP
            best = max(split_sentences(p["text"]) or [p["text"]], key=lambda s: len(cw & _words(s)), default="")
            if best:
                out.append(f'Closest sentence in [{p["chunk_id"]}]: "{best}"')
    if cited:
        have = set().union(*(_words(p["text"]) for p in cited))
        missing = sorted(w for w in _words(claim) - have if w not in _STOP and w not in _ALLOWED)
        if missing:
            out.append("Words in your sentence that the cited passage does not contain: " + ", ".join(missing))
    return out


def redraft_messages(question: str, passages: list[dict], draft: str, failures: list[tuple[str, str]],
                     share_captions: bool = False) -> list[dict]:
    blocks = []
    for s, reason in failures:
        h = "\n".join(f"    {x}" for x in hints(s, passages))
        blocks.append(f'- "{s}" — {reason}' + (f"\n{h}" if h else ""))
    lines = "\n".join(blocks)
    return messages(question, passages, share_captions) + [
        {"role": "assistant", "content": draft},
        {"role": "user", "content": f"Your previous answer had unsupported sentences:\n{lines}\n{p5r.REDRAFT_INSTRUCTION} "
                                    "Where a row or sentence is shown above, reuse its exact labels, column names and "
                                    "wording, and leave out the listed words."},
    ]


def as_prompts_module():
    from types import SimpleNamespace
    return SimpleNamespace(**{k: v for k, v in vars(p4).items() if not k.startswith("__")}
                           | {"SYSTEM": SYSTEM, "messages": messages, "redraft_messages": redraft_messages,
                              "PROMPT_VERSION": PROMPT_VERSION})
