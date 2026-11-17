"""Candidate prompt p6 = p4 + verifiable-phrasing rules in the system prompt + p5r's redraft instruction.
For evaluation only; production prompts stay p4 until the owner approves a change.

Root cause (assessment/e2e/report.md §4, live B12): 5 of 10 answerable questions were refused because correct drafts
were worded in ways the NLI verifier cannot match to the cited passage, although the facts were right:
  - naming a subject or adding a qualifier the passage lacks: "The original Transformer was trained on eight P100
    GPUs" 0.01 vs "The authors trained their Transformer models on one machine with 8 NVIDIA P100 GPUs" 0.96;
    "Mimi uses 8 quantizers ..." 0.40 vs "The authors use Q = 8 quantizers ..." 1.00;
  - stating a table value in prose: "The base Transformer uses 8 attention heads [T3:r01]" 0.04 vs "In Table 3, the
    base configuration has h = 8 [T3:r01]" 0.99;
  - a comparison citing one of the two rows it compares: 0.07 vs both rows cited 0.90.
The same phrasings with a wrong value are still rejected (5/5 at 0.00). Probe: scratch p5.py, recorded in
assessment/e2e/fix/probe_phrasing.txt.
"""
from __future__ import annotations

from relay.candidate_p5r import prompts as p5r
from relay.llm import prompts as p4

PROMPT_VERSION = "p6"
SYSTEM = p4.SYSTEM + """
8. Write each sentence close to the wording of the passage it cites. Keep the passage's subject: when the passage says "we" or "our model", write "the authors" or "the authors' model" (you may add the model name the passage's paper uses, e.g. "the authors' Transformer models"). Do not add qualifiers, names or descriptions that the cited passage does not contain (for example "original", "base", "new") unless that passage itself uses them.
9. For a value from a table row, write it as: "In Table N, the <row label> configuration has <column> = <value>." using the row label and column name exactly as shown in the passage.
10. A sentence that compares or relates two facts must cite every passage that states one of them."""


def messages(question: str, passages: list[dict], share_captions: bool = False) -> list[dict]:
    return [{"role": "system", "content": SYSTEM},
            {"role": "user", "content": p4.user_message(question, passages, share_captions)}]


def redraft_messages(question: str, passages: list[dict], draft: str, failures: list[tuple[str, str]],
                     share_captions: bool = False) -> list[dict]:
    lines = "\n".join(f'- "{s}" — {reason}' for s, reason in failures)
    return messages(question, passages, share_captions) + [
        {"role": "assistant", "content": draft},
        {"role": "user", "content": f"Your previous answer had unsupported sentences:\n{lines}\n{p5r.REDRAFT_INSTRUCTION}"},
    ]


def as_prompts_module():
    """p4's prompts with SYSTEM, messages and redraft_messages replaced, for an experiment harness to bind as
    relay.pipeline.prompts in its own process (production code and its import of p4 are not modified)."""
    from types import SimpleNamespace
    return SimpleNamespace(**{k: v for k, v in vars(p4).items() if not k.startswith("__")}
                           | {"SYSTEM": SYSTEM, "messages": messages, "redraft_messages": redraft_messages,
                              "PROMPT_VERSION": PROMPT_VERSION})
