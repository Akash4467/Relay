"""Candidate prompt p5r = production p4 + one change, for the improvement loop only (production prompts stay p4).

Root cause (assessment/improvement_loop/failure_registry.json, F-GEN-REDRAFT): after a rejection the redraft message
only says "Rewrite the answer following the rules. Remove or fix those sentences."; gpt-oss then often resends the
rejected sentence unchanged (Phase 12 q15, q02, t29; baseline q04, q13), so the one redraft is wasted. The verifier
accepts close restatements of the cited passage ("The models were trained on 8 NVIDIA P100 GPUs": 0.99) and rejects
added qualifiers or names the passage lacks ("The original Transformer was trained on ...": 0.00).

Change: only the redraft instruction says HOW to fix a rejected sentence. The system prompt, the first-draft
messages, the passages and the verifier are unchanged.
"""
from __future__ import annotations

from relay.llm import prompts as p4

PROMPT_VERSION = "p5r"
REDRAFT_INSTRUCTION = (
    "Rewrite the answer following the rules. For each sentence listed above, either delete it or restate it in the "
    "cited passage's own words: keep the passage's wording and subject, do not add qualifiers, names, comparisons or "
    "interpretations that the cited passage does not state, and cite the passage that contains every number you "
    "copy. Never repeat a listed sentence unchanged. Keep the sentences that were not listed.")


def redraft_messages(question: str, passages: list[dict], draft: str, failures: list[tuple[str, str]],
                     share_captions: bool = False) -> list[dict]:
    lines = "\n".join(f'- "{s}" — {reason}' for s, reason in failures)
    return p4.messages(question, passages, share_captions) + [
        {"role": "assistant", "content": draft},
        {"role": "user", "content": f"Your previous answer had unsupported sentences:\n{lines}\n{REDRAFT_INSTRUCTION}"},
    ]


def as_prompts_module():
    """p4's prompts with only redraft_messages replaced, for an experiment harness to bind as relay.pipeline.prompts
    in its own process (production code and its import of p4 are not modified)."""
    from types import SimpleNamespace
    return SimpleNamespace(**{k: v for k, v in vars(p4).items() if not k.startswith("__")}
                           | {"redraft_messages": redraft_messages, "PROMPT_VERSION": PROMPT_VERSION})
