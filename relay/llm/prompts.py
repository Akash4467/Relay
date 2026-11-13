"""Prompt templates and reply wording (DESIGN §13). Changing any text here means bumping prompt_version."""
from __future__ import annotations

from relay.retrieval.context import passage_block

PROMPT_VERSION = "p3"  # must equal config/app.yaml prompt_version (checked by a test)
# p3 (2026-11-13): plain text, no LaTeX. The NLI verifier cannot read LaTeX (q05 replay; verifier v2 check).
NOT_IN_PAPERS = "NOT_IN_PAPERS"  # control signal from the model, never shown to the user as a fact
# The model only sees the retrieved passages, so absence can only be claimed about them, never about the papers
# (p1 allowed "The papers do not state ...", which was false when retrieval missed the row: q11, 2026-11-13).
NOT_STATED = "The retrieved passages do not state"

SYSTEM = """You answer questions about exactly three papers using ONLY the passages provided.
Rules:
1. Every sentence must end with one or more citations in square brackets using the passage IDs given, e.g. [MOSHI:4.2:c017].
2. Use only facts stated in the cited passage. Copy numbers and units exactly.
3. If the passages do not contain the answer, reply with exactly: NOT_IN_PAPERS
4. If only part is answered, answer that part and add one sentence: "The retrieved passages do not state <missing part>." with no citation and no numbers.
5. Do not use outside knowledge. Do not mention passages that were not provided.
6. Be concise: at most 6 sentences.
7. Write plain text only: no LaTeX, no markdown math, no backslash commands. Write formulas with ordinary characters (for example: p_t = 0.5 * (1 - cos(q_t, k_t-1))), keep their mathematical meaning, and copy every number exactly."""

REDUCED_PREFIX = "Answered by a smaller backup model while capacity is limited."


def user_message(question: str, passages: list[dict]) -> str:
    blocks = "\n\n".join(passage_block(p) for p in passages)
    return f"Question: {question}\n\nPassages:\n{blocks}"


def messages(question: str, passages: list[dict]) -> list[dict]:
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user_message(question, passages)}]


def redraft_messages(question: str, passages: list[dict], draft: str, failures: list[tuple[str, str]]) -> list[dict]:
    """failures: (sentence, reason) pairs from the verifier (DESIGN §13.3)."""
    lines = "\n".join(f'- "{s}" — {reason}' for s, reason in failures)
    return messages(question, passages) + [
        {"role": "assistant", "content": draft},
        {"role": "user", "content": "Your previous answer had unsupported sentences:\n"
                                    f"{lines}\nRewrite the answer following the rules. Remove or fix those sentences."},
    ]


def refusal_text(passages: list[dict]) -> str:
    """Honest refusal plus one line on what was searched, never a guess (DESIGN §13.4). Worded about the
    retrieved passages only: a refusal cannot prove the papers never say it."""
    seen: list[str] = []
    for p in passages:
        where = f"{p['paper']} §{p['section_id']} {p['section_title']}".strip()
        if where not in seen:
            seen.append(where)
    searched = f" I searched: {'; '.join(seen[:4])}." if seen else ""
    return "I couldn't find this in the passages I retrieved from the three papers, so I won't guess." + searched


def unavailable_text(reset_in_s: float) -> str:
    """DESIGN §10.8: the daily budget is gone. Never a guessed answer."""
    minutes = max(1, round(reset_in_s / 60))
    return ("The research assistant has used today's model capacity, so I can't write a new answer right now. "
            f"Full answers resume in about {minutes} minute{'s' if minutes != 1 else ''}.")


def busy_text(seconds: int) -> str:
    return ("The assistant is at capacity right now and couldn't start your question in time. "
            f"Please try again in about {seconds} seconds.")
