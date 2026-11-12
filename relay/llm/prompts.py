"""Prompt templates and reply wording (DESIGN §13). Changing any text here means bumping prompt_version."""
from __future__ import annotations

from relay.retrieval.context import passage_block

PROMPT_VERSION = "p1"  # must equal config/app.yaml prompt_version (checked by a test)
NOT_IN_PAPERS = "NOT_IN_PAPERS"
NOT_STATED = "The papers do not state"  # the one sentence allowed without a citation

SYSTEM = """You answer questions about exactly three papers using ONLY the passages provided.
Rules:
1. Every sentence must end with one or more citations in square brackets using the passage IDs given, e.g. [MOSHI:4.2:c017].
2. Use only facts stated in the cited passage. Copy numbers and units exactly.
3. If the passages do not contain the answer, reply with exactly: NOT_IN_PAPERS
4. If only part is answered, answer that part and add one sentence: "The papers do not state <missing part>." with no citation.
5. Do not use outside knowledge. Do not mention passages that were not provided.
6. Be concise: at most 6 sentences."""

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
    """Honest refusal plus one line on what was searched, never a guess (DESIGN §13.4)."""
    seen: list[str] = []
    for p in passages:
        where = f"{p['paper']} §{p['section_id']} {p['section_title']}".strip()
        if where not in seen:
            seen.append(where)
    searched = f" I searched: {'; '.join(seen[:4])}." if seen else ""
    return "The three papers don't state this." + searched


def busy_text(seconds: int) -> str:
    return ("The assistant is at capacity right now and couldn't start your question in time. "
            f"Please try again in about {seconds} seconds.")
