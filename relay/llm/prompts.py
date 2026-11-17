"""Prompt templates and reply wording (DESIGN §13). Changing any text here means bumping prompt_version."""
from __future__ import annotations

from relay.retrieval.context import model_view, passage_block

PROMPT_VERSION = "p4"  # must equal config/app.yaml prompt_version (checked by a test)
# p4 (2026-11-14, phase 11B): with caption sharing, a later chunk of a table already shown is rendered without its
# caption line (Columns, rows, values and chunk ID unchanged). Instructions are unchanged.
# p3 (2026-11-13): plain text, no LaTeX. The NLI verifier cannot read LaTeX (q05 replay; verifier v2 check).
NOT_IN_PAPERS = "NOT_IN_PAPERS"  # control signal from the model, never shown to the user as a fact


def is_not_in_papers(text: str) -> bool:
    """The whole reply is the control signal, allowing only case, whitespace, punctuation and markup around it
    ("NOT_IN_PAPERS.", "**NOT_IN_PAPERS**"); a reply with any other content is an answer to verify (F7)."""
    core = text.strip().strip(".,;:!?\"'`*_ ()[]").strip()
    return core.upper() == NOT_IN_PAPERS
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


def user_message(question: str, passages: list[dict], share_captions: bool = False) -> str:
    """share_captions: render the caption-shared model view (context.model_view); the passages are not modified and
    remain the verifier's evidence. Must match how the evidence budget was counted (Evidence.caption_sharing)."""
    shown = model_view(passages) if share_captions else passages
    blocks = "\n\n".join(passage_block(p) for p in shown)
    return f"Question: {question}\n\nPassages:\n{blocks}"


def messages(question: str, passages: list[dict], share_captions: bool = False) -> list[dict]:
    return [{"role": "system", "content": SYSTEM},
            {"role": "user", "content": user_message(question, passages, share_captions)}]


def redraft_messages(question: str, passages: list[dict], draft: str, failures: list[tuple[str, str]],
                     share_captions: bool = False) -> list[dict]:
    """failures: (sentence, reason) pairs from the verifier (DESIGN §13.3)."""
    lines = "\n".join(f'- "{s}" — {reason}' for s, reason in failures)
    return messages(question, passages, share_captions) + [
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
