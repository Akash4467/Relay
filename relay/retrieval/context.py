"""Context cap (DESIGN §7 step 6): whole passages, best first, until the token budget is used."""
from __future__ import annotations

from relay.llm.tokenizer import count


def passage_block(p: dict) -> str:
    """How a passage appears in the prompt (DESIGN §13.2); the cap counts exactly this text."""
    return f"[{p['chunk_id']}] ({p['paper']} §{p['section_id']} {p['section_title']})\n{p['text']}"


def fill(passages: list[dict], cap: int) -> tuple[list[dict], int]:
    """Keep passages in rank order while they fit; a passage that does not fit is skipped, never cut,
    and smaller lower-ranked ones may still fill the gap. Returns (kept, tokens used)."""
    kept, used = [], 0
    for p in passages:
        n = count(passage_block(p)) + 2  # blank line between blocks
        if used + n <= cap:
            kept.append(p)
            used += n
    return kept, used
