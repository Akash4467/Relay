"""Context cap (DESIGN §7 step 6): whole passages, best first, until the token budget is used."""
from __future__ import annotations

from relay.llm.tokenizer import count


def passage_block(p: dict) -> str:
    """How a passage appears in the prompt (DESIGN §13.2); the cap counts exactly this text."""
    return f"[{p['chunk_id']}] ({p['paper']} §{p['section_id']} {p['section_title']})\n{p['text']}"


def _table_key(p: dict) -> tuple | None:
    return (p["paper"], p.get("table_id")) if p.get("kind") == "table" else None


def shared_view(p: dict, seen_tables: set) -> dict:
    """Phase 11B prompt-only caption sharing (Configuration G of phase 11A): a later chunk of a table whose caption is
    already in the prompt is shown without its caption line; its Columns line, row labels, values, units and chunk ID
    stay. Returns a NEW dict when it changes anything; the chunk itself is never modified."""
    key = _table_key(p)
    if key is None or key not in seen_tables:
        return p
    return dict(p, text=p["text"].split("\n", 1)[1] if "\n" in p["text"] else p["text"])


def model_view(passages: list[dict]) -> list[dict]:
    """What the model sees for these passages (in order) with caption sharing on. The verifier always gets the
    original passages, never this view."""
    out, seen = [], set()
    for p in passages:
        out.append(shared_view(p, seen))
        if _table_key(p):
            seen.add(_table_key(p))
    return out


def fill(passages: list[dict], cap: int, share_captions: bool = False) -> tuple[list[dict], int]:
    """Keep passages in rank order while they fit; a passage that does not fit is skipped, never cut,
    and smaller lower-ranked ones may still fill the gap. Returns (kept, tokens used).
    share_captions (phase 11B): tokens are counted on the model view (model_view) so more table rows fit the same
    cap; the returned passages are still the original, complete chunks (the verifier's evidence)."""
    kept, used, seen = [], 0, set()
    for p in passages:
        view = shared_view(p, seen) if share_captions else p
        n = count(passage_block(view)) + 2  # blank line between blocks
        if used + n <= cap:
            kept.append(p)
            used += n
            if share_captions and _table_key(p):
                seen.add(_table_key(p))
    return kept, used
