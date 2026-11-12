"""Local token counting for gpt-oss (DESIGN §11.4). Checked against reported usage once real calls exist."""
from __future__ import annotations

from functools import cache

import tiktoken


@cache
def encoding() -> tiktoken.Encoding:
    try:
        return tiktoken.get_encoding("o200k_harmony")  # gpt-oss's own encoding
    except ValueError:
        return tiktoken.get_encoding("o200k_base")


def count(text: str) -> int:
    return len(encoding().encode(text, disallowed_special=()))


def split_by_tokens(text: str, max_tokens: int) -> list[str]:
    """Last resort for a single sentence longer than a piece: cut on token boundaries."""
    ids = encoding().encode(text, disallowed_special=())
    return [encoding().decode(ids[i:i + max_tokens]) for i in range(0, len(ids), max_tokens)]
