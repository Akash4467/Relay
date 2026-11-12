"""Load config/*.yaml. Values are placeholders until recalibrated from measurements (DESIGN §5)."""
from __future__ import annotations

from functools import cache
from pathlib import Path

import yaml
from pydantic import BaseModel

CONFIG_DIR = Path("config")


class RetrievalSetting(BaseModel):
    piece_tokens: int
    candidates: int
    rerank: bool
    final_passages: int
    context_budget: int
    expansion: bool


@cache
def app_config(config_dir: Path = CONFIG_DIR) -> dict:
    # shortcut: plain dict, typed models get added per section as the app starts reading them
    return yaml.safe_load((config_dir / "app.yaml").read_text(encoding="utf-8"))


@cache
def retrieval_settings(config_dir: Path = CONFIG_DIR) -> dict[str, RetrievalSetting]:
    raw = yaml.safe_load((config_dir / "retrieval_settings.yaml").read_text(encoding="utf-8"))
    return {name: RetrievalSetting(**values) for name, values in raw.items()}


def piece_sizes(config_dir: Path = CONFIG_DIR) -> list[int]:
    return sorted({s.piece_tokens for s in retrieval_settings(config_dir).values()})
