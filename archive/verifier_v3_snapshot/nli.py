"""Semantic support check (DESIGN §14.5, verifier level 3): does a cited passage entail the sentence?

A local NLI cross-encoder, never an LLM call (CLAUDE.md hard rule 7). Like the other models it runs in the
server's bounded thread pool. The premise is the passage as indexed (paper + section header + text), so a
claim may name the paper even when the chunk text itself does not.
"""
from __future__ import annotations

import numpy as np


def load_nli(model_name: str):
    import torch
    from sentence_transformers import CrossEncoder  # heavy import, only when verifying
    # some checkpoints default to float16, which is slower on CPU and shifts scores; compare models in float32
    return CrossEncoder(model_name, device="cpu", model_kwargs={"dtype": torch.float32})


def scores(model, pairs: list[tuple[str, str]]) -> list[tuple[float, float]]:
    """(entailment, contradiction) probabilities per (premise, hypothesis) pair."""
    if not pairs:
        return []
    logits = np.asarray(model.predict(pairs), dtype="float64")
    probs = np.exp(logits - logits.max(axis=1, keepdims=True))
    probs /= probs.sum(axis=1, keepdims=True)
    labels = {v.lower(): int(k) for k, v in model.config.id2label.items()}
    return [(float(p[labels["entailment"]]), float(p[labels["contradiction"]])) for p in probs]
