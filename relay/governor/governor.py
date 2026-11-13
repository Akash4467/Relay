"""Rate governor (DESIGN §10.2–10.4): every hosted call reserves first, then settles.

Per model: RPM, RPD, TPM, TPD buckets. A call must get all four at once. Thread-safe: requests run in the
server's thread pool and share one governor. Pressure levels, organization-level ITPM/OTPM buckets and the
effective-limit lowering after an unexpected 429 are not built (DESIGN §20.3). A fresh process starts with full
daily buckets, so provider headers (RPD and TPM) are resynced after every call, and a per-day 429 freezes the
daily buckets until its retry-after.
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from pathlib import Path

import yaml

from relay.governor.buckets import Bucket

LIMITS = Path("config/limits.yaml")


@dataclass
class Reservation:
    model: str
    tokens: int
    at: float


class Governor:
    def __init__(self, limits: dict[str, dict], clock=time.monotonic):
        self.clock = clock
        self.lock = threading.RLock()
        self.buckets = {
            model: {"rpm": Bucket(l["rpm"], 60), "rpd": Bucket(l["rpd"], 86400),
                    "tpm": Bucket(l["tpm"], 60), "tpd": Bucket(l["tpd"], 86400)}
            for model, l in limits.items()}

    @classmethod
    def from_config(cls, path: Path = LIMITS, clock=time.monotonic) -> "Governor":
        return cls(yaml.safe_load(path.read_text(encoding="utf-8"))["models"], clock)

    def wait_estimate(self, model: str, tokens: int) -> float:
        """Seconds until this call could be dispatched, from the binding bucket (inf: it never fits)."""
        with self.lock:
            b, now = self.buckets[model], self.clock()
            return max(b["rpm"].wait_for(1, now), b["rpd"].wait_for(1, now),
                       b["tpm"].wait_for(tokens, now), b["tpd"].wait_for(tokens, now))

    def try_reserve(self, model: str, tokens: int) -> Reservation | None:
        """All four buckets or none."""
        with self.lock:
            if self.wait_estimate(model, tokens) > 0:
                return None
            b, now = self.buckets[model], self.clock()
            for name, n in (("rpm", 1), ("rpd", 1), ("tpm", tokens), ("tpd", tokens)):
                b[name].take(n, now)
            return Reservation(model, tokens, now)

    def settle(self, r: Reservation, actual_tokens: int, headers: dict[str, str] | None = None) -> int:
        """Correct token buckets to actual usage; returns the overrun (actual - reserved, >= 0)."""
        with self.lock:
            b, now = self.buckets[r.model], self.clock()
            diff = actual_tokens - r.tokens
            for name in ("tpm", "tpd"):
                if diff > 0:
                    b[name].take(diff, now)
                else:
                    b[name].give_back(-diff, now)
            self.resync(r.model, headers or {})
            return max(0, diff)

    def resync(self, model: str, headers: dict[str, str]) -> None:
        """Groq's headers: x-ratelimit-remaining-requests is per DAY, x-ratelimit-remaining-tokens per MINUTE."""
        with self.lock:
            b, now = self.buckets[model], self.clock()
            for header, bucket in (("x-ratelimit-remaining-requests", "rpd"), ("x-ratelimit-remaining-tokens", "tpm")):
                try:
                    b[bucket].resync(float(headers[header]), now)
                except (KeyError, ValueError):
                    pass  # missing or malformed header: keep the local estimate

    def on_429(self, model: str, retry_after_s: float | None) -> None:
        """A per-minute 429 is authoritative: freeze the per-minute buckets until retry-after (60 s if not given)."""
        with self.lock:
            until = self.clock() + (retry_after_s if retry_after_s is not None else 60.0)
            for name in ("rpm", "tpm"):
                self.buckets[model][name].freeze_until(until)

    def on_daily_429(self, model: str, retry_after_s: float | None) -> None:
        """A per-day 429: the daily budget is gone until retry-after (an hour if not given)."""
        with self.lock:
            until = self.clock() + (retry_after_s if retry_after_s is not None else 3600.0)
            for name in ("rpd", "tpd"):
                self.buckets[model][name].freeze_until(until)

    def daily_exhausted_for_s(self, model: str) -> float:
        """Seconds until the daily buckets unfreeze (0 if they are not frozen by a per-day 429)."""
        with self.lock:
            now = self.clock()
            return max(0.0, max(self.buckets[model][n].frozen_until for n in ("rpd", "tpd")) - now)
