"""Token bucket (DESIGN §10.1): capacity C refilled continuously at C/period per second.

A local estimate of the provider's quota, not the quota itself: settle() corrects it to real usage,
resync() lowers it to what the provider's headers say is left, freeze_until() honours a 429.
Times are monotonic seconds passed in by the caller, so tests control the clock.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Bucket:
    capacity: float
    period_s: float
    level: float = field(default=-1.0)       # -1 = start full
    updated: float = 0.0
    frozen_until: float = 0.0

    def __post_init__(self) -> None:
        if self.level < 0:
            self.level = self.capacity

    def _refill(self, now: float) -> None:
        if now > self.updated:
            self.level = min(self.capacity, self.level + (now - self.updated) * self.capacity / self.period_s)
            self.updated = now

    def available(self, now: float) -> float:
        self._refill(now)
        return 0.0 if now < self.frozen_until else self.level

    def wait_for(self, n: float, now: float) -> float:
        """Seconds until n units are available (inf if n exceeds the capacity)."""
        if n > self.capacity:
            return float("inf")
        self._refill(now)
        frozen = max(0.0, self.frozen_until - now)
        missing = max(0.0, n - self.level)
        return max(frozen, missing * self.period_s / self.capacity)

    def take(self, n: float, now: float) -> None:
        self._refill(now)
        self.level -= n

    def give_back(self, n: float, now: float) -> None:
        self._refill(now)
        self.level = min(self.capacity, self.level + n)

    def freeze_until(self, t: float) -> None:
        self.frozen_until = max(self.frozen_until, t)

    def resync(self, remaining: float, now: float) -> None:
        """The provider says only `remaining` is left: never let the local estimate be higher."""
        self._refill(now)
        self.level = min(self.level, remaining)
