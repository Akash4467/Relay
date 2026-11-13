import pytest

from relay.governor.buckets import Bucket
from relay.governor.governor import Governor


class Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


def test_bucket_refills_continuously_and_caps():
    b = Bucket(60, 60)          # 1 unit per second
    b.take(60, 0)
    assert b.available(0) == 0
    assert b.available(30) == pytest.approx(30)
    assert b.available(1000) == 60
    assert b.wait_for(10, 1000) == 0
    assert b.wait_for(61, 0) == float("inf")


def test_bucket_freeze_and_resync():
    b = Bucket(100, 60)
    b.freeze_until(10)
    assert b.available(5) == 0 and b.wait_for(1, 5) == pytest.approx(5)
    b.resync(40, 20)
    assert b.available(20) == 40
    b.resync(90, 20)             # the provider never raises the local estimate
    assert b.available(20) == 40


LIMITS = {"m": {"rpm": 2, "rpd": 1000, "tpm": 1000, "tpd": 200000}}


def test_reservation_is_all_or_nothing_and_respects_rpm():
    clock = Clock()
    g = Governor(LIMITS, clock)
    assert g.try_reserve("m", 400) and g.try_reserve("m", 400)
    assert g.try_reserve("m", 10) is None                 # rpm 2 used up
    assert g.buckets["m"]["tpm"].available(0) == pytest.approx(200)  # failed attempt took nothing
    clock.t = 30                                          # one request back after 30 s
    assert g.try_reserve("m", 10) is not None


def test_tokens_bind_before_requests():
    clock = Clock()
    g = Governor(LIMITS, clock)
    assert g.try_reserve("m", 900)
    assert g.wait_estimate("m", 900) == pytest.approx(48)  # needs 800 more tokens at 1000/60 per s
    assert g.wait_estimate("m", 5000) == float("inf")


def test_settle_corrects_to_actual_usage_and_reports_overrun():
    clock = Clock()
    g = Governor(LIMITS, clock)
    r = g.try_reserve("m", 500)
    assert g.settle(r, 300) == 0
    assert g.buckets["m"]["tpm"].available(0) == pytest.approx(700)
    r = g.try_reserve("m", 100)
    assert g.settle(r, 250) == 150
    assert g.buckets["m"]["tpm"].available(0) == pytest.approx(450)


def test_headers_resync_daily_requests_and_minute_tokens_only():
    clock = Clock()
    g = Governor(LIMITS, clock)
    g.resync("m", {"x-ratelimit-remaining-requests": "7", "x-ratelimit-remaining-tokens": "120",
                   "x-ratelimit-limit-requests": "garbage"})
    assert g.buckets["m"]["rpd"].available(0) == 7      # requests header is per DAY
    assert g.buckets["m"]["tpm"].available(0) == 120    # tokens header is per MINUTE
    g.resync("m", {"x-ratelimit-remaining-tokens": "not a number"})  # malformed: keep the estimate
    assert g.buckets["m"]["tpm"].available(0) == 120


def test_429_freezes_per_minute_buckets():
    clock = Clock()
    g = Governor(LIMITS, clock)
    g.on_429("m", 7.0)
    assert g.wait_estimate("m", 10) == pytest.approx(7)
    clock.t = 7.1
    assert g.try_reserve("m", 10) is not None
    g.on_429("m", None)                                  # no retry-after: assume a minute
    assert g.wait_estimate("m", 10) == pytest.approx(60, abs=0.2)


def test_repo_limits_load():
    g = Governor.from_config()
    assert set(g.buckets) == {"openai/gpt-oss-120b", "openai/gpt-oss-20b"}
