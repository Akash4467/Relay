"""Phase 10 live findings, fixed offline (no Groq): L1 gpt-oss 【ID】 citations (verifier v4.1) and L2 the rate
governor rejecting after it had slept the estimated wait (coarse Windows clock)."""
import math
import time

import pytest

from relay.config import app_config
from relay.governor.governor import Governor
from relay.llm.hosted import MIN_WAIT_S, GovernorReject, HostedLLM
from relay.pipeline import answer_from_evidence
from relay.verify.citations import normalize_brackets
from tests.unit.test_client import LIMITS, MODEL, OK, client
from tests.unit.test_pipeline import FakeLLM, ev, verifier
from tests.unit.test_verifier import CORPUS, P, run

CLAIM = {"Mimi uses 8 quantizers.": 0.95}


# ---- L1: both citation formats, same strict checks
def test_config_is_v4_1_or_later():  # v4.2 (phase 10C) keeps the L1 normalisation
    assert app_config()["verification_policy_version"] in ("v4.1", "v4.2")


@pytest.mark.parametrize("cite", ["[MOSHI:3.3.1:c030]", "【MOSHI:3.3.1:c030】"])
def test_valid_retrieved_citation_accepted_in_both_formats(cite):
    r = run(f"Mimi uses 8 quantizers {cite}.", CLAIM)
    assert r.ok and r.cited() == ["MOSHI:3.3.1:c030"]


@pytest.mark.parametrize("cite", ["[AIAYN:3.2.2:c014]", "【AIAYN:3.2.2:c014】"])
def test_existing_but_not_retrieved_chunk_rejected_in_both_formats(cite):
    r = run(f"Mimi uses 8 quantizers {cite}.", CLAIM)
    assert not r.ok and any("was not retrieved" in why for _, why in r.failures())


@pytest.mark.parametrize("cite", ["[MOSHI:9.9:c999]", "【MOSHI:9.9:c999】"])
def test_nonexistent_chunk_rejected_in_both_formats(cite):
    r = run(f"Mimi uses 8 quantizers {cite}.", CLAIM)
    assert not r.ok and any("does not exist" in why for _, why in r.failures())


@pytest.mark.parametrize("cite", ["[MOSHI-3.3.1]", "【MOSHI-3.3.1】", "【MOSHI:3.3.1:c030†L1-L3】", "【source】"])
def test_malformed_ids_are_not_citations_in_either_format(cite):
    r = run(f"Mimi uses 8 quantizers {cite}.", CLAIM)
    assert not r.ok and r.cited() == [] and any(why == "no citation" for _, why in r.failures())


@pytest.mark.parametrize("cite", ["[MOSHI:3.3.1:c030]", "【MOSHI:3.3.1:c030】"])
def test_unsupported_claim_still_fails_entailment_in_both_formats(cite):
    r = run(f"Mimi runs at 2.4 kbps {cite}.", CLAIM)
    assert not r.ok


def test_mixed_bracket_with_one_unretrieved_id_is_rejected():
    r = run("Mimi uses 8 quantizers 【MOSHI:3.3.1:c030; AIAYN:3.2.2:c014】.", CLAIM)
    assert not r.ok and any("AIAYN:3.2.2:c014, which was not retrieved" in why for _, why in r.failures())


def test_normalize_only_touches_chunk_id_brackets():
    assert normalize_brackets("a 【MOSHI:3.3.1:c030, AIAYN:3.2.2:c014】.") == "a [MOSHI:3.3.1:c030, AIAYN:3.2.2:c014]."
    for text in ("a 【note】.", "a 【MOSHI:3.3.1:c030†L1】.", "no brackets", "[0, 1, 2]"):
        assert normalize_brackets(text) == text


def test_pipeline_answers_with_normalized_citations():
    llm = FakeLLM("Mimi uses 8 quantizers 【MOSHI:3.3.1:c030】.")
    r = answer_from_evidence("How many quantizers?", ev(P), llm, verifier(CLAIM), CORPUS)
    assert (r.status, r.text, r.citations, r.llm_calls) == \
        ("answered", "Mimi uses 8 quantizers [MOSHI:3.3.1:c030].", ["MOSHI:3.3.1:c030"], 1)


# ---- L2: a coarse clock (Windows monotonic ticks are 15.625 ms) and deadline-bounded waiting
TICK = 0.015625


class CoarseTime:
    """True time advanced only by sleep(); the governor sees it rounded down to a clock tick."""

    def __init__(self, start=0.0):
        self.true, self.sleeps = start, []

    def clock(self):
        return math.floor(self.true / TICK) * TICK

    def sleep(self, s):
        self.sleeps.append(s)
        self.true += s


def short_governor(t: CoarseTime, missing_tokens: float) -> Governor:
    gov = Governor(LIMITS, clock=t.clock)
    tpm = gov.buckets["m"]["tpm"]
    tpm.take(tpm.capacity - (RESERVE - missing_tokens), t.clock())
    return gov


RESERVE = 45 + 770  # tokens_in_local for one "q" message (+ overhead 40) and the output reserve, as HostedLLM computes


def test_root_cause_sleeping_the_estimate_once_can_leave_capacity_short():
    """What the old code did: one sleep of the estimate, then a strict check. 134 missing tokens = 1.005 s, which the
    coarse clock reads as 1.0 s after sleeping, so the bucket is still a few tokens short."""
    t = CoarseTime()
    gov = short_governor(t, 134)
    w = gov.wait_estimate("m", RESERVE)
    t.sleep(w)
    assert gov.try_reserve("m", RESERVE) is None


def test_hosted_llm_keeps_waiting_and_succeeds_on_a_coarse_clock(tmp_path):
    t = CoarseTime()
    gov = short_governor(t, 134)
    llm = HostedLLM(client([OK], tmp_path), gov, MODEL, 770, request_overhead=40, max_wait_s=30, sleep=t.sleep)
    assert llm([{"role": "user", "content": "q"}]) == "Mimi runs at 12.5Hz."
    assert len(t.sleeps) >= 2 and all(s >= MIN_WAIT_S for s in t.sleeps[1:])  # re-checked instead of rejecting


def test_waiting_stops_at_the_requests_allowance(tmp_path):
    t = CoarseTime()
    gov = short_governor(t, 134)  # ~1 s needed
    llm = HostedLLM(client([OK], tmp_path), gov, MODEL, 770, request_overhead=40, max_wait_s=0.5, sleep=t.sleep)
    with pytest.raises(GovernorReject, match="estimated wait"):
        llm([{"role": "user", "content": "q"}])
    assert t.sleeps == []


def test_waiting_cannot_spin_or_overrun_the_deadline(tmp_path):
    """Capacity that never comes back (another consumer keeps taking it): the loop gives up within the allowance,
    every sleep is at least MIN_WAIT_S, so the number of passes is bounded."""
    t = CoarseTime()
    gov = short_governor(t, 134)
    real_estimate = gov.wait_estimate
    gov.try_reserve = lambda model, tokens: None  # never fits
    gov.wait_estimate = lambda model, tokens: min(real_estimate(model, tokens), 0.0)  # and claims 0 s each time
    llm = HostedLLM(client([OK], tmp_path), gov, MODEL, 770, request_overhead=40, max_wait_s=2.0, sleep=t.sleep,
                    deadline=time.monotonic() + 60)
    with pytest.raises(GovernorReject, match="no capacity after waiting"):
        llm([{"role": "user", "content": "q"}])
    assert sum(t.sleeps) <= 2.0 and len(t.sleeps) <= 2.0 / MIN_WAIT_S and min(t.sleeps) >= MIN_WAIT_S


def test_waiting_respects_a_deadline_shorter_than_max_wait(tmp_path):
    t = CoarseTime()
    gov = short_governor(t, 134)
    gov.try_reserve = lambda model, tokens: None
    gov.wait_estimate = lambda model, tokens: 0.0
    llm = HostedLLM(client([OK], tmp_path), gov, MODEL, 770, request_overhead=40, max_wait_s=60, sleep=t.sleep,
                    deadline=time.monotonic() + 0.3)
    with pytest.raises(GovernorReject):
        llm([{"role": "user", "content": "q"}])
    assert sum(t.sleeps) <= 0.3 + 1e-9


def test_governor_default_clock_is_high_resolution():
    assert Governor(LIMITS).clock is time.perf_counter
    assert time.get_clock_info("perf_counter").resolution < 1e-3


def test_concurrent_reservations_never_oversubscribe_the_token_bucket():
    """Phase 10C: 16 threads reserve a typical call (2,700 tokens) at once from an 8,000-TPM bucket; reservations
    are atomic under the governor lock, so exactly 2 fit (3 x 2,700 > 8,000) and the rest must wait."""
    import threading
    gov = Governor(LIMITS, clock=lambda: 0.0)  # frozen clock: no refill during the race
    barrier, got = threading.Barrier(16), []

    def worker():
        barrier.wait()
        got.append(gov.try_reserve("m", 2700))
    threads = [threading.Thread(target=worker) for _ in range(16)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert sum(r is not None for r in got) == 2
    assert gov.buckets["m"]["tpm"].available(0.0) == pytest.approx(8000 - 2 * 2700)
