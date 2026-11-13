"""Tables 1, 2a, 2b and 3 from results/results.csv only (DESIGN §18.4; CLAUDE.md rule 4: never hand-edited).
Table 3b (retrieval-only coverage) comes from results/retrieval_eval.csv (scripts/make_retrieval_csv.py), which is
not per-request data and is never mixed into the request tables.

Percentiles: numpy linear interpolation; N is reported; p99 with N < 100 is marked "*" (explained under Table 2a). Times in ms,
percentages with one decimal, "not measured" where the CSV has no data for a cell. Writes results/tables.md.

    uv run python scripts/make_tables.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

CSV = Path("results/results.csv")
OUT = Path("results/tables.md")
RETRIEVAL_CSV = Path("results/retrieval_eval.csv")  # retrieval-only coverage per (report, setting); not per request
LOAD_SCENARIOS = ["real_smoke", "steady", "burst", "overload", "rpm_halved", "daily_exhausted"]
PRICES = {"openai/gpt-oss-120b": (0.15, 0.60), "openai/gpt-oss-20b": (0.075, 0.30)}  # $/1M in, out; read 2026-11-11
NM = "not measured"


def pct(values: pd.Series, q: float) -> float | None:
    v = pd.to_numeric(values, errors="coerce").dropna()
    return None if v.empty else float(np.percentile(v, q, method="linear"))


def trio(values: pd.Series) -> str:
    v = pd.to_numeric(values, errors="coerce").dropna()
    if v.empty:
        return NM
    p50, p95, p99 = (round(pct(v, q)) for q in (50, 95, 99))
    return f"{p50} / {p95} / {p99}{'*' if len(v) < 100 else ''} (N={len(v)})"


def duo(values: pd.Series) -> str:
    v = pd.to_numeric(values, errors="coerce").dropna()
    return NM if v.empty else f"{round(pct(v, 50))} / {round(pct(v, 95))} (N={len(v)})"


def share(mask: pd.Series) -> str:
    return f"{100 * mask.mean():.1f}" if len(mask) else NM


def cost_usd(df: pd.DataFrame) -> float:
    total = 0.0
    for _, r in df.iterrows():
        price_in, price_out = PRICES.get(r["model"], (0.0, 0.0)) if isinstance(r["model"], str) else (0.0, 0.0)
        total += (r["tokens_in"] * price_in + r["tokens_out"] * price_out) / 1e6
    return total


def table(header: list[str], rows: list[list]) -> str:
    out = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    out += ["| " + " | ".join("" if c is None else str(c) for c in row) + " |" for row in rows]
    return "\n".join(out)


def table1(df: pd.DataFrame) -> str:
    real = df[df["source"] == "real"]
    n = len(real)
    lim_req = pd.to_numeric(real["ratelimit_limit_requests"], errors="coerce").dropna()
    lim_tok = pd.to_numeric(real["ratelimit_limit_tokens"], errors="coerce").dropna()
    # one request below the daily limit, the reset header says how long one request takes to come back
    one_used = real[pd.to_numeric(real["ratelimit_remaining_requests"], errors="coerce")
                    == pd.to_numeric(real["ratelimit_limit_requests"], errors="coerce") - 1]
    refill = pd.to_numeric(one_used["ratelimit_reset_requests_s"], errors="coerce").dropna()
    hits = int(pd.to_numeric(real["provider_429s"], errors="coerce").fillna(0).sum())
    rows = [
        ["Requests per minute", "30 (documented)", f"Groq docs; no RPM header; not probed ({n} real calls)", "low"],
        ["Tokens per minute", f"{int(lim_tok.iloc[0])}" if not lim_tok.empty else NM,
         f"x-ratelimit-limit-tokens header, all {len(lim_tok)} responses", "high"],
        ["Requests per day", f"{int(lim_req.iloc[0])}" if not lim_req.empty else NM,
         f"x-ratelimit-limit-requests header, all {len(lim_req)} responses", "high"],
        ["Tokens per day", "200,000 (documented)", "Groq docs; no header; not probed", "low"],
        ["Requests in flight at once", NM, "no concurrency test run", "n/a"],
        ["What it returns when a limit is hit", f"not observed ({hits} HTTP 429 in {n} real calls)",
         "docs: HTTP 429 + retry-after + x-ratelimit-* headers; body never seen", "medium (docs only)"],
        ["Time until it recovers", (f"daily: one request back every {refill.iloc[0]:.1f} s (continuous)"
                                    if not refill.empty else NM),
         "x-ratelimit-reset-requests after 1 of 1,000; per-minute recovery not observed",
         "medium"],
    ]
    return table(["Limit", "Value you found", "How you found it", "Confidence (high, medium, low)"], rows)


def table2(df: pd.DataFrame) -> tuple[str, str]:
    a, b = [], []
    for name in LOAD_SCENARIOS:
        d = df[df["scenario"] == name]
        if d.empty:
            a.append([name, NM, 0, NM, NM, NM, NM])
            b.append([name] + [NM] * 7)
            continue
        source = "/".join(sorted(d["source"].unique()))
        dur = pd.to_numeric(d["scenario_duration_s"], errors="coerce").dropna()
        if dur.empty:  # real runs: span of arrivals
            times = pd.to_datetime(d["start_time"], utc=True, format="mixed")
            duration = (times.max() - times.min()).total_seconds() or None
        else:
            duration = float(dur.iloc[0])
        rate = f"{len(d) / duration * 60:.2f}" if duration else NM
        a.append([name, source, len(d), f"{duration:.0f}" if duration else NM, rate,
                  trio(d["first_token_ms"]), trio(d["total_ms"])])
        ok = d[d["status"] == "ok"]
        hosted = ok[(ok["tokens_in"] + ok["tokens_out"]) > 0]
        tokens = f"{(hosted['tokens_in'] + hosted['tokens_out']).mean():.0f} (N={len(hosted)})" if len(hosted) else NM
        peak = pd.to_numeric(d["queue_length_at_arrival"], errors="coerce").dropna()
        answers = len(ok)
        cost = f"${1000 * cost_usd(d) / answers:.2f}" if answers else NM
        b.append([name, share(d["status"] == "ok"), share(d["status"] == "rate_limited"),
                  share(d["status"] == "timeout"), f"{pd.to_numeric(d['retries']).mean():.2f}",
                  f"{int(peak.max())}" if not peak.empty else NM, tokens, cost])
    ta = table(["Scenario", "Real or stand-in", "Requests sent", "Duration (s)", "Offered rate (per min)",
                "First token p50 / p95 / p99 (ms)", "Total time p50 / p95 / p99 (ms)"], a)
    tb = table(["Scenario", "Succeeded %", "Rate-limited %", "Timed out or given up %", "Retries per request",
                "Peak queue length", "Tokens per answer", "Cost per 1,000 answers"], b)
    return ta, tb


def table3(df: pd.DataFrame) -> str:
    rows = []
    for setting, piece, passages, rerank in (("A", 256, 8, "no"), ("B", 256, 8, "yes"), ("C", 512, 4, "yes")):
        d = df[df["scenario"] == f"retrieval_{setting}"]
        graded = d[d["answer_grade"].fillna("") != ""]
        quality = [share(graded["answer_grade"] == g) for g in ("correct", "wrong", "refused_correctly")] \
            if len(graded) else [NM + "*"] * 3
        ok = d[(d["status"] == "ok") & ((d["tokens_in"] + d["tokens_out"]) > 0)]
        tokens = f"{(ok['tokens_in'] + ok['tokens_out']).mean():.0f}" if len(ok) else NM
        rows.append([f"{setting}: {piece} tokens, {passages} pieces, re-rank {rerank}", len(d), *quality, tokens,
                     duo(d["first_token_ms"]), duo(d["total_ms"])])
    return table(["Setting (piece size, pieces fetched, re-rank yes or no)", "Questions asked", "Correct and cited %",
                  "Wrong %", "Correctly refused %", "Tokens per answer", "First token p50 / p95 (ms)",
                  "Total time p50 / p95 (ms)"], rows)


def coverage(path: Path = RETRIEVAL_CSV) -> str:
    """Gold-passage coverage per setting, retrieval only (did the evidence come back), from results/retrieval_eval.csv.
    Not answer quality and not request-level data; the c3 test column is a post-rebuild diagnostic, the frozen test
    result is the c1 one. Quote recall and context tokens stay in the CSV."""
    if not path.exists():
        return NM
    df = pd.read_csv(path)
    rows = []
    for setting in ("A", "B", "C"):
        row = [setting]
        for split, corpus in (("dev", "c3"), ("test", "c3"), ("test", "c1")):
            r = df[(df["split"] == split) & (df["corpus_version"] == corpus) & (df["setting"] == setting)]
            row.append(NM if r.empty else
                       f"{r.iloc[0]['any_gold_retrieved_pct']:.1f} (N={r.iloc[0]['answerable_questions']})")
        rows.append(row)
    return (table(["Setting", "Dev, c3: any gold passage retrieved %", "Test, c3 (post-rebuild diagnostic) %",
                   "Test, c1 (frozen result) %"], rows) +
            "\n\nN = answerable questions; quote recall and context tokens in results/retrieval_eval.csv.")


def components(df: pd.DataFrame) -> str:
    """Where the time goes, per scenario: median of each component (not a required table; supports heading 4)."""
    cols = ["queue_wait_ms", "retrieval_ms", "governor_wait_ms", "first_byte_ms", "model_ttft_ms", "generation_ms",
            "verification_ms", "first_token_ms", "total_ms"]
    rows = []
    for name in LOAD_SCENARIOS + ["retrieval_A", "retrieval_B", "retrieval_C"]:
        d = df[df["scenario"] == name]
        if not d.empty:
            rows.append([name] + [("-" if pct(d[c], 50) is None else round(pct(d[c], 50))) for c in cols])
    return table(["Scenario (medians, ms)"] + cols, rows)


def tiers(df: pd.DataFrame) -> str:
    """What the users actually got. "Succeeded" in Table 2b counts refusals too; this separates them."""
    rows = []
    for name in LOAD_SCENARIOS + ["retrieval_A", "retrieval_B", "retrieval_C"]:
        d = df[df["scenario"] == name]
        if d.empty:
            continue
        tier = d["answer_tier"].astype("string").fillna("")
        rows.append([name, len(d), int((tier == "normal").sum()), int((tier == "refused").sum()),
                     int((tier == "unavailable").sum()), int((d["status"] == "timeout").sum())])
    return table(["Scenario", "Requests", "Verified answer", "Refusal", "Busy / unavailable", "Timed out or gave up"], rows)


def verification(df: pd.DataFrame) -> str:
    """NLI verification time where NLI actually ran (> 50 ms). Under 50 ms means no sentence reached NLI: the model
    said NOT_IN_PAPERS, or level 1 rejected every sentence (e.g. a fixture citing a chunk this setting did not
    retrieve)."""
    rows = []
    for name in LOAD_SCENARIOS + ["retrieval_A", "retrieval_B", "retrieval_C"]:
        d = df[df["scenario"] == name]
        v = pd.to_numeric(d["verification_ms"], errors="coerce").dropna()
        ran = v[v > 50]
        if len(v):
            rows.append([name, len(v), len(ran), duo(ran) if len(ran) else "-"])
    return table(["Scenario", "Requests that reached verification", "NLI ran", "Verification p50 / p95 (ms) when NLI ran"],
                 rows)


def main() -> str:
    df = pd.read_csv(CSV)
    for c in ("tokens_in", "tokens_out", "retries"):
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)
    ta, tb = table2(df)
    md = "\n\n".join([
        "# Results tables (generated by scripts/make_tables.py from results/results.csv; do not edit)",
        "## Table 1: Limits you found in the real service", table1(df),
        "## Table 2a: Latency, one row per load scenario", ta,
        r"\* N < 100: p99 is not a stable percentile.",
        "## Table 2b: Outcomes, same rows", tb,
        "Succeeded includes refusals. Cost = 1,000 x list-price cost of hosted-call tokens (gpt-oss-120b, $0.15 / $0.60 "
        "per 1M in / out, read 2026-11-11) / succeeded requests; actual free-tier spend $0.",
        "## Table 3: Retrieval settings", table3(df),
        r"\* Not measured: stand-in replies are fixtures, so answer quality needs real-model runs.",
        "## Table 3b: Retrieval only, gold-passage coverage (not answer quality)", coverage(),
        "## Component medians (supporting data for 'Where the time goes')", components(df),
        "## Reply tiers (supporting: what users received)", tiers(df),
        "## NLI verification time when it ran (supporting)", verification(df)])
    OUT.write_text(md + "\n", encoding="utf-8")
    return md


if __name__ == "__main__":
    print(main())
