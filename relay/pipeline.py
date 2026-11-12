"""One request end to end (DESIGN phase 2): retrieve -> structural sufficiency -> draft -> verify -> [redraft]
-> answer or refuse. Phase 3 wraps these same steps as LangGraph nodes.

The LLM is any callable taking chat messages and returning the reply text; only the draft and redraft steps
call it (CLAUDE.md hard rule 7). A structural failure refuses before any LLM call.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from relay.config import RetrievalSetting
from relay.llm import prompts
from relay.retrieval.search import Evidence, Index, search
from relay.verify import sufficiency
from relay.verify.verifier import Report, verify

LLM = Callable[[list[dict]], str]


@dataclass
class Result:
    status: str                      # "answered" | "refused"
    text: str
    citations: list[str] = field(default_factory=list)
    refusal_reason: str | None = None
    llm_calls: int = 0
    drafts: list[str] = field(default_factory=list)
    reports: list[Report] = field(default_factory=list)
    evidence: Evidence | None = None


@dataclass
class Verifier:
    nli_model: object
    entail_min: float
    entail_min_flagged: float


def _refuse(reason: str, ev: Evidence, **kw) -> Result:
    return Result("refused", prompts.refusal_text(ev.passages), refusal_reason=reason, evidence=ev, **kw)


def answer_from_evidence(question: str, ev: Evidence, llm: LLM, verifier: Verifier, corpus_ids: set[str],
                         allow_redraft: bool = True) -> Result:
    suff = sufficiency.check(question, ev.passages)
    if not suff.ok:
        return _refuse(f"insufficient evidence: {suff.reason}", ev)

    drafts, reports = [], []
    msgs = prompts.messages(question, ev.passages)
    for attempt in range(2 if allow_redraft else 1):
        draft = llm(msgs).strip()
        drafts.append(draft)
        if draft == prompts.NOT_IN_PAPERS:
            return _refuse("model: not in papers", ev, llm_calls=len(drafts), drafts=drafts, reports=reports)
        report = verify(draft, ev.passages, corpus_ids, verifier.nli_model,
                        verifier.entail_min, verifier.entail_min_flagged)
        reports.append(report)
        if report.ok:
            return Result("answered", draft, report.cited(), llm_calls=len(drafts), drafts=drafts,
                          reports=reports, evidence=ev)
        msgs = prompts.redraft_messages(question, ev.passages, draft, report.failures())
    reason = "verification failed" + (" after redraft" if len(drafts) > 1 else "")
    return _refuse(reason, ev, llm_calls=len(drafts), drafts=drafts, reports=reports)


def run(question: str, setting: RetrievalSetting, context_cap: int, index: Index, llm: LLM, verifier: Verifier,
        reranker=None, allow_redraft: bool = True) -> Result:
    ev = search(question, setting, context_cap, index, reranker)
    return answer_from_evidence(question, ev, llm, verifier, set(index.chunks), allow_redraft)
