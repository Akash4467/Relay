"""Paper-derived symbolic table-header glossary and query expansion (phase 11B; Configuration G of phase 11A).

Promoted from experiments/phase11/glossary.py with the logic unchanged (only the data path and the default corpus for
check() differ); relay/retrieval/symbolic_glossary.json is a byte-for-byte copy of experiments/phase11/glossary.json.
Every entry names its paper and table and quotes the sentence of that paper (or the table caption) that defines the
symbol; its meaning phrases are verbatim substrings of that sentence. No outside synonyms.

  expand(question)  -> {"original", "expanded", "terms", "matches"}: the original question is never changed; the
                       expanded retrieval query is the original plus "[h = 1]"-style terms. Used by search() for BM25,
                       dense retrieval and the cross-encoder when retrieval_policy.symbolic_query_expansion is on.
  annotate(chunk)   -> kept for the experiments only; Phase 11A rejected it (not used by production).

Trigger rule: an entry fires when the question contains at least min(2, n) of the n content words of its meaning
phrases within a 6-token span, and the question does not name a different paper. A number within 2 words of the
matched words (digits or one..twelve, "single") binds as "symbol = number"."""
from __future__ import annotations

import json
import re
from pathlib import Path

from relay.retrieval.paper_mentions import named_papers
from relay.verify.sentences import split_sentences

HERE = Path(__file__).parent
ENTRIES = json.loads((HERE / "symbolic_glossary.json").read_text(encoding="utf-8"))["entries"]
PROCESSED = Path("data/processed")  # production corpus (c3)
SPAN = 6  # tokens within which the matched meaning words must occur together
STOP = set("a an the of or and has have is are with in on to for by as".split())
NUMBER_WORDS = {"single": "1", "one": "1", "two": "2", "three": "3", "four": "4", "five": "5", "six": "6", "seven": "7",
                "eight": "8", "nine": "9", "ten": "10", "eleven": "11", "twelve": "12"}


def stem(w: str) -> str:
    return w[:-1] if len(w) > 3 and w.endswith("s") and not w.endswith("ss") else w


def tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+(?:\.[0-9]+)?", text.lower())


def meaning_words(entry: dict) -> set[str]:
    return {stem(w) for phrase in entry["meaning"] for w in tokens(phrase) if w not in STOP}


def check(corpus: Path = PROCESSED) -> list[str]:
    """Every problem with the glossary against the corpus (empty = verified)."""
    paras = [json.loads(l) for l in (corpus / "paragraphs.jsonl").open(encoding="utf-8")]
    tables = {(t["paper"], t["table_id"]): t for t in map(json.loads, (corpus / "tables.jsonl").open(encoding="utf-8"))}
    errs = []
    for e in ENTRIES:
        t = tables.get((e["paper"], e["table_id"]))
        if t is None or e["symbol"] not in [h.split(" / ")[0] for h in t["headers"]]:
            errs.append(f"{e['symbol']}: not a header of {e['paper']} {e['table_id']}")
        if e["source"] == "caption":
            found = t is not None and e["sentence"] in t["caption"]
        else:
            found = any(e["sentence"] == s for p in paras if p["paper"] == e["paper"] and p["section_id"] == e["section_id"]
                        for s in split_sentences(p["text"]))
        if not found:
            errs.append(f"{e['symbol']}: defining sentence not found in {e['paper']} {e.get('section_id', 'caption')}")
        if not re.search(rf"(?<![\w]){re.escape(e['symbol'])}(?![\w])", e["sentence"]):
            errs.append(f"{e['symbol']}: symbol not in its defining sentence")
        for phrase in e["meaning"]:
            if phrase not in e["sentence"]:
                errs.append(f"{e['symbol']}: meaning {phrase!r} is not verbatim in the sentence")
    return errs


def _number_near(words: list[str], positions: list[int]) -> str | None:
    for i in positions:
        for j in list(range(i - 1, max(-1, i - 3), -1)) + list(range(i + 1, min(len(words), i + 3))):
            w = words[j]
            if re.fullmatch(r"\d+(?:\.\d+)?", w):
                return w
            if w in NUMBER_WORDS:
                return NUMBER_WORDS[w]
    return None


def expand(question: str) -> dict:
    """{"original", "expanded", "terms", "matches"}; expanded == original when nothing fires."""
    words = tokens(question)
    stems = [stem(w) for w in words]
    papers = named_papers(question)
    terms, matches = [], []
    for e in ENTRIES:
        if papers and e["paper"] not in papers:
            continue
        need = meaning_words(e)
        # the meaning words must appear together: min(2, n) distinct ones within one SPAN-token window
        windows = [{s for s in stems[i:i + SPAN] if s in need} for i, s in enumerate(stems) if s in need]
        hit = max(windows, key=len, default=set())
        if len(hit) < min(2, len(need)):
            continue
        start = next(i for i, s in enumerate(stems) if s in need and hit <= set(stems[i:i + SPAN]))
        n = _number_near(words, [i for i in range(start, min(len(stems), start + SPAN)) if stems[i] in hit])
        term = f"{e['symbol']} = {n}" if n is not None else e["symbol"]
        if term not in terms:
            terms.append(term)
            matches.append({"symbol": e["symbol"], "paper": e["paper"], "table_id": e["table_id"],
                            "matched_words": sorted(hit), "number": n, "evidence": e["sentence"]})
    expanded = f"{question} [{'; '.join(terms)}]" if terms else question
    return {"original": question, "expanded": expanded, "terms": terms, "matches": matches}


def annotate(chunk: dict) -> str:
    """Approach F: the cross-encoder's view of a table chunk; non-table chunks and tables without entries unchanged."""
    if chunk.get("kind") != "table":
        return chunk["text"]
    defs = {e["symbol"]: ", ".join(e["meaning"]) for e in ENTRIES
            if e["paper"] == chunk["paper"] and e["table_id"] == chunk.get("table_id")}
    if not defs:
        return chunk["text"]
    lines = chunk["text"].split("\n")
    for k, line in enumerate(lines):
        if line.startswith("Columns: "):
            cols = [c.strip() for c in line[len("Columns: "):].split(" | ")]
            lines[k] = "Columns: " + " | ".join(f"{c} ({defs[c]})" if c in defs else c for c in cols)
    return "\n".join(lines)
