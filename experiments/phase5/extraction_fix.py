"""Phase 5, investigation C (offline): the proposed tick/cross extraction fix, tried in memory only.

Root cause: pylatexenc renders an unknown \\ding{N} as its argument ("51", "55") and \\checkmark as "".
Proposed fix (not applied to relay/): in latex_parse.to_text, before pylatexenc, map \\ding{51}/\\ding{52} and
\\checkmark to "✓", \\ding{55}/\\ding{56} to "✗". A paper macro defined as empty (MOSHI's \\xmark) stays empty, as
in its PDF. This script re-parses the three papers with and without the fix (nothing under data/ is written) and
reports every table cell and prose paragraph that changes."""
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, ".")
from relay.retrieval import latex_parse as lp

MARKS = {"51": "✓", "52": "✓", "55": "✗", "56": "✗"}


def fix(tex: str) -> str:
    tex = re.sub(r"\\ding\s*\{\s*(\d+)\s*\}", lambda m: MARKS.get(m.group(1), m.group(0)), tex)
    return re.sub(r"\\checkmark(?![A-Za-z])", "✓", tex)


class FixedL2T:
    def __init__(self, inner):
        self.inner = inner

    def latex_to_text(self, tex, *a, **k):
        return self.inner.latex_to_text(fix(tex), *a, **k)


def parse_all(patched: bool) -> dict:
    original = lp._L2T
    lp._L2T = FixedL2T(original) if patched else original
    try:
        return {code: lp.parse_paper(code) for code in ("AIAYN", "MOSHI", "HNET")}
    finally:
        lp._L2T = original


before, after = parse_all(False), parse_all(True)
report = {"tables": [], "prose_paragraphs_changed": 0, "examples": []}
for code in before:
    tb = {t["table_id"]: t for t in before[code].tables}
    ta = {t["table_id"]: t for t in after[code].tables}
    for tid in tb:
        rb, ra = tb[tid]["rows"], ta[tid]["rows"]
        changed = sum(json.dumps(x, ensure_ascii=False) != json.dumps(y, ensure_ascii=False) for x, y in zip(rb, ra))
        if changed or len(rb) != len(ra):
            report["tables"].append({"table": f"{code}:{tid}", "caption": tb[tid]["caption"][:80], "rows_changed": changed,
                                     "example_before": rb[0], "example_after": ra[0]})
    pb, pa = before[code].paragraphs, after[code].paragraphs
    report["prose_paragraphs_changed"] += sum(x["text"] != y["text"] for x, y in zip(pb, pa))
assert all(len(before[c].tables) == len(after[c].tables) for c in before)

hnet = next(t for t in after["HNET"].tables if t["table_id"] == "T6")
report["hnet_t6_hnet_row_after"] = next(r for r in hnet["rows"] if "H-Net" in json.dumps(r))
moshi_t3 = next(t for t in after["MOSHI"].tables if "Mimi codec" in t["caption"])
report["moshi_mimi_ablation_rows_after"] = moshi_t3["rows"][:3]
out = Path(f"experiments/phase5/out/extraction_fix_{datetime.now(timezone.utc).strftime('%Y-%m-%dT%H-%M-%SZ')}.json")
out.parent.mkdir(parents=True, exist_ok=True)
with out.open("x", encoding="utf-8") as f:
    json.dump(report, f, indent=1, ensure_ascii=False)
for t in report["tables"]:
    print(f"{t['table']}: {t['rows_changed']} rows changed | {t['caption']}")
print("prose paragraphs changed:", report["prose_paragraphs_changed"])
print("HNET T6 H-Net row after fix:", report["hnet_t6_hnet_row_after"])
print("MOSHI Mimi ablation, first row after fix:", report["moshi_mimi_ablation_rows_after"][0])
print(out)
