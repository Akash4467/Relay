"""Phase 11C corpus integrity audit (REAL-PAPER OFFLINE): c3 (data/processed, production, extraction e1) vs c6
(experiments/phase10/corpus/c6, extraction e4 = e2 + e3 + siunitx units, c4 chunking). Read only.

Every difference is listed and classified. Allowed (documented, intended) classes:
  e2_mark          a table cell that e1 extracted as a dingbat number (51/52/55/56) or lost entirely becomes ✓, ✗ or
                   (blank)  [Phase 6]
  e2_empty_kept    an empty table position e1 dropped is kept as "" (not shown in chunk text)  [Phase 6]
  e2_multirow      a \\multirow value carried to the rows it spans (equals a value of the same column above/below) [Phase 6]
  e3_phantom       \\phantom digits removed: 142.2->42.2, 140.3->40.3, 136.8->36.8 (MOSHI T5), 001.0->1.0 (MOSHI T6),
                   and the "x" swatches in MOSHI T18's caption  [Phase 9A]
  e4_units         a siunitx unit inserted after a number (ms, min, s)  [Phase 10C]
  rechunk          a table's rows are split over a different number of chunks because row lines changed length; the
                   table's row lines (in order) are what is compared, not chunk boundaries
Anything else is "UNEXPLAINED" and makes the audit fail. Writes experiments/phase11c/out/corpus_audit_{UTC}.json.
"""
from __future__ import annotations

import difflib
import hashlib
import json
import re
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

C3 = Path("data/processed")
C6 = Path("experiments/phase10/corpus/c6/processed")
C6_MANIFEST = Path("experiments/phase10/corpus/c6/manifest.json")
C4_MANIFEST = Path("data/corpus/c4/manifest.json")
RAW = Path("data/raw")
PAPERS = ("AIAYN", "MOSHI", "HNET")
MARKS = {"✓", "✗", "(blank)"}
PHANTOM = {("142.2", "42.2"), ("140.3", "40.3"), ("136.8", "36.8"), ("001.0", "1.0")}
UNITS = {"ms", "min", "s"}


def load(d: Path, kind: str) -> list[dict]:
    return [json.loads(l) for l in (d / f"{kind}.jsonl").open(encoding="utf-8")]


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def words(t: str) -> list[str]:
    return re.findall(r"\S+", t)


def unit_insertion_only(a: str, b: str) -> bool:
    """b = a with unit words inserted after numbers (and nothing else)."""
    sm = difflib.SequenceMatcher(a=words(a), b=words(b), autojunk=False)
    wa, wb = words(a), words(b)
    for op, i1, i2, j1, j2 in sm.get_opcodes():
        if op == "equal":
            continue
        if op == "insert":  # only unit words, each right after a word that ends in a digit
            if not all(w.rstrip(",.;") in UNITS for w in wb[j1:j2]) or j1 == 0 or not re.search(r"\d$", wb[j1 - 1]):
                return False
            continue
        if op != "replace":
            return False
        new, old = " ".join(wb[j1:j2]), " ".join(wa[i1:i2])
        stripped = re.sub(r"(?<=\d) (?:ms|min|s)\b", "", new)
        if stripped.replace(" ", "") != old.replace(" ", ""):
            return False
    return True


def phantom_caption_only(a: str, b: str) -> bool:
    """Only the T18 swatch letters removed: every word change is "x," -> "," or "x." -> "."."""
    wa, wb = words(a), words(b)
    ops = [o for o in difflib.SequenceMatcher(a=wa, b=wb, autojunk=False).get_opcodes() if o[0] != "equal"]
    return bool(ops) and all(op == "replace" and wa[i1:i2] in (["x,"], ["x."]) and wb[j1:j2] == [wa[i1][1:]]
                             for op, i1, i2, j1, j2 in ops)


def classify_cell(old: str | None, new: str, column_values: list[str]) -> str:
    if old == new:
        return "same"
    if old in (None, "") and new == "":
        return "e2_empty_kept"
    if new in MARKS and (old in (None, "", "51", "52", "55", "56") or old in MARKS):
        return "e2_mark"
    if (old, new) in PHANTOM:
        return "e3_phantom"
    if old in (None, "") and new in column_values:
        return "e2_multirow"
    return "UNEXPLAINED"


def audit() -> dict:
    out: dict = {"sources": {}, "counts": {}, "differences": [], "classes": Counter()}
    # 1. sources: the raw files both corpora were built from
    c4m = json.loads(C4_MANIFEST.read_text(encoding="utf-8"))
    c6m = json.loads(C6_MANIFEST.read_text(encoding="utf-8"))
    for code in PAPERS:
        rec = {f.name: sha(f) for f in sorted((RAW / code).glob("*")) if f.is_file()}
        recorded = {k.replace("_sha256", ""): v for k, v in c4m["sources"][code].items() if k.endswith("_sha256") and k != "src_tree_sha256"}
        out["sources"][code] = {"current": rec, "equal_to_c4_manifest": all(rec.get(k) == v for k, v in recorded.items()),
                                "c6_src_tree_equals_c4": c6m["sources"][code]["src_tree_sha256"] == c4m["sources"][code]["src_tree_sha256"]}
    # 2. sections, paragraphs, figures
    for kind, key in (("sections", lambda r: (r["paper"], r["section_id"])),
                      ("paragraphs", lambda r: (r["paper"], r["section_id"], r["para_idx"])),
                      ("figures", lambda r: (r["paper"], r.get("figure_id") or r.get("label")))):
        a, b = {key(r): r for r in load(C3, kind)}, {key(r): r for r in load(C6, kind)}
        out["counts"][kind] = {"c3": len(a), "c6": len(b), "missing_in_c6": sorted(map(str, set(a) - set(b))),
                               "new_in_c6": sorted(map(str, set(b) - set(a)))}
        for k in sorted(set(a) & set(b), key=str):
            for field in sorted(set(a[k]) | set(b[k])):
                if a[k].get(field) == b[k].get(field):
                    continue
                if field == "text" and kind == "paragraphs" and unit_insertion_only(a[k]["text"], b[k]["text"]):
                    cls = "e4_units"
                else:
                    cls = "UNEXPLAINED"
                out["differences"].append({"what": kind, "key": str(k), "field": field, "class": cls,
                                           "c3": str(a[k].get(field))[:240], "c6": str(b[k].get(field))[:240]})
                out["classes"][cls] += 1
    # 3. tables: caption, headers, rows, every cell
    ta = {(t["paper"], t["table_id"]): t for t in load(C3, "tables")}
    tb = {(t["paper"], t["table_id"]): t for t in load(C6, "tables")}
    out["counts"]["tables"] = {"c3": len(ta), "c6": len(tb), "same_ids": set(ta) == set(tb)}
    cells_checked = 0
    for k in sorted(set(ta) & set(tb)):
        a, b = ta[k], tb[k]
        for field in ("label", "section_id", "headers"):
            if a[field] != b[field]:
                out["differences"].append({"what": "table", "key": f"{k[0]}:{k[1]}", "field": field, "class": "UNEXPLAINED",
                                           "c3": a[field], "c6": b[field]})
                out["classes"]["UNEXPLAINED"] += 1
        if a["caption"] != b["caption"]:
            cls = "e3_phantom" if k == ("MOSHI", "T18") and phantom_caption_only(a["caption"], b["caption"]) else "UNEXPLAINED"
            out["differences"].append({"what": "table", "key": f"{k[0]}:{k[1]}", "field": "caption", "class": cls,
                                       "c3": a["caption"][:240], "c6": b["caption"][:240]})
            out["classes"][cls] += 1
        if len(a["rows"]) != len(b["rows"]):
            out["differences"].append({"what": "table", "key": f"{k[0]}:{k[1]}", "field": "row count",
                                       "class": "UNEXPLAINED", "c3": len(a["rows"]), "c6": len(b["rows"])})
            out["classes"]["UNEXPLAINED"] += 1
            continue
        for i, (ra, rb) in enumerate(zip(a["rows"], b["rows"])):
            if ra["row_header"] != rb["row_header"]:
                cls = "e2_mark" if ra["row_header"] == "" or rb["row_header"].startswith("Row ") else "UNEXPLAINED"
                out["differences"].append({"what": "row label", "key": f"{k[0]}:{k[1]}#{i}", "class": cls,
                                           "c3": ra["row_header"], "c6": rb["row_header"]})
                out["classes"][cls] += 1
            for col in rb["cells"]:
                cells_checked += 1
                colvals = [r["cells"].get(col, "") for r in b["rows"]]
                cls = classify_cell(ra["cells"].get(col), rb["cells"][col], colvals)
                if cls == "same":
                    continue
                out["classes"][cls] += 1
                if cls != "e2_empty_kept":  # thousands of kept empty positions: counted, not listed
                    out["differences"].append({"what": "cell", "key": f"{k[0]}:{k[1]}#{i}", "row": rb["row_header"],
                                               "column": col, "class": cls, "c3": ra["cells"].get(col), "c6": rb["cells"][col]})
            for col in ra["cells"]:
                if col not in rb["cells"]:
                    out["differences"].append({"what": "cell", "key": f"{k[0]}:{k[1]}#{i}", "column": col,
                                               "class": "UNEXPLAINED", "c3": ra["cells"][col], "c6": "(column missing)"})
                    out["classes"]["UNEXPLAINED"] += 1
    out["counts"]["table_cells_checked"] = cells_checked
    # 4. chunks: IDs, duplicates, prose chunk text, table row lines per table
    for size in (256, 512):
        ca, cb = load(C3, f"chunks_{size}"), load(C6, f"chunks_{size}")
        ia, ib = [c["chunk_id"] for c in ca], [c["chunk_id"] for c in cb]
        da, db = {c["chunk_id"]: c for c in ca}, {c["chunk_id"]: c for c in cb}
        dup_text = [t for t, n in Counter(c["text"] for c in cb).items() if n > 1]
        rec = {"c3": len(ia), "c6": len(ib), "duplicate_ids": [len(ia) - len(set(ia)), len(ib) - len(set(ib))],
               "c3_ids_missing_in_c6": sorted(set(ia) - set(ib)), "new_ids_in_c6": sorted(set(ib) - set(ia)),
               "duplicate_texts_c6": len(dup_text),
               "prose_ids_identical": [c for c in ia if da[c]["kind"] != "table"] == [c for c in ib if db[c]["kind"] != "table"]}
        for cid in ia:
            if da[cid]["kind"] == "table" or da[cid]["text"] == db[cid]["text"]:
                continue
            if unit_insertion_only(da[cid]["text"], db[cid]["text"]):
                cls = "e4_units"
            elif phantom_caption_only(da[cid]["text"], db[cid]["text"]):
                cls = "e3_phantom"
            else:
                cls = "UNEXPLAINED"
            out["differences"].append({"what": f"chunk_{size}", "key": cid, "class": cls,
                                       "c3": da[cid]["text"][:200], "c6": db[cid]["text"][:200]})
            out["classes"][cls] += 1
        # table chunks: same caption+columns header per chunk, and the table's row labels in the same order
        for k in sorted(set(ta)):
            pa = [c for c in ca if c["kind"] == "table" and (c["paper"], c.get("table_id")) == k]
            pb = [c for c in cb if c["kind"] == "table" and (c["paper"], c.get("table_id")) == k]
            la = [r["configuration"] for c in pa for r in c.get("rows") or []]
            lb = [r["configuration"] for c in pb for r in c.get("rows") or []]
            if len(pa) != len(pb):
                out["classes"]["rechunk"] += 1
                out["differences"].append({"what": f"table_chunks_{size}", "key": f"{k[0]}:{k[1]}", "class": "rechunk",
                                           "c3": [c["chunk_id"] for c in pa], "c6": [c["chunk_id"] for c in pb]})
            if len(la) != len(lb) or any(x != y and not (x == "" or y.startswith("Row ")) for x, y in zip(la, lb)):
                out["differences"].append({"what": f"table_chunks_{size}", "key": f"{k[0]}:{k[1]}", "class": "UNEXPLAINED",
                                           "c3": la, "c6": lb})
                out["classes"]["UNEXPLAINED"] += 1
            # every structured row value in a c6 chunk equals the table record (values never detached)
            rows_b = {}
            for r in tb[k]["rows"]:
                rows_b.setdefault(r["row_header"], []).append(r["cells"])
            for c in pb:
                for r in c.get("rows") or []:
                    vals = {v["column"]: v["value"] for v in r["values"]}
                    if not any(all(cells.get(col) == v for col, v in vals.items()) for cells in rows_b.get(r["configuration"], [])):
                        out["differences"].append({"what": f"table_chunks_{size}", "key": c["chunk_id"], "class": "UNEXPLAINED",
                                                   "c3": None, "c6": f"row {r['configuration']} not in the table record: {vals}"})
                        out["classes"]["UNEXPLAINED"] += 1
        out["counts"][f"chunks_{size}"] = rec
    out["classes"] = dict(out["classes"])
    out["unexplained"] = [d for d in out["differences"] if d["class"] == "UNEXPLAINED"]
    out["passed"] = (not out["unexplained"] and all(v["equal_to_c4_manifest"] and v["c6_src_tree_equals_c4"]
                                                     for v in out["sources"].values()))
    return out


def main() -> int:
    r = audit()
    Path("experiments/phase11c/out").mkdir(parents=True, exist_ok=True)
    p = Path(f"experiments/phase11c/out/corpus_audit_{datetime.now(timezone.utc).strftime('%Y-%m-%dT%H-%M-%SZ')}.json")
    p.write_text(json.dumps(r, indent=1, ensure_ascii=False, default=str), encoding="utf-8")
    print("sources:", {k: (v["equal_to_c4_manifest"], v["c6_src_tree_equals_c4"]) for k, v in r["sources"].items()})
    print("counts:", json.dumps(r["counts"], ensure_ascii=False)[:900])
    print("classes:", r["classes"])
    print("unexplained:", len(r["unexplained"]))
    for d in r["unexplained"][:15]:
        print("  ", json.dumps(d, ensure_ascii=False)[:300])
    print("PASSED" if r["passed"] else "FAILED", p.as_posix())
    return 0 if r["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
