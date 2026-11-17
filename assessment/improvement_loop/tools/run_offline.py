"""Run every offline suite (same list and parsers as experiments/phase11f/run_tests.py) and write the results to the
path given (never to the phase 11F file). usage: run_offline.py <out.json>"""
import json
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, "experiments/phase11f")
from run_tests import ROOT, RUNS, counts  # noqa: E402

EXTRA = [("phase12", ["uv", "run", "python", "-m", "py_compile", "experiments/phase12/run_phase12.py",
                      "experiments/phase12/analyse_phase12.py"], ".")]


def main(out: Path) -> None:
    res = []
    for name, cmd, cwd in RUNS + EXTRA + [tuple(x) for x in json.loads(sys.argv[2])] if len(sys.argv) > 2 else RUNS + EXTRA:
        p = subprocess.run(cmd, cwd=ROOT / cwd, capture_output=True, text=True, encoding="utf-8", errors="replace",
                           timeout=1800)
        text = (p.stdout or "") + (p.stderr or "")
        keep = [l for l in text.splitlines() if not re.search(r"Warning|warn\(|Loading weights|Batches", l)]
        res.append({"name": name, "command": " ".join(cmd), "cwd": cwd, "exit_status": p.returncode,
                    "counts": counts(name, "\n".join(keep[-3:]) if not name.startswith("go") and "integration" not in name
                                     else "\n".join(keep)), "output_tail": keep[-4:]})
        print(name, p.returncode, res[-1]["counts"], flush=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"run_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "results": res,
                               "all_exit_zero": all(r["exit_status"] == 0 for r in res)}, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main(Path(sys.argv[1]))
