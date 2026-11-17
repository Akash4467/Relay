"""Phase 11F: run every permitted offline suite and record the exact command, exit status and the counts parsed from
its own output (nothing inferred). Raw output tails kept. Writes experiments/phase11f/test_results.json."""
import json
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RUNS = [
    ("phase11e", ["uv", "run", "pytest", "-q", "experiments/phase11e"], "."),
    ("phase11a", ["uv", "run", "pytest", "-q", "experiments/phase11"], "."),
    ("phase11b", ["uv", "run", "pytest", "-q", "experiments/phase11b"], "."),
    ("phase11c", ["uv", "run", "pytest", "-q", "experiments/phase11c"], "."),
    ("phase10", ["uv", "run", "pytest", "-q", "experiments/phase10"], "."),
    ("phase9b", ["uv", "run", "pytest", "-q", "experiments/phase9b"], "."),
    ("full_python_suite", ["uv", "run", "pytest", "-q", "tests/unit", "tests/integration"], "."),
    ("phase10_offline_integration", ["uv", "run", "python", "experiments/phase10/offline_integration.py"], "."),
    ("go_vet_standin", ["go", "vet", "./..."], "standin"),
    ("go_test_standin", ["go", "test", "-count=1", "./..."], "standin"),
    ("go_vet_loadgen", ["go", "vet", "./..."], "loadgen"),
    ("go_test_loadgen", ["go", "test", "-count=1", "./..."], "loadgen"),
]


def counts(name: str, text: str) -> dict:
    out = {}
    for k in ("passed", "failed", "skipped", "errors", "error", "warnings", "warning"):
        m = re.search(rf"(\d+) {k}\b", text.splitlines()[-1] if text.strip() else "")
        if m:
            out[k] = int(m.group(1))
    m = re.search(r"passed (\d+) of (\d+)", text)
    if m:
        out |= {"scenarios_passed": int(m.group(1)), "scenarios": int(m.group(2))}
    if name.startswith("go_test"):
        out["packages_ok"] = len(re.findall(r"^ok\s", text, re.M))
        out["packages_failed"] = len(re.findall(r"^FAIL\s", text, re.M))
        out["packages_without_tests"] = len(re.findall(r"no test files", text))
    return out


def main():
    res = []
    for name, cmd, cwd in RUNS:
        p = subprocess.run(cmd, cwd=ROOT / cwd, capture_output=True, text=True, encoding="utf-8", errors="replace",
                           env=None, timeout=1800)
        text = (p.stdout or "") + (p.stderr or "")
        keep = [l for l in text.splitlines() if not re.search(r"Warning|warn\(|Loading weights|Batches", l)]
        res.append({"name": name, "command": " ".join(cmd), "cwd": cwd, "exit_status": p.returncode,
                    "counts": counts(name, "\n".join(keep[-3:]) if not name.startswith("go") and "integration" not in name
                                     else "\n".join(keep)),
                    "output_tail": keep[-4:]})
        print(name, p.returncode, res[-1]["counts"])
    out = {"run_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "results": res,
           "all_exit_zero": all(r["exit_status"] == 0 for r in res)}
    (ROOT / "experiments/phase11f/test_results.json").write_text(json.dumps(out, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
