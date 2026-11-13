#!/usr/bin/env bash
# Runs the load scenarios one after another against a running stand-in (:8001) and app (:8000).
# Each scenario starts like a fresh day: the scenario's own admin step resets the stand-in, and the app's
# governor is reset here. Stand-in only: the app refuses /admin/reset against a real provider.
#
#   bash scripts/run_scenarios.sh steady burst overload rpm_halved daily_exhausted retrieval_A retrieval_B retrieval_C
set -euo pipefail
cd "$(dirname "$0")/.."
for name in "$@"; do
  echo "=== $name $(date -u +%H:%M:%S)"
  curl -fsS -X POST http://127.0.0.1:8000/admin/reset; echo
  ./.bin/loadgen.exe -scenario "config/scenarios/$name.json" -root . 2>&1 | grep -v '^ *"\|^{\|^}' || true
  sleep 10  # let in-flight work that outlived its users finish before the next reset
done
echo "=== all done $(date -u +%H:%M:%S)"
