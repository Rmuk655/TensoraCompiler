#!/usr/bin/env bash
# Drift guard: every CASE("<id>", ...) in the test sources must be
# documented in docs/testing/TEST_MATRIX.md. The case IDs live in the
# .cpp (CMake discovers them; see tensora_discover_cases in CMakeLists.txt),
# so this is the one check that keeps the test matrix from silently
# drifting away from the actual suite. Registered as the CTest
# INT-TESTMATRIX-SYNC. Linux-only, like the rest of the project (INTEG-001).
set -uo pipefail
cd "$(dirname "$0")/.."
MATRIX=docs/testing/TEST_MATRIX.md

ids=$(grep -rhoE '(CASE|TEST_CASE)\([A-Za-z0-9_]+, *"[^"]+"' tests --include='*.cpp' \
        | sed -E 's/.*"([^"]+)"/\1/' | sort -u)

missing=0
for id in $ids; do
  grep -qF -- "$id" "$MATRIX" || { echo "MISSING from $MATRIX: $id"; missing=1; }
done

n=$(printf '%s\n' "$ids" | grep -c .)
if [ "$missing" -eq 0 ]; then
  echo "OK: all $n test-case IDs from tests/*.cpp are documented in TEST_MATRIX.md"
else
  echo "TEST_MATRIX.md is out of sync with the test sources (add the rows above)."
fi
exit "$missing"
