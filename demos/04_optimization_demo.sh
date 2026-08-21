#!/usr/bin/env bash
# Level-1 graph optimization demo. It shows each of the 6 team-authored
# passes: one case where the pass changes the graph, and one near-miss
# case where it does not. This runs through the real CTest binary
# (tests/optimization/test_passes.cpp). It is the real pass code under
# test, not a copy of it.
# (A seventh pass, constant folding, was built and then removed. See
# docs/architecture/DECISIONS.md ARCH-014. The team could not prove it
# always keeps RM 3.21's exact output rule, because IEEE-754 f32 math
# is not associative.)
set -euo pipefail
cd "$(dirname "$0")/.."
FILE="${1:-examples/mlp_concrete.tns}"
BUILD="${2:-build}"

echo "=== Optimization demo: $FILE ==="
echo "--- Typed Computation Graph after Level-1 passes (tensora --emit-optimized-graph) ---"
echo "(compare against 03_semantic_demo.sh's --emit-graph output: same graph here since"
echo " this example has nothing to fold or eliminate; the pass tests below prove each"
echo " pass on inputs built to trigger it)"
"$BUILD/tensora" --emit-optimized-graph "$FILE"
echo
echo "=== Level-1 optimization pass demo ==="
echo "(each line: [PASS] <case id>: <what it demonstrates> (<checks>))"
echo
for case_id in \
  OPT-SF-001 OPT-SF-002 \
  OPT-DCE-001 OPT-DCE-002 \
  OPT-RC-001 OPT-RC-003 \
  OPT-TC-001 OPT-TC-002 \
  OPT-EF-001 OPT-EF-002 \
  OPT-LBA-001 OPT-LBA-002; do
  "$BUILD/test_passes" "$case_id"
done

echo
echo "=== Level-2: project-authored MLIR tiling pass (LOOP-001/002/003/004) ==="
"$BUILD/test_tiling_decision" LOOP-001-A
"$BUILD/test_tiling_decision" LOOP-001-B
"$BUILD/test_tiling_e2e"
