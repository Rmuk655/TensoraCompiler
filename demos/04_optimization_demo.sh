#!/usr/bin/env bash
# Level-1 + Level-2 optimization demo. It SHOWS each project-authored
# optimization actually firing on a small program (before -> after),
# then runs the real CTest pass binaries (tests/optimization/
# test_passes.cpp) which are the authoritative correctness checks.
# The demo visualizes; the OPT-* / LOOP-* tests verify.
# (A seventh pass, constant folding, was built and then removed. See
# docs/architecture/DECISIONS.md ARCH-014.)
set -euo pipefail
cd "$(dirname "$0")/.."
BUILD="${2:-build}"
OPT="examples/optimizations"

hr() { printf '%s\n' "------------------------------------------------------------"; }

# Show a graph-level optimization: op count and op lines, before vs after.
show_graph_opt() {
  local file="$1" title="$2"
  hr; echo "$title"; echo "  program: $file"
  local before after
  before=$("$BUILD/tensora" --emit-graph "$file" 2>/dev/null | grep -c ' = ' || true)
  after=$("$BUILD/tensora" --emit-optimized-graph "$file" 2>/dev/null | grep -c ' = ' || true)
  echo "  ops BEFORE (--emit-graph):"
  "$BUILD/tensora" --emit-graph "$file" 2>/dev/null | grep ' = ' | sed 's/^/      /'
  echo "  ops AFTER  (--emit-optimized-graph):"
  "$BUILD/tensora" --emit-optimized-graph "$file" 2>/dev/null | grep ' = ' | sed 's/^/      /' || echo "      (none -- fully eliminated)"
  echo "  => op count $before -> $after"
  echo
}

echo "=== Level-1 graph optimizations: SEE them fire (before -> after) ==="
echo
show_graph_opt "$OPT/dead_code.tns"       "DeadOpElimination: an unused op is dropped"
show_graph_opt "$OPT/transpose_cancel.tns" "TransposeCancellation: inverse transposes cancel"
show_graph_opt "$OPT/reshape_cancel.tns"   "ReshapeCancellation: a reshape undone by its inverse cancels"
show_graph_opt "$OPT/shape_fold.tns"       "ShapeFolding: a no-op reshape is folded away"

hr
echo "LinearBiasActivationFusion: linear+bias+relu fuse into ONE pass."
echo "  Visible in MLIR -- one linalg.generic does arith.addf (bias) then"
echo "  arith.maximumf (relu) over the matmul result:"
"$BUILD/tensora" --emit-mlir "$OPT/linear_bias_activation.tns" 2>/dev/null \
  | grep -E "linalg.matmul|linalg.generic|arith.addf|arith.maximumf" | sed 's/^/      /'
echo
hr
echo "ElementwiseFusion: a chain of elementwise ops is tagged as one fusable"
echo "  group (a graph fact consumed at lowering); verified by OPT-EF below."
echo

echo "=== Level-2: shape-guided matmul tiling -- SEE it fire (small vs large) ==="
hr
sm=$(printf 'input a: Tensor[m=8,k=8]\ninput b: Tensor[k=8,n=8]\ngraph g { c = a @ b\n return c }\n')
echo "$sm" > /tmp/tensora_small_mm.tns
echo "  8x8 matmul (below tiling threshold): scf.for loops in MLIR = $("$BUILD/tensora" --emit-mlir /tmp/tensora_small_mm.tns 2>/dev/null | grep -c scf.for)"
echo "  256x256 matmul (examples/matmul_large.tns, above threshold): scf.for loops = $("$BUILD/tensora" --emit-mlir examples/matmul_large.tns 2>/dev/null | grep -c scf.for)"
echo "  (measured speedup: run 'make benchmark')"
echo

echo "=== Authoritative correctness checks (the real pass tests) ==="
echo "(each line: [PASS] <case id>: <what it demonstrates>)"
for case_id in \
  OPT-SF-001 OPT-SF-002 OPT-DCE-001 OPT-DCE-002 OPT-RC-001 OPT-RC-003 \
  OPT-TC-001 OPT-TC-002 OPT-EF-001 OPT-EF-002 OPT-LBA-001 OPT-LBA-002; do
  "$BUILD/test_passes" "$case_id"
done
"$BUILD/test_tiling_decision" LOOP-001-A
"$BUILD/test_tiling_decision" LOOP-001-B
"$BUILD/test_codegen_e2e" LOOP-002
