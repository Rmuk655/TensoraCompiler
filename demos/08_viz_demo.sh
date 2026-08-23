#!/usr/bin/env bash
# Visualization demo. Renders Tensora's graph views to images:
#   - the AST                         (tensora --emit-ast-dot)
#   - the Typed Computation Graph     (tensora --emit-graph-dot)
#   - the graph AFTER Level-1 passes  (tensora --emit-optimized-graph-dot)
#   - the MLIR op DAG                 (mlir-opt --view-op-graph)
# The before/after TCG pair is the clearest picture of what optimization
# removed or fused. Needs Graphviz (`dot`); install with:
#   sudo apt install graphviz   (see tools/models/requirements.txt)
set -euo pipefail
cd "$(dirname "$0")/.."
BUILD="${2:-build}"
FILE="${1:-examples/optimizations/dead_code.tns}"
OUT="demos/artifacts/viz"
LLVM_BIN="${LLVM_BIN:-/usr/lib/llvm-21/bin}"
PATH="$LLVM_BIN:$PATH"

if ! command -v dot >/dev/null 2>&1; then
  echo "error: Graphviz 'dot' not found. Install it, then re-run:"
  echo "  sudo apt install graphviz"
  exit 1
fi
mkdir -p "$OUT"

render() { # <dot-text-producer-cmd...>  <out-name>  <title>
  # out-name/title are the LAST two arguments, not $2/$3 -- the producer
  # command itself can have any number of tokens (binary, flag, file, ...).
  local args=("$@") n=$#
  local title="${args[$((n - 1))]}" out="${args[$((n - 2))]}"
  local cmd=("${args[@]:0:$((n - 2))}")
  if "${cmd[@]}" 2>/dev/null > "$OUT/$out.dot" && [ -s "$OUT/$out.dot" ]; then
    dot -Tpng "$OUT/$out.dot" -o "$OUT/$out.png" 2>/dev/null && echo "  wrote $OUT/$out.png   ($title)"
  fi
}

echo "=== Tensora visualization demo: $FILE ==="
render "$BUILD/tensora" --emit-ast-dot "$FILE"             ast          "AST"
render "$BUILD/tensora" --emit-graph-dot "$FILE"           tcg_before   "Typed Computation Graph (before optimization)"
render "$BUILD/tensora" --emit-optimized-graph-dot "$FILE" tcg_after    "Typed Computation Graph (after Level-1 passes)"
if command -v mlir-opt >/dev/null 2>&1; then
  "$BUILD/tensora" --emit-mlir "$FILE" 2>/dev/null | mlir-opt --view-op-graph -o /dev/null 2>"$OUT/mlir.dot" || true
  [ -s "$OUT/mlir.dot" ] && dot -Tpng "$OUT/mlir.dot" -o "$OUT/mlir.png" 2>/dev/null && echo "  wrote $OUT/mlir.png   (MLIR op DAG)"
fi
echo "Open the PNGs in $OUT/ (tcg_before vs tcg_after shows the optimization)."
