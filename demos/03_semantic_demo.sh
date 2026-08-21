#!/usr/bin/env bash
# Semantic analysis phase demo: AST -> Typed Computation Graph. It also
# shows a deliberate compile-fail example (axis-name mismatch), so you
# see the reject path too, not just the accept path.
set -euo pipefail
cd "$(dirname "$0")/.."
FILE="${1:-examples/mlp.tns}"
BUILD="${2:-build}"

echo "=== Semantic analysis demo: $FILE ==="
echo "--- check ---"
"$BUILD/tensora" --check "$FILE"
echo "--- symbol table (--emit-symbols): declarations, name resolution, RM 3.6 ---"
"$BUILD/tensora" --emit-symbols "$FILE"
echo "--- Typed Computation Graph (--emit-graph) ---"
"$BUILD/tensora" --emit-graph "$FILE"
echo "--- Typed Computation Graph as Graphviz DOT (--emit-graph-dot) ---"
"$BUILD/tensora" --emit-graph-dot "$FILE"
echo "--- annotated per RM section (--explain) ---"
"$BUILD/tensora" --explain "$FILE"

echo
echo "=== Pipeline evidence: the same program, symbols -> semantic check -> codegen ==="
CONCRETE="examples/mlp_concrete.tns"
echo "--- $CONCRETE: --emit-symbols ---"
"$BUILD/tensora" --emit-symbols "$CONCRETE"
echo "--- $CONCRETE: --check ---"
"$BUILD/tensora" --check "$CONCRETE"
echo "--- $CONCRETE: --emit-mlir (first 10 lines) ---"
"$BUILD/tensora" --emit-mlir "$CONCRETE" | head -10

echo
echo "=== Negative example: axis-name mismatch (RM 3.8) ==="
TMP="$(mktemp --suffix=.tns)"
trap 'rm -f "$TMP"' EXIT
cat > "$TMP" <<'EOF'
input a: Tensor[batch=32, feat=64]
input b: Tensor[feat=64, batch=32]

graph bad {
    z = a + b
    return z
}
EOF
cat "$TMP"
echo "--- check (expected to fail) ---"
"$BUILD/tensora" --check "$TMP" || echo "(exit code $?: rejected, as expected)"

echo
echo "=== Symbol-table error demonstration (RM 3.6) ==="

echo "--- undefined symbol: relu(does_not_exist) ---"
TMP2="$(mktemp --suffix=.tns)"
trap 'rm -f "$TMP2"' EXIT
cat > "$TMP2" <<'EOF'
input x: Tensor[batch=4, feat=8]

graph g {
    y = relu(does_not_exist)
    return y
}
EOF
"$BUILD/tensora" --emit-symbols "$TMP2" || echo "(exit code $?: undefined name reported, symbol table shown, no crash)"

echo "--- redeclaration: input x declared twice ---"
TMP3="$(mktemp --suffix=.tns)"
trap 'rm -f "$TMP3"' EXIT
cat > "$TMP3" <<'EOF'
input x: Tensor[batch=32, feat=64]
input x: Tensor[batch=32, feat=64]

graph g { return x }
EOF
"$BUILD/tensora" --emit-symbols "$TMP3" || echo "(exit code $?: redeclaration rejected, only the first x recorded)"

echo "--- valid shadowing: h = relu(x); h = relu(h) (RM 3.6 permits reassigning a local Value) ---"
TMP4="$(mktemp --suffix=.tns)"
trap 'rm -f "$TMP4"' EXIT
cat > "$TMP4" <<'EOF'
input x: Tensor[batch=4, feat=8]

graph g {
    h = relu(x)
    h = relu(h)
    return h
}
EOF
"$BUILD/tensora" --emit-symbols "$TMP4"
