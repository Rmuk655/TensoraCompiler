#!/usr/bin/env bash
# Parser phase demo: token stream -> AST.
set -euo pipefail
cd "$(dirname "$0")/.."
FILE="${1:-examples/mlp_concrete.tns}"
BUILD="${2:-build}"

echo "=== Parser demo: $FILE ==="
echo "--- source ---"
cat "$FILE"
echo "--- raw ANTLR parse tree (tensora --emit-parse-tree) ---"
"$BUILD/tensora" --emit-parse-tree "$FILE"
echo "--- AST (tensora --emit-ast) ---"
"$BUILD/tensora" --emit-ast "$FILE"
echo "--- AST as Graphviz DOT (tensora --emit-ast-dot) ---"
"$BUILD/tensora" --emit-ast-dot "$FILE"
