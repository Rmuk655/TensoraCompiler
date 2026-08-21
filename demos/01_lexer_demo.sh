#!/usr/bin/env bash
# Lexer phase demo: source text -> token stream.
set -euo pipefail
cd "$(dirname "$0")/.."
FILE="${1:-examples/mlp_concrete.tns}"
BUILD="${2:-build}"

echo "=== Lexer demo: $FILE ==="
echo "--- source ---"
cat "$FILE"
echo "--- tokens (tensora --emit-tokens) ---"
"$BUILD/tensora" --emit-tokens "$FILE"
