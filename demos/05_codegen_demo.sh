#!/usr/bin/env bash
# Code generation demo: Typed Computation Graph -> MLIR -> LLVM IR.
# Requires a concrete-shaped source file (see README.md).
set -euo pipefail
cd "$(dirname "$0")/.."
FILE="${1:-examples/mlp_concrete.tns}"
BUILD="${2:-build}"

echo "=== Codegen demo: $FILE ==="
echo "--- MLIR (tensora --emit-mlir; Level-1 passes + Level-2 tiling already applied) ---"
"$BUILD/tensora" --emit-mlir "$FILE"
echo
echo "--- LLVM IR (tensora --emit-llvm) ---"
set +o pipefail
"$BUILD/tensora" --emit-llvm "$FILE" | head -20
set -o pipefail
echo "... (truncated; run 'tensora --emit-llvm $FILE' directly for the full module)"
