#!/usr/bin/env bash
# Independent-pipeline demo. Shows that Tensora emits STANDARD MLIR and
# LLVM IR that the upstream LLVM tools consume on their own -- a
# completely different IR world than PyTorch's (whose FX/TorchScript/
# Inductor graphs are NOT accepted by mlir-opt/llc), yet the same math.
# It runs a real-data matmul through Tensora and then hands Tensora's
# emitted IR to the neutral upstream tools.
set -euo pipefail
cd "$(dirname "$0")/.."
BUILD="${1:-build}"
LLVM_BIN="${LLVM_BIN:-/usr/lib/llvm-21/bin}"
PATH="$LLVM_BIN:$PATH"
MM=tests/independent/matmul.tns
DATA=tests/independent/matmul
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

echo "=== Tensora independent-pipeline demo ==="
echo

echo "1) Tensora runs a REAL-data matmul (varied values, not all-ones):"
"$BUILD/tensora" --run "$MM" --data-dir "$DATA" | sed 's/^/     /'
echo "   (hand-check: a=[[1,2,3,4],[5,6,7,8]] @ b=[[1,0,0],[0,1,0],[0,0,1],[1,1,1]] -> c[0,0]=5)"
echo

echo "2) Tensora emits standard MLIR; upstream mlir-opt accepts + transforms it:"
if command -v mlir-opt >/dev/null 2>&1; then
  "$BUILD/tensora" --emit-mlir "$MM" 2>/dev/null > "$TMP/mm.mlir"
  mlir-opt "$TMP/mm.mlir" -o /dev/null && echo "     mlir-opt: parsed + verified OK (valid standard MLIR)"
  mlir-opt "$TMP/mm.mlir" \
    --one-shot-bufferize="bufferize-function-boundaries function-boundary-type-conversion=identity-layout-map" \
    --convert-linalg-to-loops --convert-scf-to-cf --expand-strided-metadata --lower-affine \
    --finalize-memref-to-llvm --convert-func-to-llvm --convert-arith-to-llvm \
    --convert-cf-to-llvm --reconcile-unrealized-casts 2>/dev/null | grep -q "llvm.func @mm" \
    && echo "     mlir-opt: fully lowered linalg -> LLVM dialect, upstream (llvm.func @mm)"
else
  echo "     (mlir-opt not found; install LLVM 21 tools to see this)"
fi
echo

echo "3) Tensora emits standard LLVM IR; upstream llvm tools accept it:"
if command -v llvm-as >/dev/null 2>&1; then
  "$BUILD/tensora" --emit-llvm "$MM" 2>/dev/null > "$TMP/mm.ll"
  llvm-as "$TMP/mm.ll" -o /dev/null && echo "     llvm-as: assembled OK (valid standard LLVM IR)"
  if command -v opt >/dev/null 2>&1; then
    opt -S -O2 "$TMP/mm.ll" -o /dev/null 2>/dev/null && echo "     opt -O2: optimized OK"
  fi
else
  echo "     (llvm-as not found; install LLVM 21 tools to see this)"
fi
echo
echo "PyTorch's internal graph is not consumable by mlir-opt/llc; Tensora's is."
echo "Same math, independently re-processable IR. (Numeric agreement with"
echo "PyTorch on real trained weights is shown by the make demo-* demos.)"
