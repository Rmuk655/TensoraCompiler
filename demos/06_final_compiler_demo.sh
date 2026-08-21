#!/usr/bin/env bash
# The final compiler: source text all the way to a running or linkable
# result, across every workload this phase checks. You can check each
# --run result by hand against all-ones synthetic input data (see the
# header comment in each examples/*.tns file for the math).
set -euo pipefail
cd "$(dirname "$0")/.."
BUILD="${1:-build}"
OUT="${2:-/tmp/tensora_demo_objs}"
mkdir -p "$OUT"

run_one() {
  local file="$1" expected="$2"
  echo "--- $file (expect element[0] = $expected) ---"
  "$BUILD/tensora" --run "$file"
}

echo "=== MLP (linear -> relu -> linear) ==="
run_one examples/mlp_concrete.tns 145

echo
echo "=== Matmul, large enough to trigger Level-2 tiling (256x256x256) ==="
run_one examples/matmul_large.tns 256

echo
echo "=== CNN classifier (conv2d -> relu -> flatten -> linear -> softmax) ==="
run_one examples/cnn.tns 0.5

echo
echo "=== Single-head scaled dot-product attention ==="
run_one examples/attention.tns 1

echo
echo "=== Ahead-of-time native object files (no JIT, no MLIR/LLVM at link time) ==="
for f in mlp_concrete matmul_large cnn attention; do
  "$BUILD/tensora" -o "$OUT/$f.o" "examples/$f.tns"
  file "$OUT/$f.o"
done
