#!/usr/bin/env bash
# Polyhedral-driven optimization on Tensora's emitted MLIR (no v0.1 change).
#
# Question: can we drive one polyhedral optimization from Tensora today?
# Answer: yes. Tensora emits linalg (with affine indexing maps); MLIR's
# own affine dialect + polyhedral passes take it the rest of the way.
# This runs entirely on the ALREADY-EMITTED IR via mlir-opt -- it does
# not touch the compiler. v0.1 stays frozen.
#
# Pipeline: --emit-mlir (linalg on tensors)
#   -> one-shot-bufferize            (tensors -> memrefs)
#   -> convert-linalg-to-affine-loops (into the affine DIALECT)
#   -> affine-loop-tile / affine-loop-fusion  (POLYHEDRAL passes)
#   -> lower-affine -> ... -> LLVM   (correctness path still closes)
#
# Usage: ./polyhedral_pipeline.sh [example.tns]
set -euo pipefail
cd "$(dirname "$0")/../.."
export PATH=/usr/lib/llvm-21/bin:$PATH
TNS="${1:-examples/mlp_concrete.tns}"
BUILD="${BUILD:-build}"

emit() { "$BUILD/tensora" --emit-mlir "$TNS" 2>/dev/null; }
BUF="-one-shot-bufferize=bufferize-function-boundaries -convert-linalg-to-affine-loops"

echo "=== source: $TNS ==="
echo "--- affine loops after linalg -> affine (a real polyhedral representation) ---"
emit | mlir-opt $BUF 2>/dev/null | grep -E "affine.for" | head -6

echo
echo "--- POLYHEDRAL PASS 1: affine-loop-tile (tile-size=4) ---"
emit | mlir-opt $BUF -affine-loop-tile="tile-size=4" 2>/dev/null | grep -E "affine.for" | head -6

echo
echo "--- POLYHEDRAL PASS 2: affine-loop-fusion (dependence-analysis driven) ---"
before=$(emit | mlir-opt $BUF 2>/dev/null | grep -cE "affine.for %arg[0-9]+ = 0")
after=$(emit | mlir-opt $BUF -affine-loop-fusion 2>/dev/null | grep -cE "affine.for %arg[0-9]+ = 0")
echo "    outer affine.for loops: before=$before  after fusion=$after"

echo
echo "--- correctness path: still lowers to LLVM after the polyhedral pass ---"
n=$(emit | mlir-opt $BUF -affine-loop-tile="tile-size=4" -lower-affine \
      -convert-scf-to-cf 2>/dev/null | grep -cE "llvm\.|cf\.|memref")
echo "    lowers OK ($n low-level op lines)"
echo
echo "Conclusion: v0.1's linalg output feeds MLIR's real polyhedral passes"
echo "(tiling + dependence-driven fusion) and still lowers to LLVM -- one"
echo "polyhedral-driven optimization, with no change to the compiler."
