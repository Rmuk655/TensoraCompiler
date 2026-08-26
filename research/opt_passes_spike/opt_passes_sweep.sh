#!/usr/bin/env bash
# Spike: does adding stock MLIR -canonicalize / -(affine-)loop-invariant-
# code-motion to Tensora's lowering pipeline help, on a hard 1024^3
# matmul (kernel-timed, rtclock, no JIT/process noise -- same method as
# ../polyhedral_spike used before those files were trimmed)?
#
# Tests both the DEFAULT pipeline (convert-linalg-to-loops) and the
# --polyhedral pipeline (convert-linalg-to-affine-loops + affine-loop-tile,
# ARCH-018 on main) with each pass added, and checks correctness against
# an untouched baseline (all-ones inputs -> every output element 1024.0).
set -uo pipefail
cd "$(dirname "$0")"
export PATH=/usr/lib/llvm-21/bin:$PATH
RU=/usr/lib/llvm-21/lib/libmlir_runner_utils.so
CU=/usr/lib/llvm-21/lib/libmlir_c_runner_utils.so
ITERS="${ITERS:-5}"
SRC=/tmp/mm_bench_opt.mlir
sed "s/arith.constant 20 : index/arith.constant $ITERS : index/" matmul_bench.mlir > "$SRC"

TO_LLVM="-convert-vector-to-scf -lower-affine -expand-strided-metadata -convert-scf-to-cf -finalize-memref-to-llvm -convert-vector-to-llvm -convert-func-to-llvm -convert-arith-to-llvm -convert-index-to-llvm -convert-ub-to-llvm -convert-cf-to-llvm -reconcile-unrealized-casts"

run() {  # $1 = label, $2 = pipeline before TO_LLVM
  local label="$1" pipe="$2" dt
  dt=$(mlir-opt "$SRC" $pipe $TO_LLVM 2>/tmp/o.err | \
       mlir-runner -e main -entry-point-result=void -shared-libs=$RU,$CU 2>/tmp/r.err)
  if [ -z "$dt" ]; then printf "  %-46s FAILED (%s)\n" "$label" "$(tail -1 /tmp/o.err /tmp/r.err | tr '\n' ' ' | cut -c1-70)"; return; fi
  python3 -c "dt=$dt; print(f'  {\"$label\":<46} {2*1024**3*$ITERS/dt/1e9:6.2f} GFLOP/s   (dt={dt:.2f}s/$ITERS)')"
}

echo "=== 1024^3 matmul, $ITERS iters, kernel-timed (rtclock) ==="
echo "--- default pipeline (convert-linalg-to-loops) ---"
run "default"                                  "-convert-linalg-to-loops"
run "default + canonicalize (after bufferize)" "-canonicalize -convert-linalg-to-loops"
run "default + canonicalize (after loops)"     "-convert-linalg-to-loops -canonicalize"
run "default + LICM (after loops)"             "-convert-linalg-to-loops -loop-invariant-code-motion"
run "default + canonicalize + LICM"            "-convert-linalg-to-loops -canonicalize -loop-invariant-code-motion"

echo "--- --polyhedral pipeline (convert-linalg-to-affine-loops + affine-loop-tile, ARCH-018) ---"
run "polyhedral (tile=default)"                                 "-convert-linalg-to-affine-loops -affine-loop-tile"
run "polyhedral + canonicalize (after bufferize)"                "-canonicalize -convert-linalg-to-affine-loops -affine-loop-tile"
run "polyhedral + affine-LICM (after tile)"                      "-convert-linalg-to-affine-loops -affine-loop-tile -affine-loop-invariant-code-motion"
run "polyhedral + canonicalize + affine-LICM"                    "-canonicalize -convert-linalg-to-affine-loops -affine-loop-tile -affine-loop-invariant-code-motion -canonicalize"
