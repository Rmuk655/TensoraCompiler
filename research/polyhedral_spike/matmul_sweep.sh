#!/usr/bin/env bash
# Polyhedral perf sweep on a hard 1024^3 matmul (branch only).
# Measures KERNEL GFLOP/s (rtclock-timed loop in matmul_bench.mlir) for the
# default lowering vs a range of affine (polyhedral) schedules: tiling at
# several sizes, tiling + super-vectorization, tiling + unroll-jam. The
# harness is identical for every config; only the mlir-opt pipeline differs.
set -uo pipefail
cd "$(dirname "$0")"
export PATH=/usr/lib/llvm-21/bin:$PATH
RU=/usr/lib/llvm-21/lib/libmlir_runner_utils.so
CU=/usr/lib/llvm-21/lib/libmlir_c_runner_utils.so
ITERS="${ITERS:-5}"
SRC=/tmp/mm_bench.mlir
sed "s/arith.constant 20 : index/arith.constant $ITERS : index/" matmul_bench.mlir > "$SRC"

# Shared tail: affine/vector -> LLVM. convert-vector-to-scf and
# convert-ub-to-llvm are no-ops for the non-vector configs, but are needed
# once super-vectorize introduces vector ops and a ub.poison init value.
TO_LLVM="-convert-vector-to-scf -lower-affine -expand-strided-metadata -convert-scf-to-cf -finalize-memref-to-llvm -convert-vector-to-llvm -convert-func-to-llvm -convert-arith-to-llvm -convert-index-to-llvm -convert-ub-to-llvm -convert-cf-to-llvm -reconcile-unrealized-casts"

run() {  # $1 = label, $2 = pipeline before TO_LLVM
  local label="$1" pipe="$2" dt
  dt=$(mlir-opt "$SRC" $pipe $TO_LLVM 2>/tmp/o.err | \
       mlir-runner -e main -entry-point-result=void -shared-libs=$RU,$CU 2>/tmp/r.err)
  if [ -z "$dt" ]; then printf "  %-40s FAILED (%s)\n" "$label" "$(tail -1 /tmp/o.err /tmp/r.err | tr '\n' ' ' | cut -c1-70)"; return; fi
  python3 -c "dt=$dt; print(f'  {\"$label\":<40} {2*1024**3*$ITERS/dt/1e9:6.2f} GFLOP/s   (dt={dt:.2f}s/$ITERS)')"
}

echo "=== 1024^3 matmul, $ITERS iters, kernel-timed (rtclock) ==="
run "default (convert-linalg-to-loops)"        "-convert-linalg-to-loops"
for ts in 16 32 64 128 256; do
  run "affine tile=$ts"                          "-convert-linalg-to-affine-loops -affine-loop-tile=tile-size=$ts"
done
for cfg in "16 8" "16 16" "32 16" "64 16"; do
  set -- $cfg
  run "affine tile=$1 + super-vectorize=$2"      "-convert-linalg-to-affine-loops -affine-loop-tile=tile-size=$1 -affine-super-vectorize=virtual-vector-size=$2"
done
