#!/usr/bin/env bash
# Diagnose the affine-loop-fusion wrong result on cnn_stl10 (branch only).
# See FUSION_DIAGNOSIS.md for the conclusion.
#
# It reduces the failure to the smallest trigger: a single padded conv2d.
# For each tiny graph it fills every input/weight with 1.0, then compares
# the result of affine loops WITHOUT fusion (correct baseline) against
# affine loops WITH producer-consumer fusion, and against sibling fusion.
set -uo pipefail
cd "$(dirname "$0")/../.."
export PATH=/usr/lib/llvm-21/bin:$PATH
RU=/usr/lib/llvm-21/lib/libmlir_runner_utils.so
CU=/usr/lib/llvm-21/lib/libmlir_c_runner_utils.so
T=./build/tensora
D=$(mktemp -d); trap 'rm -rf "$D"' EXIT
BUF="-one-shot-bufferize=bufferize-function-boundaries"
TAIL="-convert-vector-to-scf -lower-affine -expand-strided-metadata -convert-scf-to-cf -finalize-memref-to-llvm -convert-vector-to-llvm -convert-func-to-llvm -convert-arith-to-llvm -convert-index-to-llvm -convert-ub-to-llvm -convert-cf-to-llvm -reconcile-unrealized-casts"

# generate a @main that fills all args with 1.0, calls the graph, prints elem 0
gen() { python3 - "$1" "$2" <<'PY'
import re, sys
mlir = open(sys.argv[1]).read()
m = re.search(r'func\.func @(\w+)\(([^)]*)\)\s*->\s*(tensor<[^>]+>)', mlir)
name, argstr, rty = m.group(1), m.group(2), m.group(3)
args = re.findall(r'%\w+:\s*(tensor<[^>]+>)', argstr)
L=["func.func @main() {","  %one = arith.constant 1.000000e+00 : f32"]; ns=[]
for i,t in enumerate(args):
    L += [f"  %e{i} = tensor.empty() : {t}",
          f"  %a{i} = linalg.fill ins(%one : f32) outs(%e{i} : {t}) -> {t}"]; ns.append(f"%a{i}")
L.append(f"  %r = call @{name}({', '.join(ns)}) : ({', '.join(args)}) -> {rty}")
rank = len(rty[len('tensor<'):-1].split('x')) - 1
L += ["  %c0 = arith.constant 0 : index",
      f"  %v = tensor.extract %r[{', '.join(['%c0']*rank)}] : {rty}",
      "  vector.print %v : f32","  return\n}"]
i = mlir.rstrip().rfind("}")
open(sys.argv[2],"w").write(mlir[:i]+"\n".join(L)+"\n}\n")
PY
}
run() { mlir-opt "$2" $1 $TAIL 2>/dev/null | mlir-runner -e main -entry-point-result=void -shared-libs=$RU,$CU 2>/dev/null; }

cat > "$D/pad.tns" <<'T'
input x: Tensor[n=1, c=1, h=8, w=8]
weight w: Tensor[oc=1, ic=1, kh=3, kw=3]
weight b: Tensor[oc=1]
graph g { c = conv2d(x, w, b, stride=1, padding=1)
          return c }
T
cat > "$D/nopad.tns" <<'T'
input x: Tensor[n=1, c=1, h=8, w=8]
weight w: Tensor[oc=1, ic=1, kh=3, kw=3]
weight b: Tensor[oc=1]
graph g { c = conv2d(x, w, b, stride=1, padding=0)
          return c }
T

for name in pad nopad; do
  $T --emit-mlir "$D/$name.tns" 2>/dev/null > "$D/g.mlir"; gen "$D/g.mlir" "$D/h.mlir"
  pads=$(grep -c 'tensor.pad' "$D/g.mlir")
  base=$(run "$BUF -convert-linalg-to-affine-loops" "$D/h.mlir")
  pc=$(run "$BUF -convert-linalg-to-affine-loops -affine-loop-fusion=mode=producer" "$D/h.mlir")
  sib=$(run "$BUF -convert-linalg-to-affine-loops -affine-loop-fusion=mode=sibling" "$D/h.mlir")
  printf "conv2d %-6s (tensor.pad=%s): no-fusion=%-4s producer-fusion=%-4s sibling-fusion=%-4s  =>  %s\n" \
    "$name" "$pads" "$base" "$pc" "$sib" \
    "$([ "$base" = "$pc" ] && echo 'fusion OK' || echo 'PRODUCER FUSION WRONG')"
done
