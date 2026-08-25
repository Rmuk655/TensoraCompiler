// Minimal reproducer for the affine-loop-fusion tensor.pad miscompilation.
// No convolution and no reduction: just tensor.pad -> elementwise add.
//
// @g pads an 8x8 input to 10x10 (pad value 0), then adds 1.0 elementwise.
// @main fills the input with 1.0, calls @g, and prints an INTERIOR element
// [1,1], which maps to input[0,0]=1. So the padded interior is 1 and the
// result is 1 + 1 = 2.
//
// Correct (no fusion): prints 2.
// Wrong (producer-consumer or greedy fusion): prints 1 -- the fused IR moves
// the affine pad-fill loop to run AFTER the intervening memref.copy that
// writes the interior, so the fill re-zeros the copied interior and the
// consumer reads the pad value 0. Likely the fusion legality check does not
// model the dependence created by the (non-affine) memref.copy; a controlled
// variant (affine_variant.mlir) that writes the interior with an affine loop
// instead fuses correctly, which supports that reading.
//
//   OPT="-one-shot-bufferize=bufferize-function-boundaries -convert-linalg-to-affine-loops"
//   TAIL="-lower-affine -expand-strided-metadata -convert-scf-to-cf \
//     -finalize-memref-to-llvm -convert-vector-to-llvm -convert-func-to-llvm \
//     -convert-arith-to-llvm -convert-index-to-llvm -convert-cf-to-llvm \
//     -reconcile-unrealized-casts"
//   L=$(llvm-config-21 --libdir)
//   RUN="mlir-runner -e main -entry-point-result=void \
//     -shared-libs=$L/libmlir_runner_utils.so,$L/libmlir_c_runner_utils.so"
//
//   mlir-opt repro_minimal.mlir $OPT                        $TAIL | $RUN  # => 2 (correct)
//   mlir-opt repro_minimal.mlir $OPT -affine-loop-fusion    $TAIL | $RUN  # => 1 (wrong)
func.func @g(%x: tensor<8x8xf32>) -> tensor<10x10xf32> {
  %pad = arith.constant 0.000000e+00 : f32
  %one = arith.constant 1.000000e+00 : f32
  %p = tensor.pad %x low[1, 1] high[1, 1] {
    ^bb0(%i: index, %j: index):
      tensor.yield %pad : f32
  } : tensor<8x8xf32> to tensor<10x10xf32>
  %e = tensor.empty() : tensor<10x10xf32>
  %r = linalg.generic {
        indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>],
        iterator_types = ["parallel", "parallel"]}
        ins(%p : tensor<10x10xf32>) outs(%e : tensor<10x10xf32>) {
    ^bb0(%in: f32, %o: f32):
      %s = arith.addf %in, %one : f32
      linalg.yield %s : f32
  } -> tensor<10x10xf32>
  return %r : tensor<10x10xf32>
}
func.func @main() {
  %one = arith.constant 1.000000e+00 : f32
  %c1 = arith.constant 1 : index
  %e = tensor.empty() : tensor<8x8xf32>
  %x = linalg.fill ins(%one : f32) outs(%e : tensor<8x8xf32>) -> tensor<8x8xf32>
  %r = call @g(%x) : (tensor<8x8xf32>) -> tensor<10x10xf32>
  %v = tensor.extract %r[%c1, %c1] : tensor<10x10xf32>
  vector.print %v : f32
  return
}
