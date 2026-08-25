#map = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#map1 = affine_map<(d0, d1, d2, d3) -> (d1)>
module {
  func.func @g(%arg0: tensor<1x1x8x8xf32>, %arg1: tensor<1x1x3x3xf32>, %arg2: tensor<1xf32>) -> tensor<1x1x8x8xf32> attributes {llvm.emit_c_interface} {
    %cst = arith.constant 0.000000e+00 : f32
    %padded = tensor.pad %arg0 low[0, 0, 1, 1] high[0, 0, 1, 1] {
    ^bb0(%arg3: index, %arg4: index, %arg5: index, %arg6: index):
      tensor.yield %cst : f32
    } : tensor<1x1x8x8xf32> to tensor<1x1x10x10xf32>
    %0 = tensor.empty() : tensor<1x1x8x8xf32>
    %cst_0 = arith.constant 0.000000e+00 : f32
    %1 = linalg.fill ins(%cst_0 : f32) outs(%0 : tensor<1x1x8x8xf32>) -> tensor<1x1x8x8xf32>
    %2 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} ins(%padded, %arg1 : tensor<1x1x10x10xf32>, tensor<1x1x3x3xf32>) outs(%1 : tensor<1x1x8x8xf32>) -> tensor<1x1x8x8xf32>
    %3 = tensor.empty() : tensor<1x1x8x8xf32>
    %4 = linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%2, %arg2 : tensor<1x1x8x8xf32>, tensor<1xf32>) outs(%3 : tensor<1x1x8x8xf32>) {
    ^bb0(%in: f32, %in_1: f32, %out: f32):
      %5 = arith.addf %in, %in_1 : f32
      linalg.yield %5 : f32
    } -> tensor<1x1x8x8xf32>
    return %4 : tensor<1x1x8x8xf32>
  }
func.func @main() {
  %one = arith.constant 1.000000e+00 : f32
  %e0 = tensor.empty() : tensor<1x1x8x8xf32>
  %a0 = linalg.fill ins(%one : f32) outs(%e0 : tensor<1x1x8x8xf32>) -> tensor<1x1x8x8xf32>
  %e1 = tensor.empty() : tensor<1x1x3x3xf32>
  %a1 = linalg.fill ins(%one : f32) outs(%e1 : tensor<1x1x3x3xf32>) -> tensor<1x1x3x3xf32>
  %e2 = tensor.empty() : tensor<1xf32>
  %a2 = linalg.fill ins(%one : f32) outs(%e2 : tensor<1xf32>) -> tensor<1xf32>
  %r = call @g(%a0, %a1, %a2) : (tensor<1x1x8x8xf32>, tensor<1x1x3x3xf32>, tensor<1xf32>) -> tensor<1x1x8x8xf32>
  %c0 = arith.constant 0 : index
  %v = tensor.extract %r[%c0, %c0, %c0, %c0] : tensor<1x1x8x8xf32>
  vector.print %v : f32
  return
}
}
