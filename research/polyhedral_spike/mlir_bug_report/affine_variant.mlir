func.func @main() {
  %cst = arith.constant 0.000000e+00 : f32   // pad
  %src = arith.constant 5.000000e+00 : f32   // "source" interior value
  %one = arith.constant 1.000000e+00 : f32
  %c1 = arith.constant 1 : index
  %alloc = memref.alloc() : memref<10x10xf32>
  affine.for %i = 0 to 10 { affine.for %j = 0 to 10 {
    affine.store %cst, %alloc[%i, %j] : memref<10x10xf32>
  }}
  // interior write as an AFFINE loop (not memref.copy): alloc[i+1,j+1]=src
  affine.for %i = 0 to 8 { affine.for %j = 0 to 8 {
    affine.store %src, %alloc[%i + 1, %j + 1] : memref<10x10xf32>
  }}
  %e = memref.alloc() : memref<10x10xf32>
  affine.for %i = 0 to 10 { affine.for %j = 0 to 10 {
    %0 = affine.load %alloc[%i, %j] : memref<10x10xf32>
    %1 = arith.addf %0, %one : f32
    affine.store %1, %e[%i, %j] : memref<10x10xf32>
  }}
  %v = memref.load %e[%c1, %c1] : memref<10x10xf32>   // expect 5+1 = 6
  vector.print %v : f32
  return
}
