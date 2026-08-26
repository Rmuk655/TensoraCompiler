// Kernel-time microbenchmark for the canonicalize/LICM pass spike.
// Same 1024x1024x1024 linalg.matmul shape and rtclock-timing method used
// in ../polyhedral_spike's (now-removed) perf sweep, recreated here for
// this spike so the numbers are directly comparable to that earlier
// tile/vectorize sweep (default 0.74 GFLOP/s, best tuned ~13.35 GFLOP/s).
func.func private @rtclock() -> f64

func.func @kernel(%A: memref<1024x1024xf32>, %B: memref<1024x1024xf32>,
                  %C: memref<1024x1024xf32>) {
  linalg.matmul ins(%A, %B : memref<1024x1024xf32>, memref<1024x1024xf32>)
                outs(%C : memref<1024x1024xf32>)
  return
}

func.func @main() {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %iters = arith.constant 20 : index
  %f1 = arith.constant 1.000000e+00 : f32
  %f0 = arith.constant 0.000000e+00 : f32

  %A = memref.alloc() : memref<1024x1024xf32>
  %B = memref.alloc() : memref<1024x1024xf32>
  %C = memref.alloc() : memref<1024x1024xf32>
  linalg.fill ins(%f1 : f32) outs(%A : memref<1024x1024xf32>)
  linalg.fill ins(%f1 : f32) outs(%B : memref<1024x1024xf32>)
  linalg.fill ins(%f0 : f32) outs(%C : memref<1024x1024xf32>)

  func.call @kernel(%A, %B, %C) : (memref<1024x1024xf32>, memref<1024x1024xf32>, memref<1024x1024xf32>) -> ()

  %t0 = func.call @rtclock() : () -> f64
  scf.for %i = %c0 to %iters step %c1 {
    func.call @kernel(%A, %B, %C) : (memref<1024x1024xf32>, memref<1024x1024xf32>, memref<1024x1024xf32>) -> ()
  }
  %t1 = func.call @rtclock() : () -> f64
  %dt = arith.subf %t1, %t0 : f64
  vector.print %dt : f64

  memref.dealloc %A : memref<1024x1024xf32>
  memref.dealloc %B : memref<1024x1024xf32>
  memref.dealloc %C : memref<1024x1024xf32>
  return
}
