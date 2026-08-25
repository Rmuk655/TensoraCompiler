Paste EVERYTHING below the line into the GitHub issue "Add a description"
box (the Write tab). It is self-contained -- the reproducers are inlined
in code blocks, so no file attachment is needed. Put the title in the
"Add a title" field:

  [mlir][affine] affine-loop-fusion reorders tensor.pad fill across memref.copy and miscompiles

--------------------------------------------------------------------------

### Observed behavior (measured, self-contained)

`repro_minimal.mlir` (inlined below) pads an all-ones `8x8` input to
`10x10` (pad value 0), then adds `1.0` elementwise. `@main` prints
interior element `[1,1]`, which maps to `input[0,0] = 1`, so the result
must be `1 + 1 = 2`.

| pipeline | printed |
|---|---|
| `-convert-linalg-to-affine-loops` (no fusion) | **2** (correct) |
| `... -affine-loop-fusion` (greedy) | **1** (wrong) |
| `... -affine-loop-fusion=mode=producer` | **1** (wrong) |
| `... -affine-loop-fusion=mode=sibling` | **2** (correct) |

The only difference between the correct and wrong runs is the presence of
`-affine-loop-fusion`. The fused module is verifier-valid (`mlir-opt`
exits 0).

### Steps to reproduce

```sh
OPT="-one-shot-bufferize=bufferize-function-boundaries -convert-linalg-to-affine-loops"
TAIL="-lower-affine -expand-strided-metadata -convert-scf-to-cf \
  -finalize-memref-to-llvm -convert-vector-to-llvm -convert-func-to-llvm \
  -convert-arith-to-llvm -convert-index-to-llvm -convert-cf-to-llvm \
  -reconcile-unrealized-casts"
L=$(llvm-config --libdir)
RUN="mlir-runner -e main -entry-point-result=void \
  -shared-libs=$L/libmlir_runner_utils.so,$L/libmlir_c_runner_utils.so"

mlir-opt repro_minimal.mlir $OPT                     $TAIL | $RUN   # => 2 (correct)
mlir-opt repro_minimal.mlir $OPT -affine-loop-fusion $TAIL | $RUN   # => 1 (wrong)
```

### The transformation (IR before/after `-affine-loop-fusion`)

After bufferization + `-convert-linalg-to-affine-loops`, the padded
producer writes `%alloc` in two ops -- an affine **fill** loop (pad
value), then a **`memref.copy`** of the source into the interior subview
-- and the consumer reads `%alloc`:

```mlir
// BEFORE fusion  (order: fill -> copy overwrites interior -> consumer reads)
%alloc = memref.alloc() : memref<10x10xf32>
affine.for %i = 0 to 10 { affine.for %j = 0 to 10 {
  affine.store %cst, %alloc[%i, %j]                 // pad fill (0)
}}
%subview = memref.subview %alloc[1, 1] [8, 8] [1, 1] ...
memref.copy %arg0, %subview                          // interior <- source
affine.for %i = 0 to 10 { affine.for %j = 0 to 10 {
  %0 = affine.load %alloc[%i, %j]
  %1 = arith.addf %0, %cst_0                         // + 1.0
  affine.store %1, %alloc_1[%i, %j]
}}
```

```mlir
// AFTER fusion  (copy hoisted above the loop; fill fused into consumer,
//                so the fill now runs AFTER the copy and re-zeros the interior)
%alloc = memref.alloc() : memref<10x10xf32>
%subview = memref.subview %alloc[1, 1] [8, 8] [1, 1] ...
memref.copy %arg0, %subview                          // interior <- source
affine.for %i = 0 to 10 { affine.for %j = 0 to 10 {
  affine.store %cst, %alloc[%i, %j]                  // pad fill CLOBBERS the copied interior
  %0 = affine.load %alloc[%i, %j]                    // reads 0, not the copied value
  %1 = arith.addf %0, %cst_0
  affine.store %1, %alloc_1[%i, %j]
}}
```

The fused IR moves the pad-fill stores to run *after* the intervening
`memref.copy` that writes the interior of the same buffer, so the fill
overwrites the copied interior and the consumer reads the pad value. The
copy is not dropped -- it is clobbered by the reordered fill.

### Likely issue (hypothesis + controlled experiment)

The likely issue is that the fusion legality check does not model the
dependence created by the `memref.copy` (a non-affine op) on `%alloc`, so
the reordering is considered legal. I have not read the pass's dependence
analysis, so I state this as a hypothesis -- but a controlled variant
supports it (`affine_variant.mlir`, inlined below): keep the same
three-part structure (full fill -> interior write -> elementwise consumer)
but write the interior with an **affine loop** instead of `memref.copy`.
With the affine interior write, fusion is correct:

| interior write | no fusion | + fusion |
|---|---|---|
| `memref.copy` (`repro_minimal.mlir`) | 2 | **1 (wrong)** |
| affine loop (`affine_variant.mlir`) | 6 | 6 (correct) |

The miscompile toggles precisely on `memref.copy` vs an affine interior
write. `tensor.pad` lowering to `alloc + linalg.fill + buffer copy` is
documented MLIR bufferization behavior, so this is a natural case.

### Environment

```
$ mlir-opt --version
Ubuntu LLVM version 21.1.8
$ llvm-config-21 --version
21.1.8
$ apt-cache policy llvm-21
llvm-21:  Installed: 1:21.1.8-6ubuntu1
$ apt-cache policy libmlir-21-dev
libmlir-21-dev:  Installed: 1:21.1.8-6ubuntu1
```

Packaged 21.1.8 release; not re-run on current trunk.

### repro_minimal.mlir

```mlir
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
```

### affine_variant.mlir (control: interior via affine loop, fuses correctly)

```mlir
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
```
