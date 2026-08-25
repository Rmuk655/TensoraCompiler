# MLIR bug report (draft, ready to submit)

Submit at: https://github.com/llvm/llvm-project/issues (label: `mlir`).
This draft is NOT submitted automatically -- review it, then post it under
your own account.

Reproducers in this directory (standard dialects only, no external deps):
- `repro_minimal.mlir` -- **headline**: `tensor.pad` -> elementwise add,
  no conv, no reduction.
- `repro.mlir` -- the original discovery: a single padded `conv2d`.

---

**Title:** [mlir][affine] affine-loop-fusion reorders a pad-fill loop past
an intervening `memref.copy`, giving wrong results

**Observed behavior (measured, self-contained)**

On `repro_minimal.mlir` -- pad an all-ones `8x8` input to `10x10` (pad
value 0), then add `1.0` elementwise; `@main` prints interior element
`[1,1]`, which maps to `input[0,0] = 1`, so the result must be
`1 + 1 = 2`:

| pipeline | printed |
|---|---|
| `-convert-linalg-to-affine-loops` (no fusion) | **2** (correct) |
| `... -affine-loop-fusion` (greedy) | **1** (wrong) |
| `... -affine-loop-fusion=mode=producer` | **1** (wrong) |
| `... -affine-loop-fusion=mode=sibling` | **2** (correct) |

The only difference between the correct and wrong runs is the presence of
`-affine-loop-fusion`. The fused module is verifier-valid (`mlir-opt`
exits 0). Measured on **Ubuntu LLVM 21.1.8** (`mlir-opt --version`).

Commands:

```
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

**The transformation (IR before/after `-affine-loop-fusion`)**

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

**Hypothesized root cause**

`memref.copy` is not an affine op, so it is invisible to the affine
producer/consumer dependence analysis that drives fusion. The pass fuses
the pad-fill loop into the consumer loop and, in doing so, moves that
store past the `memref.copy` that writes the same buffer `%alloc`. The
fused fill then re-writes the pad value over the interior the copy
produced, and the consumer reads the pad value. So the copy is not
dropped -- it is clobbered by a reordered fill. (This is the mechanism the
before/after IR shows; the label "root cause" is my reading of it, offered
for triage, not asserted as confirmed.)

This matches the general area where `affine-loop-fusion` has other known
miscompiles (opaque/external effects and aliasing it does not model). The
`tensor.pad` -> `fill + memref.copy` lowering is a common way to hit it.

**Original context** (`repro.mlir`): first seen in a convolution
pipeline. A single `linalg.conv_2d_nchw_fchw` on a `tensor.pad` input
(`padding=1`, all-ones) should give a corner output of `4 + bias(1) = 5`;
with producer-consumer fusion it gives `1` (bias only). A convolution
with no padding (no `tensor.pad`) fuses correctly.
