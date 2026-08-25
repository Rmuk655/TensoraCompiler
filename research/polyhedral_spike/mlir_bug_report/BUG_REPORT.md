# MLIR bug report (draft, ready to submit)

Submit at: https://github.com/llvm/llvm-project/issues (label: `mlir`).
This draft is NOT submitted automatically -- review it, then post it under
your own account. `repro.mlir` in this directory is the self-contained
reproducer (standard dialects only; no Tensora needed).

---

**Title:** [mlir][affine] affine-loop-fusion (producer-consumer)
miscompiles a `tensor.pad` producer: drops the interior copy, keeps only
the pad value

**Summary**

`-affine-loop-fusion` in producer-consumer mode produces wrong results
when the fused producer writes its buffer in two regions -- the pattern
that a padded convolution lowers to (`tensor.pad` = fill the whole buffer
with the pad value, then copy the source into the interior). After fusion
the consumer reads only the pad value; the interior copy is dropped, so
every "interior" element reads as the padding constant.

Reduced to the minimal trigger: a single `linalg.conv_2d_nchw_fchw` whose
input is a `tensor.pad`. Sibling-mode fusion is fine; only
producer-consumer fusion is affected. A convolution with no padding (no
`tensor.pad`) fuses correctly.

**Environment**

- LLVM/MLIR 21.1.8 (Ubuntu package `llvm-21`). Also seen with the default
  greedy fusion mode.

**Reproducer** (`repro.mlir`: all-ones `1x1x8x8` input, `1x1x3x3`
all-ones kernel, bias 1.0, `padding=1`; `@main` prints output element
`[0,0,0,0]`, a corner = 4 taps + bias = **5**).

Correct (no fusion) -- prints `5`:

```
mlir-opt repro.mlir \
  -one-shot-bufferize=bufferize-function-boundaries \
  -convert-linalg-to-affine-loops \
  -lower-affine -expand-strided-metadata -convert-scf-to-cf \
  -finalize-memref-to-llvm -convert-vector-to-llvm -convert-func-to-llvm \
  -convert-arith-to-llvm -convert-index-to-llvm -convert-cf-to-llvm \
  -reconcile-unrealized-casts \
| mlir-runner -e main -entry-point-result=void \
  -shared-libs=$LLVM/lib/libmlir_runner_utils.so,$LLVM/lib/libmlir_c_runner_utils.so
# => 5
```

Wrong (add producer-consumer fusion) -- prints `1` (bias only):

```
mlir-opt repro.mlir \
  -one-shot-bufferize=bufferize-function-boundaries \
  -convert-linalg-to-affine-loops \
  -affine-loop-fusion=mode=producer \
  ...same tail... 
# => 1
```

**Expected**: both print `5`.
**Actual**: with producer-consumer fusion it prints `1`.

**Analysis**

`tensor.pad` bufferizes to a producer that writes the destination in two
regions: a full `linalg.fill` with the pad value, then a copy of the
source into the interior. When producer-consumer fusion computes the
per-iteration source slice and sinks the producer into the conv reduction
nest, the generated slice keeps only the pad `linalg.fill`
(`affine.store %pad, %slice[0,0,0,0]` immediately reloaded) and omits the
interior copy of the real input. So each input read returns the pad
constant (0), the reduction accumulates 0, and the result is bias only.

Confirmed by inspecting the fused IR: the conv's input load reads a
`memref<1x1x1x1xf32>` that is only ever written with the pad constant.

**Notes**

- Sibling-mode fusion (`mode=sibling`) is unaffected.
- Likely the fusion slice/legality computation does not account for a
  producer with multiple disjoint write regions (partial-store init +
  interior copy).
