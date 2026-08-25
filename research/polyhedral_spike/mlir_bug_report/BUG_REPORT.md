# MLIR bug report (draft, ready to submit)

Submit at: https://github.com/llvm/llvm-project/issues (label: `mlir`).
This draft is NOT submitted automatically -- review it, then post it under
your own account.

Reproducers in this directory (standard dialects only, no Tensora):
- `repro_minimal.mlir` -- **headline**: `tensor.pad` -> elementwise add,
  no conv, no reduction.
- `repro.mlir` -- the original discovery: a single padded `conv2d`.

---

**Title:** [mlir][affine] affine-loop-fusion miscompiles a `tensor.pad`
producer: the interior copy is dropped, consumer reads the pad value

**Summary**

`-affine-loop-fusion` produces wrong results when it fuses a `tensor.pad`
producer into its consumer. `tensor.pad` bufferizes to a producer that
writes its destination in two regions: fill the whole buffer with the pad
value, then copy the source into the interior. After producer-consumer
fusion the consumer reads only the pad value -- the interior copy is
dropped -- so every interior element reads as the padding constant.

This affects the default (greedy) mode and `mode=producer`.
`mode=sibling` is unaffected.

**Environment**

- LLVM/MLIR 21.1.8 (Ubuntu package `llvm-21`).

**Minimal reproducer** (`repro_minimal.mlir`): pad an all-ones `8x8`
input to `10x10` (pad value 0), then add `1.0` elementwise. `@main`
prints interior element `[1,1]`, which maps to `input[0,0] = 1`, so the
result must be `1 + 1 = 2`.

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

**Expected**: both print `2`.
**Actual**: with fusion it prints `1` (the interior element reads the pad
value `0`, then `+1`).

**Analysis**

`tensor.pad` bufferizes to a producer with two disjoint write regions:
a full fill with the pad value and a copy of the source into the interior.
When producer-consumer fusion computes the per-iteration slice of that
producer and sinks it into the consumer, the generated slice keeps only
the pad fill and omits the interior copy. So each consumer read returns
the pad constant.

Confirmed by inspecting the fused IR on the conv variant (`repro.mlir`):
the fused input load reads a `memref<1x1x1x1xf32>` that is only ever
written with the pad constant. Likely the fusion slice/legality logic
does not handle a producer with multiple disjoint write regions
(partial-store init + interior copy).

**Original context** (`repro.mlir`): found in a convolution pipeline. A
single `linalg.conv_2d_nchw_fchw` on a `tensor.pad` input (`padding=1`,
all-ones) should give a corner output of `4 + bias(1) = 5`; with
producer-consumer fusion it gives `1` (bias only). A convolution with no
padding (no `tensor.pad`) fuses correctly.
