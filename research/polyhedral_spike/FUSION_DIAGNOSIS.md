# Diagnosis: the affine-loop-fusion wrong result on cnn_stl10

To reproduce, run `./fusion_diagnosis.sh`. The script builds the CNN
harness. `matmul_bench.mlir` is a different file.

## Symptom

The first `--polyhedral` version used `affine-loop-fusion`. It gave `2305`
for every `cnn_stl10` output element. The default path gives `1.54e12`.

`2305` is `2304 * 1 + 1`. This value means that the final matmul read an
all-ones `flat` vector and added the bias. So the conv and pool layers
gave constant results.

## What the bug is not

Experiments remove these causes:

- Not a Tensora bug, and not the affine conversion. Affine loops without
  fusion give the correct `1.54e12`. Only fusion changes the result.
- Not the pass order. Plain `mlir-opt` on the emitted MLIR shows the same
  bug (`one-shot-bufferize`, `convert-linalg-to-affine-loops`,
  `affine-loop-fusion`).
- Not the MatmulTilingPass. `cnn_stl10` has no `scf` op in the emitted IR.
  The matmul is below the tiling threshold.
- Not all fusion. `affine-loop-fusion=mode=sibling` is correct. Only
  producer-consumer fusion is wrong (`mode=producer` and the default
  `greedy`).

## Root cause: a single padded conv2d

The failure needs only one `conv2d` with `padding=1`:

| graph | `tensor.pad` | no fusion | producer fusion | sibling fusion |
|---|---|---|---|---|
| conv2d padding=1 | 1 | **5** | **1** (wrong) | 5 |
| conv2d padding=0 | 0 | 10 | 10 | 10 |

The padding is the trigger. Tensora lowers a padded conv to a
`tensor.pad`. The `tensor.pad` bufferizes to a producer that writes its
buffer in two parts. First it fills the whole buffer with the pad value.
Then it copies the real input into the interior.

The before/after IR of the minimal case shows the mechanism directly (see
`mlir_bug_report/BUG_REPORT.md`). The producer writes the buffer `%alloc`
with two ops. The first op is an affine fill loop with the pad value. The
second op is a `memref.copy` that writes the interior. The `memref.copy`
is not an affine op.

Fusion moves the fill loop into the consumer loop. This moves the fill
after the `memref.copy`. The fill then writes the pad value over the
interior again. So the consumer reads the pad value. The copy is not
removed. The reordered fill writes over its result.

A control experiment supports the reason. It writes the same interior
with an affine loop in place of the `memref.copy`. With the affine loop,
fusion is correct. The bug changes only with `memref.copy` against an
affine write (see `mlir_bug_report/affine_variant.mlir`). The likely
reason is that the fusion legality check does not model the dependence of
the `memref.copy`.

The padded conv shows the same class of bug with one more effect. Fusion
also shrinks the producer buffer to a `1x1x1x1` slice that holds only the
pad value. So each input pixel reads `0`. The reduction adds `0`. The
output is the bias only. Over four conv blocks the error reaches the
matmul.

Conclusion: this is a miscompilation in the MLIR affine producer-consumer
fusion. It happens when the pass fuses a padded producer across the
`memref.copy` that writes the interior. The output IR passes the verifier
but gives the wrong result. This is a correctness bug in the transform. It
is not a legality problem that Tensora semantics could have prevented. The
value evidence (`2` against `1`) and the before/after IR make it a direct
demonstration.

## Effect on the V0.3 research direction

This failure does not support the idea that Tensora semantics make fusion
legal. Do not use it as evidence for that idea. The transform was not
legal-but-unsafe because of missing semantic facts. The pass simply moved
the fill loop incorrectly.

`SEMANTIC_POLYHEDRAL.md` states the rule: diagnose first. The diagnosis
removes this case as evidence.

Two results stay useful:

1. Practical: do not use upstream affine producer-consumer fusion on
   padded convolutions. The shipped `--polyhedral` flag uses tiling. The
   bug does not affect tiling. The parity test passes on all 7 examples.
   For fusion, use linalg-level fusion or tile-and-fuse. These run before
   the pad becomes a fill and a copy.
2. Research: the semantic-legality question stays open. It needs a
   different example. The example must be a transform that is legal only
   because of a verified fact (for example an `L[b]` extent or a
   prefix-sum contract) that the lowered program cannot recover. This
   tooling bug is not that example.
