# Diagnosis: the affine-loop-fusion wrong result on cnn_stl10

Reproduce: `./fusion_diagnosis.sh`. Full CNN repro: the harness in this
directory (`matmul_bench.mlir` is separate; the CNN harness is generated
by the script).

## Symptom

`--polyhedral` with **`affine-loop-fusion`** (the first version, before it
was switched to tiling) gave `2305` for every `cnn_stl10` output element;
the default path gives `1.54e12`. `2305 = 2304·1 + 1` = the final matmul
fed an **all-ones** `flat` vector plus bias -- i.e. the whole conv/pool
stack produced constants.

## What it is NOT (ruled out by experiment)

- **Not a Tensora bug and not the affine conversion.** Affine loops
  *without* fusion give the correct `1.54e12`. Only fusion changes it.
- **Not our pass composition.** Reproduced with plain `mlir-opt` on the
  emitted MLIR: `one-shot-bufferize -convert-linalg-to-affine-loops
  -affine-loop-fusion`.
- **Not MatmulTilingPass interaction.** `cnn_stl10` has no `scf` in the
  emitted IR; the matmul is below the tiling threshold.
- **Not fusion in general.** `-affine-loop-fusion=mode=sibling` is
  correct. Only **producer-consumer** fusion (`mode=producer`, and the
  default `greedy`) is wrong.

## Root cause (minimal trigger: a single padded conv2d)

The failure reduces to **one `conv2d` with `padding=1`**:

| graph | `tensor.pad` | no-fusion | producer-fusion | sibling-fusion |
|---|---|---|---|---|
| conv2d padding=1 | 1 | **5** | **1** (wrong) | 5 |
| conv2d padding=0 | 0 | 10 | 10 | 10 |

Padding is the trigger. Tensora lowers a padded conv to a `tensor.pad`,
which bufferizes to a producer that writes its buffer in **two regions**:
a zero-fill of the whole padded buffer, then a copy of the real input `x`
into the interior. In the affine IR the conv then reads that padded
buffer inside its `kh,kw,ic` reduction.

MLIR's affine **producer-consumer** fusion sinks that padded producer
into the conv reduction and computes a per-iteration *slice* of it. For
this two-region producer it mis-slices: the fused code keeps only the
**zero-fill** (`affine.store %cst, %alloc[0,0,0,0]` then reads the same
`0` back) and **drops the copy of `x`**. So every input pixel reads as
`0`, the conv reduction accumulates `0`, and the output is bias only.
Chained over four conv blocks the error propagates to the constant seen
at the matmul.

Conclusion: this is a **miscompilation in MLIR's affine
producer-consumer loop fusion for a padded (multi-region-write)
producer** -- upstream `affine-loop-fusion` is a known experimental /
under-maintained pass. It is a real correctness bug in the transform, not
a subtle illegality that Tensora's semantics failed to expose.

## Consequence for the V0.3 research narrative (important, honest)

This failure does **not** support the "Tensora semantics would make
fusion provably legal" thesis, and it must not be cited as evidence for
it. The transform was not illegal-yet-plausible due to missing semantic
information; the pass simply computed a wrong slice for a padded producer.
Per the caveat in [SEMANTIC_POLYHEDRAL.md](SEMANTIC_POLYHEDRAL.md), the
diagnosis had to come first -- and it removes this case as evidence.

Two clean takeaways survive:

1. **Practical**: do not use upstream affine producer-consumer fusion on
   padded convolutions. The shipped `--polyhedral` flag correctly uses
   tiling, which is unaffected (parity test passes on all 7 examples).
   For real fusion, prefer linalg-level fusion / tiling-and-fuse, which
   operate before the pad is split into fill+copy.
2. **Research**: the semantic-legality question still stands, but it needs
   a *different* motivating example -- one where a transform is genuinely
   legal only because of a fact (e.g. a verified `L[b]` extent or
   prefix-sum contract) that the lowered program cannot recover. A tooling
   bug in a padded-producer slice is not that example.
