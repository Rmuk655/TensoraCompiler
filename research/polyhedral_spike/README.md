# Polyhedral optimization spike (v0.1 not changed)

**Question.** Can Tensora drive one polyhedral optimization today, without
a change to the frozen v0.1 compiler?

**Answer: yes.** v0.1 emits `linalg` with affine indexing maps. The MLIR
affine dialect and its polyhedral passes do the rest.
`polyhedral_pipeline.sh` runs on the already-emitted IR through
`mlir-opt`. It does not change the compiler.

## Result (run `./polyhedral_pipeline.sh`)

Pipeline: `tensora --emit-mlir` (linalg), then `one-shot-bufferize`, then
`convert-linalg-to-affine-loops` (the affine dialect), then a polyhedral
pass, then lower to LLVM.

| stage | observed |
|---|---|
| linalg to affine | 24 affine ops; real `affine.for`/`affine.load`/`affine.store` nests |
| `affine-loop-tile` (`tile-size=4`) | matmul nest is tiled: `step 4` plus tile-bound maps |
| `affine-loop-fusion` (dependence-driven) | outer loops go from 14 to 5 |
| correctness path | still lowers to LLVM after the pass (106 low-level op lines) |

So a polyhedral transform on Tensora's output works with no v0.1 change
and no rebuild.

## What v0.1 has and what a polyhedral pass adds

- v0.1 has: linalg with affine indexing maps, and Level-2 matmul tiling
  through `scf::tileUsingSCF`. The tile sizes come from verified static
  shapes (`src/codegen/MatmulTilingPass.cpp`, `LoopTiling.cpp`).
- v0.1 does not have: affine-dialect loops, or any pass whose transform
  comes from polyhedral dependence analysis. The Level-2 tiling uses a
  fixed size from a shape heuristic, not a dependence model.
- The polyhedral pass adds exactly that. `affine-loop-fusion` fuses only
  where the dependence model proves it legal. `affine-loop-tile` tiles the
  affine domain.

## Two ways to ship this

1. This spike (`polyhedral_pipeline.sh`) is a pipeline over `--emit-mlir`
   output. It needs no compiler change. It is "one polyhedral
   optimization". It uses the installed LLVM/MLIR 21 `mlir-opt`
   (`/usr/lib/llvm-21/bin`).
2. The in-compiler `--polyhedral` flag adds the same idea inside the
   driver. It is now merged into `main` (v0.1). See below.

This file and the other files in this directory stay on the
`v0.3-research` branch. `main` holds only the frozen v0.1 compiler and
the roadmap, plus the merged `--polyhedral` flag described next.

## In-compiler result (`--polyhedral` flag) — now in `main`

The `--polyhedral` flag started as a spike on this branch. It uses no
Tensora-specific semantics — it only runs stock MLIR passes on the same
`linalg` IR the default path already emits — so it carried no research
risk, and it is now merged into `main` (v0.1) as **ARCH-018**
(`docs/architecture/DECISIONS.md`). The flag, its usage text, and its
correctness tests all live in `main` now, not here:

- The flag itself: `src/driver/main.cpp` (`main`).
- The correctness check: `tests/cli/cli_tests.txt`, the `INT-POLY-*`
  CTest cases (`main`). Each reuses its `INT-FULL-*-RUN` twin's expected
  value, so a pass proves `--polyhedral --run` is numerically identical
  to the default `--run`, on mlp, matmul, cnn, attention, and the larger
  `cnn_stl10`.
- The decision record: ARCH-018.
- User docs: `README.md` and `docs/development/BUILD.md` (`main`).

This directory keeps only what is genuinely research, not shipped: the
`mlir-opt`-only pipeline above (`polyhedral_pipeline.sh`), the fusion
finding below, and the performance sweep.

### Finding: dependence-driven fusion was not safe here

The first version used `affine-loop-fusion`. It matched the default on 6
of 7 examples. But it gave a wrong result on the larger CNN
(`cnn_stl10`): the default gives `1.54e12`, the fused version gives `2305`
for every output element. The default is v0.1's PyTorch-verified path, so
the fuser is the one that is wrong. Do not apply it to every graph. The
flag therefore uses the safe transform (loop tiling). The fusion result
is a recorded finding, not shipped.

**Now diagnosed** (see [FUSION_DIAGNOSIS.md](FUSION_DIAGNOSIS.md) and
`fusion_diagnosis.sh`). The smallest trigger is one padded `conv2d`. The
MLIR affine producer-consumer fusion moves the `tensor.pad` fill after the
`memref.copy` that writes the interior. So the fill writes the pad value
over the interior again, and the conv reads the pad value. This is a
correctness bug in upstream `affine-loop-fusion` for padded producers. It
is not a Tensora bug, and not a missing-semantics case. So it is not
evidence for the semantic-legality idea.

Takeaway: a polyhedral optimization is reachable and correct in Tensora
today (tiling). A polyhedral optimization can also be silently wrong if
the framework does not enforce the legality guarantee (fusion). Both are
useful lessons for the roadmap.

## Performance sweep on a hard matmul (`matmul_sweep.sh`)

Does a tuned polyhedral schedule beat the default lowering? The
`--polyhedral` flag with tiling only and a default tile size was slower.
So this sweep measures real kernel time on a hard 1024x1024x1024 matmul.
`matmul_bench.mlir` times N calls with `rtclock`, so there is no JIT or
process noise. `matmul_sweep.sh` applies each schedule with `mlir-opt`,
then runs it with `mlir-runner`.

| schedule | GFLOP/s | vs default |
|---|---|---|
| default (`convert-linalg-to-loops`) | 0.74 | 1.0x |
| affine tile=16 | 1.90 | 2.6x |
| affine tile=32 | 1.68 | 2.3x |
| affine tile=64 | 1.41 | 1.9x |
| affine tile=16 + super-vectorize=16 | 13.23 | 17.9x |
| **affine tile=32 + super-vectorize=16** | **13.35** | **18.0x** |

The vectorized winner was checked for correctness on its own. With
all-ones inputs, every output element is `1024.0`, the same as the default
path.

What this shows:

- Tuned polyhedral tiling and vectorization is about 18x faster than
  Tensora's naive default lowering here. This is a large, real, correct
  speedup.
- Tiling alone is not enough (only about 2.6x, and slower through the JIT
  `--run` path, because that path adds compile cost). The speedup needs
  vectorization as well. This is why the tiling-only `--polyhedral` flag
  looked slow.
- This is a win over the naive baseline, not over BLAS. About 13 GFLOP/s
  is still far below a tuned single-thread CPU GEMM, and far below cuBLAS.
  This agrees with the earlier finding: do not try to beat vendor GEMMs.
  Instead, take the headroom that naive lowering leaves, plus region-level
  fusion (see `SEMANTIC_POLYHEDRAL.md`).
