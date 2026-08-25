# Polyhedral optimization spike (v0.1 untouched)

**Question.** Can we drive one polyhedral-model optimization from Tensora
today, without changing the frozen v0.1 compiler?

**Answer: yes.** v0.1 emits `linalg` (with affine indexing maps). MLIR's
own **affine dialect** and its **polyhedral passes** take that the rest
of the way. `polyhedral_pipeline.sh` runs entirely on the already-emitted
IR through `mlir-opt`; it does not touch the compiler.

## Result (runnable: `./polyhedral_pipeline.sh`)

Pipeline: `tensora --emit-mlir` (linalg) → `one-shot-bufferize` →
`convert-linalg-to-affine-loops` (into the **affine dialect**) →
**polyhedral pass** → lower to LLVM.

| stage | observed |
|---|---|
| linalg → affine | 24 affine ops; real `affine.for`/`affine.load`/`affine.store` nests |
| **affine-loop-tile** (`tile-size=4`) | matmul nest tiled: `step 4` + polyhedral tile-bound maps |
| **affine-loop-fusion** (dependence-driven) | outer loops **14 → 5** (fused) |
| correctness path | still lowers to LLVM after the pass (106 low-level op lines) |

So a polyhedral-driven transform on Tensora's output is achievable with
**no v0.1 change and no rebuild**.

## What v0.1 has vs what a polyhedral pass adds

- v0.1 **has**: linalg with affine indexing maps; Level-2 matmul tiling
  via `scf::tileUsingSCF` with tile sizes chosen from verified static
  shapes (`src/codegen/MatmulTilingPass.cpp`, `LoopTiling.cpp`).
- v0.1 **lacks**: affine-dialect loops and any pass whose transform is
  *decided by polyhedral dependence analysis*. The Level-2 tiling is a
  fixed-size SCF transform from a shape heuristic, not a
  dependence-model decision.
- The polyhedral pass supplies exactly that: `affine-loop-fusion` fuses
  only where the dependence model proves it legal; `affine-loop-tile`
  tiles the affine domain.

## Two ways to ship this

1. **This spike** (`polyhedral_pipeline.sh`, on `main`) — a pipeline over
   `--emit-mlir` output. Zero compiler change. This IS "one
   polyhedral-driven optimization." Uses the installed LLVM/MLIR 21
   `mlir-opt` (`/usr/lib/llvm-21/bin`).
2. **In-compiler `--polyhedral` flag** (branch `spike/polyhedral`, v0.1
   NOT touched) — the same idea in-process in the driver. Done and tested;
   see below.

## In-compiler result (branch `spike/polyhedral`)

A `--polyhedral` modifier flag was added to the `tensora` driver in a
throwaway git worktree (branch `spike/polyhedral`), so the graded v0.1 on
`main` stays frozen. The change is small and additive
(`src/driver/main.cpp` only, no CMake change — the driver already links
the aggregate `MLIR` lib): when the flag is set, the linalg lowering goes
`convert-linalg-to-affine-loops` → **`affine-loop-tile`** → `lower-affine`
→ scf → LLVM, instead of the default `convert-linalg-to-loops`.

Verified (`test_polyhedral_parity.sh`, on the branch):

- **Correctness**: for all **7** runnable examples (mlp, cnn, attention,
  matmul, ...), `--polyhedral --run` gives a result **numerically
  identical** to the default `--run` (exact, `np.array_equal`).
- **Real transform, not a no-op**: the generated LLVM IR differs (e.g.
  `mlp_concrete` 416→598 lines / 42→84 branches from the tile loops;
  `matmul_large` 312→338 / 24→30).

### Honest finding: dependence-driven FUSION was not safe here

The first attempt used **`affine-loop-fusion`** (the dependence-analysis
pass). It matched the default on 6 of 7 examples but produced a **wrong**
result on the larger CNN (`cnn_stl10`): default `1.54e12` vs fused `2305`
for every output element. Since the default is v0.1's PyTorch-verified
path, the fuser is the one that is wrong — it must not be applied blindly
to every graph. The flag therefore demonstrates the safe, classic
polyhedral transform (**loop tiling**), and the fusion result is recorded
as a cautionary spike finding, not shipped.

Takeaway: a polyhedral optimization is reachable and correct in Tensora
today (tiling), and a polyhedral optimization can also be **silently
wrong** if applied without the legality guarantee the framework does not
enforce for you (fusion). Both are useful lessons for the roadmap.
