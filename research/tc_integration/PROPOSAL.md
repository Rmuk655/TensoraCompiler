# V0.3 research item: integrate Tensor Comprehensions' proven idea, not TC itself

## Status of TC (checked, not assumed)

Tensor Comprehensions (Vasilache et al., arXiv:1802.04730) is **archived**
(`facebookresearch/TensorComprehensions`, archived 2023-04-28, read-only).
It was never part of core PyTorch -- only an optional, separate
integration -- and it is not used by vLLM (vLLM uses hand-tuned kernels,
`torch.compile`, Triton, and CUTLASS instead). So this item is not "add a
dependency on TC"; TC as a project is dead. The question is whether its
proven *idea* still has value combined with Tensora's differentiator.

## The idea, and the gap each side has

TC contributes: a general Einstein-notation surface syntax
(`C(i,j) += A(i,k)*B(k,j)`) that lets a user define a **new** operator and
have its output shape inferred from the index algebra, then a full
polyhedral schedule **search with autotuning** over that operator,
targeting CPU and GPU.

Tensora contributes the one thing TC's own paper does not have: **named,
statically checked axes carried into the IR** (`Tensor[batch=B, feat=D]`),
confirmed by inspecting TC's syntax (plain letters, no names, no static
check) against `include/tensora/shared/Ir.h`'s frozen, hand-coded 15-op
set.

Neither side has both. Tensora is closed (15 built-in ops, no user
extensibility); TC is open (any index expression) but unverified.

## The research question

Does generalizing Tensora's operator set to TC-style index notation --
**with Tensora's static axis-name checking kept on top of it** -- pay for
its own engineering cost, once lowered through the MLIR affine/polyhedral
path already proven in `research/polyhedral_spike/` (tiling +
vectorization, ~18x on a hard matmul, ARCH-018 on `main`)?

This is not a new pipeline. It reuses:
- the `--polyhedral` flag's lowering path (linalg -> affine ->
  affine-loop-tile) as the backend, unchanged;
- the same kernel-time measurement method as `research/polyhedral_spike/`
  used for its perf sweep;
- the axis-role framing already sketched as §8.1 "Verified index-notation
  operators" in `docs/ProjectMgmt/16_FUTURE_CONSTRUCTS_AND_INTEROP_DETAIL.md`.

## What would make this worth building (falsification-first)

Build the smallest possible version first: parse ONE user-defined
operator via index notation (e.g. a custom weighted reduction Tensora
does not have a built-in for), check every index against a declared axis
(the static rule §8.1 already specifies), lower it through the *existing*
linalg/affine path with no new codegen machinery, and compare against a
hand-coded `linalg.generic` doing the identical computation.

- If the generic, verified-index-notation version produces **the same
  MLIR** (or the same performance after `--polyhedral`) as the hand-coded
  op, the contribution is **extensibility with safety**, not performance
  -- a real result, but a different one to claim.
- If it produces meaningfully worse code (the generic lowering can't
  reach the same tiling/vectorization decision the hand-coded one gets),
  that is the actual engineering cost of generality, and it must be
  weighed against how many new operators users would realistically need.
- Kill this item if the one-operator spike shows no daylight between
  "hand-code it as a 16th built-in `OpKind`" and "add a general index-
  notation feature" -- extensibility only earns its cost if users need
  more operators than are worth hand-coding one at a time.

## Relationship to other V0.3 items

This is a **Part 1a** candidate (see `docs/ProjectMgmt/11_OPEN_SOURCE_AND_RESEARCH_ROADMAP.md`
on `main`, §3): it is about whether Tensora's semantic knowledge (here,
verified index algebra) adds something a shape-erasing extensible system
(TC, or MLIR's own `linalg.generic` with no name checking) does not
already have. It depends on nothing beyond what already exists on this
branch (the polyhedral lowering path, the axis-contract checker in
`src/semantic/`).
