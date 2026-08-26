# Spike: does adding stock MLIR canonicalize/LICM help Tensora's lowering?

Question: two stock MLIR passes are one line each to add to the driver's
lowering pipeline (`-canonicalize`, and `-loop-invariant-code-motion` /
`-affine-loop-invariant-code-motion`). Do they measurably help, on top of
the default path and the `--polyhedral` path (ARCH-018, on `main`)?

## Result: no measurable benefit, correctness confirmed

**Correctness** (the stronger check): the passes were added directly to
a local build of the real driver's pipeline (both the default and
`--polyhedral` branches), and the full CTest suite was run. **190/190
pass**, unchanged, across every existing example (mlp, cnn, cnn_stl10,
attention, attn_text_classifier, matmul, mlp_placement) -- confirms both
passes are safe (as expected; they're semantics-preserving), but does not
by itself show a speedup. This confirms the passes are a legal, low-risk
addition -- it's the performance case that doesn't hold up.

**Performance** (`opt_passes_sweep.sh`, `matmul_bench.mlir`, a hard
1024x1024x1024 matmul, kernel-timed with `rtclock` -- the same method the
now-removed polyhedral perf sweep used):

| pipeline | GFLOP/s |
|---|---|
| default | 0.78 |
| default + canonicalize | 0.76-0.78 |
| default + LICM | 0.77 |
| default + canonicalize + LICM | 0.75 |
| polyhedral (tile only) | 1.71 |
| polyhedral + canonicalize | 1.68 |
| polyhedral + affine-LICM | 1.68 |
| polyhedral + canonicalize + affine-LICM | 1.72 |

Every variant is within measurement noise of its baseline. Neither pass
gives a real speedup on this workload.

## Why, honestly

`linalg`'s generated loop nests (via `convert-linalg-to-loops` or
`convert-linalg-to-affine-loops`) are already structurally minimal --
perfectly nested, with no redundant address computation or genuinely
loop-invariant loads inside the hot loop for these passes to find. That
is exactly the kind of redundancy canonicalize/LICM are built to remove
from **hand-written or heavily-transformed** imperative code, which is
not what `linalg`'s own lowering produces. A one-line pass addition being
free correctness-wise does not make it a free performance win.

## Conclusion

**Not recommended for `main`.** No measured benefit on the one workload
where a benefit would be easiest to see (a hard, compute-bound matmul),
and no reason to expect a different result on the smaller example
graphs. Kept here as a negative result, not proposed as an ARCH change.
A generic **graph-level CSE pass** (deduplicating structurally identical
`GraphPass` nodes, distinct from anything tested here) remains a
separate, still-open "minimal effort" candidate -- it targets a real gap
(no such pass exists among the six frozen `GraphPasses.cpp` passes),
whereas canonicalize/LICM target a kind of redundancy Tensora's `linalg`
output does not have.
