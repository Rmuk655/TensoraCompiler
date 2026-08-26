# V0.3 research item: an explicit, searchable schedule (Halide/TVM/PolyMage precedent)

## The idea

A recurring, proven pattern across TC (arXiv:1802.04730), Halide, TVM
(TensorIR + MetaSchedule), and PolyMage: separate the correctness-
critical **algorithm** (what is computed) from the performance-only
**schedule** (tile sizes, fusion choices, loop order, vectorization
width), then search the schedule space instead of hand-picking one.

Tensora does not have this separation today. `LoopTiling.cpp`'s
`TilingDecision` picks **one fixed tile size**, from a static-shape
heuristic, with no schedule object and no search. The `--polyhedral`
flag (ARCH-018, `main`) is the same shape: one hard-coded transform
(`affine-loop-tile` at its default size), not a chosen point in a space.

## Why this is a real gap, not a restatement of the polyhedral spike

The already-run perf sweep (numbers folded into
`../polyhedral_spike/SEMANTIC_POLYHEDRAL.md`, Finding 2) found tile=32 +
vectorize=16 gives ~18x over the naive default on a hard matmul, and
tile alone gives only ~2.6x. That sweep was **manual** -- a person tried
a handful of `(tile, vector-width)` pairs and read off the best one. A
real schedule-search system automates exactly that: define the space
(tile sizes, vector widths, loop order), and search it (grid, random, or
a cost model), the way TVM's MetaSchedule or Halide's autoscheduler do.

## The research question

Does representing Tensora's lowering choices as an explicit, separate
schedule object, searched automatically, find schedules that are as
good as (or better than) hand-picked ones from the same space, at a
search cost cheap enough to be worth it -- and does knowing Tensora's
own verified facts (shape, axis role, whether an axis is a reduction)
let the search be **pruned** cheaper than a generic shape-blind search
over the same space could manage?

The second half is what would make this a Tensora-specific result and
not just "TVM again": a generic autotuner (TVM MetaSchedule) already
searches schedule spaces without any static semantic verification. The
open question is whether Tensora's contracts *cut the search*, not
whether search itself is novel (it is not).

## Minimal feasibility spike (do this before building anything)

Reuse `../polyhedral_spike`'s benchmark method (`rtclock`-timed kernel,
1024x1024x1024 matmul) to grid-search `(tile-size, vector-width)` pairs
automatically via a script, instead of by hand. Compare: (a) time to
find a schedule within some tolerance of the best one a human found by
hand, (b) whether a coarse legality/shape check (Tensora already has
the shape; a reduction axis's size bounds a sane tile-size range) prunes
the grid meaningfully before timing anything.

## Falsification

Kill this if the manual sweep already found the near-optimal point in a
handful of tries (as the polyhedral spike's own numbers suggest it may
have) -- if a small, cheap manual sweep is already competitive with an
automated search, building a general schedule-search system is not
worth its own engineering cost for a language this small.

## Alpha language note (Rajopadhye), for context, not a direct borrow

The Alpha language (systems of affine recurrence equations, SAREs) is
the older academic precedent for compiling a functional, index-based
specification through polyhedral techniques into scheduled code -- a
lineage that runs through Alpha -> Halide/PolyMage/TC's "algorithm vs
schedule" split above. No direct borrowing is planned; it is cited here
only as the historical root of the idea, alongside the
Tensor-Comprehensions item (`../tc_integration/PROPOSAL.md`), which is
the more directly actionable relative of this one (TC bundles verified-
enough index notation *and* schedule search; this item is the schedule-
search half on its own, decoupled from the index-notation question).

## Relationship to other V0.3 items

Part 1a (`docs/ProjectMgmt/11_OPEN_SOURCE_AND_RESEARCH_ROADMAP.md` on
`main`, §3): another candidate test of whether Tensora's semantic
knowledge changes an optimization decision a semantics-blind system
cannot reach as cheaply. Builds on `../polyhedral_spike/` (the backend
and the numbers) and is a sibling to `../tc_integration/PROPOSAL.md`
(index-notation extensibility), not a duplicate of it.
