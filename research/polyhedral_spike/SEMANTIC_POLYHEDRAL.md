# V0.3 direction: semantic-guided polyhedral compilation for ML regions

This is a V0.3 research direction, not a v0.1 feature. It builds on the
spikes in this directory ([README.md](README.md), the perf sweep) and the
ragged/extent work in [../ragged_spike/](../ragged_spike/PROPOSAL.md).

## The framing (what is and is not interesting)

Polyhedral compilation is an established, still-active area. LLVM/Polly,
Pluto, Tiramisu, PolyDL and MLIR's own affine dialect already do
polyhedral transformations. So the interesting contribution is **not**
"Tensora uses polyhedral transformations." The perf sweep in this
directory shows Tensora's linalg output already feeds those transforms
and gets an ~18x speedup from tuned tiling + vectorization -- with **no
Tensora-specific semantics involved**. That is a good engineering result,
but it is not research novelty.

The interesting question is:

> Can ML-specific semantics that Tensora has **before lowering** help
> choose, constrain, or **prove the legality/profitability** of
> transformations that are hard to recover once the program is a pile of
> loops?

## Correction from our own experiments: labels alone are not enough

Tensora knows that in `C[b,m,n] = Σ_k A[b,m,k]·B[b,k,n]`, the axes
`b,m,n` index independent outputs and `k` is a reduction. But
[../ragged_spike/tiramisu/islpy_ragged.py](../ragged_spike/tiramisu/islpy_ragged.py)
showed polyhedral **dependence analysis already derives** basic
independence and reduction structure from the loops. So simply attaching
`axis=reduction` / `axis=batch` labels will most likely **not** yield
novel performance -- the backend can already find those facts.

## Where Tensora could actually add information

The candidates that are **hard or impossible to reconstruct after
lowering**, ordered by how interesting they are:

| semantic fact Tensora supplies | likely value |
|---|---|
| "`k` is reduction" | low -- dependence analysis derives it |
| "`batch` is independent" | low -- same reason |
| tensor/operator identity | useful for cost models / specialized lowering |
| axis equivalence across ops | maybe useful for fusion / layout choices |
| `L[b]` is the valid extent of this axis | **interesting** |
| `Lout[b] = f(Lin[b])`, verified | **more interesting** |
| offsets are verified prefix sums of `L[b]` | **interesting for packed/ragged** |
| work for item `b` is `~L[b]^2` | **interesting for scheduling / load balance** |

The last row is the ragged-distribution result
([../ragged_spike/distal/ragged_balance.py](../ragged_spike/distal/ragged_balance.py)):
a scheduler that sees only 32 batch items may give 8 to each of 4
workers, but if attention work is `W_b ∝ L[b]^2`, equal counts mean very
unequal work. Knowing the semantic extents lets the scheduler balance by
`Σ L[b]^2` -- information the padded `[B, Lmax, ...]` shape does not carry.

## Two research questions, kept separate

- **Verification question**: can Tensora preserve ML-specific semantic
  contracts that let it **prove a transformation legal** which cannot be
  established reliably from the lowered representation?
- **Performance question**: given those verified facts, can a scheduler
  choose transformations that **measurably outperform** the same backend
  *without* those facts?

## The experiment that isolates the contribution (the ablation)

Not "Tensora vs PyTorch." The same program, the same polyhedral backend,
semantics **OFF** vs **ON**:

```
             SAME PROGRAM
                  |
          +-------+-------+
          v               v
      semantic         semantics
      facts OFF          ON
          |               |
          +---- same polyhedral backend ----+
                          |
                   compare: legality? schedule? runtime?
```

Outcomes: identical schedule/IR -> the result is a type/checking result,
not a codegen one. Semantics ON enables a legal transform or a better
schedule that OFF cannot -> a compiler result. This is the same A/B
discipline as [../ragged_spike/PROPOSAL.md](../ragged_spike/PROPOSAL.md).

## The fusion failure is a lead, not yet evidence

The `affine-loop-fusion` failure on `cnn_stl10` (see [README.md](README.md))
is suggestive but does **not yet** prove that Tensora semantics would fix
it. The wrong result could be a lowering issue, a pass precondition, an
alias/memory-effect gap, wrong pass composition, or a plain bug. **It
must be diagnosed first.** The scientifically interesting shape would be:

```
generic lowered MLIR  -> cannot establish property P
Tensora TCG           -> has / verifies property P  -> fusion provably legal
```

Only then is the fusion story real evidence.

## The performance opportunity is regions, not single GEMMs

The sweep confirms the earlier lesson: do not target beating cuBLAS on a
single matmul. The opportunity is **regions** -- e.g. `matmul -> bias ->
activation -> normalization`. A library runs several kernels with
round-trips to memory between them:

```
kernel1 -> memory -> kernel2 -> memory -> kernel3
```

A compiler with enough semantic + dependence information can emit one
fused, tiled region:

```
A,B -> [ fused / tiled region ] -> output
```

The gain is not faster multiply; it is removing intermediate memory
traffic, launches, padding, and conversions. This is exactly where a
small Futhark-style Tensora kernel language
([../ragged_spike/POSITIONING.md](../ragged_spike/POSITIONING.md)) fits.

## Target phrasing and fit

Frame the direction as **semantic-guided polyhedral compilation for ML
regions**, not "polyhedral Tensora." The key result to aim for: a
specific semantic fact that (1) survives at the TCG level, (2) cannot be
cheaply/reliably recovered after lowering, and (3) changes transformation
**legality or schedule quality** enough to produce a measurable result.
This fits Prof. Upadrasta's sub-polyhedral (UTVPI) scheduling work, which
is the technique for keeping the schedule search fast (see
`docs/ProjectMgmt/15_FUTURE_RESEARCH_DIRECTIONS.md`).
