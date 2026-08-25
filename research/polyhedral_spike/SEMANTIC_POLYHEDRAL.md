# V0.3 direction: semantic-guided polyhedral compilation for ML regions

This is a V0.3 research direction. It is not a v0.1 feature. It builds on
the spikes in this directory ([README.md](README.md) and the perf sweep)
and on the ragged and extent work in
[../ragged_spike/](../ragged_spike/PROPOSAL.md).

## The idea: what is new and what is not

Polyhedral compilation is an old and active area. LLVM/Polly, Pluto,
Tiramisu, PolyDL, and the MLIR affine dialect already do polyhedral
transformations. So the new part is not "Tensora uses polyhedral
transformations."

The perf sweep in this directory proves this point. Tensora's linalg
output already feeds those transformations. Tuned tiling and
vectorization give about an 18x speedup. No Tensora semantics are needed
for this. This is good engineering, but it is not new research.

The research question is different:

> Can ML semantics that Tensora holds before lowering help the compiler
> choose a transformation, limit a transformation, or prove that a
> transformation is legal or profitable, when the lowered loops no longer
> show those facts?

## A correction from our own experiments: labels are not enough

Tensora knows the axes of `C[b,m,n] = sum_k A[b,m,k] * B[b,k,n]`. The
axes `b`, `m`, and `n` index independent outputs. The axis `k` is a
reduction. But
[../ragged_spike/tiramisu/islpy_ragged.py](../ragged_spike/tiramisu/islpy_ragged.py)
shows that polyhedral dependence analysis already finds the basic
independence and reduction structure from the loops.

So a simple label such as `axis=reduction` or `axis=batch` most likely
gives no new performance. The backend can already find these facts.

## Where Tensora can add information

These facts are hard or impossible to rebuild after lowering. The list
goes from low value to high value:

| semantic fact Tensora supplies | likely value |
|---|---|
| "`k` is a reduction" | low; dependence analysis finds it |
| "`batch` is independent" | low; same reason |
| tensor or operator identity | useful for cost models and special lowering |
| axis equivalence across ops | maybe useful for fusion or layout |
| `L[b]` is the valid extent of an axis | **interesting** |
| `Lout[b] = f(Lin[b])`, verified | **more interesting** |
| offsets are verified prefix sums of `L[b]` | **interesting for packed data** |
| work for item `b` is about `L[b]^2` | **interesting for load balance** |

The last row is the ragged-distribution result
([../ragged_spike/distal/ragged_balance.py](../ragged_spike/distal/ragged_balance.py)).
A scheduler that sees only 32 batch items can give 8 items to each of 4
workers. But attention work is about `L[b]^2`. So equal item counts can
give very unequal work. The semantic extents let the scheduler balance by
`sum L[b]^2`. The padded shape `[B, Lmax, ...]` does not carry this fact.

## Two research questions, kept apart

- Verification question: can Tensora keep ML semantic contracts that let
  it prove a transformation legal, when the lowered form cannot show that
  the transformation is legal?
- Performance question: with those verified facts, can a scheduler choose
  transformations that run faster than the same backend without the facts?

## The experiment that isolates the contribution

Do not compare "Tensora against PyTorch". Use the same program and the
same polyhedral backend. Turn the semantics OFF, then ON:

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

Read the results this way. If the schedule and IR are the same, the result
is a type or checking result, not a codegen result. If the semantics ON
case allows a legal transformation or a better schedule that the OFF case
cannot, the result is a compiler result. This is the same A/B method as
[../ragged_spike/PROPOSAL.md](../ragged_spike/PROPOSAL.md).

## The fusion failure is diagnosed, and it is not evidence

We diagnosed the `affine-loop-fusion` failure on `cnn_stl10` (see
[FUSION_DIAGNOSIS.md](FUSION_DIAGNOSIS.md)). The smallest trigger is one
padded `conv2d`. The MLIR affine producer-consumer fusion reorders the
`tensor.pad` fill after the `memref.copy` that writes the interior. So the
conv reads the pad value. This is a correctness bug in the upstream pass.
It is not a case where a transformation was legal only because of a
semantic fact that the lowered form hides.

So this failure does not support the semantic-legality idea. Do not cite
it for that idea. The useful research shape is still this:

```
generic lowered MLIR  -> cannot establish property P
Tensora TCG           -> has or verifies property P  -> transform provably legal
```

But it needs a different example. The example must be a transformation
that is legal only with a verified fact, such as an `L[b]` extent or a
prefix-sum `offsets` contract, that the lowered program cannot recover.
To find that example is the real V0.3 task. The padded-conv fusion bug is
not that example.

## The performance opportunity is regions, not single matmuls

The sweep confirms the earlier lesson. Do not try to beat cuBLAS on a
single matmul. The opportunity is a region, for example `matmul -> bias
-> activation -> normalization`. A library runs several kernels and writes
to memory between them:

```
kernel1 -> memory -> kernel2 -> memory -> kernel3
```

A compiler with enough semantic and dependence information can emit one
fused and tiled region:

```
A,B -> [ fused / tiled region ] -> output
```

The gain is not a faster multiply. The gain is less memory traffic, fewer
launches, less padding, and fewer conversions. A small Futhark-style
Tensora kernel language fits here
([../ragged_spike/POSITIONING.md](../ragged_spike/POSITIONING.md)).

## Name and fit

Name the direction "semantic-guided polyhedral compilation for ML
regions". Do not name it "polyhedral Tensora". Aim for one result: a
semantic fact that (1) stays at the TCG level, (2) the lowered form cannot
recover cheaply, and (3) changes the legality or the schedule quality
enough to give a measurable result. This fits Prof. Upadrasta's
sub-polyhedral (UTVPI) scheduling work. That work keeps the schedule
search fast (see `docs/ProjectMgmt/15_FUTURE_RESEARCH_DIRECTIONS.md`).
