# V0.3 direction: semantic-guided polyhedral compilation for ML regions

This is a V0.3 research direction. It is not a v0.1 feature. It builds on
the spikes in this directory ([README.md](README.md) and the perf sweep)
and on the ragged and extent work in
[../ragged_spike/](../ragged_spike/PROPOSAL.md).

## The question, stated carefully

Polyhedral compilation is an old and active area. LLVM/Polly, Pluto,
Tiramisu, PolyDL, and the MLIR affine dialect already derive legal loop
transformations from iteration domains, affine maps, and dependences. The
MLIR `linalg` dialect already carries the iteration-to-data maps, the
iterator kinds, and the accesses. So tiling and producer-consumer fusion
already work on that information.

So do not frame the project as this:

> "We add axis names and shapes to polyhedral compilation."

That is not enough, and it is not new. The perf sweep in this directory
gets about an 18x speedup with no Tensora semantics at all. That is good
engineering, not new research.

Frame the project as this instead:

> What semantic information does an ML program know that a generic
> polyhedral representation does not? Can that information change the
> optimization decision and still keep the ability to prove legality?

## Prior art that bounds the novelty (compare, do not reinvent)

You must compare against these systems. Each already covers part of the
space:

| system | what it already does | do not claim as new |
|---|---|---|
| MLIR Affine / Linalg | iteration domains, affine maps, iterator kinds, tiling, fusion | "we bring polyhedral structure to tensors" |
| TVM TensorIR | explicit `spatial` vs `reduce` block axes; schedule transforms; MetaSchedule search | "we label axes spatial/reduction to guide scheduling" |
| TACO | separates compute from storage; split, collapse, reorder; sparse formats | "we do iteration-space transforms for tensor algebra" |
| Tiramisu | polyhedral compiler for dense/sparse data-parallel code; CPU/GPU/FPGA/distributed | "we use polyhedral transforms on tensor programs" |
| TorchInductor | fusion scheduling by memory traffic, graph distance, backend rules | "we add semantics-aware fusion" |
| JAX / StableHLO | symbolic shapes, shape constraints, runtime shape checks | "we add shape correctness" |

TensorIR is the closest prior art. It already shows that `spatial` and
`reduce` axis facts help scheduling. So treat axis roles as known, not
new.

## Axis names alone are not enough

A name such as `batch` or `sequence` gives the optimizer little. If the
lowered form is only `d0, d1, d2`, the name is just documentation. The
[../ragged_spike/tiramisu/islpy_ragged.py](../ragged_spike/tiramisu/islpy_ragged.py)
result confirms this: dependence analysis already finds independence and
reduction structure from the loops.

The value is in the semantic **properties** on the axes, for example:

```
batch:     independent, parallel
head:      independent, uniform
sequence:  ordered, maybe causal
feature:   reduction / contraction candidate
spatial:   locality-sensitive
reduction: associative operation
```

Two facts of this kind that a shape-only system does not have:

- Numeric equality is not semantic interchangeability. If `batch = 64` and
  `feature = 64`, a shape-only view may treat the two axes as the same.
  The semantics keep them apart, so the compiler does not swap them.
- Collapse only when the contract permits it. If the operation is
  independent across `batch` and `head`, the compiler can collapse
  `[B,H,S,D]` to `[B*H,S,D]` for a lower-level kernel. It may do this only
  because the semantics prove the two axes are independent instances of
  the same computation.

The high-value facts (hard to rebuild after lowering) are these:

| semantic fact | value |
|---|---|
| axis role (batch/head/sequence/feature) | low on its own; TensorIR has it |
| `L[b]` is the valid extent of an axis | interesting |
| `Lout[b] = f(Lin[b])`, verified | more interesting |
| offsets are verified prefix sums of `L[b]` | interesting for packed data |
| work for item `b` is about `L[b]^2` | interesting for load balance |

The last row is the ragged-distribution result
([../ragged_spike/distal/ragged_balance.py](../ragged_spike/distal/ragged_balance.py)).
Equal item counts can give unequal work. The semantic extents let a
scheduler balance by `sum L[b]^2`. The padded shape does not carry this.

## Four candidate contributions

### A. Semantic-contract-guided optimization

Make the shape, the named axes, the axis roles, the operator algebra, and
the provenance into one explicit contract. A scheduler then asks a
sharper question: which transformations are legal AND semantically good?
This is more specific than "shape-aware optimization".

### B. Optimization provenance

Give every transformation a provenance chain, for example
`VF-op-27 -> MLIR-op-82 -> affine-loop-12 -> LLVM-region-41`. After a
profile run, map a hot kernel back to a source concept, for example an
attention head projection. Then answer: which high-level choice caused
this hardware behavior?

### C. Counterfactual optimization (search with semantic pruning)

Compile the same typed graph several ways (schedule A, B, C), profile
each, and pick the winner. Search over schedules is not new (TVM
MetaSchedule). The open question is narrower: can semantic contracts cut
the search space and help a schedule choice transfer across shapes and
hardware?

### D. Semantic optimization failure detection

This uses the fusion bug directly. The source language knows facts such as
`padding = 1`, the conv output shape, and "interior values are not the
pad value". After an MLIR transformation, a semantic validator can check
that the transformed IR still keeps these invariants:

```
optimizer -> semantic validator -> accept / reject transformation
```

The fusion bug (Finding 3 above, filed in `mlir_bug_report/`) is a real
motivating failure: a verifier-valid transform gave a wrong result. Check
the literature on translation validation and compiler correctness before
you claim novelty.

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

If the schedule and IR are the same, the result is a type or checking
result. If semantics ON allows a legal or better schedule that OFF cannot,
the result is a compiler result. This is the A/B method of
[../ragged_spike/PROPOSAL.md](../ragged_spike/PROPOSAL.md).

## What to measure

Compare the baseline (MLIR/polyhedral optimizer) against the
semantic-guided version (contracts plus the same optimizer):

1. number of candidate schedules
2. compile and search time
3. count of invalid transformations detected
4. performance
5. memory traffic
6. behavior across shapes
7. behavior across hardware

## The paper-sized question

Do not ask "does semantic tensor information improve polyhedral
compilation" (too broad). Ask this:

> Can source-level tensor semantics -- axis roles, dependence classes,
> reduction semantics, and shape constraints -- be kept as explicit
> optimization contracts, and used to prune or validate lower-level
> transformation choices?

## Architecture for the extension

```
                 Tensora (VeriFlow)
                        |
                 Semantic analysis
                        |
          +-------------+-------------+
          |                           |
      tensor facts               semantic facts
   shape / rank / dtype       axis role / dependence
          |                    algebraic properties
          +-------------+-------------+
                        |
                Typed semantic IR
                        |
                Optimization planner
                        |
          +-------+-----+-----+-------+
          v       v           v
       fusion   tiling      layout
          +-------+-----------+
                        |
                       MLIR
                        |
              transformation space
                   |          |
             candidate A  candidate B
                   |          |
                 LLVM       LLVM
                   +----+-----+
                        |
                    profiler
                        |
                  provenance map
                        |
                 semantic validator
                   |          |
                accept      reject
                   |
                 result
```

## What the spike already established

- Finding 1: polyhedral tiling works on Tensora output. This is now in
  `main` (v0.1) as the `--polyhedral` flag, ARCH-018
  (`docs/architecture/DECISIONS.md`). It uses no Tensora-specific
  semantics, so it is settled infrastructure, not an open V0.3 question.

- Finding 2: naive tiling is not faster on its own; tiling plus
  vectorization wins. Measured on a hard 1024x1024x1024 matmul (kernel
  time only, `rtclock`, no JIT or process noise):

  | schedule | GFLOP/s | vs default |
  |---|---|---|
  | default (`convert-linalg-to-loops`) | 0.74 | 1.0x |
  | affine tile=16 | 1.90 | 2.6x |
  | affine tile=32 | 1.68 | 2.3x |
  | affine tile=16 + super-vectorize=16 | 13.23 | 17.9x |
  | affine tile=32 + super-vectorize=16 | 13.35 | 18.0x |

  So optimization is not "use one transformation": tiling alone gives
  about 2.6x; tiling plus vectorization gives about 18x. About 13
  GFLOP/s is still far below a tuned CPU GEMM and far below cuBLAS, so
  this is a win over the naive baseline, not over BLAS. It supports the
  "regions, not single GEMMs" point above -- do not try to out-tile
  vendor libraries; look for gains at the region level instead.

- Finding 3: fusion gave a wrong result on a padded conv. The minimal
  reproducer, the before/after IR, and a controlled experiment are filed
  upstream against LLVM/MLIR (`mlir_bug_report/`, already submitted).
  Summary: MLIR's affine producer-consumer fusion moves the `tensor.pad`
  fill loop to run after the `memref.copy` that writes the interior, so
  the fill writes the pad value over the interior again and the consumer
  reads the pad value. This is a miscompilation in upstream MLIR, not a
  Tensora bug, and not a case where a transform was legal only because
  of a semantic fact the lowered form hides. **It is not evidence for
  the semantic-legality idea in this document** -- it only shows that a
  verifier-valid transform can still be wrong, which is why any semantic
  validator (Direction D) needs its own proof, not a borrowed one.

- Finding 4: semantic information may constrain or guide a
  transformation. This is the question to pursue next.

## Name and fit

Name the direction "semantic-guided polyhedral compilation for ML
regions". Do not name it "polyhedral Tensora". Aim for one result: a
semantic fact that (1) stays at the typed-graph level, (2) the lowered
form cannot recover cheaply, and (3) changes the legality or the schedule
quality enough to give a measurable result. This fits Prof. Upadrasta's
sub-polyhedral (UTVPI) scheduling work, which keeps the schedule search
fast (see `docs/ProjectMgmt/15_FUTURE_RESEARCH_DIRECTIONS.md`).

## References

- MLIR Affine dialect: https://mlir.llvm.org/docs/Dialects/Affine/
- TVM TensorIR: https://tvm.apache.org/docs/deep_dive/tensor_ir/index.html
- TVM TensorIR learning (spatial/reduce axes):
  https://tvm.apache.org/docs/deep_dive/tensor_ir/learning.html
- TorchInductor (fusion scheduling): https://docs.pytorch.org/assets/pytorch2-2.pdf
- JAX shape polymorphism: https://docs.jax.dev/en/latest/export/shape_poly.html
- TACO: https://compilers.stanford.edu/software/taco/
- Tiramisu: https://github.com/Tiramisu-Compiler/tiramisu
