# Future Research Directions

For a research collaboration (for example with Prof. R. Upadrasta).
These are post-V0.2. V0.2's spikes (see
[14_V0.2_CLOSEOUT.md](14_V0.2_CLOSEOUT.md)) showed there is no
single-node runtime win to open-source that PyTorch does not already
give. So these directions are framed as research, not as adoption
claims.

## The core thesis

One idea unifies all of these: **semantic axis facts, statically
verified and preserved into the IR, are optimization information.** Not
the names — the facts. For example:

```
batch  = independent          (no cross-batch dependence)
head   = independent
model  = reduction<sum>       (associative reduction)
Q.batch == K.batch == V.batch (axis identity across tensors)
head   : shardable
model  : do not shard         (it is a reduction)
```

The claim to test: these facts let a downstream compiler prove
parallelism, prove distribution legality, cut its search space, or
reject illegal schedules — things a shape-only IR must rediscover.

The framing, in order (do not lead with "beat PyTorch"):

```
more semantic knowledge
   -> less ambiguity
   -> fewer legal schedules / less search
   -> less unnecessary computation
   -> potentially better runtime
```

Use PyTorch, CoRa, and TACO as reference points, not as the thing to
beat everywhere. If Tensora beats PyTorch in one narrow important regime
(highly heterogeneous sequence lengths), that is the headline. If not,
"semantic facts reduce search or computation" is still a real result.

## Direction 1 — Semantic-guided distributed execution (strongest)

The one direction V0.2's spikes could not dominate, because it is
distributed. The analytical spike (`research/ragged_spike/shard_analysis.py`)
showed a semantic fact decides communication cost: sharding the
independent `batch` axis needs zero communication; sharding the
`hidden` reduction axis needs a large all-reduce. Feed these facts
(independence, reduction algebra, axis identity, distribution
constraints) to a distributed mapper.

- Best fit: **DISTAL** (distributed tensor algebra; it already separates
  data distribution from computation scheduling). Tensora would supply
  a verified higher-level constraint system; DISTAL derives the machine
  mapping.
- First step needs no cluster: emit chosen distribution, estimated
  communication, estimated local work; compare against a search
  baseline. A real speedup claim needs a real cluster later.

## Direction 2 — Semantic-guided polyhedral scheduling

Feed axis roles to a polyhedral scheduler as search constraints and
priorities, rather than letting it rediscover them.

- Best fit: **Tiramisu** (polyhedral, multi-target). Tensora supplies
  "batch → parallel candidate, height/width → tile candidate, channel →
  vector/reduction candidate"; Tiramisu does the legal transformation
  and schedule search.
- The professor's **sub-polyhedral (UTVPI) scheduling** (POPL 2013;
  IMPACT 2011/2012) is the technique for keeping that search fast.
- Ragged/variable-length scheduling connects here; see
  [`research/ragged_spike/V0.3_PITCH.md`](../../research/ragged_spike/V0.3_PITCH.md).

## Direction 3 — Verification as the product

Un-dominated by any V0.2 spike, because the spikes disprove a speed
niche, not a correctness one. Static named-axis / contract checking that
PyTorch does not do: catch wrong-axis contractions, computing over
padding, illegal fusion of same-shape-different-meaning tensors. V0.1's
semantic analyzer already demonstrates the base (`demos/03`). A
`fusion-legality` experiment: two tensors with the same shape but
different logical axes (`A[batch,hidden]` vs `B[hidden,batch]`) — Tensora
rejects the illegal transform statically; a shape-only view cannot.

## Direction 4 — Quantization-axis semantics

The same tensor shape does not fix legal or efficient low-level
transforms. `W[out,in]` with `quantization_axis = out` vs `= in` are
both valid shapes but mean different per-channel quantization. Test
whether Tensora can carry `out = quantization axis` through
matmul → bias → activation → requantize without losing the
correspondence. A second, non-ragged domain for the same thesis.

## Direction 5 — Semantics-preserving multi-target ML compiler

Evolve Tensora from a language compiler into a model-to-deployment
compiler with one semantic IR and many importers/backends:

```
PyTorch / ONNX / TF  ->  Tensora Semantic IR  ->  CPU / GPU / Web / MCU / Apple
```

Do not pitch this as "convert models to every platform" — that is
**IREE** and **TVM** already. The research novelty must be specific: the
IR preserves axis identity, quantization contracts, layout, precision,
sparsity, raggedness, and distribution as explicit facts, reused by
target backends instead of rediscovered. Paper question: does preserving
these facts across the high-level-to-low-level boundary cut compiler
search cost or improve generated code across heterogeneous targets,
versus a conventional importer pipeline?

## The recommended single experiment family

Rather than six unrelated experiments, one benchmark family: **attention
with increasing semantic information.** Measure what each layer buys.

| version | new information | expected effect |
|---|---|---|
| plain shape | none | baseline |
| named axes | axis identity | better verification |
| axis roles | reduction / independence | fewer legal-schedule candidates |
| cross-tensor identity | axis equivalence | safer fusion / distribution |
| ragged | variable extent | less actual computation |

Metrics beyond runtime: transformations considered, transformations
rejected, dependence analyses performed, IR rewrites, compile time,
communication volume, kernel launches, memory traffic. These test the
semantic-IR thesis more cleanly than trying to beat a GEMM kernel.

## Prior art to cite (do not reinvent)

- **CoRa** — ragged tensor compilation with minimal padding, MLSys 2022.
  1.6x over PyTorch (Transformer encoder, V100), 1.37x over TensorFlow
  (MHA, 64-core ARM). Extends tensor-compiler abstractions to understand
  dimension relationships and variable loop bounds.
  https://www.pdl.cmu.edu/PDL-FTP/BigLearning/MLSys-2022-the-cora-tensor-compiler.pdf
- **TACO** — the Tensor Algebra Compiler, OOPSLA 2017. Compiles
  dense/sparse tensor algebra from index notation; index roles (free vs
  contraction) drive the generated loops. http://tensor-compiler.org
- **DISTAL** — distributed tensor algebra; explicit mapping of tensor
  dimensions to machine dimensions, with scheduling for tiling,
  distribution, and communication.
- **Tiramisu** — polyhedral compiler; loop and data-layout transforms
  for CPU/GPU/FPGA/distributed.
- **Sub-polyhedral scheduling** — Upadrasta & Cohen, POPL 2013; IMPACT
  2011/2012. Fast (UTVPI) scheduling.
- **PolyDL** — polyhedral optimizations for DL primitives.
- **IREE, TVM** — the existing multi-target ML compiler baselines for
  Direction 5.
- **"A Faster Generalized Two-Stage Approximate Top-K"** (TMLR 2026) —
  an example where axis roles (reduction axis vs top-k axis) decide
  fusion and which accelerator subsystem is the bottleneck. Supports the
  "axis semantics as optimization information" thesis.

## Appendix — what a polyhedral compiler does (reference)

For `for i,j: C[i,j] = A[i,j] + B[i,j]`, the compiler treats each run as
a point `(i,j)` in the iteration space and transforms it while keeping
dependencies legal:

- **Tiling** — split into blocks (e.g. 32x32) so a small region stays in
  cache/SRAM.
- **Interchange** — swap loop order (`i,j` -> `j,i`) when dependencies
  allow, to match memory layout.
- **Fusion** — merge two loops over the same space so an intermediate is
  consumed where produced, avoiding materializing it.
- **Parallelization** — run independent points on different threads
  (legal here; illegal for `A[i]=A[i-1]+1`).
- **Distribution** — split points across machines; legal with no
  cross-partition dependence, else the compiler must add communication
  or reject the split.

The semantic facts above are what let a compiler know, up front, which
of these are legal and worth doing.
